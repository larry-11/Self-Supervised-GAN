import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import numpy as np
import os
from utils.tensorboard_logger import Logger
from itertools import chain
from torchvision import utils
from ops import conv2d, deconv2d, Residual_G, Residual_D
import torch.nn.functional as F
import torchvision.datasets as dset
from utils.fid_score import calculate_fid_given_images

SAVE_PER_TIMES = 200

class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1))
            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):
    def __init__(self, channels, ssup):
        super().__init__()
        self.ssup = ssup
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.main_module = nn.Sequential(
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))

        self.output2 = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=4, kernel_size=4, stride=1, padding=0))


    def forward(self, x):
        x = self.main_module(x)
        gan_logits = self.output(x).view(-1, 1)
        if self.ssup:
            rot_logits = self.output2(x).view(-1, 4)
            rot_prob = self.softmax(rot_logits)
            return self.relu(1 - gan_logits), self.relu(1 + gan_logits), gan_logits, rot_logits, rot_prob
        else:
            return self.relu(1 - gan_logits), self.relu(1 + gan_logits), gan_logits
        # return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)


class SSLGAN_GP(object):
    def __init__(self, args):
        print("init model.")
        print(args)
        self.G = Generator(args.channels)
        self.D = Discriminator(args.channels, args.ssup)
        self.C = args.channels
        self.ssup = args.ssup
        self.loss_type = args.loss

        if self.ssup:
            self.save_path = 'sslgan_gp_ssup'
        else:
            self.save_path = 'sslgan_gp'

        # Check if cuda is available
        self.check_cuda(args.cuda)

        # WGAN values from paper
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = 64

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        # Set the logger
        self.logger = Logger('./logs')
        self.logger.writer.flush()
        self.number_of_images = 10

        self.generator_iters = args.generator_iters
        self.critic_iter = 5
        self.lambda_term = 10
        self.weight_rotation_loss_d = 1.0
        self.weight_rotation_loss_g = 0.5
        self.print_iter = 50

    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)

    def check_cuda(self, cuda_flag=False):
        print(cuda_flag)
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            print("Cuda enabled flag: {}".format(self.cuda))
        else:
            self.cuda = False

    def train(self, train_loader):
        self.t_begin = t.time()

        if not os.path.exists('{}'.format(self.save_path)):
            os.makedirs('{}'.format(self.save_path))

        # Now batches are callable self.data.next()
        self.data = self.get_infinite_batches(train_loader)

        self.frozen_latent = self.get_torch_variable(torch.randn(1000, 100, 1, 1))
        self.forzen_images = self.data.__next__()
        self.get_torch_variable(self.forzen_images)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)

        for g_iter in range(self.generator_iters):
            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True

            d_loss_real = 0
            d_loss_fake = 0
            Wasserstein_D = 0
            # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                images = self.data.__next__()
                # Check for batch to have full batch_size
                if (images.size()[0] != self.batch_size):
                    continue

                z = torch.rand((self.batch_size, 100, 1, 1))
                images, z = self.get_torch_variable(images), self.get_torch_variable(z)
                # Train discriminator
                if self.ssup:
                    # rot real image
                    x = images
                    x_90 = x.transpose(2, 3)
                    x_180 = x.flip(2, 3)
                    x_270 = x.transpose(2, 3).flip(2, 3)
                    images = torch.cat((x, x_90, x_180, x_270), 0)

                    _, __, d_loss_real, d_real_rot_logits, d_real_rot_prob = self.D(images)
                    d_loss_real = torch.mean(d_loss_real[:self.batch_size])
                    d_loss_real.backward(mone, retain_graph=True)

                    z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
                    fake_images = self.G(z)

                    # rot fake image
                    x = fake_images
                    x_90 = x.transpose(2, 3)
                    x_180 = x.flip(2, 3)
                    x_270 = x.transpose(2, 3).flip(2, 3)
                    fake_images = torch.cat((x, x_90, x_180, x_270), 0)

                    _, __, d_loss_fake, g_fake_rot_logits, g_fake_rot_prob = self.D(fake_images)
                    d_loss_fake = torch.mean(d_loss_fake[:self.batch_size])
                    d_loss_fake.backward(one)

                    gradient_penalty = self.calculate_gradient_penalty(images[:self.batch_size].data,
                                                                       fake_images[:self.batch_size].data,
                                                                       self.ssup)
                    gradient_penalty.backward()

                    rot_labels = torch.zeros(4 * self.batch_size).cuda()
                    for i in range(4 * self.batch_size):
                        if i < self.batch_size:
                            rot_labels[i] = 0
                        elif i < 2 * self.batch_size:
                            rot_labels[i] = 1
                        elif i < 3 * self.batch_size:
                            rot_labels[i] = 2
                        else:
                            rot_labels[i] = 3

                    rot_labels = F.one_hot(rot_labels.to(torch.int64), 4).float()
                    d_real_class_loss = torch.mean(F.binary_cross_entropy_with_logits(input=d_real_rot_logits,
                                                                                      target=rot_labels)) * self.weight_rotation_loss_d
                    d_real_class_loss.backward(one)

                    d_loss = d_loss_fake - d_loss_real + gradient_penalty + d_real_class_loss

                    self.d_optimizer.step()
                    if g_iter % self.print_iter == 0:
                        print(f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss: {d_loss}, gradient_penalty: {gradient_penalty}, loss_rot: {d_real_class_loss}')

                else:
                    _, __, d_loss_real = self.D(images)
                    d_loss_real = torch.mean(d_loss_real)
                    d_loss_real.backward(mone)

                    z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
                    fake_images = self.G(z)
                    _, __, d_loss_fake = self.D(fake_images)
                    d_loss_fake = torch.mean(d_loss_fake)
                    d_loss_fake.backward(one)

                    gradient_penalty = self.calculate_gradient_penalty(images.data, fake_images.data, self.ssup)
                    gradient_penalty.backward()

                    d_loss = d_loss_fake - d_loss_real + gradient_penalty

                self.d_optimizer.step()
                if g_iter % self.print_iter == 0:
                    print(f'  Discriminator iteration: {d_iter}/{self.critic_iter}, d_loss: {d_loss}, gradient_penalty: {gradient_penalty}')

            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.G.zero_grad()

            # train generator
            z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
            fake_images = self.G(z)

            if self.ssup:
                # rot fake image
                x = fake_images
                x_90 = x.transpose(2, 3)
                x_180 = x.flip(2, 3)
                x_270 = x.transpose(2, 3).flip(2, 3)
                fake_images = torch.cat((x, x_90, x_180, x_270), 0)

                _, __, g_loss, g_fake_rot_logits, g_fake_rot_prob = self.D(fake_images)
                g_loss = g_loss[:self.batch_size].mean()
                g_loss.backward(mone, retain_graph=True)
                g_loss = -g_loss

                rot_labels = torch.zeros(4 * self.batch_size).cuda()
                for i in range(4 * self.batch_size):
                    if i < self.batch_size:
                        rot_labels[i] = 0
                    elif i < 2 * self.batch_size:
                        rot_labels[i] = 1
                    elif i < 3 * self.batch_size:
                        rot_labels[i] = 2
                    else:
                        rot_labels[i] = 3

                rot_labels = F.one_hot(rot_labels.to(torch.int64), 4).float()
                g_fake_class_loss = torch.mean(F.binary_cross_entropy_with_logits(input=g_fake_rot_logits,
                                                                                  target=rot_labels)) * self.weight_rotation_loss_g
                g_fake_class_loss.backward(one)

                g_loss += g_fake_class_loss
                self.g_optimizer.step()
                if g_iter % self.print_iter == 0:
                    print(
                        f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss}, rot_loss: {g_fake_class_loss}')

            else:
                _, __, g_loss = self.D(fake_images)
                g_loss = g_loss.mean()
                g_loss.backward(mone)
                g_cost = -g_loss
                self.g_optimizer.step()
                if g_iter % self.print_iter == 0:
                    print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss}')

            # Saving model and sampling images every 1000th generator iterations
            if (g_iter) % SAVE_PER_TIMES == 0:
                self.save_model()
                fake_images = self.G(self.frozen_latent)

                # Denormalize images and save them in grid 8x8
                z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
                samples = self.G(z)
                samples = samples.mul(0.5).add(0.5)
                samples = samples.data.cpu()[:64]
                grid = utils.make_grid(samples)
                utils.save_image(grid, '{}/img_generatori_iter_{}.png'.format(self.save_path, str(g_iter).zfill(3)))

                # Testing
                time = t.time() - self.t_begin
                # fid
                fid_value = calculate_fid_given_images(fake_images, self.forzen_images)
                f = open('{}/fid.txt'.format(self.save_path), 'a')
                f.write('Iter:{} Fid:{}\n'.format(str(g_iter), str(fid_value)))
                f.close()
                print("Generator iter: {}, Time: {}, FID: {}".format(g_iter, time, fid_value))

                # ============ TensorBoard logging ============#
                # (1) Log the scalar values
                info = {
                    # 'Wasserstein distance': Wasserstein_D.data,
                    'Loss D': d_loss.data,
                    'Loss G': g_loss.data,
                    'Loss D Real': d_loss_real.data,
                    'Loss D Fake': d_loss_fake.data
                }

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, g_iter + 1)

                # (3) Log the images
                info = {
                    'real_images': self.real_images(images, self.number_of_images),
                    'generated_images': self.generate_img(z, self.number_of_images)
                }

                for tag, images in info.items():
                    self.logger.image_summary(tag, images, g_iter + 1)

        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        # self.file.close()

        # Save the trained parameters
        self.save_model()

    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, 'dgan_model_image.png')

    def calculate_gradient_penalty(self, real_images, fake_images, ssup):
        eta = torch.FloatTensor(self.batch_size, 1, 1, 1).uniform_(0, 1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        if self.cuda:
            eta = eta.cuda(self.cuda_index)
        else:
            eta = eta

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        if self.cuda:
            interpolated = interpolated.cuda(self.cuda_index)
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        if ssup:
            _, __, prob_interpolated, ___, ____ = self.D(interpolated)
        else:
            _, __, prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(
                                      prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                      prob_interpolated.size()),
                                  create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, 32, 32)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, 32, 32)[:self.number_of_images])

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, 32, 32))
            else:
                generated_images.append(sample.reshape(32, 32))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        torch.save(self.G.state_dict(), '{}/generator.pkl'.format(self.save_path))
        torch.save(self.D.state_dict(), '{}/discriminator.pkl'.format(self.save_path))
        print('Models save to ./generator.pkl & ./discriminator.pkl ')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images

    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        number_int = 10
        # interpolate between twe noise(z1, z2).
        z_intp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1 * alpha + z2 * (1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5)  # denormalize
            images.append(fake_im.view(self.C, 32, 32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int)
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images.")
