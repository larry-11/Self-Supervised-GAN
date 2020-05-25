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

class Discriminator(nn.Module):
    def __init__(self, channel, ssup):
        super(Discriminator, self).__init__()

        spectral_normed = True
        self.ssup = ssup
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.conv1 = conv2d(channel, 64, kernel_size = 3, stride = 1, padding = 1,
                            spectral_normed = spectral_normed)
        self.conv2 = conv2d(64, 128, spectral_normed = spectral_normed,
                            padding = 0)
        self.conv3 = conv2d(128, 256, spectral_normed = spectral_normed,
                            padding = 0)
        self.conv4 = conv2d(256, 512, spectral_normed = spectral_normed,
                            padding = 0)
        self.fully_connect_gan1 = nn.Linear(512, 1)
        self.fully_connect_rot1 = nn.Linear(512, 4)
        self.softmax = nn.Softmax()

        self.re1 = Residual_D(channel, 128, spectral_normed = spectral_normed,
                            down_sampling = True, is_start = True)
        self.re2 = Residual_D(128, 128, spectral_normed = spectral_normed,
                            down_sampling = True)
        self.re3 = Residual_D(128, 128, spectral_normed = spectral_normed)
        self.re4 = Residual_D(128, 128, spectral_normed = spectral_normed)
        self.fully_connect_gan2 = nn.Linear(128, 1)
        self.fully_connect_rot2 = nn.Linear(128, 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        re1 = self.re1(x)
        re2 = self.re2(re1)
        re3 = self.re3(re2)
        re4 = self.re4(re3)
        re4 = self.relu(re4)
        re4 = torch.sum(re4,dim = (2,3))
        gan_logits = self.fully_connect_gan2(re4)
        if self.ssup:
            rot_logits = self.fully_connect_rot2(re4)
            rot_prob = self.softmax(rot_logits)

        if self.ssup:
            return self.relu(1 - gan_logits), self.relu(1 + gan_logits), gan_logits, rot_logits, rot_prob
        else:
            return self.relu(1 - gan_logits), self.relu(1 + gan_logits), gan_logits



class Generator(nn.Module):
    def __init__(self, channel):
        super(Generator, self).__init__()
        # s = 4
        self.output_size = 32
        # s = self.output_size / 8
        self.s = 4
        self.z_size = 100
        self.fully_connect = nn.Linear(100, self.s * self.s * 256)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.deconv1 = deconv2d(256, 256, padding = 0)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = deconv2d(256, 128, padding = 0) 
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = deconv2d(128, 64, padding = 0)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = conv2d(64, channel, padding = 1, kernel_size = 3, stride = 1)
        self.conv_res4 = conv2d(256,channel, padding = 1, kernel_size = 3, stride = 1)

        self.re1 = Residual_G(256, 256, up_sampling = True)
        self.re2 = Residual_G(256, 256, up_sampling = True)
        self.re3 = Residual_G(256, 256, up_sampling = True)
        self.bn = nn.BatchNorm2d(256)

    def forward(self, x):
        # print(x.shape)
        x = x.view(x.size(0), -1)
        d1 = self.fully_connect(x)
        d1 = d1.view(-1, 256, self.s, self.s)
        d2 = self.re1(d1)
        d3 = self.re2(d2)
        d4 = self.re3(d3)
        d4 = self.relu(self.bn(d4))
        d5 = self.conv_res4(d4)

        return self.tanh(d5)

class SSLGAN_SN(object):
    def __init__(self, args):
        print("init model.")
        print(args)
        self.G = Generator(args.channels)
        self.D = Discriminator(args.channels, args.ssup)
        self.C = args.channels
        self.ssup = args.ssup
        self.loss_type = args.loss
        
        if self.ssup:
            self.save_path = 'sslgan_sn_ssup'
        else:
            self.save_path = 'sslgan_sn'

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
                # fake_images = self.G(z)
                # Train discriminator
                # WGAN - Training discriminator more iterations than generator
                if self.ssup:
                    # rot real image
                    x = images
                    x_90 = x.transpose(2,3)
                    x_180 = x.flip(2,3)
                    x_270 = x.transpose(2,3).flip(2,3)
                    images = torch.cat((x,x_90,x_180,x_270),0)

                    d_loss_real, _, __, d_real_rot_logits, d_real_rot_prob = self.D(images)
                    d_loss_real = torch.mean(d_loss_real[:self.batch_size])
                    d_loss_real.backward(one, retain_graph=True)

                    z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
                    fake_images = self.G(z)

                    # rot fake image
                    x = fake_images
                    x_90 = x.transpose(2,3)
                    x_180 = x.flip(2,3)
                    x_270 = x.transpose(2,3).flip(2,3)
                    fake_images = torch.cat((x, x_90, x_180, x_270),0)

                    _, d_loss_fake, __, g_fake_rot_logits, g_fake_rot_prob = self.D(fake_images)
                    d_loss_fake = torch.mean(d_loss_fake[:self.batch_size])
                    d_loss_fake.backward(one)

                    rot_labels = torch.zeros(4*self.batch_size).cuda()
                    for i in range(4*self.batch_size):
                        if i < self.batch_size:
                            rot_labels[i] = 0
                        elif i < 2*self.batch_size:
                            rot_labels[i] = 1
                        elif i < 3*self.batch_size:
                            rot_labels[i] = 2
                        else:
                            rot_labels[i] = 3

                    rot_labels = F.one_hot(rot_labels.to(torch.int64), 4).float()
                    d_real_class_loss = torch.mean(F.binary_cross_entropy_with_logits(input = d_real_rot_logits, target = rot_labels)) * self.weight_rotation_loss_d
                    d_real_class_loss.backward(one)

                    d_loss = d_loss_real + d_loss_fake + d_real_class_loss

                    self.d_optimizer.step()
                    if g_iter % self.print_iter == 0:
                        print(f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss: {d_loss}, loss_rot: {d_real_class_loss}')

                else:
                    d_loss_real, _, __ = self.D(images)
                    d_loss_real = torch.mean(d_loss_real)
                    d_loss_real.backward(one)

                    z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
                    fake_images = self.G(z)
                    _, d_loss_fake, __ = self.D(fake_images)
                    d_loss_fake = torch.mean(d_loss_fake)
                    d_loss_fake.backward(one)

                    d_loss = d_loss_real + d_loss_fake

                    self.d_optimizer.step()
                    if g_iter % self.print_iter == 0:
                        print(f'  Discriminator iteration: {d_iter}/{self.critic_iter}, d_loss: {d_loss}')

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
                x_90 = x.transpose(2,3)
                x_180 = x.flip(2,3)
                x_270 = x.transpose(2,3).flip(2,3)
                fake_images = torch.cat((x, x_90, x_180, x_270),0)

                _, __, g_loss, g_fake_rot_logits, g_fake_rot_prob = self.D(fake_images)
                g_loss = g_loss[:self.batch_size].mean()
                g_loss.backward(mone, retain_graph=True)
                g_loss = -g_loss

                rot_labels = torch.zeros(4*self.batch_size).cuda()
                for i in range(4*self.batch_size):
                    if i < self.batch_size:
                        rot_labels[i] = 0
                    elif i < 2*self.batch_size:
                        rot_labels[i] = 1
                    elif i < 3*self.batch_size:
                        rot_labels[i] = 2
                    else:
                        rot_labels[i] = 3
                
                rot_labels = F.one_hot(rot_labels.to(torch.int64), 4).float()
                g_fake_class_loss = torch.mean(F.binary_cross_entropy_with_logits(input = g_fake_rot_logits, target = rot_labels)) * self.weight_rotation_loss_g
                g_fake_class_loss.backward(one)

                g_loss += g_fake_class_loss
                self.g_optimizer.step()
                if g_iter % self.print_iter == 0:
                    print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss}, rot_loss: {g_fake_class_loss}')
                
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
        #self.file.close()

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
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            images.append(fake_im.view(self.C,32,32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int )
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images.")
