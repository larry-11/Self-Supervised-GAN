from cal_fid import calculate_fid_given_images
from dataloaders import get_cifar_dataloaders
from model import Generator
import numpy as np
import torch
np.set_printoptions(threshold=np.inf)

def cal_fid_score(images):
    # generator data
    generator_datas = images.cpu().data.numpy()
    # original data
    _, data_loader = get_cifar_dataloaders(batch_size=100)
    data_iter = iter(data_loader)
    original_datas, __ = next(data_iter)
    original_datas = original_datas.data.numpy()
    # fid
    fid = calculate_fid_given_images(generator_datas, original_datas)
    return fid

if __name__ == "__main__":
    weight = './Experiment/cifar_9.pt'

    # load model
    print('load model')
    generator = Generator(resnet = True, z_size = 128, channel = 3)
    generator.load_state_dict(torch.load(weight))
    generator.eval()

    # generate data
    print('generate data')
    z = torch.randn((100, 128))
    generator_datas = generator(z).cpu().data.numpy()

    # original data
    print('original data')
    _, data_loader = get_cifar_dataloaders(batch_size=100)
    data_iter = iter(data_loader)
    original_datas, __ = next(data_iter)
    original_datas = original_datas.data.numpy()

    # cal fid
    print('cal fid')
    fid = calculate_fid_given_images(generator_datas, original_datas)
    print('fid:', fid)

