# Self-Supervised-GAN
A pytorch implementation of "Self-Supervised GANs via Auxiliary Rotation Loss" 

## Approach

Paper:

[Self-Supervised GANs via Auxiliary Rotation Loss](https://arxiv.org/abs/1811.11212)

[Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957)

[Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)

## DataSetUsed

- mnist
- cifar10

## Install

Most of the modules can be installed by executing the code below.

```
pip install -r requirements.txt
```

Other modules are extra options to install.

## Usage

Executing the train.sh to train the network

```
sh train.sh
```

You can also directly execute the main.py to customize your training.

```python
CUDA_VISIBLE_DEVICES=0 python main.py  --model SSLGAN_SN \
                                       --is_train True \
                                       --download False \
                                       --dataroot ./datasets/cifar \
                                       --dataset cifar \
                                       --generator_iters 25000 \
                                       --cuda True \
                                       --batch_size 64 \
                                       --ssup
```