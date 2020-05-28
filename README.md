# Self-Supervised-GAN
A pytorch implementation of "Self-Supervised GANs via Auxiliary Rotation Loss" 

## Method

![](https://github.com/larry-11/Self-Supervised-GAN/tree/master/img/0.png)

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

## Result

| **Setting** | **Model** | **Loss**  | **LR**   | **D_iter** | **ssup** |
| ----------- | --------- | --------- | -------- | ---------- | -------- |
| A1          | WGAN-GP   | Wgan-gp   | 1e-4     | 5          |          |
| A2          | WGAN-GP   | Wgan-gp   | 1e-4     | 5          | √        |
| B1          | SNGAN     | hinge     | 1e-4     | 5          |          |
| B2          | SNGAN     | hinge     | 1e-4     | 5          | √        |
| C1          | SNGAN     | hinge     | 2e-4     | 5          |          |
| C2          | SNGAN     | hinge     | 2e-4     | 5          | √        |
| **D1**      | **SNGAN** | **hinge** | **2e-4** | **1**      |          |
| **D2**      | **SNGAN** | **hinge** | **2e-4** | **1**      | √        |

![](https://github.com/larry-11/Self-Supervised-GAN/tree/master/img/1.png)