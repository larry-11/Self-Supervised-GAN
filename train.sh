CUDA_VISIBLE_DEVICES=0 python main.py --model WGAN-GP \
                                       --is_train True \
                                       --download False \
                                       --dataroot ./datasets/cifar \
                                       --dataset cifar \
                                       --generator_iters 25000 \
                                       --cuda True \
                                       --batch_size 64 \
                                        --ssup