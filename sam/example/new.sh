python train.py --gpu --task cifar10 --net vgg11y --optimizer nerov3 --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.02 --prefix div_by_sqrt_clamp_avgpool_

python train.py --gpu --task cifar10 --net resnet18y --optimizer nerov3 --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.02 --loss mse --prefix kushal_scaling_srelu_clamp5_no_bn_
