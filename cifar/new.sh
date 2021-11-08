python train.py --gpu --task cifar10 --net vgg11y --optimizer nerov3 --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.02 --prefix div_by_sqrt_clamp_avgpool_

python train.py --gpu --task cifar10 --net resnet18y --optimizer nerov3 --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.02 --loss mse --prefix kushal_scaling_srelu_clamp5_no_bn_

python train.py --gpu --task cifar10 --net vgg11 --optimizer nerov3 --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0


python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --spread 10 --t 0.0 --max_p 1.0 --sch 2



====GPU 0
CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 0 --t 0.0 --max_p 1.0 

CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 2 --t 0.0 --max_p 1.0

CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 4 --t 0.0 --max_p 1.0

CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 6 --t 0.0 --max_p 1.0

CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 8 --t 0.0 --max_p 1.0

CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 10 --t 0.0 --max_p 1.0

CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 12 --t 0.0 --max_p 1.0

CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 14 --t 0.0 --max_p 1.0

CUDA_VISIBLE_DEVICES=0 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 16 --t 0.0 --max_p 1.0

sleep 1h

====GPU 1
CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 0 --t 0.1 --max_p 1.0 

CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 2 --t 0.1 --max_p 1.0

CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 4 --t 0.1 --max_p 1.0

CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 6 --t 0.1 --max_p 1.0

CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 8 --t 0.1 --max_p 1.0

CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 10 --t 0.1 --max_p 1.0

CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 12 --t 0.1 --max_p 1.0

CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 14 --t 0.1 --max_p 1.0

CUDA_VISIBLE_DEVICES=1 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 16 --t 0.1 --max_p 1.0

sleep 1h

====GPU 2
CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 0 --t 0.2 --max_p 1.0 

CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 2 --t 0.2 --max_p 1.0

CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 4 --t 0.2 --max_p 1.0

CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 6 --t 0.2 --max_p 1.0

CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 8 --t 0.2 --max_p 1.0

CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 10 --t 0.2 --max_p 1.0

CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 12 --t 0.2 --max_p 1.0

CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 14 --t 0.2 --max_p 1.0

CUDA_VISIBLE_DEVICES=2 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 16 --t 0.2 --max_p 1.0

sleep 1h

====GPU 3
CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 0 --t 0.3 --max_p 1.0 

CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 2 --t 0.3 --max_p 1.0

CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 4 --t 0.3 --max_p 1.0

CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 6 --t 0.3 --max_p 1.0

CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 8 --t 0.3 --max_p 1.0

CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 10 --t 0.3 --max_p 1.0

CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 12 --t 0.3 --max_p 1.0

CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 14 --t 0.3 --max_p 1.0

CUDA_VISIBLE_DEVICES=3 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 16 --t 0.3 --max_p 1.0

sleep 1h

====GPU 4
CUDA_VISIBLE_DEVICES=4 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 0 --t 0.4 --max_p 1.0 

CUDA_VISIBLE_DEVICES=4 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 2 --t 0.4 --max_p 1.0

CUDA_VISIBLE_DEVICES=4 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 4 --t 0.4 --max_p 1.0

CUDA_VISIBLE_DEVICES=4 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 6 --t 0.4 --max_p 1.0

CUDA_VISIBLE_DEVICES=4 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 8 --t 0.4 --max_p 1.0

CUDA_VISIBLE_DEVICES=4 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 10 --t 0.4 --max_p 1.0

CUDA_VISIBLE_DEVICES=4 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 12 --t 0.4 --max_p 1.0

CUDA_VISIBLE_DEVICES=4 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 14 --t 0.4 --max_p 1.0

CUDA_VISIBLE_DEVICES=4 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 16 --t 0.4 --max_p 1.0

sleep 1h

====GPU 5
CUDA_VISIBLE_DEVICES=5 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 0 --t 0.6 --max_p 1.0 

CUDA_VISIBLE_DEVICES=5 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 2 --t 0.6 --max_p 1.0

CUDA_VISIBLE_DEVICES=5 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 4 --t 0.6 --max_p 1.0

CUDA_VISIBLE_DEVICES=5 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 6 --t 0.6 --max_p 1.0

CUDA_VISIBLE_DEVICES=5 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 8 --t 0.6 --max_p 1.0

CUDA_VISIBLE_DEVICES=5 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 10 --t 0.6 --max_p 1.0

CUDA_VISIBLE_DEVICES=5 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 12 --t 0.6 --max_p 1.0

CUDA_VISIBLE_DEVICES=5 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 14 --t 0.6 --max_p 1.0

CUDA_VISIBLE_DEVICES=5 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 16 --t 0.6 --max_p 1.0

sleep 1h

====GPU 6
CUDA_VISIBLE_DEVICES=5 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 0 --t 0.8 --max_p 1.0 

CUDA_VISIBLE_DEVICES=5 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 2 --t 0.8 --max_p 1.0

CUDA_VISIBLE_DEVICES=5 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 4 --t 0.8 --max_p 1.0

CUDA_VISIBLE_DEVICES=5 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 6 --t 0.8 --max_p 1.0

CUDA_VISIBLE_DEVICES=5 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 8 --t 0.8 --max_p 1.0

CUDA_VISIBLE_DEVICES=5 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 10 --t 0.8 --max_p 1.0

CUDA_VISIBLE_DEVICES=5 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 12 --t 0.8 --max_p 1.0

CUDA_VISIBLE_DEVICES=5 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 14 --t 0.8 --max_p 1.0

CUDA_VISIBLE_DEVICES=5 python train.py --gpu --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0 --lr 0.02 --da 1 --loss st --sch 2 --spread 16 --t 0.8 --max_p 1.0

sleep 1h
