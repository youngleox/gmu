python train.py --gpu --task cifar10 --net vgg11y --optimizer nerov3 --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.02 --prefix div_by_sqrt_clamp_avgpool_

python train.py --gpu --net vgg11 --optimizer nero --lr 0.02 --sch 2 --da 1 --t 0.1 --spread 10 --max_p 1.0 --t2 0.0 --spread2 4 --max_p2 1.0

CUDA_VISIBLE_DEVICES=0 python train.py --gpu --net vgg11 --optimizer nero --lr 0.02 --sch 2 --da 1 --t 0.0 --spread 8 --max_p 1.0 --t2 0.0 --spread2 8 --max_p2 1.0 --n 4 --seed 0


CUDA_VISIBLE_DEVICES=2 python train.py --gpu --net vgg11 --optimizer nero --lr 0.02 --sch 2 --da 1 --t 0.0 --spread 8 --max_p 1.0 --t2 0.0 --spread2 8 --max_p2 1.0 --n 4 --seed 1


CUDA_VISIBLE_DEVICES=4 python train.py --gpu --net vgg11 --optimizer nero --lr 0.02 --sch 2 --da 1 --t 0.0 --spread 8 --max_p 1.0 --t2 0.0 --spread2 8 --max_p2 1.0 --n 4 --seed 2


CUDA_VISIBLE_DEVICES=1 python train.py --gpu --net vgg11 --optimizer nero --lr 0.02 --sch 2 --da 1 --t 0.0 --spread 12 --max_p 1.0 --t2 0.0 --spread2 12 --max_p2 1.0 --n 4 --seed 0 --prefix new_sample_


CUDA_VISIBLE_DEVICES=3 python train.py --gpu --net vgg11 --optimizer nero --lr 0.02 --sch 2 --da 1 --t 0.0 --spread 12 --max_p 1.0 --t2 0.0 --spread2 12 --max_p2 1.0 --n 4 --seed 1 --prefix new_sample_


CUDA_VISIBLE_DEVICES=5 python train.py --gpu --net vgg11 --optimizer nero --lr 0.02 --sch 2 --da 1 --t 0.0 --spread 12 --max_p 1.0 --t2 0.0 --spread2 12 --max_p2 1.0 --n 4 --seed 2 --prefix new_sample_



CUDA_VISIBLE_DEVICES=1 python train.py --gpu --net resnet18 --optimizer sgd --lr 0.1 --momentum 0.9 --sch 2 --da 1 --t 0.0 --spread 8 --max_p 1.0 --t2 0.0 --spread2 8 --max_p2 1.0 --n 4 --prefix new_sample_baseline_ --loss st --seed 0

CUDA_VISIBLE_DEVICES=1 python train.py --gpu --net resnet18 --optimizer sgd --lr 0.1 --momentum 0.9 --sch 2 --da 1 --t 0.0 --spread 8 --max_p 1.0 --t2 0.0 --spread2 8 --max_p2 1.0 --n 4 --prefix new_sample_baseline_ --loss st --seed 1

CUDA_VISIBLE_DEVICES=1 python train.py --gpu --net resnet18 --optimizer sgd --lr 0.1 --momentum 0.9 --sch 2 --da 1 --t 0.0 --spread 8 --max_p 1.0 --t2 0.0 --spread2 8 --max_p2 1.0 --n 4 --prefix new_sample_baseline_ --loss st --seed 2

sleep 1h

CUDA_VISIBLE_DEVICES=3 python train.py --gpu --net resnet18 --optimizer sgd --lr 0.1 --momentum 0.9 --sch 2 --da 0 --max_p 0.9 --n 4 --prefix ls_baseline_ --loss sce --seed 0

CUDA_VISIBLE_DEVICES=3 python train.py --gpu --net resnet18 --optimizer sgd --lr 0.1 --momentum 0.9 --sch 2 --da 0 --max_p 0.9 --n 4 --prefix ls_baseline_ --loss sce --seed 1

CUDA_VISIBLE_DEVICES=3 python train.py --gpu --net resnet18 --optimizer sgd --lr 0.1 --momentum 0.9 --sch 2 --da 0 --max_p 0.9 --n 4 --prefix ls_baseline_ --loss sce --seed 2

sleep 1h


CUDA_VISIBLE_DEVICES=5 python train.py --gpu --net resnet18 --optimizer sgd --lr 0.1 --momentum 0.9 --sch 2 --da 0 --n 4 --prefix baseline_ --seed 0

CUDA_VISIBLE_DEVICES=5 python train.py --gpu --net resnet18 --optimizer sgd --lr 0.1 --momentum 0.9 --sch 2 --da 0 --n 4 --prefix baseline_ --seed 1

CUDA_VISIBLE_DEVICES=5 python train.py --gpu --net resnet18 --optimizer sgd --lr 0.1 --momentum 0.9 --sch 2 --da 0 --n 4 --prefix baseline_ --seed 2

sleep 1h



CUDA_VISIBLE_DEVICES=0 python train.py --gpu --net resnet18 --optimizer nerov3 --wd 0.01 --lr 0.02 --sch 2 --da 0 --n 4 --seed 0


CUDA_VISIBLE_DEVICES=2 python train.py --gpu --net resnet18 --optimizer nerov3 --wd 0.01 --lr 0.02 --sch 2 --da 0 --n 4 --seed 1


CUDA_VISIBLE_DEVICES=4 python train.py --gpu --net resnet18 --optimizer nerov3 --wd 0.01 --lr 0.02 --sch 2 --da 0  --n 4 --seed 2


== nerov3 label smoothing 0.1

CUDA_VISIBLE_DEVICES=1 python train.py --gpu --net resnet18 --optimizer nerov3 --wd 0.01 --lr 0.02 --sch 2 --da 0 --n 4 --max_p 0.9 --n 4 --prefix ls_baseline_ --loss sce --seed 0


CUDA_VISIBLE_DEVICES=3 python train.py --gpu --net resnet18 --optimizer nerov3 --wd 0.01 --lr 0.02 --sch 2 --da 0 --n 4 --max_p 0.9 --n 4 --prefix ls_baseline_ --loss sce --seed 1


CUDA_VISIBLE_DEVICES=5 python train.py --gpu --net resnet18 --optimizer nerov3 --wd 0.01 --lr 0.02 --sch 2 --da 0  --n 4 --max_p 0.9 --n 4 --prefix ls_baseline_ --loss sce --seed 2



CUDA_VISIBLE_DEVICES=0 python train.py --gpu --net resnet18 --optimizer nerov3 --lr 0.02 --sch 2 --da 1 --t 0.0 --spread 8 --max_p 1.0 --t2 0.0 --spread2 8 --max_p2 1.0 --loss fl --g 2 --n 4 --seed 0


CUDA_VISIBLE_DEVICES=2 python train.py --gpu --net resnet18 --optimizer nerov3 --lr 0.02 --sch 2 --da 1 --t 0.0 --spread 8 --max_p 1.0 --t2 0.0 --spread2 8 --max_p2 1.0 --loss fl --g 2 --n 4 --seed 1


CUDA_VISIBLE_DEVICES=4 python train.py --gpu --net resnet18 --optimizer nerov3 --lr 0.02 --sch 2 --da 1 --t 0.0 --spread 8 --max_p 1.0 --t2 0.0 --spread2 8 --max_p2 1.0 --loss fl --g 2 --n 4 --seed 2


== nero focal loss gamma = 2 or 3
CUDA_VISIBLE_DEVICES=0 python train.py --gpu --net resnet18 --optimizer nerov3 --lr 0.02 --wd 0.01 --sch 2 --da 1 --t 1.0 --spread 8 --max_p 1.0 --t2 0.0 --spread2 8 --max_p2 1.0 --loss fl --g 0 --n 4 --seed 0


CUDA_VISIBLE_DEVICES=2 python train.py --gpu --net resnet18 --optimizer nerov3 --lr 0.02 --wd 0.01 --sch 2 --da 1 --t 1.0 --spread 8 --max_p 1.0 --t2 0.0 --spread2 8 --max_p2 1.0 --loss fl --g 0 --n 4 --seed 1


CUDA_VISIBLE_DEVICES=4 python train.py --gpu --net resnet18 --optimizer nerov3 --lr 0.02 --wd 0.01 --sch 2 --da 1 --t 1.0 --spread 8 --max_p 1.0 --t2 0.0 --spread2 8 --max_p2 1.0 --loss fl --g 0 --n 4 --seed 2


== sgd soft loss

CUDA_VISIBLE_DEVICES=1 python train.py --gpu --net resnet18 --optimizer sgd --lr 0.1 --momentum 0.9 --sch 2 --da 1 --t 1.0 --spread 8 --max_p 1.0 --n 4 --prefix new_sample_sgd_ --seed 0

CUDA_VISIBLE_DEVICES=3 python train.py --gpu --net resnet18 --optimizer sgd --lr 0.1 --momentum 0.9 --sch 2 --da 1 --t 1.0 --spread 8 --max_p 1.0 --n 4 --prefix new_sample_sgd_ --seed 1

CUDA_VISIBLE_DEVICES=5 python train.py --gpu --net resnet18 --optimizer sgd --lr 0.1 --momentum 0.9 --sch 2 --da 1 --t 1.0 --spread 8 --max_p 1.0 --n 4 --prefix new_sample_sgd_ --seed 2


===nerov3 soft loss 500 epoch

CUDA_VISIBLE_DEVICES=0 python train.py --gpu --net resnet18 --optimizer nerov3 --lr 0.02 --wd 0.01 --sch 2 --da 1 --t 1.0 --spread 8 --max_p 1.0  --loss st --g 2 --n 4 --prefix 500_epoch_ --seed 0

CUDA_VISIBLE_DEVICES=2 python train.py --gpu --net resnet18 --optimizer nerov3 --lr 0.02 --wd 0.01 --sch 2 --da 1 --t 1.0 --spread 8 --max_p 1.0  --loss st --g 2 --n 4 --prefix 500_epoch_ --seed 1

CUDA_VISIBLE_DEVICES=4 python train.py --gpu --net resnet18 --optimizer nerov3 --lr 0.02 --wd 0.01 --sch 2 --da 1 --t 1.0 --spread 8 --max_p 1.0  --loss st --g 2 --n 4 --prefix 500_epoch_ --seed 2


=== resnet18v2 baseline

CUDA_VISIBLE_DEVICES=1 python train.py --gpu --net resnet18v2 --da 0 --prefix baseline_ --seed 0

CUDA_VISIBLE_DEVICES=3 python train.py --gpu --net resnet18v2 --da 0 --prefix baseline_ --seed 1

CUDA_VISIBLE_DEVICES=5 python train.py --gpu --net resnet18v2 --da 0 --prefix basline_ --seed 2


=== resnet18v2 cutout baseline

CUDA_VISIBLE_DEVICES=0 python train.py --gpu --net resnet18v2 --da 1 --prefix cutout_baseline_ --seed 0

CUDA_VISIBLE_DEVICES=2 python train.py --gpu --net resnet18v2 --da 1 --prefix cutout_baseline_ --seed 1

CUDA_VISIBLE_DEVICES=4 python train.py --gpu --net resnet18v2 --da 1 --prefix cutout_baseline_ --seed 2

