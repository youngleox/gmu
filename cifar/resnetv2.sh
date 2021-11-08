=== resnet18v2 baseline

CUDA_VISIBLE_DEVICES=1 python train.py --gpu --net resnet18v2 --da 0 --prefix baseline_ --seed 0

CUDA_VISIBLE_DEVICES=3 python train.py --gpu --net resnet18v2 --da 0 --prefix baseline_ --seed 1

CUDA_VISIBLE_DEVICES=5 python train.py --gpu --net resnet18v2 --da 0 --prefix baseline_ --seed 2


=== resnet18v2 cutout baseline

python train.py --da 1 --prefix new_schedule_cutout_baseline_ --seed 0

python train.py --da 1 --prefix new_schedule_cutout_baseline_ --seed 1

python train.py --da 1 --prefix new_schedule_cutout_baseline_ --seed 2

=== resnet18v2 cutout nerov3

python train.py --da 1 --optimizer nerov3 --lr 0.02 --wd 0.01 --prefix new_schedule_cutout_baseline_ --seed 0

python train.py --da 1 --optimizer nerov3 --lr 0.02 --wd 0.01 --prefix new_schedule_cutout_baseline_ --seed 1

python train.py --da 1 --optimizer nerov3 --lr 0.02 --wd 0.01 --prefix new_schedule_cutout_baseline_ --seed 2

=== resnet18v2 cutout noise mask

CUDA_VISIBLE_DEVICES=1 python train.py --gpu --net resnet18v2 --da 1 --mask 1 --prefix cutout_noisemask_ --seed 0

CUDA_VISIBLE_DEVICES=3 python train.py --gpu --net resnet18v2 --da 1 --mask 1 --prefix cutout_noisemask_ --seed 1

CUDA_VISIBLE_DEVICES=5 python train.py --gpu --net resnet18v2 --da 1 --mask 1 --prefix cutout_noisemask_ --seed 2

=== resnet18v2 soft loss crop and cutout noise mask

CUDA_VISIBLE_DEVICES=0 python train.py --gpu --net resnet18v2 --loss st --da 2 --mask 1 --crop 10 --pcrop 4 --seed 0

CUDA_VISIBLE_DEVICES=2 python train.py --gpu --net resnet18v2 --loss st --da 2 --mask 1 --crop 10 --pcrop 4 --seed 1

CUDA_VISIBLE_DEVICES=4 python train.py --gpu --net resnet18v2 --loss st --da 2 --mask 1 --crop 10 --pcrop 4 --seed 2

=== resnet18v2 soft loss crop and cutout zero mask

CUDA_VISIBLE_DEVICES=0 python train.py --gpu --net resnet18v2 --loss st --da 2 --mask 0 --crop 10 --pcrop 4 --seed 0

CUDA_VISIBLE_DEVICES=2 python train.py --gpu --net resnet18v2 --loss st --da 2 --mask 0 --crop 10 --pcrop 4 --seed 1

CUDA_VISIBLE_DEVICES=4 python train.py --gpu --net resnet18v2 --loss st --da 2 --mask 0 --crop 10 --pcrop 4 --seed 2


=== resnet18v2 soft loss crop 

CUDA_VISIBLE_DEVICES=1 python train.py --gpu --net resnet18v2 --loss st --da 2 --mask 1 --crop 10 --pcrop 4 --l 0 --seed 0

CUDA_VISIBLE_DEVICES=3 python train.py --gpu --net resnet18v2 --loss st --da 2 --mask 1 --crop 10 --pcrop 4 --l 0 --seed 1

CUDA_VISIBLE_DEVICES=5 python train.py --gpu --net resnet18v2 --loss st --da 2 --mask 1 --crop 10 --pcrop 4 --l 0 --seed 2


=== single run


===  grid search
export CUDA_VISIBLE_DEVICES=0
export BGCROP=1
export MASK=1
export PCROP=1
export L=0
export SEED=1

python train.py --loss st --da 2 --bgcrop $BGCROP --mask $MASK --crop 0 --pcrop $PCROP --l $L --seed $SEED

python train.py --loss st --da 2 --bgcrop $BGCROP --mask $MASK --crop 2 --pcrop $PCROP --l $L --seed $SEED

python train.py --loss st --da 2 --bgcrop $BGCROP --mask $MASK --crop 4 --pcrop $PCROP --l $L --seed $SEED

python train.py --loss st --da 2 --bgcrop $BGCROP --mask $MASK --crop 6 --pcrop $PCROP --l $L --seed $SEED

python train.py --loss st --da 2 --bgcrop $BGCROP --mask $MASK --crop 8 --pcrop $PCROP --l $L --seed $SEED

python train.py --loss st --da 2 --bgcrop $BGCROP --mask $MASK --crop 10 --pcrop $PCROP --l $L --seed $SEED

python train.py --loss st --da 2 --bgcrop $BGCROP --mask $MASK --crop 12 --pcrop $PCROP --l $L --seed $SEED

python train.py --loss st --da 2 --bgcrop $BGCROP --mask $MASK --crop 14 --pcrop $PCROP --l $L --seed $SEED

python train.py --loss st --da 2 --bgcrop $BGCROP --mask $MASK --crop 16 --pcrop $PCROP --l $L --seed $SEED

===  cutout 
export CUDA_VISIBLE_DEVICES=0
export BGCROP=1
export MASK=1
export PCROP=2
export L=0
export SEED=1

python train.py --loss st --da 3 --bgcrop $BGCROP --mask $MASK --crop 0 --pcrop $PCROP --l 4 --cut 0.0 --seed $SEED

python train.py --loss st --da 3 --bgcrop $BGCROP --mask $MASK --crop 0 --pcrop $PCROP --l 8 --cut 0.0 --seed $SEED

python train.py --loss st --da 3 --bgcrop $BGCROP --mask $MASK --crop 0 --pcrop $PCROP --l 12 --cut 0.0 --seed $SEED

python train.py --loss st --da 3 --bgcrop $BGCROP --mask $MASK --crop 0 --pcrop $PCROP --l 16 --cut 0.0 --seed $SEED

python train.py --loss st --da 3 --bgcrop $BGCROP --mask $MASK --crop 0 --pcrop $PCROP --l 4 --cut 0.1 --seed $SEED

python train.py --loss st --da 3 --bgcrop $BGCROP --mask $MASK --crop 0 --pcrop $PCROP --l 8 --cut 0.1 --seed $SEED

python train.py --loss st --da 3 --bgcrop $BGCROP --mask $MASK --crop 0 --pcrop $PCROP --l 12 --cut 0.1 --seed $SEED

python train.py --loss st --da 3 --bgcrop $BGCROP --mask $MASK --crop 0 --pcrop $PCROP --l 16 --cut 0.1 --seed $SEED



python train.py --da 0 --mixup 1 --seed $SEED --a 0.1 --prefix last_10_

python train.py --da 0 --mixup 1 --seed $SEED --a 0.2 --prefix last_10_

python train.py --da 0 --mixup 1 --seed $SEED --a 0.3 --prefix last_10_

python train.py --da 0 --mixup 1 --seed $SEED --a 0.4 --prefix last_10_

python train.py --da 0 --mixup 1 --seed $SEED --a 0.5 --prefix last_10_

python train.py --da 0 --mixup 1 --seed $SEED --a 0.6 --prefix last_10_

python train.py --da 0 --mixup 1 --seed $SEED --a 0.7 --prefix last_10_

python train.py --da 0 --mixup 1 --seed $SEED --a 0.8 --prefix last_10_

python train.py --da 0 --mixup 1 --seed $SEED --a 0.9 --prefix last_10_

python train.py --da 0 --mixup 1 --seed $SEED --a 1.0 --prefix last_10_


