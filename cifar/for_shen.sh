
== best configs:

python train.py --gpu --net resnet18 --optimizer nerov3 --lr 0.02 --wd 0.01 --sch 2 --da 1 --t 1.0 --spread 8 --max_p 1.0 --pow 3 --loss st  --n 4 --seed 0

python train.py --gpu --net resnet18 --optimizer sgd --lr 0.1 --wd 0.0005 --sch 2 --da 1 --t 1.0 --spread 8 --max_p 1.0 --pow 3 --loss st  --n 4 --seed 0


===========
Important configs for best resnet 18 performance:

--loss st (st: soft target loss, fl: focal loss (wihout alpha, TODO?), sce: soft cross entropy (to use with label smoothing), default is ce: cross entropy)
--sch 2 (schedule multiplier, I keep it at 2 to have 400 training epochs)
--da 1 (da=0 uses origial DA, da=1 uses custom DA)
--t 1 (overlap threshold, below which the ground truth probability is set to chance, 1.0 seems to work well)
--spread 8 - 10 (for a 32 pixel wide image, a spread of 10 roughly makes sure all crops have -30 to +30 offsets, matching the width of the images)
--pow 2 - 4 (controls the power of the probability-overlap curve, higher the power, flatter the curve, when pow=inf, it degenerates to hard target)

SGD:
--wd 0.0005 or 0.0001 (not thoroughly tested)
--lr 0.1 - 0.25 (only tested 0.1, as it is the suggested value of many papers) 

Nero_v3:
--wd 0.01 - 1 (special weight regularizer that only act on affine gains, robust, works in a wide range)
--lr 0.02  
