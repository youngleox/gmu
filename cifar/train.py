# train.py
#!/usr/bin/env	python3

""" train network using pytorch

Adapted by Yang Liu, original repo:
https://github.com/weiaicunzai/pytorch-cifar100

"""

import os
import sys
import argparse
import time
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.autograd import Variable
from conf import settings
from utils import get_network, get_training_dataloader, \
                  get_test_dataloader, WarmUpLR, soft_target, \
                  soft_target_mixup, smooth_crossentropy,\
                  mixup_data,mixup_data2,mixup_criterion,mixup_criterion2

import math
import sys
sys.path.append('../optim')

from nero import Nero,Nero_abl
from lamb import Lamb
from lambcs import LambCS
from madam import Madam
from madamcs import MadamCS
from nero_v3 import Nero_v3


def train(epoch):
    pf = 0 #False
    start = time.time()
    net.train()

    for batch_index, (images, labels) in enumerate(cifar_training_loader):
        # if epoch <= args.warm:
        #     warmup_scheduler.step()
        if args.cos:
            train_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        alpha = args.a
        #alpha = torch.clamp(torch.tensor(args.a * current_lr / args.lr),min=0,max=1-1e-8)

        if epoch >= settings.EPOCH * 0.95:
            alpha = 0

        if args.mixup == 1:
            images, labels_a, labels_b, lam = mixup_data(images, labels.long(),
                                                       alpha, True)
            #print(lam)
            images, labels_a, labels_b = map(Variable, (images,
                                                      labels_a, labels_b))
        elif args.mixup == 2:
            images, labels_a, labels_b, lam = mixup_data2(images, labels.long(),
                                                       alpha, True)

            images, labels_a, labels_b = map(Variable, (images,
                                                      labels_a, labels_b))
        #mean = images.mean(dim=(0,2,3))
        #std = images.std(dim=(0,2,3))

        optimizer.zero_grad()
        outputs = net(images)

        if args.loss == "mse" or args.loss  == "bce" or args.loss  == "mses":
            if args.task == 'cifar100':
                num_classes = 100
            else:
                num_classes = 10
            y = torch.eye(num_classes) 
            one_hot = y[labels.long()].cuda()
            if args.sm:
                normed = softmax(outputs)
                loss = loss_function(normed, one_hot)
            else:
                loss = loss_function(outputs, one_hot)

        elif args.loss == 'st':
            if args.mixup == 1 or args.mixup == 2:
                #loss = soft_target_mixup(outputs,labels_a,labels_b)
                loss = soft_target(outputs, labels_a) + soft_target(outputs, labels_b) 
            else:
                loss = soft_target(outputs, labels)

        elif args.loss == 'sce':
            loss = smooth_crossentropy(outputs, labels.long(), smoothing=1-args.max_p)

        elif args.loss == 'fl':
            ce_loss = torch.nn.functional.cross_entropy(outputs, labels.long(), reduction='none') # important to add reduction='none' to keep per-batch-item loss
            pt = torch.exp(-ce_loss)
            loss = ((1-pt)**args.g * ce_loss).mean() # mean over the batch

        else:
            if args.mixup == 1 :
                loss = mixup_criterion(loss_function, outputs, labels_a.long(), labels_b.long(), lam)
            elif args.mixup == 2 :
                loss = mixup_criterion2(loss_function, outputs, labels_a.long(), labels_b.long(), lam)
            else:
                loss = loss_function(outputs, labels.long())

        loss.backward()

        optimizer.step()

        n_iter = (epoch - 1) * len(cifar_training_loader) + batch_index + 1

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar_training_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(dataloader=None, train=False, epoch=None):

    start = time.time()
    net.eval()

    test_loss = 0.0 
    correct = 0.0

    for (images, labels) in dataloader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
        
        #mean = images.mean(dim=(0,2,3))
        #std = images.std(dim=(0,2,3))

        outputs = net(images)
        if args.loss == "mse" or args.loss  == "bce" or args.loss  == "mses":
            if args.task == 'cifar100':
                num_classes = 100
            else:
                num_classes = 10
            y = torch.eye(num_classes) 
            one_hot = y[labels.long()].cuda()
            if args.sm:
                normed = softmax(outputs)
                loss = loss_function(normed, one_hot)
            else:
                loss = loss_function(outputs, one_hot)
        elif args.loss == 'st':
            loss = soft_target(outputs, labels) 
        elif args.loss == 'sce':
            loss = smooth_crossentropy(outputs, labels.long(), smoothing=1-args.max_p)
        elif args.loss == 'fl':
            ce_loss = torch.nn.functional.cross_entropy(outputs, labels.long(), reduction='none') # important to add reduction='none' to keep per-batch-item loss
            pt = torch.exp(-ce_loss)
            loss = ((1-pt)**args.g * ce_loss).mean() # mean over the batch   
        else:
            loss = loss_function(outputs, labels.long())
        test_loss += loss.item() * len(labels)
        _, preds = outputs.max(1)
        correct += preds.eq(labels.long()).sum()

    finish = time.time()
    print('Evaluating Network.....')
    acc = correct.float() / len(dataloader.dataset)
    mean_loss = test_loss / len(dataloader.dataset)

    name = "train" if train else "test"
    print('{} set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        name, mean_loss, acc, finish - start ))

    #add informations to tensorboard
    if train:
        writer.add_scalar('Train/Average loss', mean_loss, epoch)
        writer.add_scalar('Train/Accuracy', acc, epoch)
    else:
        writer.add_scalar('Test/Average loss', mean_loss, epoch)
        writer.add_scalar('Test/Accuracy', acc, epoch)

    return acc, mean_loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=-1, help='random seed')
    parser.add_argument('--net', type=str, default='resnet18v2', help='net type')
    parser.add_argument('--nogpu', action='store_false', default=True, dest="gpu", 
                        help='use gpu or not')
    parser.add_argument('--b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--warm', type=int, default=0, help='warm up training phase')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    # Yang added:
    parser.add_argument('--momentum', default=0.9 , type=float,help='momentum/beta1')
    parser.add_argument('--beta', default=0.999, type=float,help='beta2')

    parser.add_argument('--prefix', default='', type=str)
    parser.add_argument('--gamma', default=0.2 , type=float,help='lr decay')
    parser.add_argument('--wd', default=0.0005 , type=float)

    parser.add_argument('--c1', action='store_true', dest="c1", default=False, 
                        help='Nero mean constraint')
    parser.add_argument('--c2', action='store_true', dest="c2", default=False, 
                        help='Nero norm constraint')

    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--task', default='cifar10' , type=str)
    parser.add_argument('--da', type=int, default=0, 
                        help='data augmentation options,-1: no DA, 0: standard, 1: custom')


    parser.add_argument('--noise', type=float, default=0.0, 
                        help='sigma of noise distribution, default: no noise')

    parser.add_argument('--pnoise', type=float, default=1.0, 
                        help='power of noise to probability function')

    parser.add_argument('--bgnoise', type=float, default=1, 
                        help='background type, default: 1, zeros. 0, noise')

    parser.add_argument('--crop', type=float, default=10.0, 
                        help='sigma of crop offset, \
                            default: 10, so that 3 sigma roughly covers the full width')

    parser.add_argument('--pcrop', type=float, default=4.0, 
                        help='power of crop overlap to probability function')

    parser.add_argument('--bgcrop', type=float, default=1, 
                        help='crop background type, default: 1, zeros. 0, noise')

    parser.add_argument('--cut', type=float, default=0.0, 
                        help='sigma of cutout length distribution, default: fixed length')

    parser.add_argument('--pcut', type=float, default=4.0, 
                        help='power of cutout overlap to probability function \
                            not being used')

    parser.add_argument('--l', type=float, default=16.0, 
                        help='mean cutout length')

    parser.add_argument('--mask', type=int, default=0, 
                        help='cutout mask type, default: 0, zero mask. 1, noise mask')

    parser.add_argument('--max_p', type=float, default=1.0, 
                        help='roughly equal to 1 - smooth in label smoothing')

    parser.add_argument('--loss', default='ce', type=str,
                        help="mse or crossentropy (ce) loss," )

    parser.add_argument('--g', default=2, type=float,help='gamma for focal loss')

    parser.add_argument('--sch', type=float, default=1, 
                        help='learning schedule scaling, default 1: 200 epochs')
    
    parser.add_argument('--step', action='store_false', dest="cos", default=True, 
                        help='Step or Cosine annealing scheduler')

    parser.add_argument('--n', type=int, default=16, help='number of workers')

    parser.add_argument('--no-nag', action='store_false', dest="nag", default=True, 
                        help='disable Nesterov for SGD')

    parser.add_argument('--mixup', type=int, default=0, 
                        help='0: no mixup, 1: original mixup, 2: custom mixup')

    parser.add_argument('--a', default=1.0, type=float,help='mixup alpha')
    args = parser.parse_args()
    if args.seed == -1:
        args.seed = random.randint(0,1000)
        print("Generating random seed!")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    settings.MILESTONES = [int(mst * args.sch) for mst in settings.MILESTONES]
    settings.EPOCH = int(settings.EPOCH * args.sch)
    print(settings.MILESTONES)

    net = get_network(args)

    #data preprocessing:
    if args.task == "cifar100":
        mean = settings.CIFAR100_TRAIN_MEAN
        std = settings.CIFAR100_TRAIN_STD
    elif args.task == "cifar10":
        # mean = (0.4914, 0.4822, 0.4465)
        # std = (0.2023, 0.1994, 0.2010)
        # mean = (0, 0, 0)
        # std = (1, 1, 1)
        # mean = (0.49139968, 0.48215841, 0.44653091)
        # std = (0.24703223, 0.24348513, 0.26158784)
        # below from cutout official repo
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]]
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]

    else:
        print("invalid task!!")

    cifar_training_loader = get_training_dataloader(
        mean,
        std,
        num_workers=args.n,
        batch_size=args.b,
        shuffle=True,
        task = args.task,
        da = args.da,
        sigma_noise=args.noise, pow_noise=args.pnoise,
        sigma_crop=args.crop, pow_crop=args.pcrop,
        sigma_cut=args.cut, pow_cut=args.pcut,
        length_cut=args.l, mask_cut=args.mask
    )

    cifar_test_loader = get_test_dataloader(
        mean,
        std,
        num_workers=args.n,
        batch_size=args.b,
        shuffle=False,
        task = args.task
    )
    #test training acc
    cifar_train_test_loader = get_test_dataloader(
        mean,
        std,
        num_workers=args.n,
        batch_size=args.b,
        shuffle=False,
        task = args.task,
        train = True
    )

    if args.loss == "ce":
        loss_function = nn.CrossEntropyLoss()
    elif args.loss == "nll":
        loss_function = nn.NLLLoss()
        args.prefix += args.loss
    elif args.loss == "mses":
        print("using mse loss + softmax!!")
        loss_function = nn.MSELoss()
        args.prefix += args.loss
        args.sm =True
    elif args.loss == "mse":
        print("using mse loss!!")
        loss_function = nn.MSELoss()
        args.prefix += args.loss
        args.sm = False
    elif args.loss == "bce":
        print("using bce loss!!")
        loss_function = nn.BCEWithLogitsLoss()
        args.prefix += args.loss
    elif args.loss == "fl":
        print("using bce loss!!")
        loss_function = nn.CrossEntropyLoss()
        args.prefix += args.loss + "_gamma_" + str(args.g)
    elif args.loss == "st":
        print("using soft target loss!!")
        args.prefix += args.loss 
    elif args.loss == "sce":
        print("using smooth crossentropy loss!!")
        args.prefix += args.loss

    else:
        print("undefined loss")

    if args.optimizer == 'sgd':
        print("using sgd!")
        optimizer = optim.SGD(net.parameters(), lr=args.lr, nesterov=args.nag, momentum=args.momentum, weight_decay=args.wd)
    
    elif args.optimizer == 'adam':
        print("using adam!")
        optimizer = optim.Adam(net.parameters(), lr=args.lr,betas=(args.momentum, args.beta), weight_decay=args.wd)
    
    elif args.optimizer == 'lamb':
        print("using lamb!")
        optimizer = Lamb(net.parameters(), lr=args.lr,betas=(args.momentum, args.beta), weight_decay=args.wd)
    elif args.optimizer == 'lambcs':
        print("using lambcs!")
        optimizer = LambCS(net.parameters(), lr=args.lr,betas=(args.momentum, args.beta), weight_decay=args.wd,
                            constraints=True)

    elif args.optimizer == 'madam':
        print("using madam!")
        optimizer = Madam(net.parameters(), lr=args.lr)

    elif args.optimizer == 'madamcs':
        print("using madamcs!")
        optimizer = MadamCS(net.parameters(), lr=args.lr,constraints=True)
   
    elif args.optimizer == 'nero':
        print("using nero!")
        optimizer = Nero(net.parameters(),lr=args.lr,constraints=True)
    
    elif args.optimizer == 'nerov3':
        print("using nero!")
        optimizer = Nero_v3(net.parameters(),net,lr=args.lr,constraints=True)

    elif args.optimizer == 'neroabl':
        print("using nero ablated!")
        optimizer = Nero_abl(net.parameters(),lr=args.lr,
                        c1=args.c1,c2=args.c2)
    iter_per_epoch = len(cifar_training_loader)
    if args.cos:
        Max_it = iter_per_epoch * (settings.EPOCH+1) + 1
        train_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,Max_it)
    else:
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=args.gamma) #learning rate decay
        
    #warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    args.prefix += "_Mix{}_A{}_".format(args.mixup,args.a) if args.mixup else ""
    args.prefix += "_COS_" if args.cos else ""

    if args.da == 0:
        args.prefix = "seed" + str(args.seed) + "_DA" + str(args.da) + \
                    "_MaxP" + str(args.max_p) + "_L" + str(args.l) + \
                    "_sch" + str(args.sch) + args.prefix + '_'

    elif args.da == 1:
        args.prefix = "seed" + str(args.seed) + "_DA" + str(args.da) + \
                    "_MaxP" + str(args.max_p) + \
                    "_L" + str(args.l) + "Mask" + str(args.mask) + \
                    "_sch" + str(args.sch) + args.prefix + '_'
    elif args.da == 2 or args.da == 3:
        args.prefix = "seed" + str(args.seed) + "_DA" + str(args.da) + \
                    "_Noise" + str(args.noise) + "P" + str(args.pnoise) + "BG" + str(args.bgnoise) + \
                    "_Crop" + str(args.crop) + "P" + str(args.pcrop) + "BG" + str(args.bgcrop) + \
                    "_Cut" + str(args.cut) + "P" + str(args.pcut) + \
                    "_L" + str(args.l) + "Mask" + str(args.mask) + \
                    "_sch" + str(args.sch) + args.prefix + '_'

    # elif args.da == 2:
    #     args.prefix = "seed" + str(args.seed) + "_DA" + str(args.da) + \
    #                 "_T" + str(args.t) + \
    #                 "spread" + str(args.spread) + \
    #                 "_maxP" + str(args.max_p) + \
    #                 "_T2" + str(args.t2) + \
    #                 "spread2" + str(args.spread2) + \
    #                 "_maxP2" + str(args.max_p2) + \
    #                 "_sch" + str(args.sch) + args.prefix + '_'
    nag = 'NAG' if args.nag else ""
    if args.optimizer == "sgd":
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.task, args.net, 
                        args.prefix + args.optimizer + str(args.lr)+ nag +#'_g_'+str(args.gamma)+
                        '_momentum_'+str(args.momentum)+'_wd_'+str(args.wd),
                        settings.TIME_NOW)
    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.task, args.net, 
                        args.prefix + args.optimizer + str(args.lr)+ #'_g_'+str(args.gamma)+
                        #'_beta1_'+str(args.momentum)+'_beta2_'+str(args.beta)+
                        '_wd_'+str(args.wd),
                        settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    if args.optimizer == "sgd":
        writer = SummaryWriter(log_dir=os.path.join(
                            settings.LOG_DIR, args.task, args.net,
                            args.prefix + args.optimizer + str(args.lr)+nag +#'_g_'+str(args.gamma)+
                            '_momentum_'+str(args.momentum)+'_wd_'+str(args.wd),
                            settings.TIME_NOW))
    else:
        writer = SummaryWriter(log_dir=os.path.join(
                            settings.LOG_DIR, args.task, args.net,
                            args.prefix + args.optimizer + str(args.lr)+#'g'+str(args.gamma)+
                            #'_beta1_'+str(args.momentum)+'_beta2_'+str(args.beta)+
                            '_wd_'+str(args.wd),
                            settings.TIME_NOW))
    #input_tensor = torch.Tensor(1, 3, 32, 32).cuda()
    #writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    best_test_acc = 0.0
    best_test_acc_epoch = 0

    best_test_loss = 10.0
    best_test_loss_epoch = 0
        
    best_train_acc = 0.0
    best_train_acc_epoch = 0

    best_train_loss = 10.0
    best_train_loss_epoch = 0

    for epoch in range(0, settings.EPOCH+1):
        if not args.cos:
            train_scheduler.step(epoch)
        writer.add_scalar("lr",optimizer.param_groups[0]['lr'],epoch)
        # if epoch == settings.MILESTONES[-1] and args.da == 2:
        #     cifar_training_loader = get_training_dataloader(
        #                                 mean,
        #                                 std,
        #                                 num_workers=args.n,
        #                                 batch_size=args.b,
        #                                 shuffle=True,
        #                                 alpha = 0.0,
        #                                 task = args.task,
        #                                 da = args.da,
        #                                 T=args.t2, 
        #                                 spread=args.spread2, 
        #                                 max_p=args.max_p2
        #                             )

        train(epoch)
        if epoch % 5 == 0:
            test_acc, test_loss = eval_training(dataloader=cifar_test_loader,train=False,epoch=epoch)
            train_acc, train_loss = test_acc, test_loss
            #train_acc, train_loss = eval_training(dataloader=cifar_training_loader,train=True,epoch=epoch)
            print(writer.log_dir)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_acc_epoch = epoch

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_loss_epoch = epoch
            
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_train_acc_epoch = epoch

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_train_loss_epoch = epoch

            #start to save best performance model after learning rate decay to 0.01
            writer.add_scalar('Test/Best Average loss', best_test_loss, epoch)
            writer.add_scalar('Test/Best Accuracy', best_test_acc, epoch)
            writer.add_scalar('Train/Best Average loss', best_train_loss, epoch)
            writer.add_scalar('Train/Best Accuracy', best_train_acc, epoch)


        if epoch > settings.MILESTONES[1] and best_acc < test_acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = test_acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    writer.close()
