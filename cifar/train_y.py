# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
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
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR

import math
import sys
sys.path.append('..')
import resnet
from polynomial_scheduler import PolyLR

from constant_sum_v2 import MadamV2,SGDV2,AdamV2
from constant_sum_v3 import CSV3
from constant_sum_v4 import CSV4
from constant_sum_v5 import CSV5
from constant_sum_v6 import CSV6
from constant_sum_v7 import CSV7,SGDV7,FromageCSV7
from constant_sum_v8 import CSV8,CSV8n,SGDCS,AdamCS,FromageCS,Chimera
from nero import Nero,Nero_abl,Nero_op
from nero_pre import Nero as Nero_pre
from lamb import Lamb
from lambcs import LambCS
from fromage import Fromage
from madam import Madam
from madamcs import MadamCS
from nero_v3 import Nero_v3
def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes) 
    return y[labels]
def train(epoch):
    pf = 0 #False
    start = time.time()
    net.train()

    for batch_index, (images, labels) in enumerate(cifar_training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images) * args.os # scaled output
        
        if args.loss == "mse" or args.loss  == "bce":
            if args.task == 'cifar100':
                num_classes = 100
            else:
                num_classes = 10
            y = torch.eye(num_classes) 
            one_hot = y[labels].cuda()
            if args.pos:
                # only compute loss from positive classes, scale up by nclass to compensate.
                loss = loss_function(outputs*one_hot*num_classes, one_hot*num_classes)
            else:
                loss = loss_function(outputs, one_hot)
        else:
            loss = loss_function(outputs, labels)
            if math.isnan(loss.item()):
                print(outputs)
                print(labels) 
                break
        if pf >0:
            out_slice = outputs[0,:]
            # print("output mean: {}, var: {}".format(outputs.mean(dim=0).mean(),outputs.var(dim=0).mean()))
            # print("target mean: {}, var: {}".format(one_hot.mean(dim=0).mean(),one_hot.var(dim=0).mean()))
            print("output mean: {}, std: {}, dim :{}".format(outputs.mean(dim=0),outputs.std(dim=0),outputs.mean(dim=0).shape))
            print("output slice: {}, std: {}, dim :{}".format(out_slice,out_slice.std(dim=0),out_slice.mean(dim=0).shape))
            # print("target mean: {}, var: {}".format(one_hot.mean(dim=0),one_hot.var(dim=0)))
            print(outputs.shape)
            pf -= 1 #False
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar_training_loader) + batch_index + 1

        # last_layer = list(net.children())[-1]
        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar_training_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)
    #optimizer.clean_weight()
    # for name, param in net.named_parameters():
    #     layer, attr = os.path.splitext(name)
    #     attr = attr[1:]
    #     writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

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

        outputs = net(images) * args.os # scaled output
        if args.loss == "mse" or args.loss  == "bce":
            if args.task == 'cifar100':
                num_classes = 100
            else:
                num_classes = 10
            y = torch.eye(num_classes) 
            one_hot = y[labels].cuda()
            if args.pos:
                # only compute loss from positive classes, scale up by nclass to compensate.
                loss = loss_function(outputs*one_hot*num_classes, one_hot*num_classes)
            else:
                loss = loss_function(outputs, one_hot)
            
        else:
            loss = loss_function(outputs, labels)

        test_loss += loss.item() * len(labels)
        if args.loss == "mse" or args.loss  == "bce":
            _, preds = (outputs-1).abs().min(1)
        else:
            _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

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
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=5, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    #Yang
    parser.add_argument('--init_bias', action='store_true')

    parser.add_argument('--res', action='store_true')
    parser.add_argument('--bn', action='store_true')
    parser.add_argument('--na', action='store_true')
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--beta', default=0.999, type=float,help='beta for csv4')
    parser.add_argument('--clip', default=1.0, type=float)
    parser.add_argument('--sp', default=0.0, type=float)
    parser.add_argument('--a', default=1.0 , type=float)
    parser.add_argument('--scale', default=1.0 , type=float)
    parser.add_argument('--os', default=1.0, type=float)
    parser.add_argument('--w_mul', default=1.0 , type=float)
    parser.add_argument('--nl', default='relu', type=str)
    parser.add_argument('--prefix', default='', type=str)
    parser.add_argument('--gamma', default=0.2 , type=float)
    parser.add_argument('--lr-exp', default=1.0 , type=float)
    parser.add_argument('--wd', default=0.0005 , type=float)
    parser.add_argument('--momentum', default=0.9 , type=float)
    parser.add_argument('--free', action='store_false', dest="fix_norm", default=True)
    parser.add_argument('--ci', action='store_true', dest="constrain_on_init", default=False)

    parser.add_argument('--c1', action='store_true', dest="c1", default=False)
    parser.add_argument('--c2', action='store_true', dest="c2", default=False)

    parser.add_argument('--nog', action='store_false', dest="gnorm", default=True, help="multiply by norm(p) in update")

    parser.add_argument('--gne', default=0 , type=int, 
                                help="type of gradient norm estimation: 0=none, 1=per synape, 2=per neuron, 3=per layer")
    parser.add_argument('--un', default=0 , type=int, help="type of update normazation, 0=none")

    parser.add_argument('--wb', action='store_true', help="weight buffer for csv7")
    parser.add_argument('--wn', default=0.0 , type=float, help="weight noise for csv7")
    parser.add_argument('--nlr', action='store_true', help="noise scale with lr for csv7")

    parser.add_argument('--grad', action='store_true', help="gradual learning")
    parser.add_argument('--pos', action='store_true', help="calculate gradient only for positive classes")

    parser.add_argument('--alpha', default=0.0 , type=float,help='alpha for noise')

    parser.add_argument('--loss', default='ce', type=str,help="mse or crossentropy (ce) loss," )
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr-decay-epoch', default='40,80', type=str)
    parser.add_argument('--sch', default='step', type=str)
    parser.add_argument('--lr-pow', default=6.0 , type=float)
    parser.add_argument('--task', default='cifar100' , type=str)
    
    parser.add_argument('--no-aug', action='store_false', dest="da", default=True)

    args = parser.parse_args()


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #cudnn.enabled = True
    #cudnn.benchmark = True
    
    net = get_network(args)

    weights = []
    for i in range(100):
        w = [1] * (i + 1)
        w.extend([0] * (99 - i))
        weights.append(w)
    #print(weights)
    weights = torch.FloatTensor(weights).cuda()
    #data preprocessing:
    if args.task == "cifar100":
        mean = settings.CIFAR100_TRAIN_MEAN
        std = settings.CIFAR100_TRAIN_STD
    elif args.task == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        print("invalid task!!")

    cifar_training_loader = get_training_dataloader(
        mean,
        std,
        num_workers=4,
        batch_size=args.b,
        shuffle=True,
        alpha = args.alpha,
        task = args.task,
        da = args.da
    )

    cifar_test_loader = get_test_dataloader(
        mean,
        std,
        num_workers=4,
        batch_size=args.b,
        shuffle=False,
        task = args.task
    )
    #test training acc
    cifar_train_test_loader = get_test_dataloader(
        mean,
        std,
        num_workers=4,
        batch_size=args.b,
        shuffle=False,
        task = args.task,
        train = True
    )

    multiplier = 1
    settings.MILESTONES = [ int(100 * multiplier), int(150 * multiplier), int(180 * multiplier)] 
    settings.EPOCH = int(settings.EPOCH * multiplier)
    #settings.MILESTONES = [100,150,180]
    if args.loss == "ce":
        if args.grad:
            loss_function = nn.CrossEntropyLoss(weight=weights[0])
        else:
            loss_function = nn.CrossEntropyLoss()
            print("weights: {}".format(weights[99]))
    elif args.loss == "nll":
        loss_function = nn.NLLLoss(weight=weights[0])
        args.prefix += args.loss
    elif args.loss == "mse":
        print("using mse loss!!")
        loss_function = nn.MSELoss()
        args.prefix += args.loss
    elif args.loss == "bce":
        print("using bce loss!!")
        loss_function = nn.BCEWithLogitsLoss()
        args.prefix += args.loss
    
    if args.clip == 0:
        clip = math.inf
    else:
        clip = abs(args.clip) 

    if args.optimizer == 'sgd':
        print("using sgd!")
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    
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

    elif args.optimizer == 'fromage':
        print("using fromage!")
        optimizer = Fromage(net.parameters(), lr=args.lr)

    elif args.optimizer == 'madam':
        print("using madam!")
        optimizer = Madam(net.parameters(), lr=args.lr)

    elif args.optimizer == 'madamcs':
        print("using madamcs!")
        optimizer = MadamCS(net.parameters(), lr=args.lr,constraints=True)

    elif args.optimizer == 'sgdcs':
        print("using sgd + cs!")
        optimizer = SGDCS(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    
    elif args.optimizer == 'adamcs':
        print("using adam + cs!")
        optimizer = AdamCS(net.parameters(), lr=args.lr,betas=(args.momentum, args.beta), weight_decay=args.wd)

    elif args.optimizer == 'fromagecs':
        print("using fromage + cs!")
        optimizer = FromageCS(net.parameters(), lr=args.lr)

    elif args.optimizer == 'madamcs':
        print("using madam + cs!")
        optimizer = MadamCS(net.parameters(), lr=args.lr) 

    elif args.optimizer == 'chimera':
        print("using chimera!")
        optimizer = Chimera(net.parameters(),scale=args.scale, lr=args.lr, beta1=args.momentum, 
                            beta2=args.beta, bias_clip=clip, weight_decay=args.wd,
                            noise=args.wn, weight_buffer=args.wb,noise_with_lr=args.nlr,
                            fix_norm=args.fix_norm,lr_exp_base=args.lr_exp,
                            gnorm_estimate=args.gne,update_normalization=args.un)
    elif args.optimizer == 'nero':
        print("using nero!")
        optimizer = Nero(net.parameters(),lr=args.lr,
                        constraints=args.fix_norm)
    elif args.optimizer == 'neroabl':
        print("using nero ablation!")
        optimizer = Nero_abl(net.parameters(),lr=args.lr,
                        c1=args.c1,c2=args.c2)
    elif args.optimizer == 'neropre':
        print("using nero pre!")
        optimizer = Nero_pre(net.parameters(),lr=args.lr,
                        constraints=args.fix_norm)
    elif args.optimizer == 'nero_op':
        print("using nero_op!")
        optimizer = Nero_op(net.parameters(),lr=args.lr,
                        constraints=args.fix_norm)
    elif args.optimizer == 'nerov3':
        print("using nerov3!")
        optimizer = Nero_v3(net.parameters(),net,lr=args.lr,
                        constraints=args.fix_norm)
    elif args.optimizer == 'csv4':
        print("using csv4!")
        optimizer = CSV4(net.parameters(), lr=args.lr, beta=args.beta, bias_clip=clip)
    
    elif args.optimizer == 'csv5':
        print("using csv5!")
        optimizer = CSV5(net.parameters(), lr=args.lr, beta1=args.momentum, beta2=args.beta, bias_clip=clip)
    
    elif args.optimizer == 'csv6':
        print("using csv6!")
        optimizer = CSV6(net.parameters(), lr=args.lr, beta1=args.momentum, beta2=args.beta, bias_clip=clip, weight_decay=args.wd)
    
    elif args.optimizer == 'csv7':
        print("using csv7!")
        optimizer = CSV7(net.parameters(),scale=args.scale, lr=args.lr, beta1=args.momentum, beta2=args.beta, bias_clip=clip, weight_decay=args.wd,
                            noise=args.wn, weight_buffer=args.wb,noise_with_lr=args.nlr)
    elif args.optimizer == 'sgdv7':
        print("using sgdv7!")
        optimizer = SGDV7(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd,bias_clip=clip,
                            noise=args.wn, weight_buffer=args.wb,noise_with_lr=args.nlr)
    
    elif args.optimizer == 'csv8':
        print("using csv8!")
        optimizer = CSV8(net.parameters(),scale=args.scale, lr=args.lr, beta1=args.momentum, 
                            beta2=args.beta, bias_clip=clip, weight_decay=args.wd,
                            noise=args.wn, weight_buffer=args.wb,noise_with_lr=args.nlr,
                            fix_norm=args.fix_norm,lr_exp_base=args.lr_exp,gnorm=args.gnorm)
    elif args.optimizer == 'csv8n':
        print("using csv8n!")
        optimizer = CSV8n(net.parameters(),net,scale=args.scale, lr=args.lr, beta1=args.momentum, 
                            beta2=args.beta, bias_clip=clip, weight_decay=args.wd,
                            noise=args.wn, weight_buffer=args.wb,noise_with_lr=args.nlr,
                            fix_norm=args.fix_norm,lr_exp_base=args.lr_exp,gnorm=args.gnorm)

    elif args.optimizer == 'madamv8':
        print("using madamv8!")
        optimizer = MadamV8(net.parameters(), lr=args.lr,beta1=args.momentum, beta2=args.beta,bias_clip=clip,
                            flip_thr=args.wn,fix_norm=args.fix_norm)
        # wn is used for flip threshold

    elif args.optimizer == 'fromagecsv7':
        print("using fromagecsv7!")
        optimizer = FromageCSV7(net.parameters(), lr=args.lr, weight_decay=args.wd,bias_clip=clip,
                                                beta1=args.momentum, beta2=args.beta)


    if args.sch == "step":
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=args.gamma) #learning rate decay
    elif args.sch == "poly":
        train_scheduler = PolyLR(optimizer,T_max=settings.EPOCH, eta_min=0, power=args.lr_pow)
        args.prefix += 'pow' + str(args.lr_pow) 
    elif args.sch == "cos":
        train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.EPOCH) #learning rate decay
        args.prefix += 'cos'  
    iter_per_epoch = len(cifar_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    if not args.alpha == 0 or True:
        args.prefix += '_noise' + str(args.alpha) + '_'
    args.prefix = "seed" + str(args.seed) + args.prefix + '_scale'+ str(args.scale) +"_os" + str(args.os)
    if args.optimizer == 'chimera':
        args.prefix = args.prefix + 'gne_' + str(args.gne) + 'un_' + str(args.un)
    if args.optimizer == 'nero' or args.optimizer == 'nerov2':
        args.prefix = args.prefix + 'ci_' + str(args.constrain_on_init) 
    if args.optimizer == 'neroabl':
        args.prefix = args.prefix + 'c1_' + str(args.c1) + 'c2_' + str(args.c2) 
    args.prefix += '_no_aug' if not args.da else ""
    args.prefix += 'free_norm_' if not args.fix_norm else ''
    args.prefix += 'no_gnorm_' if not args.gnorm else ''
    args.prefix += 'lr_exp{}_'.format(args.lr_exp) if not args.lr_exp == 1.0 else ""
    if args.optimizer == "sgd":
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.task, args.net, 
                        args.prefix + args.optimizer + str(args.lr)+'g'+str(args.gamma)+'momentum'+str(args.momentum)+'wd'+str(args.wd),
                        settings.TIME_NOW)
    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.task, args.net, 
                        args.prefix + args.optimizer + str(args.lr)+'g'+str(args.gamma)+'clip'+
                        str(args.clip)+'momentum'+str(args.momentum)+'beta'+str(args.beta)+'os'+
                        str(args.os)+'wd'+str(args.wd)+'wn'+str(args.wn)+'wb'+str(int(args.wb))+'nlr'+str(int(args.nlr)),
                        settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    if args.optimizer == "sgd":
        writer = SummaryWriter(log_dir=os.path.join(
                            settings.LOG_DIR, args.task, args.net,
                            args.prefix + args.optimizer + str(args.lr)+'g'+str(args.gamma)+'momentum'+str(args.momentum)+'wd'+str(args.wd),
                            settings.TIME_NOW))
    else:
        writer = SummaryWriter(log_dir=os.path.join(
                            settings.LOG_DIR, args.task, args.net,
                            args.prefix + args.optimizer + str(args.lr)+'g'+str(args.gamma)+'clip'+
                            str(args.clip)+'momentum'+str(args.momentum)+'beta'+str(args.beta)+'os'+
                            str(args.os)+'wd'+str(args.wd)+'wn'+str(args.wn)+'wb'+str(int(args.wb))+'nlr'+str(int(args.nlr)),
                            settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32).cuda()
    writer.add_graph(net, input_tensor)

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

    for epoch in range(1, settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)
        writer.add_scalar("lr",optimizer.param_groups[0]['lr'],epoch)
        if args.loss == "ce" and args.grad:
            n = 2
            print("using weights: {}".format(weights[min(99,(epoch-1)*n)]))
            loss_function = nn.CrossEntropyLoss(weight=weights[min(99,(epoch-1)*n)])
        train(epoch)
        if args.wb:
            optimizer.clean_weight()
        test_acc, test_loss = eval_training(dataloader=cifar_test_loader,train=False,epoch=epoch)
        train_acc, train_loss = eval_training(dataloader=cifar_training_loader,train=True,epoch=epoch)
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
