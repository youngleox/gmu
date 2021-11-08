import argparse
import torch

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
import sys; sys.path.append("..")
from sam import SAM,SAM1,SAM_abl

import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
sys.path.append("../..")
from optim.nero import Nero
from optim.nero_v3 import Nero_v3

def disable_bn(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()

def enable_bn(model):
    model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=0.5, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    
    parser.add_argument("--LOG_DIR", default="logs", type=str, help="log dir")
    parser.add_argument("--prefix", default="", type=str, help="prefix")
    parser.add_argument("--optimizer", default="sgd", type=str, help="base optimizer for SAM")
    parser.add_argument("--sam", default="0", type=int, 
                        help="SAM version, 0: original, 1: random sign, \
                            2: match e_w std, 3: match p.grad std")
    parser.add_argument("--seed", default=None, type=int, 
                        help="random seed")
    parser.add_argument("--bsam", default=1, type=int, 
                        help="# batches per sam update")

    args = parser.parse_args()
    
    if args.seed is None:
        args.seed = 0
        initialize(args, seed=args.seed)

    if not os.path.exists(args.LOG_DIR):
        os.mkdir(args.LOG_DIR)
    
    TIME_NOW = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')
    adaptive = "A" if args.adaptive else ""
    writer = SummaryWriter(log_dir=os.path.join(
                            args.LOG_DIR, # args.task, args.net,
                            args.prefix + 
                            adaptive + "SAM_" + str(args.sam) + 
                            "_bsam_" + str(args.bsam) +
                            "_rho_" + str(args.rho) +
                            "_" + args.optimizer + 
                            "_lr_" + str(args.learning_rate)+
                            '_m_' + str(args.momentum) + 
                            '_wd_' + str(args.weight_decay) +
                            "_seed_" + str(args.seed),
                            TIME_NOW)
                            )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = Cifar(args.batch_size, args.threads)
    log = Log(log_each=10,writer=writer)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)
    
    if args.sam == -1:
        print("SAM ablated!")
        sam = SAM_abl
    elif args.sam == 0:
        print("Original SAM!")
        sam = SAM
    elif args.sam > 0:
        sam = SAM1
    if args.optimizer == 'sgd':
        print("SGD optimizer")
        base_optimizer = torch.optim.SGD
        optimizer = sam(model.parameters(), 
                        base_optimizer, 
                        rho=args.rho, 
                        option=args.sam,
                        adaptive=args.adaptive, 
                        lr=args.learning_rate, 
                        momentum=args.momentum, 
                        weight_decay=args.weight_decay)    
    elif  args.optimizer == 'nero':
        print("Nero optimizer")
        base_optimizer = Nero
        optimizer = sam(model.parameters(), 
                        base_optimizer, 
                        rho=args.rho,
                        option=args.sam, 
                        adaptive=args.adaptive, 
                        lr=args.learning_rate)
    elif  args.optimizer == 'nerov3':
        print("NeroV3 optimizer")
        base_optimizer = Nero_v3
        optimizer = sam(model.parameters(), 
                        base_optimizer, 
                        rho=args.rho,
                        option=args.sam, 
                        adaptive=args.adaptive, 
                        lr=args.learning_rate,
                        model=model, 
                        wd=args.weight_decay)
    else:
        print("optimizer not supported!")

    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))
        n = -1
        for batch in dataset.train:
            n += 1

            inputs, targets = (b.to(device) for b in batch)

            # first forward-backward step
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets)
            loss.mean().backward()
            if not args.sam == 0:
                optimizer.step()
            else:
                if n % args.bsam == 0:
                    optimizer.first_step(zero_grad=True)

                    # second forward-backward step
                #disable_bn(model)
                    smooth_crossentropy(model(inputs), targets).mean().backward()
                    optimizer.second_step(zero_grad=True)
                #enable_bn(model)
                else:
                    optimizer.second_step(zero_grad=True)
                    #optimizer.base_optimizer.step()
                    #optimizer.zero_grad()

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        if args.sam > 0:
            optimizer.clean() # clean weight

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())

    log.flush()
