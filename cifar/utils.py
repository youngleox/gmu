""" helper function

author baiyu
"""

import sys

import numpy
from pandas.core.computation.ops import Op

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# Yang
import cifar
import torch.nn.functional as F
from matplotlib import pyplot

def get_network(args):
    """ return given network
    """
    if args.task == 'cifar10':
        nclass = 10
    elif args.task == 'cifar100':
        nclass = 100
    #Yang added none bn vggs
    if args.net == 'vgg11':
        from models.vgg import vgg11
        net = vgg11(num_classes=nclass)
    elif args.net == 'vgg13':
        from models.vgg import vgg13
        net = vgg13(num_classes=nclass)
    elif args.net == 'vgg16':
        from models.vgg import vgg16
        net = vgg16(num_classes=nclass)
    elif args.net == 'vgg19':
        from models.vgg import vgg19
        net = vgg19(num_classes=nclass) 
    
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(num_classes=nclass)
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(num_classes=nclass)
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(num_classes=nclass)
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(num_classes=nclass)
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152(num_classes=nclass)

    # Resnet v2 (pre-act)
    elif args.net == 'resnet18v2':
        from models.resnet_v2 import resnet18v2
        net = resnet18v2(num_classes=nclass)
    elif args.net == 'resnet34v2':
        from models.resnet_v2 import resnet34v2
        net = resnet34v2(num_classes=nclass)
    elif args.net == 'resnet50v2':
        from models.resnet_v2 import resnet50v2
        net = resnet50v2(num_classes=nclass)
    elif args.net == 'resnet101v2':
        from models.resnet_v2 import resnet101v2
        net = resnet101v2(num_classes=nclass)
    elif args.net == 'resnet152v2':
        from models.resnet_v2 import resnet152v2
        net = resnet152v2(num_classes=nclass)
    
    # Wide Resnet
    elif args.net == 'wideresnet28':
        from models.wideresnet import wideresnet28
        net = wideresnet28(num_classes=nclass)

    #Yang added none bn vggs
    elif args.net == 'vgg11y':
        from models.vgg_y import vgg11
        net = vgg11(num_classes=nclass)
    elif args.net == 'vgg13':
        from models.vgg import vgg13
        net = vgg13(num_classes=nclass)
    elif args.net == 'vgg16':
        from models.vgg import vgg16
        net = vgg16(num_classes=nclass)
    elif args.net == 'vgg19':
        from models.vgg import vgg19
        net = vgg19(num_classes=nclass) 

    elif args.net == 'resnet18y':
        from models.resnet_y import resnet18
        net = resnet18(num_classes=nclass)
    elif args.net == 'resnet34':
        from models.resnet_y import resnet34
        net = resnet34(num_classes=nclass)
    elif args.net == 'resnet50y':
        from models.resnet_y import resnet50
        net = resnet50(num_classes=nclass)
    elif args.net == 'resnet101y':
        from models.resnet_y import resnet101
        net = resnet101(num_classes=nclass)
    elif args.net == 'resnet152y':
        from models.resnet_y import resnet152
        net = resnet152(num_classes=nclass)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net

from PIL import Image
import os
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class CIFAR10S(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            custom_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(CIFAR10S, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        self.custom_transform = custom_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.custom_transform is not None:
            img, target = self.custom_transform(img,target)
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")

def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True,
                            task='cifar10',da=0,
                            sigma_noise=0.0, pow_noise=1.0, bg_noise=1.0,
                            sigma_crop=10.0, pow_crop=4.0, bg_crop=1.0,
                            sigma_cut=0.0, pow_cut=4.0,
                            length_cut=16, mask_cut=1):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    custom_transform = None
    if da == -1:
        print("no data augmentation!")
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    elif da == 0: # original augmentation
        print("standard data augmentation!")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif da == 1: # original augmentation
        print("standard data augmentation with cutout!")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout(n_holes=1,length=length_cut,mask=mask_cut)
        ])
    elif da == 3:
        print("original crop + custom cutout")
        transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        #custom_transform = SoftCrop(spread=spread,T=T,max_p=max_p)
        custom_transform = SoftNoiseCropCut(
                            n_class=10,
                            sigma_noise=sigma_noise, t_noise=1.0, max_p_noise=1.0, pow_noise=pow_noise, bg_noise=bg_noise,
                            sigma_crop=sigma_crop, t_crop=1.0, max_p_crop=1.0, pow_crop=pow_crop, bg_crop=bg_crop,
                            sigma_cut=sigma_cut, t_cut=1.0, max_p_cut=1.0, pow_cut=pow_cut, # this line is not being used
                            length_cut=length_cut,mask_cut=mask_cut)
    else:
        print("custom data augmentation")
        transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            #transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        #custom_transform = SoftCrop(spread=spread,T=T,max_p=max_p)
        custom_transform = SoftNoiseCropCut(
                            n_class=10,
                            sigma_noise=sigma_noise, t_noise=1.0, max_p_noise=1.0, pow_noise=pow_noise, bg_noise=bg_noise,
                            sigma_crop=sigma_crop, t_crop=1.0, max_p_crop=1.0, pow_crop=pow_crop, bg_crop=bg_crop,
                            sigma_cut=sigma_cut, t_cut=1.0, max_p_cut=1.0, pow_cut=pow_cut, # this line is not being used
                            length_cut=length_cut,mask_cut=mask_cut)
    if task == 'cifar100':
        cifar100_training = cifar.CIFAR100(root='./data', train=True, download=True, transform=transform_train)#,alpha=alpha)
    elif task == 'cifar10':
        cifar100_training = CIFAR10S(root='./data', train=True, download=True, transform=transform_train,custom_transform=custom_transform)#,alpha=alpha)
    
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size,drop_last=False)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True,task="cifar100",train=False):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    if task == "cifar100":
        cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transform_test)
    elif task == "cifar10":
        cifar100_test = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform_test)
    
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def ComputeProb(x, T=0.25, n_classes=10, max_prob=1.0, pow=2.0):
    max_prob = torch.clamp_min(torch.tensor(max_prob),1/n_classes)
    if T <=0:
        T = 1e-10

    if x > T:
        return max_prob
    elif x > 0:
        a = (max_prob - 1/float(n_classes))/(T**pow)
        return max_prob - a * (T-x) ** pow
    else:
        return np.ones_like(x) * 1/n_classes

def DecodeTargetProb(targets):
    '''
    Helper function, takes targets as input, splits it into GT classes and probability
    if a target is 7.2, then the GT class is 7 with probability 1 - 0.2 = 0.8.
    '''
    classes = targets.long()
    probs = 1 - (targets - classes)
    return classes, probs

def EncodeTargetProb(classes,probs=None):
    '''
    Helper function, takes GT classes and probabilities as input, 
    outputs a combined encoding with integer part encoding GT class
    and decimal part encoding 1-probability
    if the GT class for a sample is 7 with probability 0.8
    then target is 7 + (1- 0.8).
    caveat: input probability should be greater than 0
    otherwise the output class will be wrong
    '''
    if probs is None:
        return classes.float()
    else:
        return classes.float() + 1 - probs

# official cutout implementation
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length, mask=0):
        self.n_holes = n_holes
        self.length = length
        self.mask = mask
        print("noise mask: ", str(self.mask))

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)
        dim1 = img.size(1)
        dim2 = img.size(2)
        
        # noise mix
        bg_n = torch.rand((3,dim1,dim2))
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            if self.mask:
                img[:,int(y1): int(y2), int(x1): int(x2)] = bg_n[:,int(y1): int(y2), int(x1): int(x2)]
            else:
                mask[int(y1): int(y2), int(x1): int(x2)] = 0.

                mask = torch.from_numpy(mask)
                mask = mask.expand_as(img)
                img = img * mask

        return img

class SoftCrop: 
    def __init__(self, spread=16, p=1.0, T=0.25, max_p=0.9, pow=2.0):
        #self.size = size
        #self.half_size = size // 2
        self.p = p
        self.spread = spread
        self.flag = False
        self.T = T
        self.max_p = max_p
        self.pow = pow
        print("use soft crop")
        print("spread: ", self.spread, " T: ", self.T, " Max P: ", self.max_p)
    
    def draw(self, spread=1, limit=24, n=100):
        for d in range(n):
            x = torch.randn((1))*spread
            if abs(x) <= limit:
                return int(x)
        return int(0)

    def __call__(self, image, label):
        if torch.rand([1]).item() > self.p:
            return image,label
        #print(image.size())
        dim1 = image.size(1)
        dim2 = image.size(2)
        bg = torch.rand((3,dim1*3,dim2*3)) # create a 3x by 3x sized noise background
        bg[:,dim1:2*dim1,dim2:2*dim2] = image # put image at the center patch
        offset1 = self.draw(self.spread,dim1)
        offset2 = self.draw(self.spread,dim1)
        
        left = offset1 + dim1
        top = offset2 + dim2
        right = offset1 + dim1 * 2
        bottom = offset2 + dim2 * 2

        overlap = (dim1 - abs(offset1))*(dim2 - abs(offset2)) / (dim1 * dim2)
        prob = ComputeProb(overlap,T=self.T,max_prob=self.max_p,pow=self.pow)
        if prob is None:
            print('prob is None:')
            print(overlap)
        #print('overlap: ', overlap, " probability: ", prob)
        new_image = bg[:, left: right, top: bottom] # crop image
        if not self.flag:
            #img = Image.fromarray(np.array(new_image*2+0.5))
            self.flag = True
        return new_image, label + 1 - prob

class SoftNoiseCropCut: 
    '''
    Apply three transforms in order: 
    (1) mix with noise
    (2) cut part of the image out 
    (3) crop image
    
    '''
    def __init__(self, n_class=10,
                 sigma_noise=0.01, t_noise=1.0, max_p_noise=1.0, pow_noise=1.0, bg_noise=1.0,
                 sigma_crop=10, t_crop=1.0, max_p_crop=1.0, pow_crop=4.0, bg_crop=1.0,
                 sigma_cut=0, t_cut=1.0, max_p_cut=1.0, pow_cut=4.0,
                 length_cut=16, mask_cut=1):
        
        self.chance = 1/n_class
        # noise mix parameters
        self.sigma_noise = sigma_noise
        self.t_noise = t_noise
        self.max_p_noise = max_p_noise
        self.pow_noise = pow_noise
        self.bg_noise = bg_noise
        # crop parameters
        self.sigma_crop = sigma_crop
        self.t_crop = t_crop
        self.max_p_crop = max_p_crop
        self.pow_crop = pow_crop
        self.bg_crop = bg_crop
        # cutout parameters
        self.sigma_cut = sigma_cut
        self.t_cut = t_cut
        self.max_p_cut = max_p_cut
        self.pow_cut = pow_cut
        self.length_cut = length_cut
        self.mask_cut = mask_cut # mask type, 0: zero mask, 1: noise mask
        #for debugging
        self.flag = True
        print("use soft noise")
        print("sigma: ", self.sigma_noise, " T: ", self.t_noise, " Max P: ", self.max_p_noise,
                "bg: ", self.bg_noise)
        print("use soft crop")
        print("sigma: ", self.sigma_crop, " T: ", self.t_crop, " Max P: ", self.max_p_crop,
                "bg: ", self.bg_crop)
        print("use soft cutout")
        print("sigma: ", self.sigma_cut, " T: ", self.t_cut, " Max P: ", self.max_p_cut,
                "length: ", self.length_cut, " mask: ", self.mask_cut)
    
    def draw_offset(self, sigma=1, limit=24, n=100):
        # draw an integer from gaussian within +/- limit
        for d in range(n):
            x = torch.randn((1))*sigma
            if abs(x) <= limit:
                return int(x)
        return int(0)
    
    def draw_ratio(self, sigma, n=100):
        # returns a 0-1 ratio for mixing
        # draw n times, if not getting a good x then return 0.0
        for d in range(n):
            x = torch.randn((1))*sigma
            if abs(x) <= 1:
                return torch.abs(x)
        return torch.tensor(0.0)
    
    def draw_length(self, sigma, n=100):
        # returns a -1 - 1 ratio for mixing
        # draw n times, if not getting a good x then return 0.0
        for d in range(n):
            x = torch.randn((1))*sigma
            if abs(x) <= 1:
                return x
        return torch.tensor(0.0)

    def draw_length2(self, sigma, n=100):
        # returns a 0-1 ratio for mixing
        # draw n times, if not getting a good x then return 0.0
        for d in range(n):
            x = torch.randn((1))*sigma
            if abs(x) <= 1:
                return torch.abs(x)
        return torch.tensor(0.0)

    def __call__(self, image, label):
        # if torch.rand([1]).item() > self.p:
        #     return image,label
        #print(image.size())
        dim1 = image.size(1)
        dim2 = image.size(2)
        
        # noise mix
        bg_n = torch.rand((3,dim1,dim2)) * self.bg_noise
        ratio = 1 - self.draw_ratio(self.sigma_noise)
        #print("ratio: ", str(ratio))
        image_mix = image * ratio  + bg_n * torch.sqrt(1-ratio**2)
        prob_mix = ComputeProb(float(ratio),T=self.t_noise,max_prob=self.max_p_noise,pow=self.pow_noise)

        # cutout, using official implementation with fixed cutout size
        # one important difference being the cutout region being 
        # masked by noise instead of 0.
        h, w = dim2, dim1 # converting to official code variable names
        y = np.random.randint(h)
        x = np.random.randint(w)
        length_cut =  int(self.length_cut * (1 + self.draw_length(self.sigma_cut)))
        #length_cut = int(torch.abs(torch.randn((1))) * self.length_cut * np.sqrt(np.pi/2))
        # this set of variables are used for cut out
        y1 = np.clip(y - length_cut // 2, 0, h) # top
        y2 = np.clip(y + length_cut // 2, 0, h) # bottom
        x1 = np.clip(x - length_cut // 2, 0, w) # left
        x2 = np.clip(x + length_cut // 2, 0, w) # right
        if self.mask_cut:
            image_mix[:,x1:x2,y1:y2] = bg_n[:,x1:x2,y1:y2]
        else:
            image_mix[:,x1:x2,y1:y2] = 0

        # Soft Crop
        bg = torch.rand((3,dim1*3,dim2*3)) * self.bg_crop # create a 3x by 3x sized noise background
        bg[:,dim1:2*dim1,dim2:2*dim2] = image_mix # put image at the center patch
        offset1 = self.draw_offset(self.sigma_crop,dim1)
        offset2 = self.draw_offset(self.sigma_crop,dim1)
        
        # this set of variables are used to compute how much 
        # is cut out in the cropped image
        y1c = np.clip(y - length_cut // 2, max(0,offset2), min(h,h+offset2))
        y2c = np.clip(y + length_cut // 2, max(0,offset2), min(h,h+offset2))
        x1c = np.clip(x - length_cut // 2, max(0,offset1), min(w,w+offset1))
        x2c = np.clip(x + length_cut // 2, max(0,offset1), min(w,w+offset1))

        left = offset1 + dim1
        top = offset2 + dim2
        right = offset1 + dim1 * 2
        bottom = offset2 + dim2 * 2


        # number of pixels in orignal image kept after cropping alone
        overlap_crop = (dim1 - abs(offset1))*(dim2 - abs(offset2))
        # number of pixels in original image kept after cropping and cutout
        overlap_final = overlap_crop - (x2c - x1c) * (y2c - y1c)
        # proportion of original pixels left after cutout and cropping
        overlap = overlap_final / (dim1 * dim2)
        # now the max prob can not be larger than prob_mix
        prob_crop = ComputeProb(overlap,T=self.t_crop,max_prob=self.max_p_crop*prob_mix,pow=self.pow_crop)
        #print("crop prob")
        #print(prob_crop)
        #print('overlap: ', overlap, " probability: ", prob)
        new_image = bg[:, left: right, top: bottom] # crop image
        if not self.flag:
            print("mix ratio: ", str(ratio))
            print("offsets: ", str(offset1), str(offset2))
            print("cutout: ", str(x1), str(x2), str(y1), str(y2))
            print("clipped cutout: ", str(x1c), str(x2c), str(y1c), str(y2c))
            img = (np.array(new_image/4+0.5))
            img = (np.array(image_mix/4+0.5))
            img = np.swapaxes(img,0,2)
            img = np.swapaxes(img,0,1)
            pyplot.imshow(img)
            pyplot.show()
            self.flag = True
        new_label = label + 1 - prob_crop #max(prob_crop*prob_mix,self.chance)
        #print(new_label)
        return new_image, new_label

# mixup code
POW = 1
@torch.no_grad()
def mixup_data(x, y, alpha=1.0, use_cuda=True, pow=POW, t=1.0, n_classes=1e6):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1 - 1e-8

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    y_int = y.long()

    y_prob = (1 - y + y_int)
    prob_a = ComputeProb(lam, max_prob=y_prob, pow=pow,T=t,n_classes=n_classes)
    prob_b = ComputeProb((1 - lam), max_prob=y_prob[index], pow=pow,T=t,n_classes=n_classes)

    y_a = torch.tensor(y_int + 1 - prob_a)
    y_b = torch.tensor(y_int[index] + 1 - prob_b)
    #print("lam: ", lam)
    #print('y_a: ', y_a)
    #print('y_b: ', y_b)
    return mixed_x, y_a, y_b, lam

@torch.no_grad()
def mixup_data2(x, y, alpha=1.0, use_cuda=True, pow=POW, t=1.0, n_classes=1e6):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1 - 1e-8

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + np.sqrt(1 - lam ** 2) * x[index, :]

    y_int = y.long()

    y_prob = (1 - y + y_int)
    #print("y_prob: ", y_prob[0])
    #print("lam: ", lam)
    prob_a = ComputeProb(lam,max_prob=y_prob, pow=pow,T=t,n_classes=n_classes)
    prob_b = ComputeProb(np.sqrt(1 - lam ** 2),max_prob=y_prob[index],pow=pow,T=t,n_classes=n_classes)
    #print("prob_a: ", prob_a[0])
    y_a = torch.tensor(y_int + 1 - prob_a)
    y_b = torch.tensor(y_int[index] + 1 - prob_b)

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_criterion2(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + np.sqrt(1 - lam ** 2) * criterion(pred, y_b)


def soft_target(pred, gold, positive_only=False):
    #print(gold)
    target = gold.long()
    prob =  1 - (gold - target)
    n_class = pred.size(1)
    #print('target: ', target[0])
    one_hot = (torch.ones_like(pred) * (1 - prob.unsqueeze(1)) / (n_class - 1)).float()
    #print('prob: ', prob[0])
    #print('one hot before: ', one_hot[0,:])
    one_hot.scatter_(dim=1, index=target.unsqueeze(1), src=prob.unsqueeze(1).float())
    one_hot_binary = torch.zeros_like(pred)
    one_hot_binary.scatter_(dim=1,index=target.unsqueeze(1), value=1.0)
    #print('one hot after: ', one_hot[0,:])
    #print("one hot binary: ", one_hot_binary[0,:])
    log_prob = F.log_softmax(pred, dim=1)
    #log_prob = pred
    if positive_only:
        kl = F.kl_div(input=log_prob.float(), target=one_hot.float(), reduction='none')
        return (kl * one_hot_binary).sum(-1).mean()
    else:
        return F.kl_div(input=log_prob.float(), target=one_hot.float(), reduction='none').sum(-1).mean()

def soft_target_mixup(pred, gold_a, gold_b):
    #print(gold)
    n_class = pred.size(1)

    target_a = gold_a.long()
    prob_a =  torch.clamp_min_(1 - (gold_a - target_a), 1/n_class)

    target_b = gold_b.long()
    prob_b =  torch.clamp_min_(1 - (gold_b - target_b), 1/n_class)

    prob_denom = torch.clamp_min_(prob_a + prob_b, 1.0+1e-8)

    prob_a_norm = prob_a/prob_denom
    prob_b_norm = prob_b/prob_denom
    #print('target: ', target[0])
    one_hot = (torch.ones_like(pred) * (1 - prob_a_norm.unsqueeze(1) - prob_b_norm.unsqueeze(1) ) / (n_class - 2)).float()
    #print('prob: ', prob[0])
    #print('one hot before: ', one_hot[0,:])
    one_hot.scatter_(dim=1, index=target_a.unsqueeze(1), src=prob_a.unsqueeze(1).float())
    one_hot.scatter_(dim=1, index=target_b.unsqueeze(1), src=prob_b.unsqueeze(1).float())
    #one_hot_binary = torch.zeros_like(pred)
    #one_hot_binary.scatter_(dim=1,index=target.unsqueeze(1), value=1.0)
    #print('one hot after: ', one_hot[0,:])
    #print("one hot binary: ", one_hot_binary[0,:])
    log_prob = F.log_softmax(pred, dim=1)
  
    return F.kl_div(input=log_prob.float(), target=one_hot.float(), reduction='none').sum(-1).mean()


def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1).mean()

# alpha = 4
# concentration = torch.ones((10000,1))*alpha
# m = torch.distributions.beta.Beta(concentration,concentration)
# m = np.random.beta(alpha, alpha,size=(10000,1))

# print(m)
# from matplotlib import pyplot
# pyplot.hist(np.array(m))
# pyplot.show()
# for i in range(0,100):
#     print(ComputeProb(i*0.01,T=0.5,max_prob=1.0))