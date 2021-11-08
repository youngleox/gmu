import sys
import math
import numpy as np
import torch
from torch import Tensor
from torch.nn.init import constant_
from torch.nn.modules import Module
from torch.nn import functional as F

def stats(x):
    with torch.no_grad():
        if x.dim() == 4:
            #print(x.shape)
            #print( x.mean(dim=(0,2,3)).shape )
            if x.shape[0] > 2:
                return torch.mean(x.mean(dim=0)),torch.mean(x.std(dim=0))
            else:
                return 0,1
        elif x.dim() == 2:
            if x.shape[0] > 2:
                return torch.mean(x.mean(dim=0)),torch.mean(x.std(dim=0))
            else:
                return 0,1
        else:
            
            return 0,1

class OneshotNormalizer2D(Module):

    def __init__(self,freq=math.inf):
        super().__init__()
        self.freq = freq
        self.mean = None
        self.std = None
        self.step = 0
    def forward(self, x):
        self.step += 1
        if self.mean is None or self.step % self.freq == 0:
            self.mean,self.std = stats(x)
        
        return (x - self.mean) / self.std 