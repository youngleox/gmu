import torch
import math
from torch.optim.optimizer import Optimizer, required
def extract_name(string):
    # removes the trailing ".bias" ".weight" from the name
    split = string[::-1].split('.',1)
    if len(split) == 2:
        return split[1][::-1]
    else:
        print('cannot split, return original name')
        return string

def neuron_norm(x):
    if x.dim() > 1:
        view_shape = [x.shape[0]] + [1]*(x.dim()-1)
        x = x.view(x.shape[0],-1)
        return x.norm(dim=1).view(*view_shape)
    else:
        return x.abs()

def neuron_mean(x):
    if x.dim() > 1:
        view_shape = [x.shape[0]] + [1]*(x.dim()-1)
        x = x.view(x.shape[0],-1)
        return x.mean(dim=1).view(*view_shape)
    else:
        raise Exception("neuron_mean not defined on 1D tensors.")

def neuron_prod(x,y):
    if x.dim() > 1:
        view_shape = [x.shape[0]] + [1]*(x.dim()-1)
        x = x.view(x.shape[0],-1)
        y = y.view(y.shape[0],-1)
        return (x*y).sum(dim=1).view(*view_shape)
    else:
        raise Exception("neuron_prod not defined on 1D tensors.")

class Nero_v3(Optimizer):

    def __init__(self, params, model, lr=0.01,lr_gain=0.01, beta=0.999, 
                grad_proj=False, lr_exp_base=1.0, constraints=True,wd=0.1):
        self.pf = True
        if self.pf:
            print("lr:{}, lr grain: {}, beta:{}, lr exp base:{}, \
                    grad projection: {}, constraints:{}".
                    format(lr, lr_gain, beta, lr_exp_base,
                    grad_proj, constraints ))

        self.model = model
        self.lr = lr
        self.lr_gain = lr_gain
        self.beta = beta
        self.grad_proj = grad_proj # enable gradient projection
        self.lr_exp_base = lr_exp_base
        self.constraints = constraints # fix norm
        self.wd = wd
        defaults = dict(lr=lr)
        super(Nero_v3, self).__init__(params, defaults)
        
        self.full_names = [name for name,p in self.model.named_parameters() ]
        for name, p in self.model.named_parameters():
            state = self.state[p]            
            if len(state) == 0:
                # get its fan in if dim > 1
                if p.dim() > 1:
                    state['name'] = name
                    state['tag'] = 'weight'
                    if self.pf:
                        print('====================================================')
                    if (neuron_norm(p) == 0).any().item():
                        torch.nn.init.uniform_(p, a=-0.01, b=0.01)
                        if self.pf:
                            print("Warning: a parameter was reinitialised by CSV8n.")
                    state['lr_mul'] = 1.0
                    if self.pf:
                        print('name: {}, dim: {} ,lr_mul: {}, norm: {}'.format(name, p.dim(),state['lr_mul'],neuron_norm(p).mean()))
        
        if self.pf:
            print("====================end of dim > 1 params")

        for name, p in self.model.named_parameters():
            state = self.state[p]            
            if len(state) == 0:
                if p.dim() == 1: 
                    if self.pf:
                        print('====================================================')

                    if (('weight' in name) and (('bn' in name) or ('batch' in name) or ("norm" in name))) or ('gain' in name):
                        state['lr_mul'] = self.lr_gain
                        state['name'] = name
                        state['tag'] = 'gain'
                        if self.pf:
                            print("this is bn weight")
                    else:
                        state['lr_mul'] = 1.0
                        state['name'] = name
                        state['tag'] = 'bias'
                    if self.pf:
                        print('name: {}, dim: {} ,lr_mul: {}, norm: {}'.format(name, p.dim(),state['lr_mul'],neuron_norm(p).mean()))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if not 'step' in state:
                    state['step'] = 0
                    state['exp_avg_sq'] = torch.zeros_like(neuron_norm(p.grad))

                if self.constraints and self.grad_proj and p.dim() > 1:
                    p.grad.data -= neuron_mean(p.grad)
                    p.grad.data -= neuron_prod(p.grad, p) * p

                exp_avg_sq = state['exp_avg_sq']

                state['step'] += 1                
                bias_correction = 1 - self.beta ** state['step']

                rt_avg_sq = neuron_norm(p.grad)
                exp_avg_sq.mul_(self.beta).addcmul_(rt_avg_sq, rt_avg_sq, value=1-self.beta)
                
                denom = (exp_avg_sq/bias_correction).sqrt()
                grad_normed = p.grad / denom
                #grad_normed /= neuron_norm(grad_normed).pow(0.5).clamp(min=1.0)
                grad_normed /= (neuron_norm(grad_normed)/5).clamp(min=1.0)
                
                if state['tag'] == 'gain':
                    grad_normed += self.wd * (p.data - torch.ones_like(p.data))

                grad_normed[torch.isnan(grad_normed)] = 0
                p.data.add_(grad_normed, alpha=-group['lr'])

                if self.constraints and p.dim() > 1:                    
                    p.data -= neuron_mean(p)
                    p.data /= neuron_norm(p)
        
        return loss
