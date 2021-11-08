import torch

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, option=1, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
                self.state[p]["prev_grad"] = p.grad
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        sim = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                cos = torch.mean(self.cos(p.grad,self.state[p]["e_w"]))
                print(cos)
                #sim.extend(cos)
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

class SAM_abl(torch.optim.Optimizer):
    '''
    Fake SAM optimizer that just implements base_optimizer
    '''
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, option=1, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM_abl, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
    @torch.no_grad()
    def step(self, closure=None):
        self.base_optimizer.step()
        self.zero_grad()

class SAM1(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, option=1, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM1, self).__init__(params, defaults)
        self.option = option
        print("SAM option:",str(self.option))
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state['step'] = 0
                #state['exp_avg_sq'] = torch.zeros_like(neuron_norm(p))
                state['buffer'] = p.clone()

    @torch.no_grad()
    def step(self, closure=None):
        
        for group in self.param_groups:        
            for p in group["params"]:
                state = self.state[p]
                if p.grad is None: continue
                p = state['buffer'].clone()

        self.base_optimizer.step()

        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)            
            for p in group["params"]:
                state = self.state[p]
                if p.grad is None: continue
                state['buffer'] = p.clone()

                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                if self.option == 1: 
                    p.add_(torch.sign(torch.randn_like(p)) * e_w)  # climb to the local maximum "w + e(w)"
                elif self.option == 2:
                    p.add_(torch.randn_like(p) * torch.std(e_w))
                elif self.option == 3:
                    p.add_(torch.randn_like(p) * torch.std(p.grad) * group['rho'])
                
                self.state[p]["e_w"] = e_w

        self.zero_grad()

    @torch.no_grad()
    def clean(self, closure=None):
        for group in self.param_groups:     
            for p in group["params"]:
                state = self.state[p]
                if p.grad is None: continue
                p = state['buffer'].clone()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm