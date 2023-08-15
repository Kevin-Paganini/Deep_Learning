import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim import Optimizer



class Adam(Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    The implementation of the L2 penalty follows changes proposed in
    `Decoupled Weight Decay Regularization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)
        
        
        # initializing momentum dictionaries
        self.beta1_momentum_dict = {}
        self.beta2_momentum_dict = {}
        
        for i, p in enumerate(self.param_groups[0]['params']):
            self.beta1_momentum_dict[i] = torch.zeros_like(p)
            self.beta2_momentum_dict[i] = torch.zeros_like(p)
        self.t = 1


    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
    
    @torch.no_grad()
    def step(self):
        # print(self.param_groups[0]['lr'], '\n')
        lr = self.param_groups[0]['lr']
        betas = self.param_groups[0]['betas']
        eps = self.param_groups[0]['eps']
        weight_decay = self.param_groups[0]['weight_decay']
       
        # for each parameter set (weight layer, bias, etc.)
        for i, p in enumerate(self.param_groups[0]['params']):
            
            # getting grad
            g = p.grad
            
            # This would just be stochastic gradient descent
            # p -= lr * g
            
            # momentum 
            v = betas[0] * self.beta1_momentum_dict[i] + (1 - betas[0])* g
            self.beta1_momentum_dict[i] = v
            
            # RMSProp
            s = betas[1] * self.beta2_momentum_dict[i] + (1 - betas[1])* g**2
            self.beta2_momentum_dict[i] = s
            
            # correcting
            v_corr = v / (1 - betas[0]**self.t)
            s_corr = s / (1 - betas[1]**self.t)
            
            # updating parameter
            p -= lr * (v_corr / torch.sqrt(s_corr + eps))
            
        self.t += 1


