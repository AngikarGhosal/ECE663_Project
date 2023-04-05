import numpy as np 
import matplotlib.pyplot as plt 
import math 
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import os
from torch import distributions
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal
import pytorch_lightning as pl

from necessary_classes import *

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AffineConstantFlow(nn.Module):
    """ 
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """
    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if scale else None
        self.t = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if shift else None
        
    def forward(self, x):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class AffineHalfFlow(nn.Module):
    """
    The AffineHalfFlow module implements both RealNVP and NICE, depending on the
    scale and shift parameters, with RealNVP being the default module. These affine
    autoregressive flow methods use z=x*(e^s)+t as the linear transformation for half of the dimensions in x, while the other 
    dimensions in x are left untouched.
    We implement this here by choosing the odd dimensions and the even dimensions.
    We have a parity bit which can be even or odd.
    If the parity bit is even, the even dimensions (0,2,4,...) are left untouched,
    while the odd dimensions (1,3,5,...) are transformed by the normalizing flow modules.
    If the parity bit is odd, the odd dimensions (1,3,5,...) are left untouched,
    while the even dimensions (0,2,4,...) are transformed by the normalizing flow modules.
    This is essentially an example of bit masking.
    If the parameter scale is set to False, the scaling by (e^s) will not happen, thus it is no longer Non-Volume Preserving (NVP). 
    Thus setting scale to False is equivalent to the NICE algorithm, as taught in class. The shift due to t occurs in both RealNVP and NICE.
    """
    def __init__(self, dim, parity, net_class=MLP, nh=24, scale=True, shift=True):
        super().__init__()
        self.dim = dim
        self.parity = parity
        #in case scale is false, multiplying by e^(s_cond) will keep the tensor as it is.
        #by default, nh=24, i.e., 24 neurons in the hidden layers, as a parameter in MLP.
        self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        if scale:
            self.s_cond = net_class(self.dim // 2, self.dim // 2, nh)
        if shift:
            self.t_cond = net_class(self.dim // 2, self.dim // 2, nh)
        #s_cond and t_cond have dim//2 as they only operate on half the dimensions.
        
    def forward(self, x):
        x0, x1 = x[:,::2], x[:,1::2]
        if self.parity:
            x0, x1 = x1, x0
        #In the slides in class, F and H are the functions for transformation, they are s and t here respectively.
        s = self.s_cond(x0)
        t = self.t_cond(x0)
        z0 = x0
        #the x0 is the dimensions which are directly sent through
        z1 = torch.exp(s) * x1 + t 
        #the x1 is the dimensions which are transformed
        if self.parity:
            z0, z1 = z1, z0
        # z = torch.cat([z0, z1], dim=1)
        z = torch.zeros((x.shape[0], self.dim))  
        z[:,::2] = z0
        z[:,1::2] = z1
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z):
        #The parity bit functions the same way for the backward method as in the forward method by choosing the odd dimensions and the even dimensions.
        z0, z1 = z[:,::2], z[:,1::2]
        if self.parity:
            z0, z1 = z1, z0
        s = self.s_cond(z0)
        t = self.t_cond(z0)
        x0 = z0
        #z0 was the part of z which was directly sent through from x, hence x0 remains z0.
        x1 = (z1 - t) * torch.exp(-s) 
        #z1 was the part of z which was the transformed part of x, hence we need to invert that transformation.
        if self.parity:
            x0, x1 = x1, x0
        x = torch.zeros((z.shape[0], self.dim))  
        x[:,::2] = x0
        x[:,1::2] = x1
        log_det = torch.sum(-s, dim=1)
        return x, log_det
    
class MAF(nn.Module):
    """ Masked Autoregressive Flow that uses a MADE-style network for fast likelihood estimation, but the sampling is much slower"""
    def __init__(self, dim, parity, net_class=ARMLP, nh=24):
        super().__init__()
        self.dim = dim
        self.net = net_class(dim, dim*2, nh)
        self.parity = parity

    def forward(self, x):
        #The likelihood estimation is fast as all dimensions of z are evaluated in parallel.
        st = self.net(x)
        s, t = st.split(self.dim, dim=1)
        z = x * torch.exp(s) + t
        z = z.flip(dims=(1,)) if self.parity else z
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z):
        # The decoding and sampling of x is done sequentially.
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.size(0))
        z = z.flip(dims=(1,)) if self.parity else z
        for i in range(self.dim):
            st = self.net(x.clone()) 
            s, t = st.split(self.dim, dim=1)
            x[:, i] = (z[:, i] - t[:, i]) * torch.exp(-s[:, i])
            log_det += -s[:, i]
        return x, log_det

class IAF(MAF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        IAF reverses the flow of MAF, giving an Inverse Autoregressive Flow instead.
        The sampling is fast as all the dimensions of x are evaluated in parallel, while the dimensions of z are evaluated sequentially.
        """
        self.forward, self.backward = self.backward, self.forward