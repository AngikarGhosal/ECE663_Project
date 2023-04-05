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

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    """This is a multi-layer perceptron with 4 layers"""
    """This uses four Linear layers, with LeakyReLU as the underlying nonlinearization."""
    """The variables nin, nout, nh denote the number of input neurons, number of output neurons and the number of hidden neurons in middle layers"""
    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )
    def forward(self, x):
        return self.net(x)
   
class MaskedLinear(nn.Linear):
    """The MaskedLinear module is similar to a Linear module in Pytorch except that we can configure a mask on the weights"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)        
        self.register_buffer('mask', torch.ones(out_features, in_features))
    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))
        
    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)

class MADE(nn.Module):
    def __init__(self, nin, hidden_sizes, nout, num_masks=1, natural_ordering=False):
        """
        MADE stands for Masked Autoencoder for Density Estimation.
        Here the parameter (nin) represents the number of inputs, while hidden_sizes represents the number of units in the hidden layers.
        The outputs are parameter for the distribution we are trying to model.
        """
        
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"
        self.net = []
        #self.net is a simple MLP Model.
        hs = [nin] + hidden_sizes + [nout]
        for h0,h1 in zip(hs, hs[1:]):
            self.net.extend([
                    MaskedLinear(h0, h1),
                    nn.ReLU(),
                ])
        self.net.pop() 
        self.net = nn.Sequential(*self.net)
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0 # for cycling through num_masks orderings
        self.m = {}
        self.update_masks() 
        
    def update_masks(self):
        if self.m and self.num_masks == 1: return 
        L = len(self.hidden_sizes)
        #This method constructs a random stream based on a seed.
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks
        self.m[-1] = np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
        for l in range(L):
            self.m[l] = rng.randint(self.m[l-1].min(), self.nin-1, size=self.hidden_sizes[l])
        
        # This constructs mask matrices.
        masks = [self.m[l-1][:,None] <= self.m[l][None,:] for l in range(L)]
        masks.append(self.m[L-1][:,None] < self.m[-1][None,:])
        
        # This handles the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            masks[-1] = np.concatenate([masks[-1]]*k, axis=1)
        
        # This sets the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l,m in zip(layers, masks):
            l.set_mask(m)
    
    def forward(self, x):
        return self.net(x)

class ARMLP(nn.Module):

    """ ARMLP refers to Autoregressive MLP, which uses a wrapper around MADE."""
    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = MADE(nin, [nh, nh, nh], nout, num_masks=1, natural_ordering=True)
        
    def forward(self, x):
        return self.net(x)


