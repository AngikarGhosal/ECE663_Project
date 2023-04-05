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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 
class NormalizingFlow(nn.Module):
    """This module enables us to create a sequence of Normalizing Flows"""

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        #log_det stores the sum of the logarithms of the determinants
        #zs is an array storing the result of transformation of x for each of the flows
        m, _ = x.shape
        log_det = torch.zeros(m)
        zs = [x]
        for flow in self.flows: 
            x, ld = flow.forward(x) 
            log_det += ld  
            zs.append(x) 
        return zs, log_det

    def backward(self, z):
        #again, log_det stores the sum of the logarithms of the determinants
        #xs is an array storing the backward transformation of z for each of the flows
        #these flows occur in reverse order as shown by the self.flows[::-1]
        m, _ = z.shape
        log_det = torch.zeros(m)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
            xs.append(z)
        return xs, log_det
    

class NormalizingFlowModel(nn.Module):
    """"This wrapper is used for preprocessing input data to feed into the Normalizing Flow Model by defining the prior distribution"""
    """The forward function here also reshapes the data when the log_prob is calculated in the forward method"""
    """ A Normalizing Flow Model is a (prior, flow) pair """
    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows)
    
    def forward(self, x):
        zs, log_det = self.flow.forward(x)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z):
        xs, log_det = self.flow.backward(z)
        return xs, log_det
    
    def sample(self, num_samples): 
        #This sampling operation occurs in an inverse way where we sample from the prior, and then run the backward method.
        z = self.prior.sample((num_samples,))
        xs, _ = self.flow.backward(z)
        return xs
    