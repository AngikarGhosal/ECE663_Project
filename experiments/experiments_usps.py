import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from individual_flows import *
from normalizing_flow import *
from necessary_classes import *

import torchvision
from torchvision.datasets import USPS
from torchvision.datasets import KMNIST
from torchvision import transforms
import torch.utils.data as data

from torch.distributions import MultivariateNormal
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import distributions
import torch.nn as nn
import os
import torch
from mpl_toolkits.mplot3d import Axes3D



def show_imgs(imgs, title=None, row_size=4, filename='Myfile.jpg'):
    #This function forms a grid of pictures 
    num_imgs = imgs.shape[0] if isinstance(imgs, torch.Tensor) else len(imgs)
    is_int = imgs.dtype==torch.int32 if isinstance(imgs, torch.Tensor) else imgs[0].dtype==torch.int32
    nrow = min(num_imgs, row_size)
    ncol = int(math.ceil(num_imgs/nrow))
    imgs = torchvision.utils.make_grid(imgs, nrow=nrow, pad_value=128 if is_int else 0.5)
    np_imgs = imgs.cpu().numpy()
    # Plot the grid
    plt.figure(figsize=(1.5*nrow, 1.5*ncol))
    plt.imshow(np.transpose(np_imgs, (1,2,0)), interpolation='nearest')
    plt.axis('off')
    if title is not None:
        plt.title(title)
    myfilename=filename
    plt.savefig(myfilename)
    plt.close()

def sample_img(model, iter, myfilename):
  model.eval()
  #At a particular iteration, the model samples 20 images and displays them.
  zs = model.sample(20)
  z = zs[-1]
  z = z.cpu().detach()
  show_imgs([z[i, :].reshape(1, 16, 16) for i in range(20)], title=f'Results at epoch {iter}', filename=myfilename)

def train_img_model(model, dataloader, n_epochs):
  #This wrapper enables us to run a Flow model for a particular image dataset.
  #For a given number of epochs, we enumerate through the dataloader, and optimize the loss function.
  optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
  model.train()
  for epoch in range(n_epochs):
    epoch_loss = 0
    n_items = 0
    for i, (image,_) in enumerate(dataloader):
      x = torch.flatten(image, start_dim=1)
      zs, prior_logprob, log_det = model(x)
      logprob = prior_logprob + log_det
      loss = -torch.sum(logprob) # NLL
      model.zero_grad()
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()
      n_items += x.size(0)
    print(f'{epoch}/{n_epochs} average NLL loss: {epoch_loss/n_items}')
  return model

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((16,16)),
     transforms.Normalize((0.5,), (0.5,)),
     ])
usps_dataset = torchvision.datasets.USPS(root='./usps_16', download=True, transform=transform)

usps_idx = np.array(usps_dataset.targets) == 8
usps_dataset.data = usps_dataset.data[usps_idx]
usps_dataset.targets = np.array(usps_dataset.targets)[usps_idx]
usps_loader = data.DataLoader(usps_dataset, batch_size=64, shuffle=True)

show_imgs([usps_dataset[i][0] for i in range(8)], filename='USPS_8.jpg')

dim = 256
#Images are 16x16 dimensional, hence dim=256 (on flattening the image, we get a 256 dimensional vector)
prior = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
flows = [AffineHalfFlow(dim=dim, parity=i%2) for i in range(9)]
model = NormalizingFlowModel(prior, flows)
model = model

S=0
for i in range(30):
    S=S+100 
    model=train_img_model(model,usps_loader, 100)
    filename='After_'+str(S)+'_8.jpg'
    sample_img(model, S, filename)