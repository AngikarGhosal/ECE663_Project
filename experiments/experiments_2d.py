import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from individual_flows import *
from normalizing_flow import *
from necessary_classes import *

from torch.distributions import MultivariateNormal
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import distributions
import torch.nn as nn
import os
import torch
from mpl_toolkits.mplot3d import Axes3D


from sklearn.datasets import make_circles, make_moons

#device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print("DEVICE : ",device)

class DatasetMoons:
	def sample(self,n):
		moons=make_moons(n_samples=n, noise=0.05)[0].astype(np.float32)
		return torch.from_numpy(moons)
class DatasetCircles:
	def sample(self,n):
		circles=make_circles(n_samples=n, noise=0.05)[0].astype(np.float32)
		return torch.from_numpy(circles)
dataset=DatasetMoons()
x=dataset.sample(1000)
plt.figure(figsize=(5,5))
plt.scatter(x[:,0],x[:,1],s=5,alpha=0.5)
plt.axis('equal')
plt.savefig('Moons.jpg')

prior=MultivariateNormal(torch.zeros(2), torch.eye(2))
flows=[AffineHalfFlow(dim=2, parity=i%2) for i in range(9)]
model=NormalizingFlowModel(prior,flows)
optimizer=optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
print("number of params: ", sum(p.numel() for p in model.parameters()))

d=DatasetMoons()
total_iter=2000
for k in range(total_iter+1):
	x=d.sample(128)
	zs,prior_logprob,log_det=model(x)
	logprob=prior_logprob+log_det
	loss=-torch.sum(logprob)
	model.zero_grad()
	loss.backward()
	optimizer.step()
	if k%100==0:
		print(f'{k}/{total_iter} NLL loss: {loss.item()}')

model.eval()
d=DatasetMoons()
x=d.sample(1000)
zs,prior_logprob,log_det=model(x)
z=zs[-1]
plt.figure(figsize=(8,8))
x=x.cpu().detach().numpy()
z=z.cpu().detach().numpy()

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
axs[0,0].scatter(z[:,0], z[:,1], c='g', marker="*", s=5)
axs[0,0].set_title(r'$z=f(X)$')

z=np.random.multivariate_normal(np.zeros(2),np.eye(2),1000)
axs[0,1].scatter(z[:,0], z[:,1], c='r', marker="*", s=5)
axs[0,1].set_title(r'$z \sim p(z)')

x=d.sample(1000)
axs[1,0].scatter(x[:,0], x[:,1], c='y', marker='D', s=5)
axs[1,0].set_title(r'$X \sim p(X)')

x=model.sample(1000)[-1].detach().numpy()
axs[1,1].scatter(x[:,0], x[:,1], c='b', marker='D', s=5)
axs[1,1].set_title(r'$X = g(z)')

plt.savefig('normalizingflow_moons.jpg')










