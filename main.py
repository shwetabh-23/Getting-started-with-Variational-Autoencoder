from Data import loaders, show_images
from VAE_Model import vae, MonitorLoss, train_vae, evaluate_vae

import torch
import torch.nn as nn
import pyro
import pyro.distributions as distributions
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from tqdm import tqdm

pyro.enable_validation(True)
pyro.distributions.enable_validation(True)
pyro.clear_param_store()

SMOKE_TEST = False
cuda = False
lr = 1e-04
epochs = 1 if SMOKE_TEST else 10
z_dim = 30
h_dim = 400

data_path = r'D:\ML-Projects\Getting-started-with-Variational-Autoencoder\Data\Raw Data'
train_loader, test_loader = loaders(data_path=data_path, batch_size=16, use_cuda=cuda)

vae = vae(z_dim=z_dim, h_dim=h_dim, use_cuda=cuda)
optimizer = Adam({'lr' : lr})
svi = SVI(model=vae.model, guide=vae.guide, optim=optimizer, loss=Trace_ELBO())

loss_monitor = MonitorLoss()
print('Starting training process')
for epoch in range(epochs):
    
    train_loss = train_vae(train_loader=train_loader, svi=svi, use_cuda=cuda)
    loss_monitor.append_loss(epoch=epoch, loss=train_loss, set='train')
    print('Loss at epoch {} : {}'.format(epoch, train_loss))

if not SMOKE_TEST:
    loss_monitor.show_loss()
    
