import torch
import pyro
import pyro.distributions as dist
import torch.nn as nn

class encoder(nn.Module):
    def __init__(self, z_dim, h_dim):
        super(encoder, self).__init__()
        self.fc1 = nn.Linear(784, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.reshape(-1, 784)
        l1 = self.fc1(x)
        l2 = self.softplus(l1)
        mean = self.fc21(l2)
        var = self.softplus(self.fc22(l2))

        return mean, var
    
    def freeze(self):
        for params in self.parameters():
            params.requires_grad = False

    def unfreeze(self):
        for params in self.parameters:
            params.requires_grad = True