from .encoder import encoder
from .decoder import decoder
import torch.nn as nn
import pyro.distributions as distributions
import pyro
import torch

class vae(nn.Module):
    def __init__(self, z_dim = 30, h_dim = 400, use_cuda = False):
        super(vae, self).__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.encoder = encoder(z_dim=self.z_dim, h_dim=self.h_dim)
        self.decoder = decoder(z_dim=self.z_dim, h_dim=self.h_dim)

        if use_cuda:
            self.cuda()

    def model(self, x):
        pyro.module('decoder', self.decoder)

        with pyro.plate('data', x.shape[0]):
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_var = x.new_ones(torch.Size((x.shape[0], self.z_dim)))

            z = pyro.sample('latent', distributions.Normal(loc=z_loc, scale=z_var).to_event(1))
            img = self.decoder.forward(z)
            #print(img)
            #breakpoint()
            pyro.sample('observed', distributions.Bernoulli(img).to_event(1))

    def guide(self, x):
        pyro.module('encoder', self.encoder)
        with pyro.plate('data', x.shape[0]):
            z_loc, z_var = self.encoder.forward(x=x)
            pyro.sample('latent', distributions.Normal(loc=z_loc, scale= z_var).to_event(1))

    def reconstruct_img(self, x):
        z_loc, z_var = self.encoder(x)
        z = distributions.Normal(loc=z_loc, scale= z_var).sample()
        img = self.decoder.forward(z)

        return img
    
