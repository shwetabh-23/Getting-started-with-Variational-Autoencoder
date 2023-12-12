import torch.nn as nn

class decoder(nn.module):
    def __init__(self, z_dim, h_dim):
        super(decoder, self).__init__()

        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, 784)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        l1 = self.softplus(self.fc1(z))
        l2 = self.sigmoid(self.fc21(l1))

        return l2
    

