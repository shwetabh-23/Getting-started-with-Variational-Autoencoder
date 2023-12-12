import torch
import torchvision.datasets as dset
import torchvision.transforms as transform
from torch.utils.data import DataLoader
import os
import torchvision.utils as vutils
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def loaders(data_path, batch_size, use_cuda = False):
    trans = transform.ToTensor()
    train_set = dset.MNIST(root = data_path, train=True, transform=trans, download=True)
    test_set = dset.MNIST(root=data_path, train=False, transform= trans, download=True)
    train_loader = DataLoader(dataset= train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def show_images(dataloader, num_images):
    images, _ = next(iter(dataloader))
    grid = vutils.make_grid(images[:num_images], normalize = True, padding = 5)
    
    plt.imshow(grid.numpy().transpose(1, 2, 0))
    plt.show()
    #plt.savefig((os.path.join(os.curdir, 'test.png')))
