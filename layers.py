
'''
Generator_High: z -> (z1,z2,z3,z4)
Generator_Low: z[i] -> x[i]

Inference_High: (z1,z2,z3,z4) -> z
Inference_Low: x[i] -> z[i]

Discriminator_High: (z1,z2,z3,z4,z)
Discriminator_Low: (x[i], z[i])

Initially all visible, use batch norm in generator and inference.  Inference and generator should both have their own noise.  

'''

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as dsets

class Generator_Low(nn.Module):

    def __init__(self, z_dim, h_dim, x_dim):
        super(Generator_Low, self).__init__()
        self.fc_1 = nn.Linear(z_dim*2, h_dim)
        self.fc_2 = nn.Linear(h_dim, h_dim)
        self.fc_3 = nn.Linear(h_dim, x_dim)
        self.z_dim = z_dim
        self.activ = nn.LeakyReLU()
        self.bn_h = nn.BatchNorm1d(h_dim)

    def forward(self, z):

        z_local = Variable(torch.randn(z.size(0), self.z_dim).cuda())

        z_full = torch.cat((z_local, z), 1)

        h1 = self.activ(self.bn_h(self.fc_1(z_full)))
        h2 = self.activ(self.bn_h(self.fc_2(h1)))
        xgen = self.fc_3(h2)

        return xgen

class Discriminator_Low(nn.module):

    def __init__(self, z_dim, h_dim, x_dim):
        super(Discriminator_Low, self).__init__()
        self.fc_1 = nn.Linear(z_dim + x_dim, h_dim)
        self.fc_2 = nn.Linear(h_dim, h_dim)
        self.fc_3 = nn.Linear(h_dim, 1)
        self.z_dim = z_dim
        self.activ = nn.LeakyReLU()


if __name__ == "__main__":
    
    generator_low = Generator_Low(128,1024,28*28)

    generator_low.cuda()

    discriminator_low = Discriminator_Low(128, 1024, 28*28)

    discriminator_low.cuda()

    z_in = Variable(torch.randn(64,128).cuda())

    output = generator_low(z_in)

    print output.size()



