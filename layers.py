
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
from torch import optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as dsets
import gzip
import cPickle as pickle

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

class Discriminator_Low(nn.Module):

    def __init__(self, h_dim, x_dim):
        super(Discriminator_Low, self).__init__()
        self.fc_1 = nn.Linear(x_dim, h_dim)
        self.fc_2 = nn.Linear(h_dim, h_dim)
        self.fc_3 = nn.Linear(h_dim, 1)
        self.activ = nn.LeakyReLU()

    def forward(self, x):
        inp_concat = x#torch.cat((x,z), 1)
        print inp_concat.size()
        h1 = self.activ(self.fc_1(inp_concat))
        h2 = self.activ(self.fc_2(h1))
        out = self.fc_3(h2)

        return out

if __name__ == "__main__":
    
    mn = gzip.open("/u/lambalex/data/mnist/mnist.pkl.gz")

    train, valid, test = pickle.load(mn)

    trainx,trainy = train
    validx,validy = valid
    testx, testy = test

    generator_low = Generator_Low(128,1024,28*28)

    generator_low.cuda()

    discriminator_low = Discriminator_Low(1024, 28*28)

    discriminator_low.cuda()

    gparams = generator_low.parameters()
    dparams = discriminator_low.parameters()

    z_in = Variable(torch.randn(64,128).cuda())

    xgen = generator_low(z_in)

    disc_out_gen = discriminator_low(xgen)

    xreal = Variable(torch.from_numpy(trainx[0:64]).cuda())

    disc_out_real = discriminator_low(xreal)

    print disc_out_gen.size(), disc_out_real.size()

    gen_loss = ((disc_out_gen - 0.5)**2).sum()/64.0

    disc_loss = ((disc_out_gen)**2 + (disc_out_real - 1.0)**2).sum()/64.0

    optimizer_gen = optim.RMSprop(gparams)
    optimizer_disc = optim.RMSprop(dparams)

    
    for i in range(0,1000):

        optimizer_gen.zero_grad()
        optimizer_disc.zero_grad()



