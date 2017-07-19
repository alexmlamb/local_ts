'''
Simple MNIST classifier: test the effect of training with homogenous batches (same label in batch) vs. random batches.  
'''

import torch
from torch.autograd import Variable, grad
from torch import optim
import gzip
import cPickle as pickle
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn

'''
z -> z_1,z_2,z_3
gen_high

z_1,z_2,z_3 -> z
inf_high

z_i -> x_i
gen_low
x_i -> z_i
inf_low
'''
def init_gparams():

    params = {}

    params['W_gh_1'] = Variable(0.01 * torch.randn(128,1024).cuda(), requires_grad=True)
    params['W_gh_2'] = Variable(0.01 * torch.randn(1024,1024).cuda(), requires_grad=True)
    params['W_gh_3'] = Variable(0.01 * torch.randn(1024,128*3).cuda(), requires_grad=True)

    params['W_ih_1'] = Variable(0.01 * torch.randn(128*3,1024).cuda(), requires_grad=True)
    params['W_ih_2'] = Variable(0.01 * torch.randn(1024,1024).cuda(), requires_grad=True)
    params['W_ih_3'] = Variable(0.01 * torch.randn(1024,128).cuda(), requires_grad=True)

    params['W_gl_1'] = Variable(0.01 * torch.randn(128*2,1024).cuda(), requires_grad=True)
    params['W_gl_2'] = Variable(0.01 * torch.randn(1024,1024).cuda(), requires_grad=True)
    params['W_gl_3'] = Variable(0.01 * torch.randn(1024,784).cuda(), requires_grad=True)

    params['W_il_1'] = Variable(0.01 * torch.randn(784,1024).cuda(), requires_grad=True)
    params['W_il_2'] = Variable(0.01 * torch.randn(1024,1024).cuda(), requires_grad=True)
    params['W_il_3'] = Variable(0.01 * torch.randn(1024,128).cuda(), requires_grad=True)

    return params

'''
D(z_i,x_i)

D(z, z_1, z_2, z_3)

D_low
D_high

'''
def init_dparams():

    params = {}

    params['W_l_1'] = Variable(0.01 * torch.randn(784+128,1024).cuda(), requires_grad=True)
    params['W_l_2'] = Variable(0.01 * torch.randn(1024,1024).cuda(), requires_grad=True)
    params['W_l_3'] = Variable(0.01 * torch.randn(1024,1).cuda(), requires_grad=True)
    
    params['W_h_1'] = Variable(0.01 * torch.randn(128*4,1024).cuda(), requires_grad=True)
    params['W_h_2'] = Variable(0.01 * torch.randn(1024,1024).cuda(), requires_grad=True)
    params['W_h_3'] = Variable(0.01 * torch.randn(1024,1).cuda(), requires_grad=True)



    return params  


def network(p, x, ytrue):

    x = Variable(torch.from_numpy(x)).cuda()
    ytrue = Variable(torch.from_numpy(ytrue).type(torch.LongTensor)).cuda()

    sm = torch.nn.Softmax()
    relu = torch.nn.ReLU()
    nll = torch.nn.functional.nll_loss

    h1 = relu(torch.matmul(x, p['W1']))

    y = sm(torch.matmul(h1, p['W2']))

    loss = nll(y, ytrue).sum()

    acc = ytrue.eq(y.max(1)[1]).sum()

    return loss, acc

if __name__ == "__main__":
    mn = gzip.open("/u/lambalex/data/mnist/mnist.pkl.gz")

    train, valid, test = pickle.load(mn)

    trainx,trainy = train
    validx,validy = valid
    testx, testy = test

    gp = init_gparams()
    dp = init_dparams()

    optimizer = optim.Adam([gp['W1'], gp['W2']], lr = 0.001, betas=(0.9,0.999))

    accl = []

    mode = "diff"
    #mode = "same"
    

    for i in range(0,50000):

        optimizer.zero_grad()

        if mode == "diff":

            rind = random.randint(0,40000)

            x = trainx[rind:rind+64]
            y = trainy[rind:rind+64]

        elif mode == "same":

            rclass = random.randint(0,9)
            rind = random.randint(0,4500)

            x = trainx[trainy==rclass][rind:rind+64].astype('float32')
            y = trainy[trainy==rclass][rind:rind+64]

        loss,acc = network(p,x,y)

        loss.backward()

        optimizer.step()

        accl.append(acc.data.cpu().numpy())

        if i % 100 == 0:
            print loss
            print acc
            print np.array(accl[:-200]).mean()
            

    plt.plot(accl) 
    plt.savefig(mode + '.png')

