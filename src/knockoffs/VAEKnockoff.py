__author__ = "Ying Liu <summeryingl@gmail.com>"
__date__ = "2019/03/18"
# class of importance weighted VAE

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class BasicBlock(nn.Module):
    def __init__(self, xdim, nhin, zdim):
        super(BasicBlock, self).__init__()
        self.xdim = xdim
        self.zdim = zdim
        self.nhin = nhin
        if len(nhin) == 0:
            self.mu = nn.Linear(xdim, zdim).double()
            self.logsigma = nn.Linear(xdim, zdim).double()

        if len(nhin) > 0:
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(xdim, nhin[0]).double())
            for i in range(len(nhin) - 1):
                self.layers.append(nn.Linear(nhin[i], nhin[i + 1]).double())

            self.mu = nn.Linear(nhin[-1], zdim).double()
            self.logsigma = nn.Linear(nhin[-1], zdim).double()

    def forward(self, x):
        if len(self.nhin)>0:
            for layer in self.layers:
                x = torch.tanh(layer(x))

        mu = self.mu(x)
        logsigma = self.logsigma(x)
        sigma = torch.exp(logsigma)
        return mu, sigma


class VAE(nn.Module):
    def __init__(self, xdim, zdim, nhin=[], nhout=[]):
        super(VAE, self).__init__()
        self.xdim = xdim
        self.zdim = zdim
        self.nhin = nhin
        self.nhout = nhout

        ## encoder
        self.encoder_h1 = BasicBlock(xdim, nhin, zdim)

        ## decoder
        self.decoderlayers = nn.ModuleList()

        if len(nhout) == 0:
            self.decoderlast=nn.Linear(zdim, xdim).double()

        if len(nhout) > 0:
            self.decoderlayers.append(nn.Linear(zdim, nhout[0]).double())
            for i in range(len(nhout) - 1):
                self.decoderlayers.append(nn.Linear(nhout[i], nhout[i + 1]).double())

            self.decoderlast = nn.Linear(nhout[-1], xdim).double()

    def encoder(self, x):
        mu_h1, sigma_h1 = self.encoder_h1(x)
        eps = Variable(sigma_h1.data.new(sigma_h1.size()).normal_())
        h1 = mu_h1 + sigma_h1 * eps
        return h1, mu_h1, sigma_h1, eps

    def decoder(self, z):
        if len(self.nhout)>0:
            for layer in self.decoderlayers:
                z = torch.tanh(layer(z))

        z = torch.sigmoid(self.decoderlast(z))
        return z

    def forward(self, x):
        h1, mu_h1, sigma_h1, eps = self.encoder(x)
        p = self.decoder(h1)
        return (h1, mu_h1, sigma_h1, eps), (p)

    def generator(self, x, bin=0):
        h1, mu_h1, sigma_h1, eps = self.encoder(x)
        p = self.decoder(h1)
        #print(sigma1.size())
        if bin==0:
            sigma1 = torch.mean((x - p) ** 2, dim=0).double()
            e=torch.randn(x.size()).double()*torch.sqrt(1-sigma1).double()
            out=p+e
        if bin==1:
            out=torch.bernoulli(p)
        #print(p.size())
        #print(e.size())
        return out


    def train_loss(self, inputs, col1=0, col2=0):
        # pdb.set_trace()
        h1, mu_h1, sigma_h1, eps = self.encoder(inputs)

        p = self.decoder(h1)
        #recon = K.sum(tf.squared_difference(y_pred, y_true), axis=1)
        #kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
        recon = torch.norm(p-inputs,2)**2
        # D_KL(Q(z|X))||P(z|X)); it is closed form as both are Gaussian
        #kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
        kl=0.5*torch.sum(sigma_h1+mu_h1**2-1-torch.log(sigma_h1))
        loss = recon+kl
        # pdb.set_trace()
        if (col1 + col2) != 0:
            if col1 != 0:
                l1_regularization = torch.DoubleTensor([0])
                for param in self.parameters():
                    l1_regularization += torch.norm(param, 1)

                loss = loss + col1 * l1_regularization

            if col2 != 0:
                l2_regularization = torch.DoubleTensor([0])
                for param in self.parameters():
                    l2_regularization += torch.norm(param, 2)**2

                loss = loss + col2 * l2_regularization

        return loss