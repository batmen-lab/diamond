import torch.optim as optim
import torch
import numpy as np
import torch.nn as nn
from .VAEKnockoff import VAE
from scipy.linalg import cholesky

def train_vae_knockoff(X, n_epoch=30, mb_size=25):
    n, p = X.shape
    SIGMA = 0.1*np.ones([p,p])
    np.fill_diagonal(SIGMA, 1)
    C = cholesky(SIGMA, lower=True)
    nhin=[100, 100]
    nhout=[100, 100]
    zdim=50
    X_dim=p

    X_tr = (X.dot(C) > 0) * 1.
    X_tr=torch.from_numpy(X_tr)  
    model= VAE(X_dim,zdim,nhin,nhout)
    optimizer=optim.Adam(model.parameters())

    def train(epoch,X,col1=0,col2=0):
        model.train()
        train_loss=0
        for i in range((n - 1) // mb_size + 1):
            start_i = i * mb_size
            end_i = start_i + mb_size
            xb = X[start_i:end_i]
            optimizer.zero_grad()
            loss = model.train_loss(xb,col1,col2)
            loss.backward()
            train_loss+=loss.item()
            optimizer.step()
        print('===> Epoch: {} Average loss: {:.4f}'.format(epoch,train_loss/n))
        return train_loss/n
        
    loss_arr = []
    for epoch in range(1,n_epoch):
        loss = train(epoch,X_tr,col2=0.2)
        loss_arr.append(loss)
    return model.generator(X_tr).detach().numpy()