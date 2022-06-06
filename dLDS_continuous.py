import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class dLDS_continuous(nn.Module):

    def __init__(self, 
                 M=6, # number of dictionary elements
                 N=3 # number of latent dimensions
                ):
        super(dLDS_continuous, self).__init__()
#         self.G = nn.Parameter(torch.mul(torch.randn((M, N, N)), var), requires_grad=True)
        self.G = nn.Parameter(torch.randn((M, N, N)), requires_grad=True)
        self.G.data = self.G.data / self.G.reshape(M, -1).norm(dim=1)[:, None, None]
        self.M = M
        self.N = N

    def forward(self, x):
        out = torch.zeros(x.shape, dtype=x.dtype)
        batch_size = len(x)
        T = (self.G[None, :, :, :] * self.c[:, :, None, None]).sum(dim=1).reshape((batch_size, self.N, self.N))
        out =torch.matrix_exp(T) @ x
        return out

    def set_coefficients(self, c):
        self.c = c

    def get_G(self):
        return self.G.data

    def set_G(self,G_input):
        self.G.data = G_input


import time
import torch.nn.functional as F    

def fit_dLDS(dLDS, 
             z0, 
             z1, 
             G_lr = 1e-1, 
             zeta = 1e-1, 
             max_iter = 200, 
             weight_decay = 1,
             tol = 1e-4, 
             device='cpu'):
    
    dLDS_opt = torch.optim.SGD(dLDS.parameters(), lr=G_lr, momentum=0.7, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(dLDS_opt, gamma=0.985)

    print(f'Epoch\tLoss\t\tTime')
    i = 0
    change = 1e99
    while i < max_iter and change > tol:
        t1 = time.time()
        old_coeff = dLDS.get_G().clone()
        dLDS_opt.zero_grad()
        c_data, c_pred = infer_coefficients(z0, z1,dLDS.get_G(), zeta,
                                                          device=device)

        c_loss, steps, k, = c_data
        dLDS.set_coefficients(c_pred)
        z1_hat = dLDS(z0.unsqueeze(-1).type(torch.float)).squeeze()
        dLDS_loss = F.mse_loss(z1_hat, z1.type(torch.float), reduction='sum')
        dLDS_loss.backward()
        dLDS_opt.step()   
        scheduler.step()

        time_elapsed = time.time() - t1    
        print(f'{i}\t{dLDS_loss.item():.6f}\t{time_elapsed:.6f}')
        change = torch.norm(dLDS.get_G().data - old_coeff) / (torch.norm(old_coeff) + 1e-9)
        i += 1

    


def infer_coefficients(x0, x1, G, zeta, max_iter=800, tol=1e-5, device='cpu'):
    c = nn.Parameter(torch.mul(torch.randn((len(x0),len(G)), device=device),
                     0.02), requires_grad=True)
    
    c_opt = torch.optim.SGD([c], lr=1e-2, nesterov=True, momentum=0.9)
    opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(c_opt, gamma=0.985)
    change = 1e99
    k = 0
    while k < max_iter and change > tol:
        old_coeff = c.clone()
        c_opt.zero_grad()
        
        loss = compute_loss(c, x0, x1, G)
        loss.backward()
        c_opt.step()
        opt_scheduler.step()

        with torch.no_grad():
            c.data = soft_threshold(c, get_lr(c_opt)*zeta)

        change = torch.norm(c.data - old_coeff) / (torch.norm(old_coeff) + 1e-9)
        k += 1
#         print(c.data)
    return (loss.item(), get_lr(c_opt), k), c.data

def compute_loss(c, x0, x1, G):
    T = (G[None, :, :, :] * c[:, :, None, None]).sum(dim=1).reshape((x0.shape[0], G.shape[1], G.shape[2]))
#     x1_hat = torch.matrix_exp(T) @ x0
#     x1_hat = torch.bmm(torch.matrix_exp(T), x0[:,:,None]).squeeze()

    x1_hat = (torch.matrix_exp(T)@x0[:,:,None]).squeeze()
    loss = F.mse_loss(x1_hat, x1, reduction='sum')
    return loss


def soft_threshold(c, zeta):
    return F.relu(torch.abs(c) - zeta) * torch.sign(c)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



# class ZetaDecoder(nn.Module):
#
#     def __init__(self, latent_dim, dict_size):
#         super(ZetaDecoder, self).__init__()
#         self.model = nn.Sequential(
#                 nn.Linear(latent_dim, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 1028),
#             nn.BatchNorm1d(1028),
#             nn.ReLU(),
#             nn.Linear(1028, dict_size))
#
#     def forward(self, x):
#         return self.model(x)
#
# class ZetaDecoder_small(nn.Module):
#
#     def __init__(self, latent_dim, dict_size):
#         super(ZetaDecoder_small, self).__init__()
#         self.model = nn.Sequential(
#                 nn.Linear(latent_dim, dict_size))
#
#     def forward(self, x):
#         return self.model(x)
