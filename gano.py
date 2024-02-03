import torch
import numpy as np
import pylab as plt
import torch.nn.functional as F
import torch.nn as nn
from random_fields import *
from tqdm import trange
from neuralop.models import FNO1d
from stochasticheatequation import StochasticHeatEquation
torch.manual_seed(0)
np.random.seed(0)
batch_size=100
data=StochasticHeatEquation(100)
real_data=data.train_loader.dataset[:][0].numpy()
train_loader=data.train_loader
res=64
epochs=100
device='cpu'
λ_grad=10
grf = GaussianRF_idct(1, res, alpha=1.5, tau=1.0,device=device)
lr=1e-4
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc=FNO1d(64,100,in_channels=1)
        

    def forward(self, x):
        x=x.squeeze(-1).unsqueeze(1)
        x=self.fc(x).squeeze(1).unsqueeze(-1)
        return x
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc=FNO1d(64,100,in_channels=1)
        self.lin=nn.Linear(64,1)

    def forward(self, x):
        x=x.squeeze(-1).unsqueeze(1)
        x=self.fc(x)
        x=x.reshape(x.shape[0],-1)
        return x

G=Generator().to(device)
D=Discriminator().to(device)
G_optimizer = torch.optim.Adam(G.parameters(), lr=lr) #, weight_decay=1e-4)
D_optimizer = torch.optim.Adam(D.parameters(), lr=lr) #, weight_decay=1e-4)
fn_loss = nn.BCEWithLogitsLoss()



D.train()
G.train()
nn_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
print("Number discriminator parameters: ", nn_params)
nn_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
print("Number generator parameters: ", nn_params)

def calculate_gradient_penalty(model, real_images, fake_images, device):
    """Calculates the gradient penalty loss for GANO"""
    # Random weight term for interpolation between real and fake data
    alpha = torch.randn((real_images.size(0),1, 1), device=device)
    # Get random interpolation between real and fake data
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)

    model_interpolates = model(interpolates)
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1/np.sqrt(res * res)) ** 2)
    return gradient_penalty


def train_GANO(D, G, train_data, epochs, D_optim, G_optim, scheduler=None):
    losses_D = np.zeros(epochs)
    losses_G = np.zeros(epochs)
    for i in trange(epochs):
        loss_D = 0.0
        loss_G = 0.0
        for j, data in enumerate(train_data):
            # Train D
            x = data[0].to(device)
            D_optimizer.zero_grad()
            z=grf.sample(x.shape[0]).unsqueeze(-1)
            x_syn = G(grf.sample(x.shape[0]))
            W_loss = -torch.mean(D(x)) + torch.mean(D(x_syn.detach()))

            gradient_penalty = calculate_gradient_penalty(D, x.data, x_syn.data, device)

            loss = W_loss + λ_grad * gradient_penalty
            loss.backward()

            loss_D += loss.item()

            D_optim.step()
            
            G_optimizer.zero_grad()

            x_syn = G(grf.sample(x.shape[0]).unsqueeze(-1))

            loss = -torch.mean(D(x_syn))
            loss.backward()
            loss_G += loss.item()

            G_optim.step()
        
        losses_D[i] = loss_D / batch_size
        losses_G[i] = loss_G / batch_size
            
        print(i, "D: ", losses_D[i], "G: ", losses_G[i], "mean: ", x.mean().item(), "std: ", x.std().item())

    return losses_D, losses_G

losses_D, losses_G = train_GANO(D, G, train_loader, epochs, D_optimizer, G_optimizer)


with torch.no_grad():
    t=np.arange(64)
    z=grf.sample(600)
    x = G(z).squeeze(-1)
    fig,ax=plt.subplots()
    for i in range(600):
        ax.plot(t,x[i,:])
    fig.savefig("gano.png")

    fig,ax=plt.subplots()
    for i in range(600):
        ax.plot(t,real_data[i,:])
    fig.savefig("true.png")

def compute_acovf(z):
    from scipy.stats import binned_statistic
    z_hat = torch.fft.rfft(z)
    acf = torch.fft.irfft(torch.conj(z_hat) * z_hat)
    acf = torch.fft.fftshift(acf).mean(dim=0) / z[0].numel()
    acf_r = acf.view(-1).cpu().detach().numpy()
    lags_x=torch.arange(res) - res//2
    lags_r = torch.sqrt(lags_x**2).view(-1).cpu().detach().numpy()

    idx = np.argsort(lags_r)
    lags_r = lags_r[idx]
    acf_r = acf_r[idx]

    bin_means, bin_edges, binnumber = binned_statistic(lags_r, acf_r, 'mean', bins=np.linspace(0.0, res, 50))
    return bin_edges[:-1], bin_means


with torch.no_grad():
    z=grf.sample(1000).unsqueeze(-1)
    x = G(z).squeeze(-1)
    lags, acf = compute_acovf(x.squeeze())

    real_data=real_data.reshape(600,64,1)
    tmp=torch.tensor(real_data).clone()
    lags2, acf2 = compute_acovf(tmp.squeeze())
    fig,ax=plt.subplots()
    ax.plot(lags, acf, label="GANO")
    ax.plot(lags2, acf2, label="True")
    ax.legend()
    fig.savefig("acovf_gano.png")

x=x.numpy().reshape(1000,-1)
real_data=real_data.reshape(600,-1)
print(np.linalg.norm(np.mean(x,axis=0)-np.mean(real_data,axis=0))/np.linalg.norm(np.mean(real_data,axis=0)))
print(np.linalg.norm(np.cov(x.T)-np.cov(real_data.T))/np.linalg.norm(np.cov(real_data.T)))
acf=acf[:21]
acf2=acf2[:21]
print(np.linalg.norm(acf2-acf)/np.linalg.norm(acf2))