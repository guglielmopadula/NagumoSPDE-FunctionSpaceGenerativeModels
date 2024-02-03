from torch import nn
import numpy as np
import torch
from tqdm import trange
from stochasticheatequation import StochasticHeatEquation
import normflows as nf
import torch
base = nf.distributions.base.DiagGaussian(600,trainable=False)
import matplotlib.pyplot as plt
flows = []
num_layers = 32
res=64

for i in range(num_layers):
    # Neural network with two hidden layers having 64 units each
    # Last layer is initialized by zeros making training more stable
    param_map = nf.nets.MLP([300, 64, 64, 600], init_zeros=True)
    # Add flow layer
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    # Swap dimensions
    flows.append(nf.flows.Permute(2, mode='swap'))


model = nf.NormalizingFlow(base, flows)
xi=np.load("xi.npy")

real_data=StochasticHeatEquation(100).train_loader.dataset[:][0].numpy()
# Load data
import numpy as np
latent=np.load("coeff.npy")
latent=latent.reshape(latent.shape[0],-1)
latent=torch.tensor(latent).float()


dataset=torch.utils.data.TensorDataset(latent)
loader=torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)


BATCH_SIZE=100

data=torch.utils.data.TensorDataset(torch.tensor(np.load("coeff.npy")))
train_loader=torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 100
loss_list = []
for epoch in range(num_epochs):
    for i, x in enumerate(loader):
        optimizer.zero_grad()
        loss = model.forward_kld(x[0])
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('Epoch: {}/{}, Iter: {}/{}, Loss: {:.3f}'.format(
                epoch+1, num_epochs, i+1, len(loader), loss.item()))
    loss_list.append(loss.item())
plt.plot(np.arange(len(loss_list)),loss_list)
plt.show()

with torch.no_grad():
    t=np.arange(64)
    x=model.sample(600)[0]
    x=x.numpy()@xi.T
    fig,ax=plt.subplots()
    for i in range(600):
        ax.plot(t,x[i,:])
    fig.savefig("nf.png")

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
    x = model.sample(1000)[0]
    x=torch.tensor(x.numpy()@xi.T)
    x=x.reshape(-1,64,1)
    print(torch.min(x), torch.max(x))
    lags, acf = compute_acovf(x.squeeze())

    tmp=torch.tensor(real_data).clone()
    print(tmp.shape)
    lags2, acf2 = compute_acovf(tmp.squeeze())
    fig,ax=plt.subplots()
    ax.plot(lags, acf, label="NF+POD")
    ax.plot(lags2, acf2, label="True")
    ax.legend()
    fig.savefig("avcovf_nf.png")

x=x.numpy().reshape(1000,-1)
real_data=real_data.reshape(600,-1)
print(np.linalg.norm(np.mean(x,axis=0)-np.mean(real_data,axis=0))/np.linalg.norm(np.mean(real_data,axis=0)))
print(np.linalg.norm(np.cov(x.T)-np.cov(real_data.T))/np.linalg.norm(np.cov(real_data.T))),
acf=acf[:21]
acf2=acf2[:21]
print(np.linalg.norm(acf2-acf)/np.linalg.norm(acf2))