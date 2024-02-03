from torch import nn
import numpy as np
import torch
from tqdm import trange
from torchdiffeq import odeint
import torch
from nagumospde import NagumoSPDE
import matplotlib.pyplot as plt
flows = []
num_layers = 128
res=64
T=1
class ScoreFunction(nn.Module):
    def __init__(self, latent_size):
        super(ScoreFunction, self).__init__()
        self.latent_size = latent_size
        self.model = nn.Sequential(nn.Linear(latent_size+1, 200),nn.Tanh(),nn.Linear(200, 200),nn.Tanh(),nn.Linear(200, 200),nn.Tanh(),nn.Linear(200, 200),nn.Tanh(),nn.Linear(200, 200),nn.Tanh(),nn.Linear(200, 200),nn.Tanh(),nn.Linear(200, self.latent_size))
        self.T=T
    
    def beta(self,t):
        return torch.tensor(1.0)
    
    def int_beta(self,t):
        return torch.tensor(t)
    
    def weight(self,t):
        return torch.exp(-t)

    def drift_coef(self,t):
        return -1/2*self.beta(t)

    def diffusion(self,t):
        return torch.sqrt(self.beta(t))

    def mean_p_0t(self,x0,t):
        return x0*torch.exp(-1/2*self.int_beta(t))
    
    def std_p_0t(self,x0,t):
        return torch.max(1-torch.exp(-self.int_beta(t)),torch.tensor(0.0001))

    def der_likelihood(self,x,x0,t):
        return -1/self.std_p_0t(x0,t)*(x-self.mean_p_0t(x0,t))

    def diff_eq(self,t,x):
        return -1/2*self.beta(t)*(x+self.forward(x,t))

    def sample(self,num_samples):
        t=torch.linspace(0,1,1000).flip(0)
        x=torch.randn(num_samples,self.latent_size)
        y=odeint(self.diff_eq,x,t)
        return y[-1]

    def forward(self, x, t):
        if t.ndim<2:
            t=torch.tensor(t)
            t=t.unsqueeze(0)
            t=t.unsqueeze(0)
            t=t.repeat(x.shape[0],1)
        tmp=torch.cat((x,t),1)
        return self.model(tmp)
    






i=np.load("xi.npy")

real_data=NagumoSPDE(100).train_loader.dataset[:][0].numpy()
# Load data
import numpy as np
latent=np.load("coeff.npy")
latent=latent.reshape(latent.shape[0],-1)
latent=torch.tensor(latent).float()
xi=np.load("xi.npy")

dataset=torch.utils.data.TensorDataset(latent)
loader=torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)


BATCH_SIZE=100




data=torch.utils.data.TensorDataset(torch.tensor(np.load("coeff.npy")))
train_loader=torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)



model=ScoreFunction(600)

# Train model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
num_epochs = 1000
loss_list = []
for epoch in range(num_epochs):
    for i, x in enumerate(loader):
        optimizer.zero_grad()
        x0=x[0]
        t=torch.rand(x0.shape[0],1)*T
        x_t=model.mean_p_0t(x0,t)+torch.sqrt(model.diffusion(t))*torch.randn(x0.shape[0],600)
        loss = torch.mean(model.weight(t)*torch.linalg.norm(model(x_t,t)-model.der_likelihood(x_t,x0,t),dim=1))
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print('Epoch: {}/{}, Iter: {}/{}, Loss: {:.3f}'.format(
            epoch+1, num_epochs, i+1, len(loader), loss.item()))
    if loss.item()<100:
        loss_list.append(loss.item())
plt.plot(np.arange(len(loss_list)),loss_list)
plt.show()


with torch.no_grad():
    t=np.arange(64)
    x=model.sample(600)
    x=x.numpy()@xi.T
    x=x.reshape(-1,64)
    fig,ax=plt.subplots()
    for i in range(600):
        ax.plot(t,x[i,:])
    fig.savefig("sbm.png")

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
    x = model.sample(1000)
    x=torch.tensor(x.numpy()@xi.T)
    x=x.reshape(-1,64,1)
    lags, acf = compute_acovf(x.squeeze())

    real_data=real_data.reshape(600,64,1)
    tmp=torch.tensor(real_data).clone()
    lags2, acf2 = compute_acovf(tmp.squeeze())
    fig,ax=plt.subplots()
    ax.plot(lags, acf, label="SBM+POD")
    ax.plot(lags2, acf2, label="True")
    ax.legend()
    fig.savefig("acovf_sbm.png")

x=x.numpy().reshape(1000,-1)
real_data=real_data.reshape(600,-1)
print(np.linalg.norm(np.mean(x,axis=0)-np.mean(real_data,axis=0))/np.linalg.norm(np.mean(real_data,axis=0)))
print(np.linalg.norm(np.cov(x.T)-np.cov(real_data.T))/np.linalg.norm(np.cov(real_data.T)))
acf=acf[:21]
acf2=acf2[:21]
print(np.linalg.norm(acf2-acf)/np.linalg.norm(acf2))