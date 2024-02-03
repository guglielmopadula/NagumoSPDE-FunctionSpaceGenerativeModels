import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stochasticheatequation import StochasticHeatEquation
from tqdm import trange

torch.manual_seed(0)
np.random.seed(0)
batch_size=100
data=StochasticHeatEquation(100)
x=data.x
real_data=data.train_loader.dataset[:][0].reshape(600,-1)
train_loader=data.train_loader
res=64
device='cpu'
SAMPLE_SIZE=600
new_input_mean=torch.zeros(res)
new_input_std=torch.zeros((res*(res+1))//2,2)
new_output_mean=torch.zeros(res)
new_output_std=torch.zeros((res*(res+1))//2)

mean=real_data.mean(dim=0)
cov=torch.tensor(np.cov(real_data.T))


t=0
for i in range(res):
    new_input_mean[i]=x[i]
    new_output_mean[i]=mean[i]
    for j in range(i,res):
        new_input_std[t,0]=x[i]
        new_input_std[t,1]=x[j]
        new_output_std[t]=cov[i,j]
        t+=1


dataset_1=torch.utils.data.TensorDataset(new_input_mean.unsqueeze(-1),new_output_mean.unsqueeze(-1))
dataset_2=torch.utils.data.TensorDataset(new_input_std,new_output_std)
loader_1=torch.utils.data.DataLoader(dataset_1, batch_size=100, shuffle=True)
loader_2=torch.utils.data.DataLoader(dataset_2, batch_size=100, shuffle=True)

model1=nn.Sequential(nn.Linear(1,100),nn.ReLU(),nn.Linear(100,100),nn.ReLU(),nn.Linear(100,100),nn.ReLU(),nn.Linear(100,100),nn.ReLU(),nn.Linear(100,1))

epochs=500
optimizer = torch.optim.Adam(model1.parameters(), lr=1e-2)

for _ in range(epochs):
    my_loss=0
    for (x,y) in loader_1:
        optimizer.zero_grad()
        y_hat=model1(x)
        loss=F.mse_loss(y_hat,y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            my_loss+=(batch_size/SAMPLE_SIZE*torch.linalg.norm(y_hat-y)/torch.linalg.norm(y)).item()
    print(my_loss)


class ModelSTD(nn.Module):
    def __init__(self):
        super(ModelSTD,self).__init__()
        self.model=nn.Sequential(nn.Linear(1,200),nn.ReLU(),nn.Linear(200,200),nn.ReLU(),nn.Linear(200,200),nn.ReLU(),nn.Linear(200,200),nn.ReLU(),nn.Linear(200,100))
    def forward(self,x,y):
        return torch.sum(self.model(x)*self.model(y),axis=1)

model2=ModelSTD()
optimizer = torch.optim.Adam(model2.parameters(), lr=1e-4)

for _ in range(epochs):
    my_loss=0
    for (x,y) in loader_2:
        x1=x[:,0].reshape(-1,1)
        x2=x[:,1].reshape(-1,1)
        optimizer.zero_grad()
        y_hat=model2(x1,x2)
        y_hat=y_hat.reshape(y.shape)
        loss=F.mse_loss(y_hat,y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            my_loss+=(batch_size/SAMPLE_SIZE*torch.linalg.norm(y_hat-y)/torch.linalg.norm(y)).item()
    print(my_loss)


m=model1(new_input_mean.unsqueeze(-1))

pairs=torch.meshgrid(new_input_mean,new_input_mean,indexing="xy")
pairs=torch.stack(pairs,dim=-1)
C=model2(pairs[:,:,0].reshape(-1,1),pairs[:,:,1].reshape(-1,1)).reshape(res,res)+1E-3*torch.eye(res)
with torch.no_grad():
    m=m.numpy().reshape(-1)
    C=C.numpy().reshape(res,res)

real_data=real_data.numpy()
import matplotlib.pyplot as plt
with torch.no_grad():
    t=np.arange(64)
    x=np.random.multivariate_normal(m,C,600)
    fig,ax=plt.subplots()
    for i in range(600):
        ax.plot(t,x[i,:])
    fig.savefig("gpr.png")

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
    x=torch.tensor(np.random.multivariate_normal(m,C,1000))
    lags, acf = compute_acovf(x.squeeze())

    real_data=real_data.reshape(600,64,1)
    tmp=torch.tensor(real_data).clone()
    lags2, acf2 = compute_acovf(tmp.squeeze())
    fig,ax=plt.subplots()
    ax.plot(lags, acf, label="GPR")
    ax.plot(lags2, acf2, label="True")
    ax.legend()
    fig.savefig("acovf_gpr.png")

x=x.numpy().reshape(1000,-1)
real_data=real_data.reshape(600,-1)
print(np.linalg.norm(np.mean(x,axis=0)-np.mean(real_data,axis=0))/np.linalg.norm(np.mean(real_data,axis=0)))
print(np.linalg.norm(np.cov(x.T)-np.cov(real_data.T))/np.linalg.norm(np.cov(real_data.T)))
acf=acf[:21]
acf2=acf2[:21]
print(np.linalg.norm(acf2-acf)/np.linalg.norm(acf2))
