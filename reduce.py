from nagumospde import NagumoSPDE
import numpy as np
u_data=NagumoSPDE(100).train_loader.dataset[:][0].numpy()
u_data=u_data.reshape(600,64)
covariance=u_data@((u_data.T))/600
U,S,V=np.linalg.svd(covariance)
xi=1/np.sqrt(600)*((u_data.T@U))
for i in range(600):
    xi[:,i]=xi[:,i]/np.sqrt(S[i])
coeff=u_data@(xi)
u_rec=coeff@xi.T

np.save("coeff.npy",coeff)
np.save("xi.npy",xi)