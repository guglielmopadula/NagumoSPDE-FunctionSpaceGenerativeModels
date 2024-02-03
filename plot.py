from stochasticheatequation import StochasticHeatEquation
import numpy as np
a=StochasticHeatEquation(600).train_loader.dataset[:][0].numpy()
a=a.reshape(600,-1)
x=np.arange(0,a.shape[1])
import matplotlib.pyplot as plt

for i in range(600):
    plt.plot(x,a[i,:])

plt.show()
