import mat
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import scipy.io as scio

#W1 = np.load(r"E:\pythonprogram\SAE\W1.npy")
W1 = scio.loadmat(r"E:\pythonprogram\SAE\W1.mat")
W1 = W1['W1']
W1 = np.transpose(W1)
W1_mean = np.mean(W1)
#W1_mean = (np.mean(W1,axis=1)).reshape((64,1))
W1 = W1 - W1_mean
clim = np.linalg.norm(W1,ord=1)
#clim = np.linalg.norm(W1)
#clim = np.std(W1)
fig=plt.figure()
for i in range(1,26):
    ax = fig.add_subplot(5,5,i)
    
    #patch1 = W1 / a
    #clim = np.linalg.norm(W1,ord=np.inf)
    patch = W1[:,(i-1)] / clim
 
    huanyuan = patch.reshape(8,8)
    cmap=mpl.cm.cool
    plt.imshow(huanyuan,cmap='gray')

plt.show()


