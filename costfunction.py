import numpy as np
import mat
#import display
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io as scio

def sigmoid(x):
  y = 1 / (1 + np.exp(-x));
  return y

visibleSize = 64  # number of input units 
hiddenSize = 25    #number of hidden units 
sparsityParam = 0.01  # desired average activation of the hidden units.                     
lambdad = 0.0001    #weight decay parameter       
beta = 3            #weight of sparsity penalty term  
iter=5000
learningrate= 1.5

#W1 = np.random.random((hiddenSize, visibleSize))
#B1 = np.random.random((hiddenSize, 1))
#W2 = np.random.random((visibleSize, hiddenSize))
#B2 = np.random.random((visibleSize, 1))


W1 = scio.loadmat(r"E:\W1.mat")
W1 = W1['W1']
W2 = scio.loadmat(r"E:\W2.mat")
W2 = W2['W2']
B1 = scio.loadmat(r"E:\b1.mat")
B1 = B1['b1']
B2 = scio.loadmat(r"E:\b2.mat")
B2 = B2['b2']

W1grad = np.zeros(np.size(W1))
W2grad = np.zeros(np.size(W2))
B1grad = np.zeros(np.size(B1))
B2grad = np.zeros(np.size(B2))

data = mat.patch()

for k in range(iter):
    cost = 0;
    count = 1  
    m = np.size(data,1); #length of axis=1

    Z2 = np.dot(W1,data) + B1
    A2 = sigmoid(Z2);
    Z3 = np.dot(W2,A2) + B2
    A3 = sigmoid(Z3);
    cost = (1/m)*(1/2)*sum(sum((A3-data)*(A3-data)))

    regularization = (lambdad/2) * (sum(sum(W1*W1))+sum(sum(W2*W2)))

    #Sparsity penalty term
    averageActivationVec = (1/m)*np.sum(A2,axis=1)
    spasityPenalty = beta * sum(sparsityParam * np.log( sparsityParam  / averageActivationVec ) + (1-sparsityParam) * np.log((1-sparsityParam) /(1-averageActivationVec)) )
                 
    # Add regularization and spasity penalty
    cost = cost + regularization + spasityPenalty

    # gradients by Back Propagation Algorithm with regularization and spasity penalty
    Delta3 = (A3-data)*A3*(1-A3)
    sparsityTermInDelta2 = (beta*(-sparsityParam/averageActivationVec+(1-sparsityParam)/(1-averageActivationVec))).reshape((hiddenSize, 1))
    Delta2 = ( np.dot(np.transpose(W2),Delta3) + sparsityTermInDelta2)*A2*(1-A2)

    W1grad = np.dot(Delta2,np.transpose(data))/m + lambdad*W1
    W2grad = np.dot(Delta3,np.transpose(A2))/m + lambdad*W2
    B1grad = (np.sum(Delta2,axis=1)/m).reshape((hiddenSize, 1))
    B2grad = (np.sum(Delta3,axis=1)/m).reshape((visibleSize, 1))

    W1 = W1 - W1grad*learningrate
    B1 = B1 - B1grad*learningrate
    W2 = W2 - W2grad*learningrate
    B2 = B2 - B2grad*learningrate

    '''
    if(count%10 == 1):
        fig=plt.figure()
        for i in range(1,26):
            ax = fig.add_subplot(5,5,i)
    
        #patch1 = W1 / a
        #clim = np.linalg.norm(W1,ord=np.inf)
            patch = W1[(i-1),:] #/ clim
 
            huanyuan = patch.reshape(8,8)
            cmap=mpl.cm.cool
            plt.imshow(huanyuan,cmap='gray')

        plt.show()
    '''

    count = count + 1
    print(cost)
    if (cost<0.2):
        break

#np.save(r"E:\pythonprogram\SAE\W1.npy", W1)
#np.save(r"E:\pythonprogram\SAE\B1.npy", B1)

fig=plt.figure()
W1_mean = np.mean(W1)
W1 = W1 - W1_mean
clim = np.linalg.norm(W1,ord=1)
for i in range(1,26):
    ax = fig.add_subplot(5,5,i)
    patch = W1[(i-1),:]/clim
    huanyuan = patch.reshape(8,8)
    cmap=mpl.cm.cool
    plt.imshow(huanyuan,cmap='gray')
plt.show()
print(W1)



    



