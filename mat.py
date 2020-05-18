import scipy.io as scio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl


def patch():
    dataFile = 'E:\pythonprogram\SAE\IMAGES.mat'
    data = scio.loadmat(dataFile)
    ###图片可视化
    #print(data)
    #print (data.keys())    # 输出所有键
    #print (data.values())  # 输出所有值
    data1 = data['IMAGES']
    #data11=data1[:,:,0]
    #print(data11)
    #plt.imshow(data11,cmap='gray')
    #print(data11.shape)
    #im = Image.fromarray(data11.astype(np.uint8))
    #cmap=mpl.cm.cool
    #aa = plt.imshow(data11,cmap='gray')

    patchsize = 8;  #we'll use 8x8 patches 
    numpatches = 10000;

    # Initialize patches with zeros.  Your code will fill in this matrix--one column per patch, 10000 columns. 
    patches = np.zeros((patchsize*patchsize, numpatches))
    for i in range(numpatches):    
        imageIndex = np.random.random_integers(0,np.size(data1,2)-1)   # [0,9]
        patchIndex1 = np.random.random_integers(1,np.size(data1,0)/patchsize) - 1   # [0,63]
        patchIndex2 = np.random.random_integers(1,np.size(data1,1)/patchsize) - 1   # [0,63]
        patch = data1[patchIndex1*patchsize:patchIndex1*patchsize+8, patchIndex2*patchsize:patchIndex2*patchsize+8, imageIndex]
        patch = patch.reshape(patchsize*patchsize)
        patches[:,i] = patch


    mean = (np.mean(patches,axis=1)).reshape((patchsize*patchsize,1))
    #mean1 = (np.array(np.sum(patches, axis=1) / np.size(patches, 1))).reshape((patchsize*patchsize, 1))
    patches_mean = patches - mean

    pstd =np.array(3 * np.std(patches_mean))
    min1 = np.minimum(patches_mean,pstd)
    patches = np.maximum(min1, -pstd) 
    patches = patches / pstd

    patches = (patches + 1) * 0.4 + 0.1
    

    return patches