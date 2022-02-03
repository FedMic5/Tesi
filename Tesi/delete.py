'''
Created on 3 feb 2022

@author: federico
'''

import h5py
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean


'Posso cancellare' 
def deltaRMS(image = []):
    
    i=0    
    mean = np.ma.mean(image, axis=0)
    val = []
    std = []
    somma = 0
    sommadev=0
    
    while i<len(image):
        newima = image[i] - mean 
        val.append(np.reshape(newima[:,:], -1))
        std.append(np.std(val[i]))
        
        print("Delta RMS",i, "=", std[i])
        
        somma+=std[i]
        i=i+1
    media = somma/len(image)
    print("Mean deltaRMS = ", media)
    
    g = 0
    while g<len(image):
        sommadev+=(std[g]-media)**2
        g=g+1
        
    dev = np.sqrt(sommadev/(len(image)-1)) 
    print("Standard Deviation deltaRMS = ", dev)
        
    return std, media, dev 




'Posso cancellare'
def deltaPV(image = []):
    
    i=0    
    mean = np.ma.mean(image, axis=0)
    pv = []
    somma = 0
    sommadev=0
    
    while i<len(image):
        newima = image[i] - mean 
        a = np.amax(newima)
        b = np.amin(newima)
        pv.append(a-b)
        
        print("Delta PV",i, "=", pv[i])
        
        somma+=pv[i]
        i=i+1
    media = somma/len(image)
    print("Mean deltaPV = ", media)
    
    g = 0
    while g<len(image):
        sommadev+=(pv[g]-media)**2
        g=g+1
        
    dev = np.sqrt(sommadev/(len(image)-1)) 
    print("Standard Deviation deltaPV = ", dev)
        
    return pv, media, dev


'Posso cancellare'
def std(image):
    val = np.reshape(image[:,:], -1)
    mean = np.mean(val)
    std = np.std(val)
    var = np.var(val)
    print("Standard Deviation = ", std)
    print("Mean = ", mean)
    print("Delta = ", var)
    return std, mean

#def mediaRms():
    
    
    #v = array([std[]])
    
    #media = np.mean(v)
    #print("Mean = ", media)
    #return media