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
    
    
    '''
    RMS modo grezzo
    '''
    
def RMS(self, image):
        '''
        image deve essere una lista
        '''
        i = 0
        
        val = []
        std = []
        somma = 0
        sommadev=0
        
        mean = np.ma.mean(image, axis=0)
        val1 = []
        std1 = []
        somma1 = 0
        sommadev1 = 0
        
            
        while i<len(image):
            val.append(np.reshape(image[i][:,:], -1))
            std.append(np.std(val[i]))
            
            newima = image[i] - mean       
            val1.append(np.reshape(newima[:,:], -1))
            std1.append(np.std(val1[i]))
            
            print("RMS",i, "=", std[i])
            print("DeltaRMS",i, "=", std1[i])
            somma+=std[i]
            somma1+=std1[i]
            i=i+1
        
        print("")
        media = somma/(len(image))
        media1=somma1/len(image)
        print("Mean RMS = ", media)
        print("Mean DeltaRMS = ", media1)
        
        g = 0
        while g<len(image):
            sommadev+=(std[g]-media)**2
            sommadev1+=(std1[g]-media1)**2
            g=g+1
            
        dev = np.sqrt(sommadev/(len(image)-1)) 
        dev1 = np.sqrt(sommadev1/(len(image)-1))
        print("Standard Deviation RMS = ", dev)
        print("Standard Deviation deltaRMS = ", dev1)
        
            
        
        return std, media, dev, std1, media1, dev1
    

def circle_mask1(self, image, xc,yc,radius,perc,imagePixels):
    a=perc/100
        
    maskedd = np.zeros((imagePixels, imagePixels), dtype = np.uint8) 
    rr, cc = circle(xc, yc , int(radius*a))
    maskedd[rr,cc] = 1
    maskinv = np.logical_not(maskedd)
        
        
    new_ima = np.ma.masked_array(image, mask = maskinv)
                
        
    return maskedd, maskinv, new_ima
    
    

