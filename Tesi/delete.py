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


    def removeZernike2(self, ima, num):
        coeff, mat = zernike.zernikeFit(ima, np.linspace(1,35,35))
        surf = zernike.zernikeSurface(ima, coeff[:num], mat)
        new_ima = ima - surf
        return new_ima
    
        def removeZernike3(self, ima, num):
        coeff, mat = zernike.zernikeFit(ima, np.arange(1,35))
        i=0
        while i <= num:
            surf = zernike.zernikeSurface(ima, coeff[0:i], mat[:,0:i])
            new_ima = ima - surf
            i=i+1
        return new_ima
    
    
    def residui2(self, image):
        '''
        image deve essere una sola immagine
        '''
        diff = []
        rms = []
        residui = []
        #coeff, mat = zernike.zernikeFit(image, np.arange(1, 35))
        #surf_image = zernike.zernikeSurface(image, coeff, mat)
        i=1
        while i<36:
            surf1 = self.removeZernike(image, i)
            #diff.append(image - surf1)
            rms.append(np.std(surf1))
            residui.append(i)
            #coef, mat = ...(diff, np....)
            #print("RMS", i, " = ", rms[i] )
            i=i+1
        plt.plot(residui,rms,marker="o",color='red')
        plt.title("Calcolo dei residui")
        plt.xlabel("Numero Zernike")
        plt.ylabel("RMS")
        #plt.ylim(np.min(rms), np.max(rms))
        plt.ylim(np.min(rms)*0.75, np.max(rms)*1.25)
        plt.show()
        
        
        #plt.show(diff[i])
        
        return rms, diff
    
    
    
    def zernikeFit(img, zernike_index_vector):
    '''
    Parameters
    ----------
    img: numpy masked array
        image for zernike fit
    zernike_index_vector: numpy array
        vector containing the index of Zernike modes to be fitted starting from 1

    Returns
    -------
    coeff: numpy array
        vector of zernike coefficients
    mat: numpy array
    '''
    img1 = img.data
    x, y, r, xx, yy, maschera = geo_circle.qpupil(img)
    mm = (maschera==1)
    coeff = _surf_fit(xx[mm], yy[mm], img1[mm], zernike_index_vector)
    mat = _getZernike(xx[mm], yy[mm], zernike_index_vector)
    return coeff, mat


    def zernikeFit(img, zernike_index_vector):
    '''
    Parameters
    ----------
    img: numpy masked array
        image for zernike fit
    zernike_index_vector: numpy array
        vector containing the index of Zernike modes to be fitted starting from 1

    Returns
    -------
    coeff: numpy array
        vector of zernike coefficients
    mat: numpy array
    '''
    m = sandbox.Analysis('Prova')
    ima = m.circle_mask_ext(img,img.shape[0],100)
    img1 = ima.data
    mask = np.invert(ima.mask).astype(int)
    x, y, r = m.circle_parameters_detection(ima,img.shape[0])
    xx, yy = geo_circle.qpupil(mask)
    mm = (mask==1)
    coeff = _surf_fit(xx[mm], yy[mm], img1[mm], zernike_index_vector)
    mat = _getZernike(xx[mm], yy[mm], zernike_index_vector)
    return coeff, mat, ima


    

