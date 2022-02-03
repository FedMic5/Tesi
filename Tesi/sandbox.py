'''
Created on 3 feb 2022

@author: federico
'''

import h5py
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean


def from4D(h5filename, misura):
       
    file = h5py.File(h5filename, 'r')
    genraw = file[misura]['genraw']['data']
    data = np.array(genraw)
    mask = np.zeros(data.shape, dtype=np.bool)
    mask[np.where(data == data.max())] = True
    ima = np.ma.masked_array(data * 632.8e-9, mask=mask)
    return ima

#per ariel ptm ARIEL - PTM - After HT - F2.2 High Rep - PT - RAW.h5
def from4D2(h5filename, misura):
       
    file = h5py.File(h5filename, 'r')
    genraw = file[misura]['genraw']['opts']
    data = np.array(genraw)
    mask = np.zeros(data.shape, dtype=np.bool)
    mask[np.where(data == data.max())] = True
    ima = np.ma.masked_array(data * 632.8e-9, mask=mask)
    return ima

def from4Dmask(h5filename):
       
    file = h5py.File(h5filename, 'r')
    mask1 = file['measurement0']['Detectormask']
    
    return mask1


'Aprire tutte le misure di un fileh5 in unica lista di matrici che rappresentano le mie immagini'
    
def from4Dlist(h5filename):
       
    file = h5py.File(h5filename, 'r')
    
    misura = ['measurement0','measurement1','measurement2','measurement3','measurement4','measurement5','measurement6','measurement7','measurement8','measurement9','measurement10','measurement11','measurement12','measurement13','measurement14']
    i = 0
    ima = []
    
    while i<len(misura):
        genraw = file[misura[i]]['genraw']['data']
        data = np.array(genraw)
        mask = np.zeros(data.shape, dtype=np.bool)
        mask[np.where(data == data.max())] = True
        ima.append(np.ma.masked_array(data * 632.8e-9, mask=mask))
        
        i = i+1
    return(ima)


'Funzione per trovare RMS e delta RMS di un interferogramma. Le variabili senza 1 si riferiscono al conto per RMS, le variabili con 1 si riferiscono al conto per deltaRMS'

def RMS(image = []):
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



'Immettere coeffmedio = coefficienti di Zernike per immagine'
'ck = coefficienti di Zernike ottenuti da analisi Montecarlo'
'Il parametrdo d da la statistica dei coefficienti (d<1 ok, d>1 devo capire quale Zernike mi disturba'


#def coefzernananalysis(coeffmedio):
    
    
 #   i=4
 #   d=0
    
 #   while i<36:
        
 #       d += (coeffmedio[i]/ck[i])**2
 #       i=i+1
        
 #   return d
 
def meanimage(image = []):
    mean = np.ma.mean(image, axis=0)
    return mean
 
 


'Calcolo del PV, del delta PV e delle rispettive medie e deviazioni standard'

def PV(image = []):  
    i=0
    pv = []
    somma = 0
    sommadev = 0
    mean = np.ma.mean(image, axis=0)
    pv1 = []
    somma1 = 0
    sommadev1 = 0
    
    while i<len(image):
        a = np.amax(image[i])
        b = np.amin(image[i])
        pv.append(a-b)
        
        newima = image[i] - mean 
        c = np.amax(newima)
        d = np.amin(newima)
        pv1.append(c-d)
        
        print("PV",i, "=", pv[i])
        print("deltaPV",i, "=", pv1[i])
        somma+=pv[i]
        somma1+=pv1[i]
        i=i+1
    print("")
    media=somma/len(image)
    media1=somma1/len(image)
    print("Mean PV = ", media)
    print("Mean deltaPV = ", media1)
    
    g = 0
    while g<len(image):
        sommadev+=(pv[g]-media)**2
        sommadev1+=(pv1[g]-media1)**2
        g=g+1
        
    dev = np.sqrt(sommadev/(len(image)-1)) 
    dev1 = np.sqrt(sommadev1/(len(image)-1))
    print("Standard Deviation PV = ", dev)
    print("Standard Deviation deltaPV = ", dev1)
    
    return pv, media, dev, pv1, media1, dev1        
  

        



