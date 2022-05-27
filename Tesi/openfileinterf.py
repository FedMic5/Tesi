'''
Created on 3 feb 2022

@author: federico
'''


import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean


def from4D4020(h5filename):
       
    file = h5py.File(h5filename, 'r')
    genraw = file['measurement0']['genraw']['data']
    data = np.array(genraw)
    mask = np.zeros(data.shape, dtype=np.bool)
    mask[np.where(data == data.max())] = True
    ima = np.ma.masked_array(data * 632.8e-9, mask=mask)
    return ima

def from4D6110(i4dfilename):
        """
        Parameters
        ----------
            h5filename: string
                 path of h5 file to convert

        Returns
        -------
                ima: numpy masked array
                     masked array image
        """
        file = h5py.File(i4dfilename, 'r')
        data = file.get('/Measurement/SurfaceInWaves/Data')
        meas = data[()]
        mask = np.invert(np.isfinite(meas))
        image = np.ma.masked_array(meas * 632.8e-9, mask=mask)
        return image


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

def from4Dlist2(h5filename):
       
    file = h5py.File(h5filename, 'r')
    
    misura = ['measurement0']
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



def from4Dlist3(h5filename):
       
    file = h5py.File(h5filename, 'r')
    
    misura = ['measurement0','measurement1','measurement2','measurement3','measurement4','measurement5']
    i = 0
    ima = []
    
    while i<len(misura):
        genraw = file[misura[i]]['genraw']['opts']
        data = np.array(genraw)
        mask = np.zeros(data.shape, dtype=np.bool)
        mask[np.where(data == data.max())] = True
        ima.append(np.ma.masked_array(data * 632.8e-9, mask=mask))
        
        i = i+1
    return(ima)

