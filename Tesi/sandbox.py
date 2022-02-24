'''
Created on 3 feb 2022

@author: federico
'''
import os
import h5py
import skimage
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean
from skimage.draw import circle
from Tesi import openfileinterf
from Tesi import zernike
from pickle import NONE
fac = np.math.factorial
from Tesi import geo
from skimage.measure import EllipseModel
from skimage.draw import ellipse
from skimage.measure import CircleModel


class Measurements():
    
    def __init__(self, interf):
        self._interf = interf
        self.name = 'pippo'
        
    def misura(self):
        pass
    
        

class Analysis():
    
    '''
    originData si riferisce alla cartella Data in Scrivania
    in __init__ digitare path = cartella dove dentro ci sono le immagini (dentro Data)
    '''
    
    def __init__(self, path):
        self._originData = '/home/federico/Scrivania/Data'
        self._path = path
    '''
    In open_images_4D inserire num = numero di file nella cartella
    output = ima_list = lista delle immagini
    '''    
    def open_images_4D(self, num):
        ima_list = []
        for i in range(num):
            name = '%d.4D' %i
            final_path = os.path.join(self._originData, self._path,
                                      name)
            ima = openfileinterf.from4D6110(final_path)
            ima_list.append(ima)
        return ima_list
    
    
        
    '''
    image deve essere singola
    in [:,:] il primo è la y il secondo la x
    Per taglio 1D digitare solo il numero del pixel dove fare il taglio
    Per taglio 2D digitare numero del pixel range [pixel1 = 1:1000, pixel2 = 100:2000]
    '''  
    
    def cutprofiley(self, image, pixel1):  
        
        yprofile = image[pixel1, :]
        plt.plot(yprofile)
        return yprofile
      
    def cutprofilex(self, image, pixel1):  
        
        xprofile = image[:,pixel1]
        plt.plot(xprofile)
        return xprofile  
    
    def cutprofilexy(self, image, pixel1, pixel2, pixel3, pixel4):  
        
        xyprofile = image[pixel1:pixel2, pixel3:pixel4]
        plt.plot(xyprofile)
        return xyprofile   
    
    
    
    
    def RMS(self, image):
        '''
        image deve essere una lista
        Dev standard con a denominatore n e non n-1
        '''
        i = 0
        
        val = []
        std = []
        
        mean = np.ma.mean(image, axis=0)
        val1 = []
        std1 = []
    
            
        while i<len(image):
            val.append(np.reshape(image[i][:,:], -1))
            std.append(np.std(val[i]))
            
            newima = image[i] - mean       
            val1.append(np.reshape(newima[:,:], -1))
            std1.append(np.std(val1[i]))
            
            print("RMS",i, "=", std[i])
            print("DeltaRMS",i, "=", std1[i])
            
            i=i+1
        
        print("")
        media = np.mean(std)
        media1= np.mean(std1)
        print("Mean RMS = ", media)
        print("Mean DeltaRMS = ", media1)
        
        dev = np.std(std)
        dev1 = np.std(std1)
        print("Standard Deviation RMS = ", dev)
        print("Standard Deviation deltaRMS = ", dev1)
                
        return std, media, dev, std1, media1, dev1
    '''
    Calcolo del PV, del delta PV e delle rispettive medie 
    e deviazioni standard
    '''
    def PV(self, image):  
        i=0
        pv = []
        mean = np.ma.mean(image, axis=0)
        pv1 = []
        
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
            i=i+1
            
        print("")
        
        media=np.mean(pv)
        media1=np.mean(pv1)
        print("Mean PV = ", media)
        print("Mean deltaPV = ", media1)
        
        dev = np.std(pv)
        dev1 = np.std(pv1)
        
        print("Standard Deviation PV = ", dev)
        print("Standard Deviation deltaPV = ", dev1)
        
        return pv, media, dev, pv1, media1, dev1
    

    def meanimage(self, image):
        mean = np.ma.mean(image, axis=0)
        return mean
    
    '''
    Funzione per la rimozione di coefficienti di Zernike.
    modes deve essere un array con coefficiente >0
    '''
    
    def removeZernike(self, ima, num):
        coeff, mat = zernike.zernikeFit(ima, np.linspace(1,num,num))
        surf = zernike.zernikeSurface(ima, coeff, mat)
        new_ima = ima - surf
        return new_ima
    
    
    
    def residui(self, image):
        '''
        image deve essere una sola immagine
        '''
        diff = []
        rms = []
        residui = []
        coeff, mat = zernike.zernikeFit(image, np.arange(1, 35))
        #surf_image = zernike.zernikeSurface(image, coeff, mat)
        i=0
        while i<len(coeff)+1:
            surf1 = zernike.zernikeSurface(image, coeff[0:i], mat[:,0:i])
            diff.append(image - surf1)
            rms.append(np.std(diff))
            residui.append(i+1)
            #coef, mat = ...(diff, np....)
            print("RMS", i, " = ", rms[i] )
            i=i+1
        plt.plot(residui,rms,marker="o",color='red')
        plt.title("Calcolo dei residui")
        plt.xlabel("Numero Zernike")
        plt.ylabel("RMS")
        #plt.ylim(np.min(rms), np.max(rms))
        plt.ylim(np.min(rms)*0.75, np.max(rms)*1.25)
        plt.show()
        
        
        #imshow(diff[i])
        
        return rms, diff
    
    
    
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
      
    


    '''
    file deve essere nel formato file.txt per essere letto
    coeff = coefficienti calcolati sull'immagine
    Immettere coeffmedio = coefficienti di Zernike per immagine
    ck = coefficienti di Zernike ottenuti da analisi Montecarlo
    Il parametrdo d da la statistica dei coefficienti (d<1 ok, d>1 
    devo capire quale Zernike mi disturba
    '''

    def coefzernanalysis(self, coeff):
        
        with open('/home/federico/Scrivania/zcoeff') as file_in:
            lines = []
            for line in file_in:
                if line.strip():
                    line_tokens = []
                    for tok in line.split():
                        line_tokens.append(float(tok))
                    lines.append(line_tokens)
                    
        i=3
        d=0
        val=[]
        vali=[]
        
        while i<len(coeff):
            print("coeff = ", coeff[i], "ck = ", lines[i-3])
            a = (coeff[i]/lines[i-3])**2
            d += (coeff[i]/lines[i-3])**2
            val.append(a)
            vali.append(i+1)
            i=i+1
        
        plt.plot(vali,val,marker="o",color='red')
        plt.title("Peso dei coefficienti di Zernike")
        plt.xlabel("Numero Zernike")
        plt.ylabel("Peso")
        plt.show()
        
        
        return d
        
        
        
    def create_circular_mask(self, center_y, center_x, radius, imagePixels=None):
        '''
        Parameters
        ----------
        center_y: int
                y coordinate for circular mask
        center_x: int
                x coordinate for circular mask
        radius: int
                radius of circular mask
        
        Other Parameters
        ----------
        imagePixels: int, optional
                radius of the image in which the mask is inserted
        
        Returns
        -------
        mask: numpy array
                ones circular mask
        '''
        
        if imagePixels is None:
            imagePixels = 512
        else:
            imagePixels = imagePixels
        mask = np.ones((imagePixels, imagePixels), dtype= bool) 
        rr, cc = circle(center_y, center_x, radius)
        mask[rr,cc] = 0
        return mask
    
    
    
    def circular_mask_to_image(self, image, center_y, center_x, radius, imagePixels=None):
        '''
        Parameters
        ----------
        image must to be a single image
        
        center_y: int
                y coordinate for circular mask
        center_x: int
                x coordinate for circular mask
        radius: int
                radius of circular mask
        
        Other Parameters
        ----------
        imagePixels: int, optional but important!!!
                radius of the image in which the mask is inserted
        
        Returns
        -------
        mask: numpy array
                ones circular mask
        '''
        
        if imagePixels is None:
            imagePixels = 512
        else:
            imagePixels = imagePixels
        maskedd = np.ones((imagePixels, imagePixels), dtype= bool) 
        rr, cc = circle(center_y, center_x, radius)
        maskedd[rr,cc] = 0
        
        new_ima = np.ma.masked_array(image, mask = maskedd)
        
        return new_ima
    
    
    
    def circular_mask_to_series_images(self, image, center_y, center_x, radius, imagePixels=None):
        '''
        Parameters
        ----------
        image must to be a list
        
        center_y: int
                y coordinate for circular mask
        center_x: int
                x coordinate for circular mask
        radius: int
                radius of circular mask
        
        Other Parameters
        ----------
        imagePixels: int, optional but important!!!
                radius of the image in which the mask is inserted
        
        Returns
        -------
        mask: numpy array
                ones circular mask
        '''
        
        new_ima = []
        
        if imagePixels is None:
            imagePixels = 512
        else:
            imagePixels = imagePixels
        maskedd = np.ones((imagePixels, imagePixels), dtype= bool) 
        rr, cc = circle(center_y, center_x, radius)
        maskedd[rr,cc] = 0
        i=0
        while i<len(image):
            new_ima.append(np.ma.masked_array(image[i], mask = maskedd))
            i=i+1
        
        return new_ima

    '''
    Funzione per il calcolo dei parametri di un'ellisse direttamente
    dall'immagine usando il metodo dei minimi quadrati
    image deve essere una singola immagine
    '''

    def ellipse_parameters_detection(self, image, imagePixels):
        
        ell = EllipseModel()
        cnt = self.trova_punti_bordi_ima2(image, imagePixels)
        ell.estimate(cnt)
        xc, yc, a, b, theta = np.array(ell.params, dtype = int)
        
        return xc, yc, a, b, theta
    

    '''
    Crea una maschera ellittica rispetto all'ellissi originale in base alla
    percentuale desiderata data in input
    perc = percentuale che si vuole eliminare dall'immagine partendo 
    dal centro-> numero int
    esempio: se vogliamo togliere il 10% interno del cerchio -> perc = 10
    '''


    
    def ellipse_mask_int(self, image, imagePixels,perc):
        
        p=perc/100
        xc, yc, a, b, theta = self.ellipse_parameters_detection(image,imagePixels)
        maskedd = np.zeros((imagePixels, imagePixels), dtype = np.uint8) 
        rr, cc = ellipse(xc, yc ,int(a*p) ,int(b*p) ,shape=None , rotation = theta)
        maskedd[rr,cc] = 1
        
        new_ima = np.ma.masked_array(image, mask = maskedd)
        
        return new_ima

    '''
    Crea una maschera ellittica rispetto all'ellise originale in base alla
    percentuale desiderata data in input
    perc = percentuale che si vuole ottenere dall'immagine 
    all'esterno-> numero int
    esempio: se vogliamo solo il 90% del cerchio eliminando il 10%
    esterno -> perc = 90
    '''

  
    
    def ellipse_mask_ext(self, image, imagePixels,perc):
        
        p=perc/100        
        xc, yc, a, b, theta = self.ellipse_parameters_detection(image,imagePixels)
        maskedd = np.zeros((imagePixels, imagePixels), dtype = np.uint8) 
        rr, cc = ellipse(xc, yc ,int(a*p) ,int(b*p) ,shape=None , rotation = theta)
        maskedd[rr,cc] = 1
        maskinv = np.logical_not(maskedd)
        
        new_ima = np.ma.masked_array(image, mask = maskinv)
        
        return new_ima
    
    
    '''
    Funzione per il calcolo dei parametri di un cerchio direttamente
    dall'immagine usando il metodo dei minimi quadrati
    image deve essere una singola immagine
    '''

    def circle_parameters_detection(self, image, imagePixels):
        
        circ = CircleModel()
        cnt = self.trova_punti_bordi_ima2(image, imagePixels)
        circ.estimate(cnt)
        xc, yc, radius = np.array(circ.params, dtype = int)
        
        return xc, yc, radius
    
    
    '''
    Crea una maschera circolare rispetto al cerchio originale in base alla
    percentuale desiderata data in input
    perc = percentuale che si vuole eliminare dall'immagine partendo 
    dal centro-> numero int
    esempio: se vogliamo togliere il 10% interno del cerchio -> perc = 10
    '''
    
    def circle_mask_int(self, image, imagePixels, perc):
        a=perc/100
        xc, yc, radius = self.circle_parameters_detection(image, imagePixels)
        maskedd = np.zeros((imagePixels, imagePixels), dtype = np.uint8) 
        rr, cc = circle(xc, yc , int(radius*a))
        maskedd[rr,cc] = 1
        
        new_ima = np.ma.masked_array(image, mask = maskedd)
        
        return new_ima
    
    '''
    Crea una maschera circolare rispetto al cerchio originale in base alla
    percentuale desiderata data in input
    perc = percentuale che si vuole tenere dall'immagine 
    all'esterno-> numero int
    esempio: se vogliamo solo il 90% del cerchio eliminando il 10%
    esterno -> perc = 90
    '''
    
    def circle_mask_ext(self, image, imagePixels, perc):
        a=perc/100
        xc, yc, radius = self.circle_parameters_detection(image, imagePixels)
        maskedd = np.zeros((imagePixels, imagePixels), dtype = np.uint8) 
        rr, cc = circle(xc, yc , int(radius*a))
        maskedd[rr,cc] = 1
        maskinv = np.logical_not(maskedd)
        
        new_ima = np.ma.masked_array(image, mask = maskinv)
        
        return new_ima
    
    
    
    def trova_punti_bordi_ima(self, image, imagePixels):
        
        limit = int(imagePixels/2)
        x = image
        val = []
        wwww = []
        
        i=limit-200
        while i < limit+200:
                       
            a = x[i,:]
            aa = np.where(a.mask.astype(int)==0)
            q = np.asarray(aa)
            val.append(np.array([[i,q[0,0]],[i,q[0,q.size-1]]]))
            i = i+1
           
        cut1 = np.concatenate(val)
        
        n=limit-200
        while n < limit+200:
                       
            w = image[:,n]
            ww = np.where(w.mask.astype(int)==0)
            www = np.asarray(ww)
            wwww.append(np.array([[www[0,0],n],[www[0,www.size-1],n]]))
            n = n+1
           
        cut2 = np.concatenate(wwww)
        
        d = np.array([[cut1],[cut2]])
        f = np.concatenate(d)
        cut3 = np.concatenate(f)
        
        return cut3
    
    
    def trova_punti_bordi_ima2(self, image, imagePixels):
        
        x = image
        val = []
        wwww = []
        
        i=0
        while i < imagePixels:
                       
            a = x[i,:]
            aa = np.where(a.mask.astype(int)==0)
            q = np.asarray(aa)
            if q.size < 2:
                i = i+1
            else:
                val.append(np.array([[i,q[0,0]],[i,q[0,q.size-1]]]))
                i = i+1
           
        cut1 = np.concatenate(val)
        
        return cut1
                
    
    
    

