'''
Created on 3 mar 2022

@author: federico
'''

import numpy as np
#import image as image
import scipy
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.measure import EllipseModel
from skimage.draw import ellipse
from skimage.measure import CircleModel
from skimage.draw import circle



def qpupil_circle(image):
        
        aa = np.shape(image)
        imagePixels = aa[0]
        circ = CircleModel()
        cnt = trova_punti_bordi_ima2(image, imagePixels)
        circ.estimate(cnt)
        xc, yc, radius = np.array(circ.params, dtype = int)
        maskedd = np.zeros((imagePixels, imagePixels), dtype = np.uint8) 
        rr, cc = circle(xc, yc , int(radius))
        maskedd[rr,cc] = 1
    
    
        idx = np.where(maskedd==1)
        ss = np.shape(maskedd)
        x = np.arange(ss[0]).astype(float)
        x = np.transpose(np.tile(x, [ss[1], 1]))
        y = np.arange(ss[1]).astype(float)
        y = np.tile(y, [ss[0], 1])
        xx = x
        yy = y
        #maxv = max(xx[idx])
        #minv = min(xx[idx])
        xx = xx - xc
        xx = xx/radius
        #maxv = max(yy[idx])
        #minv = min(yy[idx])
        yy = yy - yc
        yy = yy/radius
        
        return xc, yc, radius, xx, yy
    
    
def qpupil_ellipse(image):
        
        aa = np.shape(image)
        imagePixels = aa[0]
        ell = EllipseModel()
        cnt = trova_punti_bordi_ima2(image, imagePixels)
        ell.estimate(cnt)
        xc, yc, a, b, theta = np.array(ell.params, dtype = int)
        maskedd = np.zeros((imagePixels, imagePixels), dtype = np.uint8)
        radius = max(a,b)
        rr, cc = circle(xc, yc , int(radius))
        maskedd[rr,cc] = 1 
        
        

        idx = np.where(maskedd==1)
        ss = np.shape(maskedd)
        x = np.arange(ss[0]).astype(float)
        x = np.transpose(np.tile(x, [ss[1], 1]))
        y = np.arange(ss[1]).astype(float)
        y = np.tile(y, [ss[0], 1])
        xx = x
        yy = y

        xx = xx - xc
        xx = xx/radius

        yy = yy - yc
        yy = yy/radius
        
        return xc, yc, radius, xx, yy





def trova_punti_bordi_ima2(image, imagePixels):
        
        x = image
        val = []
        
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
           
        cut = np.concatenate(val)
        
        return cut


    
    