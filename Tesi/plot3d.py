'''
Created on 30 mar 2022

@author: federico
'''

import numpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

def plot3d(image):
    
    z = image
    x,y = z.shape
    xs = np.arange(x)
    ys = np.arange(y)
    
    xx, yy = np.meshgrid(xs,ys)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection = '3d')
    
    mycmap = cm.plasma
    ax.plot_surface(xx,yy,z, linewidth = 0, antialiased = False, cmap = mycmap)
    
    return image

    






