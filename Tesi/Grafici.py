'''
Created on 16 mar 2022

@author: federico
'''
import os
import h5py
import skimage
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean
from Tesi import openfileinterf
from Tesi import zernike
from pickle import NONE
fac = np.math.factorial
from Tesi import geo
from skimage.draw import circle
from skimage.measure import EllipseModel
from skimage.draw import ellipse
from skimage.measure import CircleModel
from Tesi import zernike_circle
from Tesi import zernike_ellipse



def grafico():
    my_dpi = 300
    font_size = 9
    title_label='$C_N^2$ - Vertical profile temporal evolution'
    fig=plt.figure(figsize=(1900./my_dpi, 1550./my_dpi), dpi=my_dpi)
    ax1 = plt.subplot(1,1,1)
    plt.rcParams.update({'mathtext.default':  'regular' })
    plt.rcParams['axes.labelsize'] = font_size
    plt.rcParams['xtick.labelsize'] = font_size
    plt.rcParams['ytick.labelsize'] = font_size
    #levels=np.linspace(min_S, max_S, 100)
    #my_map.set_under("white")
    #my_map.set_over("black")
    #im=plt.contourf(dff_slice.index,heights,cn2,levels=levels,cmap=my_map,vmin=min_S, vmax=max_S, extend='max')

    print(dff_slice.index[200])
    ##im=plt.contourf(data.time, data.heights, data.cn2.transpose(), levels=levels, cmap=my_map,vmin=min_S, vmax=max_S, extend='max')
   
    plt.title(title_label,fontsize=font_size+1.4, y=1.15)
    plt.text(0, 1.09,stringadata_fig,verticalalignment='center',transform = ax1.transAxes, fontsize=font_size+1)
    #centery=(np.max(data.heights[bottom:top+1])+np.min(data.heights[bottom:top+1]))/2.
    ##start, end = ax1.get_xlim()
    ##ax1.xaxis.set_ticks(np.arange(int(start), int(end), 1,np.int64))
    ##ystepping=int(top_h)/10.
    ##ax1.yaxis.set_ticks(np.arange(int(minvalY), int(top_h)+1, ystepping,np.float64))    
    ##plt.xlabel('Hour (from simulation start)')
    ##ax1.xaxis.labelpad = 0
    plt.ylabel('Height (km) a.g.l.')
    plt.tick_params(which='both')
    plt.tick_params(which='major', length=10,width=1)
    plt.tick_params(which='minor', length=5)
    plt.minorticks_on()
    plt.xlim(startdate,enddate)
    plt.ylim(minvalY,maxvalY)

    ##
    ### Barra con scala colori    
    cbar = plt.colorbar(im, orientation='horizontal', shrink=3, pad=0.12,fraction=0.06, ax=ax1)
    cbar.ax.tick_params(labelsize=font_size+1)
    if ((round(max_S)-round(min_S)) > 0):
        labels = np.arange(round(min_S,1), max_S, (round(max_S)-round(min_S))/10)
        labels = np.round(labels,1)
        loc = labels
        cbar.set_ticks(loc)
        cbar.set_ticklabels(labels)
    cbar.set_label("CN2 ($m^{-2/3}$)", size=font_size+1)
    ####if (TIMEOFFSET != 0) | (LABELUP != ""):
    ##ax2 = ax1.twiny()
    ##ax2.set_xlim(ax1.get_xlim())
    ##ax1Ticks = ax1.get_xticks()
    ##ax2.set_xticks(ax1Ticks)
    ##ax2Ticks=ax1Ticks+TIMEOFFSET
    ######    for i in range(len(ax2Ticks)):
    ######        if (ax2Ticks[i] < 0):
    ######            ax2Ticks[i]= ax2Ticks[i]+24
    ######
    ##ax2.set_xbound(ax1.get_xbound())
    ##ax2.set_xticklabels(ax2Ticks,fontsize=font_size+1)
    ####    ax2.set_xlabel(LABELUP,fontsize=font_size+1)
    ##ax2.tick_params(which='both')
    ##ax2.tick_params(which='major', length=10,width=1)
    ##ax2.tick_params(which='minor', length=5)
    ##ax2.minorticks_on()
    plt.tight_layout()
    plt.subplots_adjust(left=0.12, right=0.96, top=0.86, bottom=0.1)
    fileout=startdate_string+"_cn2_pv_evol_time_"+".png"
    plt.savefig(fileout, dpi=my_dpi)
    print("Creato file con plot:", fileout)