'''
Created on 3 feb 2022

@author: federico
'''

import numpy as np
from Tesi import geo
fac = np.math.factorial
from Tesi import zernike


def removeZernike(ima, modes=np.array):
    coeff, mat = zernike.zernikeFit(ima, modes)
    surf = zernike.zernikeSurface(ima, coeff, mat)
    new_ima = ima - surf
    return new_ima