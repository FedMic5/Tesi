'''
Created on 3 feb 2022

@author: federico
'''

import numpy as np
from Tesi import geo
fac = np.math.factorial
from Tesi import Zernike


def removeZernike(ima, modes=np.array):
    coeff, mat = Zernike.zernikeFit(ima, modes)
    surf = Zernike.zernikeSurface(ima, coeff, mat)
    new_ima = ima - surf
    return new_ima