from osgeo import gdal
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
#import incidence_angle_corr
from math import pi


def plot_feature_space(arr1,arr2):
    plt.plot(arr1.flatten(), arr2.flatten(), 'ko', markersize=.5)
    plt.show()
    
if __name__=='__main__':
    plot_feature_space(np.array([[2,3],[3,4]]), np.array([[6,7],[8,9]]))