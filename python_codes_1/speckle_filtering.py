import plotting 
from osgeo import gdal, ogr, osr
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import incidence_angle_corr
from math import pi
import itertools
#from scipy import linalg
from sklearn import mixture

import extract_polarimetric
import glcm_sklearn
import fit_inci_model
from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage,misc
import reproject
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


import EPFS
from skimage.draw import circle
import numpy.ma as ma
import feature_selection
#plt.style.use('ggplot')
import pandas as pd
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from scipy.special import factorial
import read_binary
import classification
import UAVSAR_time_series

def test_convolve(arr, kernal):
    grad = signal.convolve2d(arr, kernal, boundary='symm', mode='valid')
    return grad

def kernal(window_size_x, window_size_y):
    k=np.ones((window_size_y,window_size_x))#.reshape(window_size_x, window_size_y)
    normalize_k=k/(window_size_x*window_size_y)
    return normalize_k

def box_car_filtering(arr, window_size_x, window_size_y):
    return test_convolve(arr, kernal(window_size_x, window_size_y))

def main():
    base_dir='../North_Sea_UAVSAR/UAV_norway'
    file_ext=['.ann','.dat','.gif','.hgt','.inc','.kmz','.slope','_hgt.tif','_pauli.tif']
    folders=['_mlc','_grd']
    Region='norway'
    Heading='007'
    Counter_num='09'
    Year='15'
    Num_flights_year='092'
    Data_take='000'
    Day='10'
    Month='06'
    Band='L'
    Steering_angle='090'
    Cross_talk='CX'
    Processing_version='02'
    Polarization='VV'
    
    wd=UAVSAR_time_series.getdir(region=Region,heading=Heading,counter_num=Counter_num,year=Year,num_flights_year=Num_flights_year,data_take=Data_take,day=Day,month=Month,band=Band,steering_angle=Steering_angle,cross_talk=Cross_talk,processing_version=Processing_version)
    
    base_file_name=UAVSAR_time_series.get_base_file_name(region=Region,heading=Heading,counter_num=Counter_num,year=Year,num_flights_year=Num_flights_year,data_take=Data_take,day=Day,month=Month,band=Band,steering_angle=Steering_angle,cross_talk=Cross_talk,processing_version=Processing_version)
    
    meta=UAVSAR_time_series.metadata_dict(base_file_name)
    
    os.chdir(wd)
    #print(os.getcwd())
    
    slc_VV=UAVSAR_time_series.get_SLC(meta,base_file_name, polarization='VV', dType='slc', mlc_cropping_list=[521,1545,4049,5233], cropping_List_MLC=True)
    
    slc_VV_boxcar=box_car_filtering(slc_VV, 3,12)
    
    
    
    
    
if __name__=='__main__':
    main()