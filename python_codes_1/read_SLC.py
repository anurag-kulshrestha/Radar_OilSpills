#python for NORSE-2015

from osgeo import gdal
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#top_left=gdal.Open('subset_0_of_S1A_IW_GRDH_1SDV_20170129T003132_20170129T003157_015039_01892E_6D04_Amplitude_VV_top_left_new.tif')

os.chdir('../North_Sea_UAVSAR/UAV_norway/SLC_004')
#print(os.getcwd())


def display(arr, x_label, y_label, title):
    imgplot=plt.imshow(arr, cmap='gray')
    #imgplot=plt.imshow(arr)
    #plt.set_yticklabels()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.colorbar()
    plt.show()

def hist_stretch(arr):
    n=arr.shape
    #new_arr=arr
    per=np.percentile(arr,[2.5, 97.5])
    per_max=per[1]
    per_min=per[0]
    min_arr=np.full(n, per_min)
    max_arr=np.full(n, per_max)
    #new_arr=arr
    new_arr=np.maximum(min_arr, np.minimum(max_arr, arr))
    new_arr=np.floor(255*(new_arr-per_min)/(per_max-per_min))
    return new_arr

def clear_list(arr):
    #del arr[:]
    del arr

def dimension_data(data):
    return data.shape

if __name__=='__main__':
    s11=gdal.Open('s11.bin')
    s11_gt=s11.GetGeoTransform()
    arr_s11=s11.ReadAsArray()
    a=np.absolute(arr_s11)
    #print(dimension_data(arr_s11))
    display(hist_stretch(a), 'Range', 'Azimuth', 'UAVSAR S11')
    #print(arr_s11)
    clear_list(a)
    clear_list(arr_s11)