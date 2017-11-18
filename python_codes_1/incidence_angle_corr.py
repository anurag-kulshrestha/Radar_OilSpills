from osgeo import gdal
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import fit_inci_model
from math import pi
import plotting


#os.chdir('../North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01/MLC_Python_New/norway_00709_15092_000_150610_L090_CX_01_mlc')
#print(os.getcwd())

peg_lati=59.950769 #m
peg_longi=2.5844668 #m
peg_height=10701.8711 #m

def convert_to_dB(arr):
    return 10*np.log10(arr)


#C33=gdal.Open('C3/C33.bin')
mat_ele=""
def set_mat_element_name(name):
    mat_ele=name

def read_image(matrix, element):
    mat_ele=matrix+'/'+element
    img=gdal.Open(mat_ele+'.bin')
    return img

def read_Raster(matrix, element):
    img=read_image(matrix, element)
    arr_img=img.ReadAsArray()
    set_mat_element_name(mat_ele)
    #print(C33.RasterXSize, C33.RasterYSize, C33.RasterCount)
    #print(arr_C33)
    #print(C33_gt)
    return arr_img

'''
def calc_inci_angle_pixel(pix_lati, pix_longi):
    lati_diff=pix_lati-peg_lati
    longi_diff=pix_longi-peg_longi
    dist=(lati_diff**2+longi_diff**2)**0.5
    dist_meters=dist*111000
    angle=math.atan(dist_meters/peg_height)
    return angle
'''

def get_inc_ang_array():
    missing=np.arange(521, 1546, 1)
    #print(extrapolate_inc_angle(missing))
    angle=fit_inci_model.extrapolate_inc_angle(missing)
    return angle

#def plot_raster():
def hist_stretch(arr, bits, clip_extremes=True):
    n=arr.shape
    #new_arr=arr
    per=np.percentile(arr,[2.5, 97.5])
    per_max=per[1]
    per_min=per[0]
    min_arr=np.full(n, per_min)
    max_arr=np.full(n, per_max)
    if(clip_extremes==False):
        new_arr=arr
    else:
        new_arr=np.maximum(min_arr, np.minimum(max_arr, arr))
    new_arr=np.floor((2**bits-1)*(new_arr-per_min)/(per_max-per_min))
    return new_arr
    

def display(arr, x_label, y_label, title):
    imgplot=plt.imshow(arr, cmap='gray')
    #plt.set_yticklabels()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.title('('+mat_ele+') '+title)
    plt.title(title)
    plt.colorbar()
    plt.show()

def plot_histogram(image_arr):
    #plt.hist(image_arr, bins=bins)
    #print(np.histogram(hist_stretch(image_arr, 5))[0].shape, np.histogram(hist_stretch(image_arr, 5))[1].shape)
    
    #H, bins=np.histogram(image_arr, list(range(0,32)))
    H, bins=np.histogram(image_arr)
    #plt.hist(arr_hist[::-1], bins='auto')
    #plt.plot(arr_hist[1], np.append(arr_hist[0],256))
    #print(bins[:-1], H)
    plt.bar(bins[:-1], H, width=0.001)
    plt.show()

def inci_correction(matrix, element):# https://doi.org/10.1016/S0034-4257(01)00279-6
    angle_arr=get_inc_ang_array()
    ref_angle=np.mean(angle_arr)
    img_raster_arr=read_Raster(matrix, element)
    #raster_arr=read_Raster()
    #raster_arr_copy=raster_arr
    #raster_arr[:,[5]]= raster_arr[:,[5]]*20
    #return raster_arr[:,[4,5]]
    for i in range(0,img_raster_arr.shape[1]):
        img_raster_arr[:,[i]]=img_raster_arr[:,[i]]*np.sin(angle_arr[i]/180*pi)
    img_raster_arr=img_raster_arr/np.sin(ref_angle/180*pi)
    #return (angle_arr.shape,raster_arr.shape)
    return img_raster_arr
    #print(angle_arr.shape)
    
#def reproject_image():
    


if __name__=='__main__':
    #print(inci_correction())
    image_arr=read_Raster('C3', 'C33')
    hist_image_arr=hist_stretch(image_arr, 5, False)
    #image_arr=inci_correction()
    #display(image_arr)
    #display(hist_stretch(image_arr,6), 'Range(pixel#)', 'Azimuth (pixel #)', 'No incidence angle correction applied')
    #display(hist_stretch(inci_correction('C3', 'C33'),6), 'Range(pixel#)', 'Azimuth(pixel#)', 'Incidence angle correction applied')
    #print(get_image_geo_details('C3', 'C33'))
    #plotting.plot_histogram(hist_image_arr, 'sigma nought (VV) (Linear Units)', 'Frequency', 'Histogram of C33 after linear stretching and clipping extreme values(2.5%, 97.5%)',32, 0.1)
    plotting.plot_histogram(hist_image_arr, 'sigma nought (VV) (Linear Units)', 'Frequency', 'Histogram of C33 after linear stretching only',32, 3)
    
    #plot_histogram(hist_image_arr)