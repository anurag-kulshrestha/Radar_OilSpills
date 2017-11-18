from osgeo import gdal
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import incidence_angle_corr
from math import pi


#os.chdir('../MLC_Python_New/norway_00709_15092_000_150610_L090_CX_01_mlc')
#the the grey levels were quantised between 0 and 255

def GLCM_array(radio_bit):
    arr=np.zeros((radio_bit,radio_bit))
    return arr

def extract_Occur(direction, arr):#array_elements=unique 1d array
    #arr=np.array([[0,9,2,1,4],[1,2,4,5,3],[1,2,3,2,0],[1,6,0,2,1],[1,60,0,2,1]])
    array_elements=np.unique(arr)
    size=arr.shape
    glcm_arr=GLCM_array(array_elements.size)
    #img_arr_hist_inci_corr=incidence_angle_corr.hist_stretch(incidence_angle_corr.inci_correction())
    #print(array_elements)
    #print(arr)
    for i in range(0,size[0]):
        for j in range(0,size[1]):
            val=arr[i,j]
            if(j>0):
                val_1=arr[i,j-1]
            else:
                val_1=-1
            if(j<size[1]-1):
                #print(j)
                val_2=arr[i,j+1]
                #val_2=array_elements[i,j]
            else:
                val_2=-1
            if(val_1!=-1):
                pos_x=np.where(array_elements==val)[0][0]
                pos_y=np.where(array_elements==val_1)[0][0]
                glcm_arr[pos_x, pos_y]=glcm_arr[pos_x, pos_y]+1
            if(val_2!=-1):
                pos_x=np.where(array_elements==val)[0][0]
                pos_y=np.where(array_elements==val_2)[0][0]
                glcm_arr[pos_x, pos_y]=glcm_arr[pos_x, pos_y]+1
                #glcm_arr[val, val_2]=glcm_arr[val, val_2]+1
    return glcm_arr



def compute_Prob(direction, arr):
    glcm_arr=extract_Occur(direction, arr)
    s=np.sum(glcm_arr)
    return glcm_arr/s
    

def calc_weight(method, arr):
    arr_uni=np.unique(arr)
    #print(arr_uni)
    radio_bit=arr_uni.size
    weight=np.zeros((radio_bit, radio_bit))
    #index=np.indices((radio_bit,radio_bit))
    if(method==0): #0=contrast
        #return (index[0]-index[1])**2
        for i in range(0,radio_bit):
            for j in range(0,radio_bit):
                weight[i,j]=(arr_uni[i]-arr_uni[j])**2
        return weight

def calc_sum(arr, method, direction):
    #glcm_arr=extract_Occur(direction, arr)
    weight=calc_weight(method,arr)
    prob=compute_Prob(direction, arr)
    return np.sum(weight*prob)
    #return weight*prob

def occurance_kernel(direction, method, window_size, stride_row, stride_col):
    img_arr_hist_inci_corr=incidence_angle_corr.hist_stretch(incidence_angle_corr.inci_correction('C3', 'C33'), 5)
    
    rows=img_arr_hist_inci_corr.shape[0]
    cols=img_arr_hist_inci_corr.shape[1]
    #res=np.zeros((math.floor(rows/(stride_row-1))-window_size, math.floor(cols/(stride_col-1))-window_size))
    res=np.zeros((math.floor(rows/stride_row), math.floor(cols/stride_col)))
    res_row=0
    res_col=0
    for i in range(0, rows-window_size, stride_row):
        for j in range(0, cols-window_size, stride_col):
            a=img_arr_hist_inci_corr[i:i+window_size, j:j+window_size]
            res[res_row,res_col]=calc_sum(a, method, direction)
            #print((i,j))
            res_col+=1
        res_col=0
        res_row+=1
        print(i)
    incidence_angle_corr.display(res, 'Range(pixel#)', 'Azimuth (pixel #)', 'Contrast GLCM feature(dir=0, window_size=15, row_stride=2, col_stride=2)')
    
    
if __name__=='__main__':
    #arr=np.array([[0,9,2,1,4],[1,2,4,5,3],[1,2,3,2,0],[1,6,0,2,1],[1,10,0,2,1]])
    #print(GLCM_array(0, 5))
    #glcm_arr=extract_Occur(direction, arr)
    #print(arr)
    #print(np.unique(arr))
    #print(extract_Occur(0, arr))
    #print(compute_Prob(0,arr))
    #print(calc_weight(0,arr))
    #print(calc_sum(arr, 0,0))
    occurance_kernel(0, 0, 15, 2, 2)