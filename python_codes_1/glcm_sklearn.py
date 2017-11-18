import numpy as np
from skimage.feature import greycomatrix, greycoprops
from numpy import pi
import incidence_angle_corr
import math
import time

image = np.array([[0, 0, 1, 1],\
                  [0, 0, 1, 1],\
                  [0, 2, 2, 2],\
                  [2, 2, 3, 3]], dtype=np.uint8)
#result = greycomatrix(image, [1,2], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=8, symmetric=True, normed=True)

#sum_=np.sum(result[...,0,0])
#print(sum_)

#contrast = greycoprops(result, 'contrast')
#print (contrast)

def return_texture_value(direction, distance, imag_arr, texture):
    level_num=max(np.unique(imag_arr))+1
    glcm= greycomatrix(imag_arr, distance, direction, levels=level_num, symmetric=True, normed=True)
    texture=greycoprops(glcm, texture)
    return texture

def occurance_kernel(distance, direction, texture_method, window_size, stride_row, stride_col):
    img_arr_hist_inci_corr=incidence_angle_corr.hist_stretch(incidence_angle_corr.inci_correction('C3', 'C33'), 5)
    
    rows=img_arr_hist_inci_corr.shape[0]
    cols=img_arr_hist_inci_corr.shape[1]
    #res=np.zeros((math.floor(rows/(stride_row-1))-window_size, math.floor(cols/(stride_col-1))-window_size))
    res=np.zeros((math.floor(rows/stride_row), math.floor(cols/stride_col),1,2))
    res_row, res_col, max_res_col, max_res_row= 0,0,0,0
    for i in range(0, rows-window_size, stride_row):
        for j in range(0, cols-window_size, stride_col):
            a=img_arr_hist_inci_corr[i:i+window_size, j:j+window_size]
            #print (a.astype(int))
            res[res_row,res_col]=return_texture_value(direction, distance, a.astype(int), texture_method)
            #print((i,j))
            res_col+=1
        max_res_col=res_col
        res_col=0
        res_row+=1
        max_res_row=res_row
    print(res.shape)
    print(max_res_col, max_res_row)
    if(max_res_row<res.shape[0]):
        res=np.delete(res, np.s_[-1*(res.shape[0]-max_res_row):],0)
    if(max_res_col<res.shape[1]):
        res=np.delete(res, np.s_[-1*(res.shape[1]-max_res_col):],1)
        
    #print(res[...,0,1])
    incidence_angle_corr.display(res[...,0,0], 'Range(pixel#)', 'Azimuth (pixel #)', 'Contrast GLCM feature(method='+texture_method+'dir='+str(direction[0])+', window_size='+str(window_size)+', row_stride='+str(stride_row)+', col_stride='+str(stride_col)+')')
    incidence_angle_corr.display(res[...,0,1], 'Range(pixel#)', 'Azimuth (pixel #)', 'Contrast GLCM feature(method='+texture_method+'dir='+str(direction[1])+', window_size='+str(window_size)+', row_stride='+str(stride_row)+', col_stride='+str(stride_col)+')')
    
if __name__=='__main__':
    #{‘contrast’, ‘dissimilarity’, ‘homogeneity’, ‘energy’, ‘correlation_NO’, ‘ASM’}
    #print(return_texture_value([0, np.pi/2],[1], image, 'energy'))
    
    occurance_kernel([1], [0, np.pi/2],'ASM', 15, 10,10)