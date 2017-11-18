from numpy import linalg as LA
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
import incidence_angle_corr
import reproject
from osgeo import gdal
#np.linalg.eig
import math

def test_convolve(arr, kernal):
    grad = signal.convolve2d(arr, kernal, boundary='symm', mode='valid')
    return grad

def kernal(window_size):
    k=np.ones(window_size*window_size).reshape(window_size, window_size)
    normalize_k=k/(window_size**2)
    return normalize_k

def read_image(matrix, element):
    mat_ele=matrix+'/'+element
    img=gdal.Open(mat_ele+'.bin')
    return img

def read_Raster(matrix, element):
    img=read_image(matrix, element)
    arr_img=img.ReadAsArray()
    #set_mat_element_name(mat_ele)
    #print(C33.RasterXSize, C33.RasterYSize, C33.RasterCount)
    #print(arr_C33)
    #print(C33_gt)
    return arr_img

def co_pol_power_ratio(Ivv, Ihh, window_size):
    k=kernal(window_size)
    Ivv_avg=test_convolve(Ivv, k)
    Ihh_avg=test_convolve(Ihh, k)
    return Ivv_avg/Ihh_avg

def co_pol_power_ratio_1(cov_arr):
    #cov_arr=extract_covariance_arr(window_size)
    return cov_arr[:,:,2,2]/cov_arr[:,:,0,0]

def extract_coherency_arr(window_size):
    elements=['T22','T33', 'T12_real', 'T12_imag', 'T13_real', 'T13_imag', 'T23_real', 'T23_imag']
    raster_arr=test_convolve(read_Raster('T3', 'T11'), kernal(window_size))
    raster_stack=np.stack(np.array([raster_arr], dtype=np.complex64))
    iota=1j
    for i in elements:
        raster_arr=test_convolve(read_Raster('T3', i),kernal(window_size))
        raster_stack=np.concatenate((raster_stack,[raster_arr]))
    t11=raster_stack[0]
    t22=raster_stack[1]
    t33=raster_stack[2]
    t12=raster_stack[3]+raster_stack[4]*iota
    t13=raster_stack[5]+raster_stack[6]*iota
    t23=raster_stack[7]+raster_stack[8]*iota
    new_shape=t11.shape
    return np.dstack((t11,t12,t13,t12,t22,t23,t13,t23,t33)).reshape(new_shape[0],new_shape[1],3,3)

def extract_covariance_arr(window_size, correction_switch):
    elements=['C22','C33', 'C12_real', 'C12_imag', 'C13_real', 'C13_imag', 'C23_real', 'C23_imag']
    if (correction_switch):
        raster_arr=test_convolve(incidence_angle_corr.inci_correction('C3', 'C11'), kernal(window_size))
    else:
        raster_arr=test_convolve(read_Raster('C3', 'C11'), kernal(window_size))
    #raster_arr=read_Raster('C3', 'C11')
    raster_stack=np.stack(np.array([raster_arr], dtype=np.complex64))
    iota=1j
    if (correction_switch):
        for i in elements:
            raster_arr=test_convolve(incidence_angle_corr.inci_correction('C3', i),kernal(window_size))
            raster_stack=np.concatenate((raster_stack,[raster_arr]))
    else:
        for i in elements:
            raster_arr=test_convolve(read_Raster('C3', i),kernal(window_size))
            raster_stack=np.concatenate((raster_stack,[raster_arr]))
            
    c11=raster_stack[0]
    c22=raster_stack[1]
    c33=raster_stack[2]
    c12=raster_stack[3]+raster_stack[4]*iota
    c13=raster_stack[5]+raster_stack[6]*iota
    c23=raster_stack[7]+raster_stack[8]*iota
    new_shape=c11.shape
    return np.dstack((c11,c12,c13,c12,c22,c23,c13,c23,c33)).reshape(new_shape[0],new_shape[1],3,3)

def determinant_cov(cov_arr):
    #cov_arr=extract_covariance_arr(window_size)
    shp=cov_arr.shape
    #return shp
    linear_cov_arr=np.reshape(cov_arr,(np.prod(shp[:2]),shp[2],shp[3]))
    det_arr=np.linalg.det(linear_cov_arr)
    return det_arr.reshape(shp[0],shp[1])

def get_eigen_values(coh_matrix):
    w=LA.eigvals(coh_matrix)
    return w

def eigen_raster_full(window_size):
    arr=extract_coherency_arr(window_size)
    eigen_arr=np.sort(np.absolute(get_eigen_values(arr)), axis=2)
    return eigen_arr

def eigen_raster(eigen_num, window_size):
    #arr=extract_coherency_arr(window_size)
    eigen_arr=eigen_raster_full(window_size)
    
    if(eigen_num==1):
        lamb1=eigen_arr[:,:,2]
        return lamb1
    if(eigen_num==2):
        lamb2=eigen_arr[:,:,1]
        return lamb2
    if(eigen_num==3):
        lamb3=eigen_arr[:,:,0]
        return lamb3

def get_p_value(eigen_full):
    #eigen_full=eigen_raster_full(window_size) 
    lamb1=eigen_full[:,:,2]
    lamb2=eigen_full[:,:,1]
    lamb3=eigen_full[:,:,0]
    lamb_sum=np.sum(eigen_full, axis=2)
    #a=eigen_full
    #p=np.divide(eigen_full, lamb_sum) #We can include a axis parameter for np.divide as well
    #return a
    p1=lamb1/lamb_sum
    p2=lamb2/lamb_sum
    p3=lamb3/lamb_sum
    #return np.dstack((p1,p2,p3))
    return [p1,p2,p3]

def entropy(eigen_full):
    p=get_p_value(eigen_full)
    p1_log=np.log(p[0])/math.log(3)
    p2_log=np.log(p[1])/math.log(3)
    p3_log=np.log(p[2])/math.log(3)
    ent=-1*(p[0]*p1_log+p[1]*p2_log+p[2]*p3_log)
    return ent
#def Cloude_Pottier():
    
def pol_fraction(eigen_full):
    #eigen_full=eigen_raster_full(window_size)
    #return eigen_full.shape
    lamb_sum=np.sum(eigen_full, axis=2)
    #return lamb_sum
    return 1-(eigen_full[:,:,0]/lamb_sum)

def anisotropy(eigen_full):
    #eigen_full=eigen_raster_full(window_size)
    return (eigen_full[:,:,1]-eigen_full[:,:,0])/(eigen_full[:,:,1]+eigen_full[:,:,0])

def pedestal_height(eigen_full):
    return eigen_full[:,:,0]/eigen_full[:,:,2]
    

#def pass_filter():
    #arr=np.array([[-6-6j, -3-3j, 0-10j,  +3 -3j, 6-6j],[-20+0j, -10+0j, 0+ 0j, +10 +0j, 10+0j],[-6+6j, -3+3j, 0+10j,  +3 +3j, 6+6j]])
    #return arr

def co_pol_cross_product(cov_arr):
    #arr=extract_covariance_arr(window_size)
    return cov_arr[:,:,0,2]
    
def co_pol_diff(cov_arr):
    #cov_arr=extract_covariance_arr(window_size)
    return cov_arr[:,:,0,0]-cov_arr[:,:,2,2]

def save_feature(name, output_array, window_size):
    reproject.save_tiff_image(name, output_array, window_size)
    
    
    
if __name__=='__main__':
    #arr=np.array([[0,9,2,1,4],[1,2,4,5,3],[1,2,3,2,0],[1,6,0,2,1],[1,60,0,2,1]])
    #Ivv=incidence_angle_corr.inci_correction('C3', 'C33')
    #Ihh=incidence_angle_corr.inci_correction('C3', 'C11')
    cov_arr=extract_covariance_arr(25, False)
    #kernal=pass_filter()
    #print(arr)
    #imgplot=plt.imshow(incidence_angle_corr.hist_stretch(Ivv,6), cmap='gray')
    #plt.show()
    #imgplot=plt.imshow(np.angle(res), cmap='gray')
    #image=reproject_new('Co-pol_pl_power_ratio', , newRasterYSize, bands, fuse_array, projection,geotransform)
    
    #arr_co_pol=co_pol_power_ratio(Ivv, Ihh, 10)
    #imgplot=plt.imshow(arr_co_pol, cmap='gray')
    #plt.show()
    
    #arr=extract_coherency_arr(9)
    #print(arr)
    #arr1=arr.reshape((1177,1017,3,3))
    #arr2=arr[0][1].reshape(3,3)
    #print(arr[0][0])
    #print(get_eigen_values([arr1,arr2]))
    #print(np.amax(np.absolute(get_eigen_values(arr)), axis=2).shape)#[1000][1000])
    #print(np.sort(np.absolute(get_eigen_values(arr)), axis=2)[:,:,2])#[1000][1000])#.shape)
    #print(entropy(9))#[1000][1000])
    #arr_lamb1=eigen_raster(3,9)
    #arr_ent=entropy(9)
    #arr_det_cov=determinant_cov(9)
    #arr_pol_frac=pol_fraction(9)
    #arr_anisotropy=anisotropy(9)
    #arr_co_cross=co_pol_cross_product(9)
    arr_co_pol_diff=co_pol_diff(cov_arr)
    #print(arr_co_cross.shape)
    #print(arr_pol_frac)
    #imgplot=plt.imshow(np.imag(arr_co_cross), cmap='gray')
    save_feature('abs_of_co_pol_diff_window_size_25', np.absolute(arr_co_pol_diff), 25)
    imgplot=plt.imshow(np.absolute(arr_co_pol_diff), cmap='gray')
    plt.colorbar()
    plt.show()
    