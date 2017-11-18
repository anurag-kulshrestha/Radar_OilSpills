from osgeo import gdal
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
#import incidence_angle_corr
from math import pi
from scipy import signal
from scipy import misc

os.chdir('../RISAT-1/RI1_SAR_L1SLC_FRS1_CR_20150610T071918_20150610T071923_17197_1515551004')

def img_to_array():
    s21=gdal.Open('s21.bin') #Srv
    s21_gt=s21.GetGeoTransform()
    s21_proj=s21.GetProjectionRef()
    arr_s21=s21.ReadAsArray()
    
    s11=gdal.Open('s11.bin') #Srh
    s11_gt=s11.GetGeoTransform()
    s11_proj=s11.GetProjectionRef()
    arr_s11=s11.ReadAsArray()
    
    return np.dstack((arr_s11, arr_s21))

def dimension_data(data):
    return data.shape

def save_figure(arr, x_label, y_label, title):
    fig1=plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    plt.imshow(arr, cmap='gray')
    #imgplot=plt.imshow(arr)
    #plt.set_yticklabels()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.colorbar()
    #plt.show()
    plt.savefig('Output/'+title+'.tiff', dpi=300)
    
    
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

def test_convolve(arr, kernal):
    grad = signal.convolve2d(arr, kernal, boundary='symm', mode='valid')
    return grad

def kernal(window_size):
    k=np.ones(window_size*window_size).reshape(window_size, window_size)
    normalize_k=k/(window_size**2)
    return normalize_k


def oil_subset(arr):
    return arr[6500:9000,5400:6800]
    
def averaging_arr(arr, window_size):
    kern=kernal(window_size)
    return test_convolve(arr, kern)

def get_phase(arr):
    return np.angle(arr)

def get_covariance_matrix(Srh_arr, Srv_arr, window_size):
    C12=averaging_arr(Srh_arr* np.conj(Srv_arr), window_size)
    C21=averaging_arr(Srv_arr* np.conj(Srh_arr), window_size)
    C11=averaging_arr(np.absolute(Srh_arr)**2, window_size)
    C22=averaging_arr(np.absolute(Srv_arr)**2, window_size)
    new_shape=C11.shape
    #return (C11[0,0],C12[0,0],C21[0,0],C22[0,0])
    return (np.dstack((C11,C12,C21,C22)).reshape(new_shape[0], new_shape[1], 2,2))#[0,0,:,:]
    
def get_stokes_vector(Srh_arr, Srv_arr, window_size):
    cross_product=averaging_arr(Srh_arr*np.conj(Srv_arr), window_size)
    q0=averaging_arr(np.absolute(Srh_arr)**2+np.absolute(Srv_arr)**2, window_size)
    q1=averaging_arr(np.absolute(Srh_arr)**2-np.absolute(Srv_arr)**2, window_size)
    q2=2*np.real(cross_product)
    q3=-2*np.imag(cross_product)
    return np.dstack((q0,q1,q2,q3))

def degreeOfPolarization(stokes_vector_arr):
    return np.sqrt(stokes_vector_arr[:,:,1]**2+ stokes_vector_arr[:,:,2]**2+ stokes_vector_arr[:,:,3]**2)/stokes_vector_arr[:,:,0]

def ellipticity_angle(stokes_vector, dop):
    return np.arcsin(-1*stokes_vector[:,:,3]/(dop*stokes_vector[:,:,0]))/2

def relative_phase(stokes_vector):
    return np.arctan(stokes_vector[:,:,3]/stokes_vector[:,:,2])/2

def alpha_angle(stokes_vector):
    return np.arctan((stokes_vector[:,:,1]+stokes_vector[:,:,2])/stokes_vector[:,:,3])/2

def circ_pol_ratio(stokes_vector):
    return (stokes_vector[:,:,0]-stokes_vector[:,:,3])/(stokes_vector[:,:,0]+stokes_vector[:,:,3])

def hybrid_pol_power_ratio(Srh_arr, Srv_arr, window_size):
    return averaging_arr(np.absolute(Srv_arr)**2, window_size)/averaging_arr(np.absolute(Srh_arr)**2,window_size)

def correlation_coeff(Srh_arr, Srv_arr, window_size):
    return averaging_arr(np.absolute(Srh_arr* np.conj(Srv_arr)), window_size)/np.sqrt(averaging_arr(np.absolute(Srh_arr)**2, window_size)*averaging_arr(np.absolute(Srv_arr)**2, window_size))

def std_phase_diff(Srh_arr, Srv_arr, window_size):
    phi_rh=get_phase(Srh_arr)
    phi_rv=get_phase(Srv_arr)
    return np.sqrt(averaging_arr((phi_rh-phi_rv)**2, window_size) + averaging_arr((phi_rh-phi_rv),window_size)**2)

def eigen_values(stokes_vector):
    lamb1=stokes_vector[:,:,0]+np.sqrt(stokes_vector[:,:,1]**2+stokes_vector[:,:,2]**2+stokes_vector[:,:,3]**2)
    lamb2=stokes_vector[:,:,0]-np.sqrt(stokes_vector[:,:,1]**2+stokes_vector[:,:,2]**2+stokes_vector[:,:,3]**2)
    return np.dstack((lamb1, lamb2))

def entropy(lambdas):
    sum_eigen=np.sum(lambdas, axis=2)
    p=lambdas/np.dstack((sum_eigen, sum_eigen))
    p_log=np.log(p)/np.log(2)
    H=-1*np.sum(p*p_log, axis=2)
    return H

def conformity_coeff(Srh_arr, Srv_arr, window_size):
    return 2*np.imag(averaging_arr(Srh_arr*np.conj(Srv_arr), window_size))/(averaging_arr(np.absolute(Srh_arr)**2, window_size)+averaging_arr(np.absolute(Srv_arr)**2, window_size))

def det_covariance_mat(cov_arr):
    shp=cov_arr.shape
    linear_cov_arr=np.reshape(cov_arr,(np.prod(shp[:2]),shp[2],shp[3]))
    det_arr=np.linalg.det(linear_cov_arr)
    return det_arr.reshape(shp[0],shp[1])

def extract_all_features(Srh_arr, Srv_arr, only_subset, window_size):
    if(only_subset==True):
        Srh_arr=oil_subset(Srh_arr)
        Srv_arr=oil_subset(Srv_arr)
    #if(do_average==True)
    #Srh_arr=averaging_arr(Srh_arr, window_size)
    #Srv_arr=averaging_arr(Srv_arr, window_size)
    save_figure(np.absolute(Srh_arr), 'Range', 'Azimuth', 'RISAT-1_Srh_Cropped')
    save_figure(hist_stretch(np.absolute(Srh_arr)), 'Range', 'Azimuth', 'RISAT-1_Srh_Cropped_stretched')
    
    save_figure(np.absolute(Srv_arr), 'Range', 'Azimuth', 'RISAT-1_Srv_Cropped')
    save_figure(hist_stretch(np.absolute(Srv_arr)), 'Range', 'Azimuth', 'RISAT-1_Srv_Cropped_stretched')
    
    stokes_vector=get_stokes_vector(Srh_arr,Srv_arr, window_size)
    
    save_figure(np.absolute(stokes_vector[:,:,0]), 'Range', 'Azimuth', 'q0_cropped')
    save_figure(hist_stretch(np.absolute(stokes_vector[:,:,0])), 'Range', 'Azimuth', 'q0_cropped_stretched')
    save_figure(np.absolute(stokes_vector[:,:,1]), 'Range', 'Azimuth', 'q1_cropped')
    save_figure(hist_stretch(np.absolute(stokes_vector[:,:,1])), 'Range', 'Azimuth', 'q1_cropped_stretched')
    save_figure(np.absolute(stokes_vector[:,:,2]), 'Range', 'Azimuth', 'q2_cropped')
    save_figure(hist_stretch(np.absolute(stokes_vector[:,:,2])), 'Range', 'Azimuth', 'q2_cropped_stretched')
    save_figure(np.absolute(stokes_vector[:,:,3]), 'Range', 'Azimuth', 'q3_cropped')
    save_figure(np.absolute(hist_stretch(stokes_vector[:,:,3])), 'Range', 'Azimuth', 'q3_cropped_stretched')
    
    DoP=degreeOfPolarization(stokes_vector)
    
    save_figure(np.absolute(DoP), 'Range', 'Azimuth', 'DoP_cropped')
    save_figure(hist_stretch(np.absolute(DoP)), 'Range', 'Azimuth', 'DoP_cropped_stretched')
    
    chi=ellipticity_angle(stokes_vector, DoP)
    save_figure(np.absolute(chi), 'Range', 'Azimuth', 'chi_cropped')
    save_figure(hist_stretch(np.absolute(chi)), 'Range', 'Azimuth', 'chi_cropped_stretched')
    
    
    hppr=hybrid_pol_power_ratio(Srh_arr,Srv_arr, window_size)
    save_figure(np.absolute(hppr), 'Range', 'Azimuth', 'HPPR_cropped')
    save_figure(hist_stretch(np.absolute(hppr)), 'Range', 'Azimuth', 'HPPR_cropped_stretched')
    
    
    corr_coef=correlation_coeff(Srh_arr,Srv_arr, window_size)
    save_figure(np.absolute(corr_coef), 'Range', 'Azimuth', 'Corr_coef_cropped')
    save_figure(hist_stretch(np.absolute(corr_coef)), 'Range', 'Azimuth', 'Corr_coef_cropped_stretched')
    
    std_phd=std_phase_diff(Srh_arr, Srv_arr, window_size)
    save_figure(np.absolute(std_phd), 'Range', 'Azimuth', 'std_phd_cropped')
    save_figure(hist_stretch(np.absolute(std_phd)), 'Range', 'Azimuth', 'std_phd_cropped_stretched')
    
    lambdas=eigen_values(stokes_vector)
    save_figure(np.absolute(lambdas[:,:,0]), 'Range', 'Azimuth', 'lambda1_cropped')
    save_figure(hist_stretch(np.absolute(lambdas[:,:,0])), 'Range', 'Azimuth', 'lambda1_cropped_stretched')
    save_figure(np.absolute(lambdas[:,:,1]), 'Range', 'Azimuth', 'lambda2_cropped')
    save_figure(hist_stretch(np.absolute(lambdas[:,:,1])), 'Range', 'Azimuth', 'lambda2_cropped_stretched')
    
    H=entropy(lambdas)
    save_figure(np.absolute(H), 'Range', 'Azimuth', 'Entropy_cropped')
    save_figure(hist_stretch(np.absolute(H)), 'Range', 'Azimuth', 'Entropy_cropped_stretched')
    
    
    con_coeff=conformity_coeff(Srh_arr, Srv_arr, window_size)
    save_figure(np.absolute(con_coeff), 'Range', 'Azimuth', 'Conformity_coeff_cropped')
    save_figure(hist_stretch(np.absolute(con_coeff)), 'Range', 'Azimuth', 'Conformity_coeff_cropped_stretched')
    
    cov_mat=get_covariance_matrix(Srh_arr, Srv_arr, window_size)
    det_arr=det_covariance_mat(cov_mat)
    save_figure(np.absolute(det_arr), 'Range', 'Azimuth', 'Det_Cov_Mat_cropped')
    save_figure(hist_stretch(np.absolute(det_arr)), 'Range', 'Azimuth', 'Det_Cov_Mat_cropped_stretched')
    #print(det_arr[0,0])
    #return(stokes_vector)
    #print(Srh_arr)
    #display(hist_stretch(np.absolute(det_arr)), 'Range', 'Azimuth', 'RISAT_S11')
    #display(hist_stretch(stokes_vector[:,:,3]), 'Range', 'Azimuth', 'RISAT-1 S11')
    

    
if __name__=='__main__':
    #s11=gdal.Open('s11.bin')
    #s11_gt=s11.GetGeoTransform()
    #arr_s11=s11.ReadAsArray()
    arr=img_to_array()
    
    arr_s11=arr[:,:,0]
    arr_s21=arr[:,:,1]
    
    extract_all_features(arr_s11, arr_s21,True,10)
    
    #a=np.absolute(arr_s21)
    #print(dimension_data(arr_s21))
    #print(s21_gt)
    #print(s21_proj)
    #save_figure(hist_stretch(a[6500:9000,5400:6800]), 'Range', 'Azimuth', 'RISAT-1 S11')
    #display(hist_stretch(a), 'Range', 'Azimuth', 'RISAT-1 S11')
    #clear_list(a)
    #clear_list(arr_s21)
    #clear_list(arr_s11)