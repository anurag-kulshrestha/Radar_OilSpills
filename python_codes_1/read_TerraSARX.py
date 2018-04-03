import os
import struct
import numpy as np  
from matplotlib import pyplot as plt
#import reproject
from scipy import ndimage
import numpy.ma as ma
from numpy import pi
from scipy import signal
from numpy import linalg as LA

#import plotting
import read_binary
from osgeo import gdal
import extract_polarimetric
import matplotlib
#To read terrasar-x data

def read_Raster(directory):
    os.chdir(directory)
    ihh=gdal.Open('i_HH.img')
    ihh_arr=ihh.ReadAsArray()
    qhh=gdal.Open('q_HH.img')
    qhh_arr=qhh.ReadAsArray()
    ivv=gdal.Open('i_VV.img')
    ivv_arr=ivv.ReadAsArray()
    qvv=gdal.Open('q_VV.img')
    qvv_arr=qvv.ReadAsArray()
    
    iota=1j
    
    amp_HH=ihh_arr+iota*qhh_arr
    amp_VV=ivv_arr+iota*qvv_arr
    return np.dstack((amp_HH,amp_VV))

def multilook_TSX(directory,window_size_y,window_size_x):
    TSX_amp=read_Raster(directory)
    amp_HH=TSX_amp[...,0]
    #print(amp_HH.shape)
    amp_VV=TSX_amp[...,1]
    kern=kernal(window_size_y,window_size_x)
    amp_HH_ML=test_convolve(np.absolute(amp_HH), kern)
    amp_VV_ML=test_convolve(np.absolute(amp_VV), kern)
    return np.dstack((amp_HH_ML,amp_VV_ML))

def multilook_TSX_1(TSX_amp,window_size_y,window_size_x):
    #TSX_amp=read_Raster(directory)
    return read_binary.multilook_C3(window_size_x, window_size_y, TSX_amp, leaping_win=True)




def test_convolve(arr, kernal):
    grad = signal.convolve2d(arr, kernal, boundary='symm', mode='valid')
    return grad

def kernal(window_size_y, window_size_x):
    k=np.ones(window_size_x*window_size_y).reshape(window_size_y, window_size_x)
    normalize_k=k/(window_size_x * window_size_y)
    return normalize_k

def get_cov_matrix(TSX_amp, window_size_x, window_size_y):
    amp_HH=TSX_amp[...,0]
    #print(amp_HH.shape)
    amp_VV=TSX_amp[...,1]
    
    c11 = amp_HH*np.conj(amp_HH)
    c12 = amp_HH*np.conj(amp_VV)
    c21 = amp_VV*np.conj(amp_HH)
    c22 = amp_VV*np.conj(amp_VV)
    #print(TSX_amp.shape)
    cov_arr = np.dstack((c11,c12,c21,c22)).reshape(*TSX_amp.shape[:2],2,2)
    cov_arr = read_binary.multilook_C3(window_size_x, window_size_y, cov_arr, leaping_win=True)
    return cov_arr

def get_coh_matrix(cov_arr):
    
    C11=cov_arr[...,0,0]
    C12=cov_arr[...,0,1]
    C21=cov_arr[...,1,0]
    C22=cov_arr[...,1,1]
    
    cov_arr[...,0,0] = (C11 + C22 + C12 + C21)/2
    cov_arr[...,0,1] = (C11 - C12 + C21 - C22)/2
    cov_arr[...,1,1] = (C11 + C22 - C12 - C21)/2
    cov_arr[...,1,0] = np.conj(cov_arr[...,0,1])
    
    return cov_arr

def get_eigen_values_vectors(arr):
    w,v=LA.eig(arr)
    return [w,v]

def eigen_raster_full(coh_arr):
    eigen_arr=np.sort(np.absolute(get_eigen_values_vectors(coh_arr)[0]), axis=2) # this method does indeed give real eigen vaues of the hermitian covariance and coherency matrices
    #eigen_arr=np.sort(get_eigen_values(arr), axis=2)
    return eigen_arr

def get_p_value(eigen_full):
    #eigen_full=eigen_raster_full(window_size) 
    lamb1=eigen_full[:,:,1]
    lamb2=eigen_full[:,:,0]
    
    lamb_sum=np.sum(eigen_full, axis=2)
    #a=eigen_full
    #p=np.divide(eigen_full, lamb_sum) #We can include a axis parameter for np.divide as well
    #return a
    p1=lamb1/lamb_sum
    p2=lamb2/lamb_sum
    #p3=lamb3/lamb_sum
    #return np.dstack((p1,p2,p3))
    return [p1,p2]

def entropy(eigen_full):
    p=get_p_value(eigen_full)
    p0_log=np.log(p[0])/np.log(2)
    p1_log=np.log(p[1])/np.log(2)
    #p2_log=np.log(p[2])/math.log(3)
    ent=-1*(p[0]*p0_log+p[1]*p1_log)
    return ent

def mean_alpha_angle(v, eigen_full):
    p = get_p_value(eigen_full)
    alpha_0 = np.arccos(np.absolute(np.mean(v[...,0,:], axis = 2)))
    alpha_1 = np.arccos(np.absolute(np.mean(v[...,1,:], axis = 2)))
    mean_alpha = (p[0]*alpha_0 + p[1]*alpha_1) * 180/np.pi
    return mean_alpha

def sd_co_pol_phase_diff(TSX_amp, window_size_x, window_size_y):
    slc_VV = TSX_amp[...,1]
    slc_HH = TSX_amp[...,0]
    phase_VV=np.angle(slc_VV)
    phase_HH=np.angle(slc_HH)
    phase_diff=phase_HH-phase_VV
    
    var_arr = np.dstack((phase_diff**2,phase_diff))
    var_arr_avg = read_binary.multilook_C3(window_size_x, window_size_y, var_arr)
    
    del(slc_VV,slc_HH, phase_VV, phase_HH, phase_diff)
    
    return np.sqrt(var_arr_avg[...,0]+var_arr_avg[...,1])


if __name__=='__main__':
    
    directory='../TerraSAR_X/dims_op_oc_dfd2_567023152_2/TSX-1.SAR.L1B/TSX1_SAR__SSC______SM_D_SRA_20150610T062401_20150610T062409/sybset/subset_0_of_TSX1_SAR_SSC_SM_D_SRA_20150610T062401_20150610T062409_Cal.data'
    cov_win_x = 9
    cov_win_y = 9
    
    
    #=================Exrtacting feature vectorr================
    TSX_amp=read_Raster(directory)
    #print(kernal(9))
    
    #TSX_amp=multilook_TSX(directory,25, 25)
    #TSX_amp=multilook_TSX_1(TSX_amp,25, 25)
    #print(TSX_amp)
    
    #plt.imshow(np.clip(10*np.log10(np.absolute(TSX_amp[...,1])),-25,0), cmap = 'gray')
    #plt.colorbar()
    #plt.show()
    
    #=================Exrtacting 2x2 cov_arr and coh_ matrix================
    cov_arr = get_cov_matrix(TSX_amp, cov_win_x, cov_win_x)
    coh_arr = get_coh_matrix(cov_arr)
    matplotlib.rcParams.update({'font.size': 5})
    #================I_hh and I_vv==============
    plt.subplot(4,4,1)
    plt.imshow(10*np.log10(np.absolute(coh_arr[...,0,0])), cmap = 'gray')
    plt.colorbar(label='dB')
    plt.title('$I_{HH}$')
    plt.ylabel('Azimuth')
    
    plt.subplot(4,4,2)
    plt.imshow(10*np.log10(np.absolute(coh_arr[...,1,1])), cmap = 'gray')
    plt.colorbar(label='dB')
    plt.title('$I_{VV}$')

    #=================determinant_cov================
    plt.subplot(4,4,3)
    det_cov = extract_polarimetric.determinant_cov(cov_arr)
    #print(det_cov)
    plt.imshow(10*np.log10(np.absolute(det_cov)), cmap = 'gray')
    plt.colorbar(label='dB')
    plt.title('$det(C_{DP})$')
    
    #=================Exrtacting eigen_values coh_matric================
    
    eigen_full = eigen_raster_full(coh_arr)
    #print(eigen_full)
    plt.subplot(4,4,4)
    plt.imshow(10*np.log10(np.absolute(eigen_full[...,0])), cmap = 'gray')
    plt.colorbar(label='dB')
    plt.title('$\lambda_{2}$')
    
    plt.subplot(4,4,5)
    plt.imshow(10*np.log10(np.absolute(eigen_full[...,1])), cmap = 'gray')
    plt.colorbar(label='dB')
    plt.title('$\lambda_{1}$')
    plt.ylabel('Azimuth')
    #=================Exrtacting eigen_values coh_matric================
    plt.subplot(4,4,6)
    ent = entropy(eigen_full)
    plt.imshow(10*np.log10(np.absolute(ent)), cmap = 'gray')
    plt.colorbar(label='dB')
    plt.title('H')
    
    #===================alpha angle================
    v = get_eigen_values_vectors(coh_arr)[1]
    plt.subplot(4,4,7)
    mean_alpha = mean_alpha_angle(v, eigen_full)
    
    plt.imshow(mean_alpha, cmap = 'gray')
    plt.colorbar(label='degrees')
    plt.title(r'mean $\alpha$')
    #===================Anisotropy================
    
    plt.subplot(4,4,8)
    A = extract_polarimetric.anisotropy(eigen_full)
    plt.imshow(A, cmap = 'gray')
    plt.colorbar()
    plt.title('A')
    
    #====================Geometric Intensity============
    
    plt.subplot(4,4,9)
    GI = np.sqrt(extract_polarimetric.determinant_cov(coh_arr))
    plt.imshow(GI, cmap = 'gray')
    plt.colorbar()
    plt.title('GI')
    plt.ylabel('Azimuth')
    #=============Co_pol_power_ratio===============
    plt.subplot(4,4,10)
    Co_pol_power_ratio = cov_arr[...,0,0]/cov_arr[:,:,1,1]
    plt.imshow(np.absolute(Co_pol_power_ratio), cmap = 'gray')
    plt.colorbar()
    plt.title('$\gamma_{CO}$')
    #plt.ylabel('Azimuth')
    #=============Co_pol_phase diff===============
    plt.subplot(4,4,11)
    sd_phase_diff = sd_co_pol_phase_diff(TSX_amp, cov_win_x, cov_win_y)
    plt.imshow(np.absolute(sd_phase_diff), cmap = 'gray')
    plt.colorbar()
    plt.title('$\phi_{CO}$')
    
    #=============Co_pol_corr_coeff===============
    plt.subplot(4,4,12)
    Co_pol_corr_coeff = np.absolute(cov_arr[...,0,1]/np.sqrt(cov_arr[...,0,0]*cov_arr[...,1,1]))
    plt.imshow(Co_pol_corr_coeff, cmap = 'gray')
    plt.colorbar()
    plt.title('co-pol correlation')
    
    #=============Co_pol_cross_product===============
    
    plt.subplot(4,4,13)
    r_co = np.real(cov_arr[...,0,1])
    plt.imshow(10*np.log10(Co_pol_corr_coeff), cmap = 'gray')
    plt.colorbar()
    plt.title('r_{CO}')
    plt.ylabel('Azimuth')
    plt.xlabel('Range')
    #=============PD===============
    
    plt.subplot(4,4,14)
    PD = np.absolute(cov_arr[...,1,1]  - cov_arr[...,0,0])
    plt.imshow(10*np.log10(PD), cmap = 'gray')
    plt.colorbar(label='dB')
    plt.title('PD')
    #plt.ylabel('Azimuth')
    plt.xlabel('Range')
    #=============span===============
    plt.subplot(4,4,15)
    span = np.real(cov_arr[...,1,1]  + cov_arr[...,0,0])
    plt.imshow(10*np.log10(np.real(span)), cmap = 'gray')
    plt.colorbar(label='dB')
    plt.title('Span')
    plt.xlabel('Range')
    
    
    plt.tight_layout()
    
    #plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Features/Polarimetric_Features/RS_X_all.tiff', dpi=300, box_inches = 'tight',papertype = 'a4',orientation = 'portrait')
    
    plt.show()
    
    #mean_alpha =     
    '''
    amp_HH=TSX_amp[...,0]
    amp_VV=TSX_amp[...,1]
    I_HH=np.absolute(amp_HH)**2
    I_VV=np.absolute(amp_VV)**2
    
    
    plt.imshow(10*np.log10(I_HH), cmap='gray')
    #plt.imshow(10*np.log10(np.absolute(amp_VV)), cmap='gray')
    plt.colorbar()
    plt.show()
    '''