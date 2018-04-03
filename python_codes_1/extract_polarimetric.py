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
import plotting
import read_binary
import os


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
    return cov_arr[:,:,0,0]/cov_arr[:,:,2,2]

def extract_coherency_arr(window_size, correction_switch, degree):
    elements=['T22','T33', 'T12_real', 'T12_imag', 'T13_real', 'T13_imag', 'T23_real', 'T23_imag']
    if (correction_switch):
        raster_arr=test_convolve(incidence_angle_corr.inci_correction('T3', 'T11', degree), kernal(window_size))
    else:
        raster_arr=test_convolve(read_Raster('T3', 'T11'), kernal(window_size))
    raster_stack=np.stack(np.array([raster_arr], dtype=np.complex64))
    iota=1j
    if (correction_switch):
        for i in elements:
            raster_arr=test_convolve(incidence_angle_corr.inci_correction('T3', i, degree),kernal(window_size))
            raster_stack=np.concatenate((raster_stack,[raster_arr]))
    else:
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
    return np.dstack((t11,t12,t13,np.conj(t12),t22,t23,np.conj(t13),np.conj(t23),t33)).reshape(new_shape[0],new_shape[1],3,3)





def extract_covariance_arr(window_size, correction_switch, degree):
    elements=['C22','C33', 'C12_real', 'C12_imag', 'C13_real', 'C13_imag', 'C23_real', 'C23_imag']
    if (correction_switch):
        raster_arr=test_convolve(incidence_angle_corr.inci_correction('C3', 'C11', degree), kernal(window_size))
    else:
        raster_arr=test_convolve(read_Raster('C3', 'C11'), kernal(window_size))
    #raster_arr=read_Raster('C3', 'C11')
    raster_stack=np.stack(np.array([raster_arr], dtype=np.complex64))
    iota=1j
    if (correction_switch):
        for i in elements:
            raster_arr=test_convolve(incidence_angle_corr.inci_correction('C3', i, degree),kernal(window_size))
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
    return np.dstack((c11,c12,c13,np.conj(c12),c22,np.conj(c23),np.conj(c13),np.conj(c23),c33)).reshape(new_shape[0],new_shape[1],3,3)

def determinant_cov(cov_arr):
    #cov_arr=extract_covariance_arr(window_size)
    shp=cov_arr.shape
    #return shp
    linear_cov_arr=np.reshape(cov_arr,(np.prod(shp[:2]),shp[2],shp[3]))
    det_arr=np.linalg.det(np.absolute(linear_cov_arr))
    return det_arr.reshape(shp[0],shp[1])

def determinant_cov_conj(cov_arr):
    #cov_arr=extract_covariance_arr(window_size)
    shp=cov_arr.shape
    #return shp
    linear_cov_arr=np.reshape(cov_arr,(np.prod(shp[:2]),shp[2],shp[3]))
    det_arr=np.linalg.det(linear_cov_arr)
    return det_arr.reshape(shp[0],shp[1])

def get_eigen_values(coh_matrix):
    w=LA.eigvals(coh_matrix)
    return w

def get_eigen_vectors(window_size, correction_switch, degree):
    coh_matrix=extract_coherency_arr(window_size, correction_switch, degree)
    w,v=np.linalg.eig(coh_matrix)
    return [w,v]
    

def eigen_raster_full(window_size, correction_switch, degree):
    arr=extract_coherency_arr(window_size, correction_switch, degree)
    #arr=extract_covariance_arr(window_size, False)
    eigen_arr=np.sort(np.absolute(get_eigen_values(arr)), axis=2) # this method does indeed give real eigen vaues of the hermitian covariance and coherency matrices
    #eigen_arr=np.sort(get_eigen_values(arr), axis=2)
    return eigen_arr

def eigen_raster(eigen_num, window_size, correction_switch, degree):
    #arr=extract_coherency_arr(window_size)
    eigen_arr=eigen_raster_full(window_size, correction_switch, degree)
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
    p0_log=np.log(p[0])/math.log(3)
    p1_log=np.log(p[1])/math.log(3)
    p2_log=np.log(p[2])/math.log(3)
    ent=-1*(p[0]*p0_log+p[1]*p1_log+p[2]*p2_log)
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
    
#def mean_alpha_angle(window_size, correction_switch, degree):
    #w,v=get_eigen_vectors(window_size, correction_switch, degree)
    #p=get_p_value(eigen_full)
    #eig_vec_mean = 
def mean_alpha_angle(eigen_full,window_size, correction_switch, degree ):
    p = get_p_value(eigen_full)
    w,v=get_eigen_vectors(window_size, correction_switch, degree)
    alpha_0 = np.arccos(np.absolute(np.mean(v[...,0,:], axis = 2)))
    alpha_1 = np.arccos(np.absolute(np.mean(v[...,1,:], axis = 2)))
    alpha_2 = np.arccos(np.absolute(np.mean(v[...,2,:], axis = 2)))
    mean_alpha = (p[0]*alpha_0 + p[1]*alpha_1+ p[2]*alpha_2) * 180/np.pi
    return mean_alpha
    

#def pass_filter():
    #arr=np.array([[-6-6j, -3-3j, 0-10j,  +3 -3j, 6-6j],[-20+0j, -10+0j, 0+ 0j, +10 +0j, 10+0j],[-6+6j, -3+3j, 0+10j,  +3 +3j, 6+6j]])
    #return arr

def co_pol_cross_product(cov_arr):
    #arr=extract_covariance_arr(window_size)
    return cov_arr[:,:,0,2]
    
def co_pol_diff(cov_arr):
    #cov_arr=extract_covariance_arr(window_size)
    return cov_arr[:,:,0,0]-cov_arr[:,:,2,2]

def co_pol_correlation(cov_arr):
    return cov_arr[...,1,1]/np.sqrt(cov_arr[...,0,0]*cov_arr[...,2,2])

def conformity_coeff(cov_arr):
    return 2.0*(np.real(cov_arr[...,2,2])-cov_arr[...,1,1]/2)/(cov_arr[...,0,0]+cov_arr[...,1,1]+cov_arr[...,2,2])

def sd_co_pol_phase_diff(directory):
    pwd=os.getcwd()
    os.chdir(directory)
    mlc_row_looks=12
    mlc_col_looks=3
    scan_lines=86417
    scan_pix=9900
    mlc_cropping_list=[521,1545,4049,5233]
    slc_cropping_list=[mlc_cropping_list[0]*mlc_col_looks,\
        mlc_cropping_list[1]*mlc_col_looks,\
            mlc_cropping_list[2]*mlc_row_looks,\
                mlc_cropping_list[3]*mlc_row_looks]
    file_name_VV="norway_00709_15092_000_150610_L090VV_CX_02.slc"
    file_name_HV="norway_00709_15092_000_150610_L090HV_CX_02.slc"
    file_name_HH="norway_00709_15092_000_150610_L090HH_CX_02.slc"
    
    slc_oil = np.load(directory+'/S_oil_SLC.npy')
    #pos=stretch_mask_to_SLC(mlc_row_looks, mlc_col_looks,0)
    #slc_VV=read_binary.read_SLC(file_name_VV, scan_lines, scan_pix, slc_cropping_list, False)
    #slc_HH=read_binary.read_SLC(file_name_HH, scan_lines, scan_pix, slc_cropping_list, False)
    #print(slc_oil.shape)
    slc_VV = slc_oil[...,2]
    slc_HH = slc_oil[...,0]
    phase_VV=np.angle(slc_VV)
    phase_HH=np.angle(slc_HH)
    phase_diff=phase_HH-phase_VV
    
    #sd_phase_diff=np.sqrt(np.var(slc_HH-slc_VV))
    #diff_avg=read_binary.multilooking(3,12,phase_diff*180/np.pi)
    var_arr = np.dstack((phase_diff**2,phase_diff))
    var_arr_avg = read_binary.multilook_C3(3,12, var_arr)
    #diff_sqr_avg=read_binary.multilooking(3,12,phase_diff**2)
    #diff_avg_sqr=read_binary.multilooking(3,12,phase_diff)**2
    os.chdir(pwd)
    del(slc_VV,slc_HH, phase_VV, phase_HH, phase_diff)
    #return diff_avg
    #return np.sqrt(diff_sqr_avg+diff_avg_sqr)
    return np.sqrt(var_arr_avg[...,0]+var_arr_avg[...,1])

def save_feature(name, output_array, window_size):
    reproject.save_tiff_image(name, output_array, window_size)
    
    
    
if __name__=='__main__':
    
    os.chdir('../North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01/MLC_Python_New/norway_00709_15092_000_150610_L090_CX_01_mlc')
    
    #arr=np.array([[0,9,2,1,4],[1,2,4,5,3],[1,2,3,2,0],[1,6,0,2,1],[1,60,0,2,1]])
    #Ivv=incidence_angle_corr.inci_correction('C3', 'C33')
    #Ihh=incidence_angle_corr.inci_correction('C3', 'C11')
    
    #================Extract C3, T3;original and inc corrected
    cov_arr=extract_covariance_arr(9, False,3)
    cov_arr_corr=extract_covariance_arr(9, True, 1)
    
    #coh_arr=extract_coherency_arr(9)
    #print(cov_arr[...,2,2])
    #print(cov_arr[...,1,2])
    #print(np.absolute(cov_arr[...,1,2]))
    #print(cov_arr[0,0,...], sep='\n')
    #print(coh_arr[0,0,...], sep='\n')
    #print(np.mean(np.mean(cov_arr,axis=1), axis=0))
    
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
    #eigen_full=eigen_raster_full(9)
    #arr_lamb1=eigen_full[:,:,2]
    #arr_lamb2=eigen_full[:,:,1]
    #arr_lamb3=eigen_full[:,:,0]
    #arr_ent=entropy(9)

    #arr_pol_frac=pol_fraction(eigen_full)
    #arr_anisotropy=anisotropy(9)
    #arr_co_cross=co_pol_cross_product(9)
    #arr_co_pol_diff=co_pol_diff(cov_arr)
    #print(arr_co_cross.shape)
    #print(arr_pol_frac)
    #co_pol_corr_arr=co_pol_correlation(cov_arr_corr)
    #imgplot=plt.imshow(np.imag(arr_co_cross), cmap='gray')
    #save_feature('abs_of_co_pol_diff_window_size_25', np.absolute(arr_co_pol_diff), 15)
    #arr_conform_coeff=conformity_coeff(cov_arr)
    #plt.imshow(np.absolute(arr_conform_coeff), cmap='gray')
    #plt.colorbar()
    #plt.show()
    
    #plt.subplots(4,1)
    
    #plt.subplot(4,1,1)
    #imgplot=plt.imshow(np.absolute(arr_lamb1), cmap='gray')
    #plt.colorbar()
    #plt.subplot(4,1,2)
    #imgplot=plt.imshow(np.absolute(arr_lamb2), cmap='gray')
    #plt.colorbar()
    #plt.subplot(4,1,3)
    #imgplot=plt.imshow(np.absolute(arr_lamb3), cmap='gray')
    #plt.colorbar()
    #plt.subplot(4,1,4)
    #imgplot=plt.imshow(np.absolute(arr_pol_frac), cmap='gray')
    #plt.colorbar()
    #plt.show()
    #=============DET============
    #arr_det_cov=determinant_cov(cov_arr)
    #arr_det_cov_corr=determinant_cov(cov_arr_corr)
    #arr_det_cov=np.absolute(determinant_cov_conj(cov_arr))
    #arr_det_cov_corr=np.absolute(determinant_cov_conj(cov_arr_corr))
    #print(eigen_raster_full(9))
    #plt.subplots(1,2)
    #plt.subplot(1,2,1)
    #imgplot=plt.imshow(arr_det_cov, cmap='gray')
    #plt.title('Determinant of Covariance matrix (conj_original)')
    #plt.colorbar()
    #plt.subplot(1,2,2)
    #imgplot1=plt.imshow(arr_det_cov_corr, cmap='gray')
    #plt.title('Determinant of Covariance matrix')
    #plt.colorbar()
    #plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Features/Polarimetric_Features/Inc_corr_applied/Det_cov_conj.tiff', dpi=300)
    #plt.show()
    #==============DET============
    
    #=============PLOT_TRANSECT=========
    #plotting.plot_transect_two_arr(cov_arr, cov_arr_corr, [500], ['row500, original_C12','row100, inc_corrected_C12'], [2,2])
    #plotting.plot_transect_two_arr(cov_arr, cov_arr_corr, 100, ['row100, original','row100, inc_corrected'], [2,2])
    '''
    
    #=============Phase_diff_sd=========
    slc_dir='/home/anurag/Documents/MScProject/SAR/OilSpill/North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_02'
    sd_phase_diff=sd_co_pol_phase_diff(slc_dir)
    #print(sd_phase_diff.dtype)
    #print(sd_phase_diff)
    plt.subplot(111)
    plt.imshow(np.real(sd_phase_diff), cmap='gray') 
    plt.colorbar()
    
    #plt.imshow(np.absolute(sd_phase_diff))
    #plt.subplot(122)
    #plt.hist(sd_phase_diff.flatten(), bins=300, rwidth=0.5,histtype='step')
    
    plt.title('Avg phase difference')
    #plt.colorbar()
    plt.show()
    '''
    
    
    
    