from osgeo import gdal, osr,ogr
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import math
#import incidence_angle_corr
from math import pi
from scipy import signal
from scipy import misc
import subprocess
import read_binary
#import reproject
#import plotting
#print(os.getcwd())

#dat_rep='/home/anurag/Documents/MScProject/SAR/OilSpill/RISAT-1/RI1_SAR_L1SLC_FRS1_CR_20150610T071918_20150610T071923_17197_1515551004/scene_RV'

metadata_filename='BAND_META.txt'


#def boxcar_filter(window_size_y, window_size_x):
    


def crop_from_binary():
    d=metadata_dict(metadata_filename)
    os.chdir('scene_RV')
    file_name='dat_01.001'
    rows=int(d['NoScans'])
    cols=int(d['NoPixels'])
    cropping_List=[5400,6800,6500,8850]
    a=read_binary.read_SLC(file_name, rows, cols, cropping_List, False)
    print(a)

def img_to_array():
    #os.chdir(directory)
    s21=gdal.Open('s21.bin') #Srv
    #s21=gdal.Open('temp/final_s21.img')
    s21_gt=s21.GetGeoTransform()
    s21_proj=s21.GetProjectionRef()
    arr_s21=s21.ReadAsArray()
    
    s11=gdal.Open('s11.bin') #Srh
    s11_gt=s11.GetGeoTransform()
    s11_proj=s11.GetProjectionRef()
    arr_s11=s11.ReadAsArray()
    #print(s21_gt, s21_proj)
    #print(arr_s21.shape)
    #print(arr_s21)
    return np.dstack((arr_s11, arr_s21))
    #return arr_s21

def get_inc_angle_array(file_loc):
    inc=gdal.Open(file_loc)
    return inc.ReadAsArray()

def metadata_dict(metadata_filename):
    meta = open(metadata_filename, 'r') 
    #print(meta.readline())
    meta_dict={}
    for i in meta.readlines():
        j=i[:-1].split('=')
        #meta_dict.update({j[0],j[1]})
        meta_dict[j[0]]=j[1]
    #print(meta_dict['ProdURLat'])
    return meta_dict

def slant_to_ground_range():
    d=metadata_dict(metadata_filename)
    look_ang=float(d['IncidenceAngle'])
    azi_res=float(d['InputResolutionAlong'])
    slant_res=float(d['InputResolutionAcross'])
    grd_res=slant_res/np.sin(look_ang/180*pi)
    return (azi_res, grd_res)

#def 

def get_corners_UTM(metadata_filename):
    d=metadata_dict(metadata_filename)
    gcp_UL= float(d['ProdULLat']),float(d['ProdULLon'])
    gcp_LL= float(d['ProdLLLat']),float(d['ProdLLLon'])
    gcp_UR= float(d['ProdURLat']),float(d['ProdURLon'])
    gcp_LR= float(d['ProdLRLat']),float(d['ProdLRLon'])
    #print(gcp_UL)
    #print(gcp_UR)
    
    gcp_UL_utm=reproject.convert_proj_sys(gcp_UL[1],gcp_UL[0], 4326, 32631)
    gcp_UR_utm=reproject.convert_proj_sys(gcp_UR[1],gcp_UR[0], 4326, 32631)
    gcp_LR_utm=reproject.convert_proj_sys(gcp_LR[1],gcp_LR[0], 4326, 32631)
    gcp_LL_utm=reproject.convert_proj_sys(gcp_LL[1],gcp_LL[0], 4326, 32631)
    return (gcp_UL_utm,gcp_UR_utm,gcp_LR_utm,gcp_LL_utm)


def calc_affine_rotation(metadata_filename):
    coord=get_corners_UTM(metadata_filename)
    LL=coord[3]
    LR=coord[2]
    perp=LR[1]-LL[1]
    base=LR[0]-LL[0]
    return np.arctan(perp/base)

def get_cell_size():
    d=metadata_dict(metadata_filename)
    rows=int(d['NoScans'])
    cols=int(d['NoPixels'])
    coord=get_corners_UTM(metadata_filename)
    UL=coord[0]
    LL=coord[3]
    UR=coord[1]
    dist_Y=np.sqrt((UL[0]-LL[0])**2+(UL[1]-LL[1])**2)
    dist_X=np.sqrt((UL[0]-LL[0])**2+(UR[1]-UR[1])**2)
    #return (np.floor(dist_X/cols),np.floor(dist_Y/rows))
    return (dist_X/cols,dist_Y/rows)

def set_geoTransform():
    coord=get_corners_UTM(metadata_filename)
    rot_angle=calc_affine_rotation(metadata_filename)+pi
    #cell_size=get_cell_size()
    cell_size=slant_to_ground_range()[::-1]
    #print(coord(metadata_filename)[0])
    gt0,gt3=coord[0]
    print(cell_size)
    gt1,gt5=(np.cos(rot_angle)*cell_size[0], np.cos(rot_angle)*cell_size[1])
    gt2,gt4=(np.sin(rot_angle)*cell_size[0]*-1, np.sin(rot_angle)*cell_size[1])
    return [gt0,gt1,gt2,gt3,gt4,gt5]

def projection(zone, is_North):
    proj = osr.SpatialReference()
    proj.SetUTM(zone, is_North)
    return proj.ExportToWkt()

def reproject_RISAT_SLC(arr):
    
    zone=31
    is_North=True
    newname='georef_RISAT1_SLC_test'
    d=metadata_dict(metadata_filename)
    newRasterYSize=int(d['NoScans'])
    newRasterXSize=int(d['NoPixels'])
    bands=1
    
    
    gt=set_geoTransform()
    proj=projection(zone, is_North)
    
    
    
    os.chdir('temp')
    #print(arr.shape)
    reproject.reproject_image_complex(newname, newRasterXSize, newRasterYSize, bands, arr, proj, gt)


def reproject_RISAT_sigma_nought(arr):
    zone=31
    is_North=True
    newname='georef_RISAT1_sigma_nought_srv_test_1'
    d=metadata_dict(metadata_filename)
    newRasterYSize=int(d['NoScans'])
    newRasterXSize=int(d['NoPixels'])
    bands=1
    gt=set_geoTransform()
    proj=projection(zone, is_North)
    os.chdir('temp')
    #print(arr.shape)
    #reproject_image(newname, newRasterXSize, newRasterYSize, bands, output_array, projection, geotransform)
    reproject.reproject_image(newname, newRasterXSize, newRasterYSize, bands, arr, proj, gt)


def read_reprojected_SLC():
    s11=gdal.Open('temp/georef_RISAT1_SLC_test.tif') #Srh
    s11_gt=s11.GetGeoTransform()
    s11_proj=s11.GetProjectionRef()
    arr_s11=s11.ReadAsArray()
    return arr_s11




def geo_reference_using_corner(output_file, metadata_filename, dim):
    d=metadata_dict(metadata_filename)
    rows=str(dim[0])
    cols=str(dim[1])
    #gcp_UL_lat= UL[0]
    gcp_UL= '0.0 0.0 '+d['ProdULLat']+' '+d['ProdULLon']
    gcp_LL=  rows+' '+'0.0 '+d['ProdLLLat']+' '+d['ProdLLLon']
    gcp_UR=  '0.0 '+cols+' '+d['ProdURLat']+' '+d['ProdURLon']
    gcp_LR=  rows+' '+cols+' '+d['ProdLRLat']+' '+d['ProdLRLon']
    print((gcp_UL,gcp_LL,gcp_UR,gcp_LR))
    cmd='gdal_translate -gcp '+gcp_UL+' -gcp '+gcp_LL+' -gcp '+gcp_UR+' -gcp '+gcp_LR+'  "s21.bin" "temp/s21.img"'
    
    cmd_1='gdalwarp -r near -tps -co COMPRESS=NONE -dstalpha "temp/s21.img" "temp/final_s21.img"'
    
    os.system(cmd_1)

def beta_nought(arr, calib_const_beta):
    return calib_const_beta*np.absolute(arr)**2

def convert_to_sigma_nought(arr, calib_const_sigma, inc_ang_arr):
    DN=np.absolute(arr)
    inc_ang_arr=inc_ang_arr*pi/180
    mid=inc_ang_arr.shape
    mid_row=np.floor(mid[0]/2)
    mid_col=np.floor(mid[1]/2)
    #print([mid_row, mid_col] )
    i_ang_const=inc_ang_arr[int(mid_row), int(mid_col)]
    sigma=20*np.log10(DN)-calib_const_sigma+10*np.log10(np.sin(inc_ang_arr)/np.sin(i_ang_const))
    #plt.imshow()
    #sigma = 10*np.log10(DN)
    return sigma
    

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
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Features/HP_RISAT1_features/naye_Features/'+title+'.tiff', dpi=300, bbox_inches='tight')
    
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
    #new_arr=np.floor(255*(new_arr-per_min)/(per_max-per_min))
    new_arr=(new_arr-per_min)/(per_max-per_min)
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

def kernal_1(window_size_y, window_size_x):
    k=np.ones(window_size_x*window_size_y).reshape(window_size_y, window_size_x)
    normalize_k=k/(window_size_x * window_size_y)
    return normalize_k

def oil_subset(arr):
    #return arr[6500:9000,5400:6800]
    return arr[6500:8850,5400:6800]

def oil_subset_1(arr):
    #return arr[6500:9000,5400:6800]
    return arr[5000:10000,4000:9000]

    
def averaging_arr(arr, window_size):
    kern=kernal(window_size)
    return test_convolve(arr, kern)

def averaging_arr_1(arr, window_size_y,window_size_x):
    kern=kernal_1(window_size_y,window_size_x)
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

def extract_all_features_1(Srh_arr, Srv_arr, only_subset, window_size):
    if(only_subset==True):
        Srh_arr=oil_subset(Srh_arr)
        Srv_arr=oil_subset(Srv_arr)
    #if(do_average==True)
    #Srh_arr=averaging_arr(Srh_arr, window_size)
    #Srv_arr=averaging_arr(Srv_arr, window_size)
    matplotlib.rcParams.update({'font.size': 5})
    
    
    plt.subplot(4,4,1)
    plt.imshow(10*np.log10(np.absolute(Srh_arr*np.conj(Srh_arr))), cmap='gray')
    plt.ylabel('Azimuth')
    plt.colorbar(label = 'dB')
    plt.title('$I_{RH}$')
    
    plt.subplot(4,4,2)
    plt.imshow(10*np.log10(np.absolute(Srv_arr*np.conj(Srv_arr))), cmap='gray')
    plt.colorbar(label = 'dB')
    plt.title('$I_{RV}$')
    
    
    stokes_vector=get_stokes_vector(Srh_arr,Srv_arr, window_size)
    
    plt.subplot(4,4,3)
    plt.imshow(10*np.log10(np.absolute(stokes_vector[:,:,0])), cmap='gray')
    plt.colorbar(label = 'dB')
    plt.title('$S_{0}$')
    
    plt.subplot(4,4,4)
    plt.imshow(10*np.log10(np.absolute(stokes_vector[:,:,1])), cmap='gray')
    plt.colorbar(label = 'dB')
    plt.title('$S_{1}$')
    
    plt.subplot(4,4,5)
    plt.imshow(10*np.log10(np.absolute(stokes_vector[:,:,2])), cmap='gray')
    plt.colorbar(label = 'dB')
    plt.title('$S_{2}$')
    plt.ylabel('Azimuth')
    
    plt.subplot(4,4,6)
    plt.imshow(10*np.log10(np.absolute(stokes_vector[:,:,3])), cmap='gray')
    plt.colorbar(label = 'dB')
    plt.title('$S_{3}$')
    
    plt.subplot(4,4,7)
    DoP=degreeOfPolarization(stokes_vector)
    plt.imshow(DoP, cmap='gray')
    plt.colorbar()
    plt.title('DoP')
    
    plt.subplot(4,4,8)
    chi=ellipticity_angle(stokes_vector, DoP)
    plt.imshow(chi, cmap='gray')
    plt.colorbar()
    plt.title('Ellipticity angle')
    
    plt.subplot(4,4,9)
    hppr=hybrid_pol_power_ratio(Srh_arr,Srv_arr, window_size)
    plt.imshow(hppr, cmap='gray')
    plt.colorbar()
    plt.title('Power Ratio')
    plt.ylabel('Azimuth')
    
    plt.subplot(4,4,10)
    corr_coef=correlation_coeff(Srh_arr,Srv_arr, window_size)
    plt.imshow(corr_coef, cmap='gray')
    plt.colorbar()
    plt.title('Correlation coefficient')
    
    plt.subplot(4,4,11)
    lambdas=eigen_values(stokes_vector)
    plt.imshow(lambdas[:,:,0], cmap='gray')
    plt.colorbar()
    plt.title('$\lambda_{1}$')
    
    plt.subplot(4,4,12)
    #lambdas=eigen_values(stokes_vector)
    plt.imshow(lambdas[:,:,1], cmap='gray')
    plt.colorbar()
    plt.title('$\lambda_{2}$')

    plt.subplot(4,4,13)
    H=entropy(lambdas)
    plt.imshow(H, cmap='gray')
    plt.colorbar()
    plt.title('H')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')

    plt.subplot(4,4,14)
    con_coeff=conformity_coeff(Srh_arr, Srv_arr, window_size)
    plt.imshow(con_coeff, cmap='gray')
    plt.colorbar()
    plt.title('Conformity Coefficient')
    plt.xlabel('Range')
    
    plt.subplot(4,4,15)
    cov_mat=get_covariance_matrix(Srh_arr, Srv_arr, window_size)
    det_arr=det_covariance_mat(cov_mat)
    plt.imshow(10*np.log10(np.absolute(det_arr)), cmap='gray')
    plt.colorbar()
    plt.title('det($C_{HP}$)')
    plt.xlabel('Range')
    
    plt.tight_layout()
    
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Features/HP_RISAT1_features/all_features.tiff', dpi=300, papertype='a4', bbox_inches='tight')
    
    plt.show()
    
    


def extract_all_features(Srh_arr, Srv_arr, only_subset, window_size):
    if(only_subset==True):
        Srh_arr=oil_subset(Srh_arr)
        Srv_arr=oil_subset(Srv_arr)
    #if(do_average==True)
    #Srh_arr=averaging_arr(Srh_arr, window_size)
    #Srv_arr=averaging_arr(Srv_arr, window_size)
    save_figure(np.absolute(Srh_arr), 'Range', 'Azimuth', 'RISAT-1_Srh_Cropped')
    save_figure(10*np.log10(np.absolute(Srh_arr)), 'Range', 'Azimuth', 'RISAT-1_Srh_Cropped_stretched')
    
    save_figure(np.absolute(Srv_arr), 'Range', 'Azimuth', 'RISAT-1_Srv_Cropped')
    save_figure(10*np.log10(np.absolute(Srv_arr)), 'Range', 'Azimuth', 'RISAT-1_Srv_Cropped_stretched')
    
    stokes_vector=get_stokes_vector(Srh_arr,Srv_arr, window_size)
    
    save_figure(np.absolute(stokes_vector[:,:,0]), 'Range', 'Azimuth', 'q0_cropped')
    save_figure(10*np.log10(np.absolute(stokes_vector[:,:,0])), 'Range', 'Azimuth', 'q0_cropped_stretched')
    save_figure(np.absolute(stokes_vector[:,:,1]), 'Range', 'Azimuth', 'q1_cropped')
    save_figure(10*np.log10(np.absolute(stokes_vector[:,:,1])), 'Range', 'Azimuth', 'q1_cropped_stretched')
    save_figure(np.absolute(stokes_vector[:,:,2]), 'Range', 'Azimuth', 'q2_cropped')
    save_figure(10*np.log10(np.absolute(stokes_vector[:,:,2])), 'Range', 'Azimuth', 'q2_cropped_stretched')
    save_figure(np.absolute(stokes_vector[:,:,3]), 'Range', 'Azimuth', 'q3_cropped')
    save_figure(10*np.log10(np.absolute(stokes_vector[:,:,3])), 'Range', 'Azimuth', 'q3_cropped_stretched')
    
    DoP=degreeOfPolarization(stokes_vector)
    
    save_figure(np.absolute(DoP), 'Range', 'Azimuth', 'DoP_cropped')
    #save_figure(hist_stretch(np.absolute(DoP)), 'Range', 'Azimuth', 'DoP_cropped_stretched')
    
    chi=ellipticity_angle(stokes_vector, DoP)
    save_figure(np.absolute(chi), 'Range', 'Azimuth', 'chi_cropped')
    #save_figure(hist_stretch(np.absolute(chi)), 'Range', 'Azimuth', 'chi_cropped_stretched')
    
    
    hppr=hybrid_pol_power_ratio(Srh_arr,Srv_arr, window_size)
    save_figure(np.absolute(hppr), 'Range', 'Azimuth', 'HPPR_cropped')
    #save_figure(hist_stretch(np.absolute(hppr)), 'Range', 'Azimuth', 'HPPR_cropped_stretched')
    
    
    corr_coef=correlation_coeff(Srh_arr,Srv_arr, window_size)
    save_figure(np.absolute(corr_coef), 'Range', 'Azimuth', 'Corr_coef_cropped')
    #save_figure(hist_stretch(np.absolute(corr_coef)), 'Range', 'Azimuth', 'Corr_coef_cropped_stretched')
    
    std_phd=std_phase_diff(Srh_arr, Srv_arr, window_size)
    save_figure(np.absolute(std_phd), 'Range', 'Azimuth', 'std_phd_cropped')
    #save_figure(hist_stretch(np.absolute(std_phd)), 'Range', 'Azimuth', 'std_phd_cropped_stretched')
    
    lambdas=eigen_values(stokes_vector)
    save_figure(np.absolute(lambdas[:,:,0]), 'Range', 'Azimuth', 'lambda1_cropped')
    save_figure(10*np.log10(np.absolute(lambdas[:,:,0])), 'Range', 'Azimuth', 'lambda1_cropped_stretched')
    save_figure(np.absolute(lambdas[:,:,1]), 'Range', 'Azimuth', 'lambda2_cropped')
    save_figure(10*np.log10(np.absolute(lambdas[:,:,1])), 'Range', 'Azimuth', 'lambda2_cropped_stretched')
    
    H=entropy(lambdas)
    save_figure(np.absolute(H), 'Range', 'Azimuth', 'Entropy_cropped')
    #save_figure(10*np.log10(np.absolute(H)), 'Range', 'Azimuth', 'Entropy_cropped_stretched')
    
    
    con_coeff=conformity_coeff(Srh_arr, Srv_arr, window_size)
    save_figure(np.absolute(con_coeff), 'Range', 'Azimuth', 'Conformity_coeff_cropped')
    #save_figure(hist_stretch(np.absolute(con_coeff)), 'Range', 'Azimuth', 'Conformity_coeff_cropped_stretched')
    
    cov_mat=get_covariance_matrix(Srh_arr, Srv_arr, window_size)
    det_arr=det_covariance_mat(cov_mat)
    save_figure(np.absolute(det_arr), 'Range', 'Azimuth', 'Det_Cov_Mat_cropped')
    save_figure(10*np.log10(np.absolute(det_arr)), 'Range', 'Azimuth', 'Det_Cov_Mat_cropped_stretched')
    #print(det_arr[0,0])
    #return(stokes_vector)
    #print(Srh_arr)
    #display(hist_stretch(np.absolute(det_arr)), 'Range', 'Azimuth', 'RISAT_S11')
    #display(hist_stretch(stokes_vector[:,:,3]), 'Range', 'Azimuth', 'RISAT-1 S11')

def m_chi_decomposition(Srh_arr,Srv_arr, window_size):
    #Srh_arr=averaging_arr(Srh_arr)
    stokes_vector=get_stokes_vector(Srh_arr,Srv_arr, window_size)
    s1=stokes_vector[...,0]
    m=degreeOfPolarization(stokes_vector)
    chi=ellipticity_angle(stokes_vector, m)
    
    b=np.sqrt(m*s1*(1-np.sin(2*chi))/2)
    r=np.sqrt(m*s1*(1+np.sin(2*chi))/2)
    g=np.sqrt(s1*(1-m))
    
    r,g,b=hist_stretch(r), hist_stretch(g), hist_stretch(b)
    #plotting.hist_stretch_all(arr, bits, clip_extremes)
    plt.imshow(np.dstack((r,g,b)))
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('$m-\chi$ - decomposition')
    #plt.show()


def m_delta_decomposition(Srh_arr,Srv_arr, window_size):
    stokes_vector=get_stokes_vector(Srh_arr,Srv_arr, window_size)
    s1=stokes_vector[...,0]
    m=degreeOfPolarization(stokes_vector)
    delta=relative_phase(stokes_vector)
    
    b=np.sqrt(m*s1*(1-np.sin(2*delta))/2)
    r=np.sqrt(m*s1*(1+np.sin(2*delta))/2)
    g=np.sqrt(s1*(1-m))
    
    r,g,b=hist_stretch(r), hist_stretch(g), hist_stretch(b)
    #plotting.hist_stretch_all(arr, bits, clip_extremes)
    plt.imshow(np.dstack((r,g,b)))
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('$m-\delta$ - decomposition')
    
def m_alpha_decomposition(Srh_arr,Srv_arr, window_size):
    stokes_vector=get_stokes_vector(Srh_arr,Srv_arr, window_size)
    s1=stokes_vector[...,0]
    m=degreeOfPolarization(stokes_vector)
    alpha=alpha_angle(stokes_vector)
    
    b=np.sqrt(m*s1*(1-np.sin(2*alpha))/2)
    r=np.sqrt(m*s1*(1+np.sin(2*alpha))/2)
    g=np.sqrt(s1*(1-m))
    
    r,g,b=hist_stretch(r), hist_stretch(g), hist_stretch(b)
    #plotting.hist_stretch_all(arr, bits, clip_extremes)
    plt.imshow(np.dstack((r,g,b)))
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title(r'$m-\alpha$ - decomposition')
    
if __name__=='__main__':
    #print(slant_to_ground_range())
    #print(set_geoTransform())
    os.chdir('../RISAT-1/RI1_SAR_L1SLC_FRS1_CR_20150610T071918_20150610T071923_17197_1515551004')
    
    
    #window_size=10
    arr=img_to_array()
    
    arr_s11=arr[:,:,0]#Srh
    arr_s21=arr[:,:,1]#Srv
    #Srh_arr=oil_subset(arr_s11)
    #Srv_arr=oil_subset(arr_s21)
    
    #meta=metadata_dict('BAND_META.txt')
    #inc_ang_arr=get_inc_angle_array('incidence_angle.bin')
    #inc_ang_oil_subset=oil_subset(inc_ang_arr)
    
    
    #==========m-chi decomposition===============
    #m_chi_decomposition(Srh_arr,Srv_arr, window_size)
    
    extract_all_features_1(arr_s11, arr_s21,True,9)
    
    #dim=dimension_data(arr_s11)
    #print (dim)
    #geo_reference_using_corner('output_file', 'BAND_META.txt', dim)
    
    #extract_all_features(arr_s11, arr_s21,True,10)
    #read_corner_coordinates('BAND_META.txt')

    #print(inc_ang_arr)
    #plt.imshow(inc_ang_arr)
    #plt.show()
    #print(inc_ang_arr.shape)
    #print(inc_ang_arr)
    
    #print(meta)
    '''
    trans_points=get_corners_UTM('BAND_META.txt')
    print(trans_points)
    print(calc_affine_rotation('BAND_META.txt'))
    
    #print(get_cell_size())
    
    print(set_geoTransform())
    '''
    
    #reproject_RISAT_SLC(arr_s11)
    #arr_s11_grec=read_reprojected_SLC()
    #print(arr_s11)
    
    #=============conversion to beta_nought/ convert_to_sigma_nought
    #beta=beta_nought(arr_s21, float(meta['Calibration_Constant_Beta0_RV']))
    #plt.imshow(beta, cmap='gray')
    #plt.show()
    #print(beta)
    #sigma=convert_to_sigma_nought(arr_s21, float(meta['Calibration_Constant_RV']), inc_ang_arr)
    #sigma=convert_to_sigma_nought(Srv_arr, float(meta['Calibration_Constant_RV']), inc_ang_oil_subset)
    #print(sigma)
    #reproject_RISAT_sigma_nought(sigma)
    #plt.imshow(np.log10(np.absolute(Srv_arr)), cmap='gray')
    #plt.colorbar()
    #plt.show()
    #a=np.absolute(arr_s21)
    #print(dimension_data(arr_s21))
    #print(s21_gt)
    #print(s21_proj)
    #save_figure(hist_stretch(a[6500:9000,5400:6800]), 'Range', 'Azimuth', 'RISAT-1 S11')
    #display(10 *np.log10(np.absolute(Srv_arr)), 'Range', 'Azimuth', 'RISAT-1 S11')
    #clear_list(a)
    #clear_list(arr_s21)
    #clear_list(arr_s11)
    
    
    #READ datafrom binary
    
    #crop_from_binary()
    
    
    
    
    