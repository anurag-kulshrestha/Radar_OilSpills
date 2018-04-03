import plotting 
from osgeo import gdal, ogr, osr
import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import incidence_angle_corr
from math import pi
import itertools
from scipy import linalg
from scipy import signal
from sklearn import mixture

import extract_polarimetric
import glcm_sklearn
import fit_inci_model
from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage,misc
import EPFS
import pandas as pd
import matplotlib

#Step1: choose pixels which overlap
#Step2: calc JM distance
    #calc mean, variance and then introduce into the formula
#Step3: calc 


#


def create_mask():
    return EPFS.get_seg_mask()[2]

def test_convolve(arr, kernal):
    grad = signal.convolve2d(arr, kernal, boundary='symm', mode='valid')
    return grad

def kernal(window_size):
    k=np.ones(window_size*window_size).reshape(window_size, window_size)
    normalize_k=k/(window_size**2)
    return normalize_k

def make_mask_max_val_1(mask_arr):
    mask_arr[np.where(mask_arr>0)]=1
    return mask_arr

def get_slick_wise_mask(window_size=0):
    P=gdal.Open('Output/P_only_pixels.tif')
    E40=gdal.Open('Output/E40_only_pixels.tif')
    E60=gdal.Open('Output/E60_only_pixels.tif')
    E80=gdal.Open('Output/E80_only_pixels.tif')
    W_near=gdal.Open('Output/W_near_1_only_pixels.tif')
    W_far=gdal.Open('Output/W_far_1_only_pixels.tif')
    W_mid=gdal.Open('Output/W_mid_1_only_pixels.tif')
    kern=kernal(window_size)
    '''
    P_arr=test_convolve(make_mask_max_val_1(P.ReadAsArray()), kern)
    E40_arr=test_convolve(make_mask_max_val_1(E40.ReadAsArray()), kern)
    E60_arr=test_convolve(make_mask_max_val_1(E60.ReadAsArray()), kern)
    E80_arr=test_convolve(make_mask_max_val_1(E80.ReadAsArray()), kern)
    W_near_arr=test_convolve(make_mask_max_val_1(W_near.ReadAsArray()), kern)
    W_far_arr=test_convolve(make_mask_max_val_1(W_far.ReadAsArray()), kern)
    W_mid_arr=test_convolve(make_mask_max_val_1(W_mid.ReadAsArray()), kern)
    '''
    P_arr=  make_mask_max_val_1(P.ReadAsArray())
    E40_arr=  make_mask_max_val_1(E40.ReadAsArray())
    E60_arr=  make_mask_max_val_1(E60.ReadAsArray())
    E80_arr=  make_mask_max_val_1(E80.ReadAsArray())
    W_near_arr=  make_mask_max_val_1(W_near.ReadAsArray())
    W_far_arr=  make_mask_max_val_1(W_far.ReadAsArray())
    W_mid_arr=  make_mask_max_val_1(W_mid.ReadAsArray())
    return np.dstack((P_arr,E40_arr, E60_arr, E80_arr,W_near_arr, W_mid_arr, W_far_arr))

def get_padded_feature_stack(window_size, correction_switch, degree, pad=-1):
    
    eigen_full=extract_polarimetric.eigen_raster_full(window_size, correction_switch, degree) #9 is the window size
    cov_arr=extract_polarimetric.extract_covariance_arr(window_size, correction_switch, degree)
    
    #padding=int(np.floor(window_size/2))
    
    padding=window_size//2 if pad!=0 else 0
    
    Ihh=np.pad(cov_arr[...,0,0], padding, 'constant')
    Ihv=np.pad(cov_arr[...,1,1], padding, 'constant')
    Ivv=np.pad(cov_arr[...,2,2], padding, 'constant')
    arr_lamb1=np.pad(eigen_full[:,:,2], padding, 'constant')
    arr_lamb2=np.pad(eigen_full[:,:,1], padding, 'constant')
    arr_lamb3=np.pad(eigen_full[:,:,0], padding, 'constant')
    
    co_pol_dif=np.pad(extract_polarimetric.co_pol_diff(cov_arr), padding, 'constant')
    arr_det_cov=np.pad(np.absolute(extract_polarimetric.determinant_cov_conj(cov_arr)), padding, 'constant')
    
    R_co = Rco_X = np.pad(np.real(cov_arr[...,0,2]), padding, 'constant')
    Ico_X = np.pad(np.absolute(np.imag(cov_arr[...,0,2])), padding, 'constant')
    
    #return np.dstack((np.real(Ihh),np.real(Ihv),np.real(Ivv),np.real(arr_lamb1), np.real(arr_lamb2), np.real(arr_lamb3), np.absolute(co_pol_dif),R_co, Ico_X, np.absolute(arr_det_cov)))
    
    return np.dstack((np.real(Ihh),np.real(Ihv),np.real(Ivv),np.real(arr_lamb1), np.real(arr_lamb2), np.real(arr_lamb3), np.absolute(co_pol_dif), np.absolute(arr_det_cov)))#,R_co, Ico_X
    
    #return np.absolute(arr_det_cov)#,R_co, Ico_X

def get_masked_arr_stack(window_size, correction_switch, degree):
    pol=get_padded_feature_stack(window_size, correction_switch, degree)
    s_mask=get_slick_wise_mask(window_size)
    ma_cond=s_mask[...,0]
    shp=ma_cond.shape
    num_features=pol.shape[-1]
    #for i in range(7):
        #a=ma.masked_where(np.repeat(s_mask[...,0]==0,num_features).reshape(shp[0],shp[1],num_features), pol)
    #print (pol.shape)
    #print(s_mask.shape)

    P_ma=ma.masked_where(np.repeat(s_mask[...,0]<1,num_features).reshape(shp[0],shp[1],num_features), pol)
    E40_ma=ma.masked_where(np.repeat(s_mask[...,1]<1,num_features).reshape(shp[0],shp[1],num_features), pol)
    E60_ma=ma.masked_where(np.repeat(s_mask[...,2]<1,num_features).reshape(shp[0],shp[1],num_features), pol)
    E80_ma=ma.masked_where(np.repeat(s_mask[...,3]<1,num_features).reshape(shp[0],shp[1],num_features), pol)
    
    W_near_ma=ma.masked_where(np.repeat(s_mask[...,-3]<1,num_features).reshape(shp[0],shp[1],num_features), pol)
    W_mid_ma=ma.masked_where(np.repeat(s_mask[...,-2]<1,num_features).reshape(shp[0],shp[1],num_features), pol)
    W_far_ma=ma.masked_where(np.repeat(s_mask[...,-1]<1,num_features).reshape(shp[0],shp[1],num_features), pol)
    #return P_ma
    return ma.dstack((P_ma,E40_ma,E60_ma,E80_ma, W_near_ma, W_mid_ma, W_far_ma))


def plot_histograms(window_size, correction_switch, degree, pol_feature_index, list_slicks=[0,1,2,3,4,5,6]):
    res=get_masked_arr_stack(window_size, correction_switch, degree)
    shp=res.shape
    #mean_arr=res.reshape(shp[0]*shp[1],shp[3]).mean(axis=0)
    #print(np.absolute(res[...,0].flatten()))
    
    #LIST_SLICKS
    #0->Plant Oil
    #1->E40
    #2->E60
    #3->E80
    #4->W_near
    #5->W_mid
    #6->W_far
    
    #FEATURE_INDEX
    #0 -> Ihh
    #1 -> Ihv
    #2 -> Ivv
    #3-> lambda_1
    #4-> lambda_2
    #5-> lambda_3
    #6-> co_pol_dif
    #7-> arr_det_cov
    num_features=5
    
    feature_labels=['$\lambda_{1}$','$\lambda_{2}$','$\lambda_{3}$','co_pol_diff', 'determinent_cov']
    slick_labels=['PO', 'E40', 'E60', 'E80', 'W_near', 'W_mid', 'W_far']
    num_bin=60
    
    plt.subplot(1,2,1)
    for i in list_slicks:
        plt.hist(np.absolute(res[...,pol_feature_index+i*num_features].compressed()), bins=num_bin, rwidth=0.5,histtype='step', label=feature_labels[pol_feature_index]+' '+slick_labels[i])

    plt.legend()
    plt.subplot(1,2,2)
    for i in list_slicks:
        plt.imshow(res[...,pol_feature_index+i*num_features], cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.show()
    
def seperability(mean_arr, var_arr,window_size, correction_switch, degree, sep_arr_indices, pol_feature_index, num_features):
    #res=get_masked_arr_stack(window_size, correction_switch, degree)

    #print(mean_arr)
    #print(var_arr)
    arr1_index=pol_feature_index+sep_arr_indices[0]*num_features
    arr2_index=pol_feature_index+sep_arr_indices[1]*num_features
    
    #print(arr1_index,arr2_index)
    
    bhatt_dist=(mean_arr[arr1_index]-mean_arr[arr2_index])**2 * (var_arr[arr1_index]+var_arr[arr2_index])/4 + 0.5* np.log(np.absolute(var_arr[arr1_index]+var_arr[arr2_index])/(2*np.sqrt(np.absolute(var_arr[arr1_index]*var_arr[arr2_index]))))
    
    JM=2*(1-np.exp(-1*bhatt_dist))
    
    fdr = (mean_arr[arr1_index]-mean_arr[arr2_index])**2/(var_arr[arr1_index]+var_arr[arr2_index])
    
    
    #return(bhatt_dist)
    
    return [JM,fdr]



def damping_ratio(window_size, correction_switch, degree, slick_num, sea_num, num_features):
    res=get_masked_arr_stack(window_size, correction_switch, degree)
    #print(res.shape)
    feature_num=2 # 0->Ihh, 1-> Ihv, 2: Ivv
    oil_slick=res[...,feature_num+slick_num*num_features]
    water=res[...,feature_num+sea_num*num_features]
    plt.imshow(oil_slick)
    plt.show()
    plt.imshow(water)
    plt.show()
    print(oil_slick.mean())
    print(water.mean())
    return -10*np.log10(oil_slick.mean()/water.mean())
    
def boundary_oil(seg_arr, window_size=3):
    shp=seg_arr.shape
    #print(shp)
    rows,cols=shp[0], shp[1]
    res_bound=np.zeros((rows, cols))
    #res_row, res_col=0,0
    for i in range(0, rows-window_size):
        for j in range(0, cols-window_size):
            a=seg_arr[i:i+window_size, j:j+window_size]
            #print(a)
            #break
            #print (a.astype(int))clear
            if((np.amin(a)==0) & (a[1,1]>0)):
                res_bound[i+1, j+1]=1
    return res_bound

def calc_slick_mean_var(res):
    shp=res.shape
    num_slicks = 7
    
    means=10*np.log10(np.absolute(res)).mean(0).mean(0)
    sd=np.sqrt(10*np.log10(np.absolute(res)).reshape(shp[0]*shp[1], shp[2]).var(0))
    
    #means=res.reshape(shp[0]*shp[1],shp[2]).mean(axis=0)
    #sd=np.sqrt(res.reshape(shp[0]*shp[1],shp[2]).var(axis=0))
    
    num_features=shp[-1]//num_slicks
    #print(num_features)
    slick_index=np.arange(0,num_slicks)
    features=np.arange(0,num_features)#determinant_cov is left out
    feature_labels=['$I_{hh}$','$I_{hv}$','$I_{vv}$','$\lambda_{1}$','$\lambda_{2}$','$\lambda_{3}$','PD', '$R_{CO}X$','$I_{CO}X$','det(C3)']
    slick_labels=['PO', 'E40', 'E60', 'E80', 'W_near', 'W_mid', 'W_far']
    col=['r','g','b','m','c','y','darkorange','navy']
    
    width_stride = 0
    width = 0.1
    '''
    for j in features:
        pol_feature_index=j
        feature_index=[pol_feature_index+i*num_features for i in slick_index]
        
        #plt.errorbar(slick_index,means[feature_index],sd[feature_index], linestyle='', color=col[j], label=feature_labels[j], linewidth=1,capsize=2,elinewidth=0.8,marker='.')#, capthick=0.5
        
        plt.bar(slick_index - width_stride, means[feature_index], width, yerr=sd[feature_index], label=feature_labels[j], linewidth=1,capsize=2)#, capthick=0.5,marker='.')
        
        width_stride = width_stride - width
        
    plt.xticks(slick_index, slick_labels)
    plt.xlabel('Slicks')
    plt.ylabel('$\sigma_{0}$ (dB)')
    plt.ylim((-40,0))
    plt.legend(bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)#bbox_to_anchor=(1, 1))
    #plt.tight_layout()
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Feature_Selection/'+'mean_variance_features_bars'+'.tiff', dpi=200)#, box_inches='tight')
    plt.show()
    '''
    ax = plt.subplot(111)
    for j in slick_index:
        #pol_feature_index=j
        slick_id = j
        
        feature_index=[pol_feature_index + slick_id*num_features for pol_feature_index in features]
        
        ax.bar(features - width_stride, means[feature_index], width, yerr=sd[feature_index], label=slick_labels[j], linewidth=1,capsize=2)#, capthick=0.5,marker='.')
        
        width_stride = width_stride - width
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    plt.xticks([i+len(features)/20 for i in features], feature_labels)
    ax.set_xlabel('Slicks')
    ax.set_ylabel('$\sigma_{0}$ (dB)')
    #ax.set_ylim((-90,0))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=len(features), prop={'size': 6})
    #plt.tight_layout()
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Feature_Selection/'+'mean_variance_features_bars_10_features'+'.tiff', dpi=150)#, box_inches='tight')
    plt.show()
    
    
def slick_area_extent():
    a=get_slick_wise_mask(window_size)
    #print (np.mean(get_slick_wise_mask(), axis=0))
    area=np.zeros((7))
    spread=np.zeros((7))
    g_cmin=a.shape[1]
    g_cmax=0
    #plt.figure(dpi=200)
    #matplotlib.rcParams.update({'font.size': 9})
    for i in range(7):
        plt.imshow(a[...,i], cmap='gray_r', alpha=0.1)
    
    for i in range(4):
        pos=np.where(a[...,i]==1)
        #plt.imshow(a[...,i], cmap='gray_r', alpha=0.1)
        area[i]=pos[0].size*7.2*5
        spread[i]=(np.amax(pos[1])-np.amin(pos[1]))*5
        if g_cmin>np.amin(pos[1]):
            g_cmin=np.amin(pos[1])
        if g_cmax<np.amax(pos[1]):
            g_cmax=np.amax(pos[1])
    print(g_cmin)
    print(g_cmax)
    #plt.axvline(x=g_cmin,  linewidth=2, color = 'k')#,xmin=0.25, xmax=0.402,)
    #plt.axvline(x=g_cmax,  linewidth=2, color = 'k')#,xmin=0.25, xmax=0.402,)
    print(area)
    print(spread)
    plt.text(207,458, 'W_near')
    plt.text(594,138, 'W_mid')
    plt.text(821,562, 'W_far')
    plt.text(594,328, 'PO')
    plt.text(564,524, 'E40')
    plt.text(532,776, 'E60')
    plt.text(411,985, 'E80')
    
    plt.arrow(1000,1000,-200,100)# arrow for wind
    
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('All slicks')
    plt.tight_layout()
    #plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Segmentation/Slicks/all_1.tiff' , dpi=300, box_inches='tight')
    '''
    #=============minchew_vaala tareeka===========
    w_top=np.zeros((a.shape[0], a.shape[1]))
    w_top[110:160,g_cmin:g_cmax]=1
    #plt.imshow(w_top, cmap='gray_r',alpha=0.1)
    plt.text(g_cmin,150, 'WATER_1')
    
    w_middle=np.zeros((a.shape[0], a.shape[1]))
    w_middle[600:650,g_cmin:g_cmax]=1
    #plt.imshow(w_middle, cmap='gray_r',alpha=0.1)
    plt.text(g_cmin,640, 'WATER_2')
    
    w_bot=np.zeros((a.shape[0], a.shape[1]))
    w_bot[1090:1140,g_cmin:g_cmax]=1
    #plt.imshow(w_bot, cmap='gray_r',alpha=0.1)
    plt.text(g_cmin,1130, 'WATER_3')
    #plt.axis('off')
    '''
    plt.show()

def all_slick_seperability(mean_arr, var_arr, plotting,window_size, correction_switch, degree):
    num_features=10
    num_slicks = 7
    
    slick_index=np.arange(0,num_slicks)
    features=np.arange(0,num_features)#determinant_cov is left out
    feature_labels=['$I_{hh}$','$I_{hv}$','$I_{vv}$','$\lambda_{1}$','$\lambda_{2}$','$\lambda_{3}$','PD', '$R_{CO}X$','$I_{CO}X$','det(C3)']
    slick_labels=['PO', 'E40', 'E60', 'E80', 'W_near', 'W_mid', 'W_far']
    sep=np.zeros((num_slicks,num_slicks,num_features), dtype=np.float64)
    fdr=np.zeros((num_slicks,num_slicks,num_features), dtype=np.float64)
    
    for i in features:
        for j in slick_index:
            for k in slick_index:
                sep_JM_fdr = seperability(mean_arr, var_arr,window_size, correction_switch, degree,sep_arr_indices=[j,k], pol_feature_index=i, num_features=num_features)
                
                sep[j,k,i]= sep_JM_fdr[0]
                
                #fdr[j,k,i]= sep_JM_fdr[1]
    #print(sep[0:4,4,:].shape)
    #for i in range(4:7)
    #fig, axes = plt.subplots(nrows=1, ncols=3)
    if(plotting==True):
        matplotlib.rcParams.update({'font.size': 8})
        
        #======================JM=======================
        
        
        plt.subplot(1,7,1)
        plt.imshow(sep[0:4,4,:].T, cmap='RdYlBu_r')
        plt.xticks(slick_index[:4], slick_labels[:4])
        plt.yticks(features, feature_labels)
        plt.title('Oil slicks vs W_near')
        #plt.colorbar(orientation='horizontal', label='JM', ticks=[0.5,1,1.5,2])
        
        plt.subplot(1,7,2)
        #plt.subplot(1,1,1)
        plt.imshow(sep[0:4,5,:].T, cmap='RdYlBu_r')
        plt.xticks(slick_index[:4], slick_labels[:4])
        #plt.yticks(features, feature_labels)
        plt.title('Oil slicks vs W_mid')
        #plt.colorbar(orientation='horizontal', label='JM', ticks=[0.5,1,1.5,2])
        
        plt.subplot(1,7,3)
        im=plt.imshow(sep[0:4,6,:].T, cmap='RdYlBu_r')
        plt.xticks(slick_index[:4], slick_labels[:4])
        #plt.yticks(features, feature_labels)
        plt.title('Oil slicks vs W_far')
        #plt.colorbar(orientation='horizontal', label='JM', ticks=[0.5,1,1.5,2])
        
        
        #=======================OIL VS OIL==============================
        
        
        
        
        plt.subplot(1,7,4)
        plt.imshow(sep[0:4,0,:].T, cmap='RdYlBu_r')
        plt.xticks(slick_index[:4], slick_labels[:4])
        #plt.yticks(features, feature_labels)
        plt.title('Oil slicks vs PO')
        plt.colorbar(orientation='horizontal', label='JM', ticks=[0.5,1,1.5,2])
        
        plt.subplot(1,7,5)
        #plt.subplot(1,1,1)
        plt.imshow(sep[0:4,1,:].T, cmap='RdYlBu_r')
        plt.xticks(slick_index[:4], slick_labels[:4])
        #plt.yticks(features, feature_labels)
        plt.title('Oil slicks vs E40')
        #plt.colorbar(orientation='horizontal', label='JM', ticks=[0.5,1,1.5,2])
        
        plt.subplot(1,7,6)
        im=plt.imshow(sep[0:4,2,:].T, cmap='RdYlBu_r')
        plt.xticks(slick_index[:4], slick_labels[:4])
        #plt.yticks(features, feature_labels)
        plt.title('Oil slicks vs E60')
        #plt.colorbar(orientation='horizontal', label='JM', ticks=[0.5,1,1.5,2])
        
        ax = plt.subplot(1,7,7)
        im=plt.imshow(sep[0:4,3,:].T, cmap='RdYlBu_r')
        plt.xticks(slick_index[:4], slick_labels[:4])
        #plt.yticks(features, feature_labels)
        ax.yaxis.tick_right()
        plt.title('Oil slicks vs E80')
        #plt.colorbar(orientation='horizontal', label='JM', ticks=[0.5,1,1.5,2])
        
        #========================FDR=========================
        
        
        #=======================OIL VS OIL==============================
        #plt.subplot(1,4,1)
        #plt.imshow(fdr[0:4,0,:].T, cmap='RdYlBu_r')
        #plt.xticks(slick_index[:4], slick_labels[:4])
        #plt.yticks(features, feature_labels)
        #plt.title('Oil slicks vs PO')
        #plt.colorbar(orientation='horizontal', label='FDR')#, ticks=[0.5,1,1.5,2])
        
        #plt.subplot(1,4,2)
        ##plt.subplot(1,1,1)
        #plt.imshow(fdr[0:4,1,:].T, cmap='RdYlBu_r')
        #plt.xticks(slick_index[:4], slick_labels[:4])
        #plt.yticks(features, feature_labels)
        #plt.title('Oil slicks vs E40')
        #plt.colorbar(orientation='horizontal', label='FDR')#, ticks=[0.5,1,1.5,2])
        
        #plt.subplot(1,4,3)
        #im=plt.imshow(fdr[0:4,2,:].T, cmap='RdYlBu_r')
        #plt.xticks(slick_index[:4], slick_labels[:4])
        #plt.yticks(features, feature_labels)
        #plt.title('Oil slicks vs E60')
        #plt.colorbar(orientation='horizontal', label='FDR')#, ticks=[0.5,1,1.5,2])
        
        #plt.subplot(1,4,4)
        #im=plt.imshow(fdr[0:4,3,:].T, cmap='RdYlBu_r')
        #plt.xticks(slick_index[:4], slick_labels[:4])
        #plt.yticks(features, feature_labels)
        #plt.title('Oil slicks vs E80')
        #plt.colorbar(orientation='horizontal', label='FDR')#, ticks=[0.5,1,1.5,2])
        
        
        
        #fig.colorbar(im,orientation='horizontal', label='JM')
        plt.tight_layout()
        plt.savefig(plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Feature_Selection/'+'separability_water_and_oil_from_oils'+'.tiff', dpi=300, box_inches='tight'))
        plt.show()
    
    return np.dstack((sep[0:4,4,:].T, sep[0:4,5,:].T,sep[0:4,6,:].T))
    
def seperability_vs_looks(correction_switch, degree,max_window_size):
    matplotlib.rcParams.update({'font.size': 5})
    
    num_features=10
    num_slicks = 7
    
    sep_all=np.empty((max_window_size//2,num_features,4,3))#(window_size, feature_num,slick_num,water_location)
    
    count=0
    
    feature_labels=['$I_{hh}$','$I_{hv}$','$I_{vv}$','$\lambda_{1}$','$\lambda_{2}$','$\lambda_{3}$','PD', '$R_{CO}X$','$I_{CO}X$','det(C3)']
    #col=['r','g','b','m','c','y','darkorange','navy']
    for i in range(1,max_window_size,2):

        window_size=i
        res=get_masked_arr_stack(window_size, correction_switch, degree)
        shp=res.shape
        mean_arr=10*np.log10(np.absolute(res)).reshape(shp[0]*shp[1],shp[2]).mean(axis=0)
        var_arr=10*np.log10(np.absolute(res)).reshape(shp[0]*shp[1],shp[2]).var(axis=0)
        #mean_arr=res.reshape(shp[0]*shp[1],shp[2]).mean(axis=0)
        #var_arr=res.reshape(shp[0]*shp[1],shp[2]).var(axis=0)
        sep=all_slick_seperability(mean_arr,var_arr, False,window_size, correction_switch, degree)
        #return sep.shape
        sep_all[count]=sep
        count+= 1
    #plt.subplot(1,2,1)
    #for j in range(0,8):
        #plt.plot(sep_all[:,j,0,0], marker='.', label=feature_labels[j])#color=col[j]

    #plt.legend()
    #plt.xlabel('Number of looks over MLC image')
    #plt.xticks(list(range(0,count)) , [str(i) for i in list(range(1,max_window_size,2))])
    #plt.ylabel('JM distance')
    #plt.title('PO vs W_near')
    #plt.tight_layout()
    ##plt.savefig(plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Feature_Selection/'+'separability_oil_vslooks_PO_W_near'+'.tiff', dpi=300, box_inches='tight'))
    
    #plt.subplot(1,2,2)

    #for j in range(0,8):
        #plt.plot(sep_all[:,j,0,2], color=col[j], marker='.', label=feature_labels[j])

    #plt.legend()
    #plt.xlabel('Number of looks over MLC image')
    #plt.xticks(list(range(0,count)) , [str(i) for i in list(range(1,max_window_size,2))])
    #plt.ylabel('JM distance')
    #plt.title('PO vs W_far')
    
    #==============All Oils vs W_mid==================
    plt.subplot(2,2,1)

    for j in range(0,num_features):
        plt.plot(sep_all[:,j,0,1], marker='.', label=feature_labels[j])#color=col[j]

    plt.legend()
    #plt.xlabel('Number of looks over MLC image')
    plt.xticks(list(range(0,count)) , [str(i) for i in list(range(1,max_window_size,2))])
    plt.ylabel('JM distance')
    plt.title('PO vs W_mid')
    
    plt.subplot(2,2,2)

    for j in range(0,num_features):
        plt.plot(sep_all[:,j,1,1], marker='.', label=feature_labels[j])#color=col[j]

    plt.legend()
    #plt.xlabel('Number of looks over MLC image')
    plt.xticks(list(range(0,count)) , [str(i) for i in list(range(1,max_window_size,2))])
    #plt.ylabel('JM distance')
    plt.title('PO vs W_mid')
    
    plt.subplot(2,2,3)

    for j in range(0,num_features):
        plt.plot(sep_all[:,j,2,1], marker='.', label=feature_labels[j])#color=col[j]

    plt.legend()
    plt.xlabel('Number of looks over MLC image')
    plt.xticks(list(range(0,count)) , [str(i) for i in list(range(1,max_window_size,2))])
    plt.ylabel('JM distance')
    plt.title('PO vs W_mid')
    
    ax = plt.subplot(2,2,4)

    for j in range(0,num_features):
        plt.plot(sep_all[:,j,3,1], marker='.', label=feature_labels[j])#color=col[j]

    plt.legend()
    #ax.legend(loc = (0.5, 0), ncol=5, labelspacing=0.)
    plt.xlabel('Number of looks over MLC image')
    plt.xticks(list(range(0,count)) , [str(i) for i in list(range(1,max_window_size,2))])
    #plt.ylabel('JM distance')
    plt.title('PO vs W_mid')
    #plt.figlegend( lines, labels, loc = 'lower center', ncol=5, labelspacing=0. )
    
    
    plt.tight_layout()
    plt.savefig(plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Feature_Selection/'+'separability_oil_vslooks_W_mid_all_slicks'+'.tiff', dpi=300, box_inches='tight'))
    plt.show()
        
    
    
    
    
if __name__=='__main__':
    os.chdir('/home/anurag/Documents/MScProject/SAR/OilSpill/North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01/MLC_Python_New/norway_00709_15092_000_150610_L090_CX_01_mlc')
    #res=EPFS.majority_smoothening(arr_1, arr_2,ncomp, window_size_smoo)
    #read_shp_file()
    #print(create_mask())
    window_size=1
    correction_switch=False
    degree=1
    JM_vs_window_size = 50
    
    
    #res=get_masked_arr_stack(window_size, correction_switch, degree)
    #shp=res.shape
    
    #mean_arr=10*np.log10(np.absolute(res)).reshape(shp[0]*shp[1],shp[2]).mean(axis=0)
    #var_arr=10*np.log10(np.absolute(res)).reshape(shp[0]*shp[1],shp[2]).var(axis=0)
    
    #plt.imshow(10*np.log10(np.absolute(res[...,59])))
    #plt.show()
    
    #mean_arr=res.reshape(shp[0]*shp[1],shp[2]).mean(axis=0)
    #var_arr=res.reshape(shp[0]*shp[1],shp[2]).var(axis=0)
    ##=========PLOT_MEAN_VAR============
    #calc_slick_mean_var(res)
    
    #import sys
    #sys.exit()
    
    
    #=========GETTING SLICK-WISE MASK-extent_AREA analysis==========
    #slick_area_extent()
    
    
    

    #============Plotting========
    
    #pol=get_padded_feature_stack(window_size, correction_switch, degree)
    #print(pol.shape)
    #plt.imshow(boundary_oil(a[...,0]), cmap='gray', alpha=1)
    #plt.imshow(boundary_oil(a[...,1]), cmap='gray', alpha=1)
    #plt.imshow(boundary_oil(a[...,2]), cmap='gray', alpha=1)
    #plt.imshow(boundary_oil(a[...,3]), cmap='gray', alpha=1)
    #plt.imshow(10*np.log10(pol[...,2]), cmap='gray', alpha=0.8)
    
    #print(np.percentile((res[...,34]), [0,25,50,75, 100]))
    #print(10*np.log10(res[...,20]), [25,50,75])
    #print((10*np.log10(res[...,4])).flatten().shape)
    #print()
    #plt.imshow(10*np.log10(res[...,34]), cmap='gray')
    #plt.colorbar()
    #plt.subplot(122)
    #plt.boxplot(((res[...,34])).flatten(),  meanline=True,notch=True,sym='')
    #plt.show()
    
    #==========seperability=========
    
    #all_slick_seperability(mean_arr,var_arr, True,window_size, correction_switch, degree)
    
    #seperability vs looks
    seperability_vs_looks(correction_switch, degree, JM_vs_window_size)
    
    
    ##=========Plot histogram==========
    #plot_histograms(window_size=window_size, correction_switch=False, degree=1, pol_feature_index=3, list_slicks=[0,4])#,2,3,6])
    
    ## damping damping_ratio
    #print(damping_ratio(window_size, correction_switch, degree, 3, 5,8))
    
#def read_shp_file():
    #shapefile = "Output/slick_polygons.shp"
    #driver = ogr.GetDriverByName("ESRI Shapefile")
    #dataSource = driver.Open(shapefile, 0)
    #layer = dataSource.GetLayer()
    #for feature in layer:
        #print (feature.GetField("Slick_name"))
    #layer.ResetReading()