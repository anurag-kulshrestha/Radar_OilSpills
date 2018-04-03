import EPFS

import plotting 
from osgeo import gdal, ogr, osr
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import incidence_angle_corr
from math import pi
import itertools
from scipy import linalg
from sklearn import mixture

import extract_polarimetric
import glcm_sklearn
import fit_inci_model
from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage,misc
import reproject
import read_RISAT1
import matplotlib.patches as mpatches
from scipy import stats
import read_TerraSARX

def read_Pol_features(TSX_amp, window_size):
    
    
    cov_arr = read_TerraSARX.get_cov_matrix(TSX_amp, window_size, window_size)
    coh_arr = read_TerraSARX.get_coh_matrix(cov_arr)
    
    I_hh = coh_arr[...,0,0]
    I_vv = coh_arr[...,1,1]
    det_cov = read_TerraSARX.extract_polarimetric.determinant_cov(cov_arr)
    eigen_full = read_TerraSARX.eigen_raster_full(coh_arr)
    lamb_1 = eigen_full[...,0]
    lamb_2 = eigen_full[...,1]
    GI = np.sqrt(extract_polarimetric.determinant_cov(coh_arr))
    PD = np.absolute(cov_arr[...,1,1]  - cov_arr[...,0,0])
    span = np.real(cov_arr[...,1,1]  + cov_arr[...,0,0])
    
    return np.dstack((np.absolute(I_hh), np.absolute(I_vv), np.absolute(det_cov), np.absolute(lamb_1),np.absolute(lamb_2), np.absolute(GI), np.absolute(PD), np.absolute(span)))

def visualize_hist(pol):
    #dop=plotting.hist_stretch_all(pol[...,0],0, False)
    #H=plotting.hist_stretch_all(pol[...,1],0, False)
    pol = 10*np.log10(pol)
    label_arr = ['$I_{HH}$', '$I_{VV}$', 'det(C2)', '$\lambda_{1}$','$\lambda_{2}$', 'GI', 'PD', 'Span']
    
    for i in range(0, pol.shape[2]):
        print(stats.shapiro(np.random.choice(pol[...,i].flatten(), size =5000, replace = False ), reta = True))
        plt.hist(plotting.hist_stretch_all(pol[...,i],0, False).flatten(),bins=200, rwidth=0.5,histtype='step', label=label_arr[i])
    #dop=pol[...,0]
    #H=pol[...,1]
    
    
    
    #plt.hist(dop.flatten(),bins=200, rwidth=0.5,histtype='step', label='DoP')
    #plt.hist(H.flatten(),bins=200, rwidth=0.5,histtype='step', label='Entropy (H)')
    plt.xlabel('Normalized values', fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.legend()
    plt.show()

def prepare_X(arr_1, arr_2,arr3):
    return np.array([arr_1.flatten(),arr_2.flatten(),arr3.flatten()]).T

def gmm_fitting(arr_1, arr_2,arr3, n_comp):
    X=prepare_X(arr_1, arr_2,arr3)
    gmm = mixture.GaussianMixture(n_components=n_comp, covariance_type='full', tol=0.0001, verbose=1, verbose_interval=1).fit(X)
    return gmm

def plot_legend(im, values,ax):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    
    
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          #fancybox=True, shadow=True, ncol=len(values), prop={'size': 6})
    
    colors = [ im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label="Segment {l}".format(l=int(values[i])) ) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=len(values), prop={'size': 6} ) 

def plot_result(arr_1, arr_2,arr3, gmm):
    X=prepare_X(arr_1, arr_2,arr3)
    shp=arr_1.shape
    res=gmm.predict(X)
    values = np.unique(res.ravel())
    im=plt.imshow(res.reshape(shp[0],shp[1]), cmap='RdYlBu')
    #plot_legend(im)
    colors = [ im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label="Segment {l}".format(l=values[i]//1) ) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.tight_layout()
    #plt.colorbar()
    #plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Segmentation/'+'Segmentation_window_9_segment_2 _inc_corr_yes_1'+'.tiff', dpi=300)
    plt.show()

def plot_result_1(res):
    
    shp=res.shape
    values = np.unique(res.ravel())
    im=plt.imshow(res.reshape(shp[0],shp[1]), cmap='RdYlBu')
    colors = [ im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i]) ) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.tight_layout()
    #plt.colorbar()
    #plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Segmentation/'+'Segmentation_window_9_segment_2 _inc_corr_yes_1'+'.tiff', dpi=300)
    plt.show()

def majority(arr):
    max_arr=np.bincount(arr.flatten())
    pos=np.where(max_arr==np.amax(max_arr))
    return pos[0][0]

def majority_smoothening(arr_1, arr_2,arr3, n_components, window_size):
    gmm=gmm_fitting(arr_1, arr_2,arr3, n_components)
    X=prepare_X(arr_1, arr_2,arr3)
    shp=arr_1.shape
    seg_image=gmm.predict(X).reshape(shp[0],shp[1])
    #print(seg_image.dtype)
    oil_class=seg_image[118,190]
    #print(oil_class)
    rows,cols=shp[0], shp[1]
    res=np.zeros((rows-window_size, cols-window_size))
    res_oil_only=np.zeros((rows-window_size, cols-window_size))
    res_row, res_col, max_res_col, max_res_row= 0,0,0,0
    for i in range(0, rows-window_size):
        for j in range(0, cols-window_size):
            a=seg_image[i:i+window_size, j:j+window_size]
            #print (a.astype(int))
            smooth_val=majority(a)
            res[res_row,res_col]=smooth_val
            if(smooth_val==oil_class):
                res_oil_only[res_row, res_col]=1
            #print((i,j))
            res_col+=1
        max_res_col=res_col
        res_col=0
        res_row+=1
        max_res_row=res_row
    return [seg_image,res, res_oil_only]

def maj_smoo_plots(res, arr):
    values = np.unique(res[0].ravel())
    
    fig = plt.figure(dpi = 150, tight_layout=True)
    #ax = plt.subplots(1,4)
    
    ax = plt.subplot(141)
    im=plt.imshow(arr, cmap='gray')
    #plt.colorbar(orientation = 'horizontal')
    ax.set_xlabel('Range')
    ax.set_ylabel('Azimuth')
    #plt.title('Entropy')
    ax = plt.subplot(142)
    im=ax.imshow(res[0], cmap='RdYlBu')
    plot_legend(im, values,ax)
    ax.set_xlabel('Range')
    ax.set_ylabel('Azimuth')
    
    ax = plt.subplot(143)
    im=ax.imshow(res[1], cmap='RdYlBu')
    plot_legend(im, values, ax)
    ax.set_xlabel('Range')
    ax.set_ylabel('Azimuth')
    
    ax = plt.subplot(1,4,4)
    im=ax.imshow(res[2], cmap='gray_r')
    values = np.unique(res[2].ravel())
    plot_legend(im, values,ax)
    ax.set_xlabel('Range')
    ax.set_ylabel('Azimuth')
    
    plt.tight_layout()
    plt.show()
    


if __name__=='__main__':
    directory = '../TerraSAR_X/dims_op_oc_dfd2_567023152_2/TSX-1.SAR.L1B/TSX1_SAR__SSC______SM_D_SRA_20150610T062401_20150610T062409/sybset/subset_0_of_TSX1_SAR_SSC_SM_D_SRA_20150610T062401_20150610T062409_Cal.data'
    
    #os.chdir(directory)
    window_size=9
    window_size_smoo=5
    n_comp=3
    #arr=read_RISAT1.img_to_array()
    TSX_amp=read_TerraSARX.read_Raster(directory)
    
    
    
    
    pol=10*np.log10(read_Pol_features(TSX_amp, window_size))
    Ivv=plotting.hist_stretch_all(pol[...,1],0, False)
    GI = plotting.hist_stretch_all(pol[...,5],0, False)
    det = plotting.hist_stretch_all(pol[...,2],0, False)
    
    #visualize_hist(pol)
    #gmm=gmm_fitting(dop, H, n_comp)
    #plot_result(dop, H, gmm)
    res=majority_smoothening(Ivv, GI, det, n_comp, window_size_smoo)
    #plot_result_1(res[1])
    maj_smoo_plots(res, Ivv)