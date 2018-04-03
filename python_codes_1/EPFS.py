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
from scipy import stats

import extract_polarimetric
import glcm_sklearn
import fit_inci_model
from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage,misc
import reproject
import matplotlib.patches as mpatches
import matplotlib
import matplotlib.gridspec as gridspec
import prob_surface_modelling

import sys

#step1: selection of features
#step2: feature transformation using histogram
#step3: gaussian fitting using EM method: number of components=3

#lambda_3; co-pol diff; contrast_C33_0deg, det(cov_arr), dissimilarity_C33_0deg; energy_C33_0deg; homogeniety_C33_0deg #asm_C33_0deg

#features checked and to be used further: co-pol diff, lambda_3, 



def read_GLCM_C33_features():
    wd = os.getcwd()
    #os.chdir('Output')
    #print(os.getcwd())
    texture_method=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'ASM']
    for method in texture_method:
        array_name=r'Contrast_GLCM_feature_method_'+method+'_dir_'+str(0)+'_pi_window_size_'+str(9)+'_row_stride_'+str(1)+'_col_stride_'+str(1)
        img=gdal.Open(array_name+'.tif')
        img_gt=img.GetGeoTransform()
        arr_img=img.ReadAsArray()
        os.chdir(wd)
        yield arr_img
    
    #print(C33.RasterXSize, C33.RasterYSize, C33.RasterCount)
    #print(arr_C33.shape)
    #return arr_img
    
def contextual_stack(log_Transform_switch = False):
    os.chdir('Output')
    glcm=read_GLCM_C33_features()
    
    #glcm_stack = np.dstack((next(glcm)[:-1,:-1], next(glcm)[:-1,:-1], next(glcm)[:-1,:-1], next(glcm)[:-1,:-1], next(glcm)[:-1,:-1]))
    
    glcm_stack = np.dstack((rescale(next(glcm)), rescale(next(glcm)), rescale(next(glcm)), rescale(next(glcm)), rescale(next(glcm))))
    if log_Transform_switch==True:
        glcm_stack = 10*log
    
    os.chdir('../')
    return glcm_stack

def read_Pol_features(window_size, correction_switch, degree, rescale_switch = False, log_Transform_switch = False):
    #os.chdir('../')
    
    eigen_full = extract_polarimetric.eigen_raster_full(window_size, correction_switch, degree) #9 is the window size
    cov_arr = extract_polarimetric.extract_covariance_arr(window_size, correction_switch, degree)
    
    arr_lamb1=np.absolute(eigen_full[:,:,2])
    arr_lamb2=np.absolute(eigen_full[:,:,1])
    arr_lamb3=np.absolute(eigen_full[:,:,0])
    co_pol_dif=np.absolute(extract_polarimetric.co_pol_diff(cov_arr))
    arr_det_cov=np.absolute(extract_polarimetric.determinant_cov_conj(cov_arr))
    Rco_X = np.real(cov_arr[...,0,2])
    Ico_X = abs(np.imag(cov_arr[...,0,2]))
    I_hh = np.absolute(cov_arr[...,0,0])
    I_hv = np.absolute(cov_arr[...,1,1])
    I_vv = np.absolute(cov_arr[...,2,2])
    
    if (log_Transform_switch ==True):
        arr_lamb1 = np.log10(arr_lamb1)
        arr_lamb2 = np.log10(arr_lamb2)
        arr_lamb3 = np.log10(arr_lamb3)
        co_pol_dif = np.log10(co_pol_dif)
        arr_det_cov = np.log10(arr_det_cov)
        Rco_X = np.log10(Rco_X)
        Ico_X = np.log10(Ico_X)
        I_hh = np.log10(I_hh)
        I_hv = np.log10(I_hv)
        I_vv = np.log10(I_vv)
    
    if(rescale_switch == True):
        arr_lamb1=rescale(arr_lamb1)
        arr_lamb2=rescale(arr_lamb2)
        arr_lamb3=rescale(arr_lamb3)
        co_pol_dif = rescale(co_pol_dif)
        arr_det_cov = rescale(arr_det_cov)
        Rco_X = rescale(Rco_X)
        Ico_X = rescale(Ico_X)
        I_hh = rescale(I_hh)
        I_hv = rescale(I_hv)
        I_vv = rescale(I_vv)
    
    
    #if(log_Transform_switch ==True):
        #return 10*np.log10(np.dstack((arr_lamb1, arr_lamb2, arr_lamb3, co_pol_dif, arr_det_cov, Rco_X, Ico_X, I_hh, I_hv, I_vv)))
    #else:
    return np.dstack((arr_lamb1, arr_lamb2, arr_lamb3, co_pol_dif, arr_det_cov, Rco_X, Ico_X, I_hh, I_hv, I_vv))
    
    #contrast_C33_0deg = glcm_sklearn.occurance_kernel(distance, direction,method, window_size, stride_row, stride_col)

def visualize_hist(pol):
    matplotlib.style.use('default')
    #glcm=contextual_stack()
    #pol = 10*np.log10(read_Pol_features(window_size, correction_switch, degree))
    #pol = read_Pol_features(window_size, correction_switch, degree,rescale = rescale_switch, log_Transform = log_Transform_switch)
    bin_size = 200
    font_size = 15
    #plt.subplot()
    #a=glcm[...,0]
    #a[np.where(a==0)]=10**-10
    #print(np.where(a==0.))
    #PLOTTING GLCM FEATURES
    #plotting.plot_histogram(glcm[...,0], 'Range', 'Azimuth', 'Contrast_glcm_hist', 'auto', width=1)
    #plotting.plot_histogram(glcm[...,1], 'Range', 'Azimuth', 'Dissimilarity_glcm_hist', 'auto', width=1)
    #plotting.plot_histogram(glcm[...,2], 'Range', 'Azimuth', 'Homogeneity_glcm_hist', 'auto', width=0.001)
    #plotting.plot_histogram(glcm[...,3], 'Range', 'Azimuth', 'Energy_glcm_hist', 'auto', width=0.001)
    #plotting.plot_histogram(glcm[...,4], 'Range', 'Azimuth', 'ASM_glcm_hist', 'auto', width=0.001)
    
    #print(pol[...,0])
    #plotting.plot_histogram(np.real(pol[...,0]), 'Range', 'Azimuth', 'Lambda_1', 'auto', width=0.001)
    #plotting.plot_histogram(np.real(pol[...,1]), 'Range', 'Azimuth', 'Lambda_2', 'auto', width=0.00001)
    #plotting.plot_histogram(np.real(pol[...,2]), 'Range', 'Azimuth', 'Lambda_3', 'auto', width=0.00001)
    #plotting.plot_histogram(np.absolute(np.real(pol[...,3])), 'Range', 'Azimuth', 'Col-pol Ratio', 'auto', width=0.0001)
    ##print(np.log(np.real(pol[...,2])))
    #plotting.plot_histogram(np.log(np.real(pol[...,4])), 'Range', 'Azimuth', 'Log of Determinant', 'auto', width=.01)
    #plt.subplot(2,2,1)
    #plt.hist(pol[...,2].flatten(),bins=150, rwidth=0.5,histtype='step')
    #plt.xlabel('$\lambda_{3}$',fontsize=15)
    #plt.ylabel('Frequency',fontsize=15)
    
    #plt.subplot(2,2,2)
    #plt.hist(pol[...,3].flatten(),bins=150, rwidth=0.5,histtype='step')
    #plt.xlabel('Co-Pol_diff', fontsize=15)
    #plt.ylabel('Frequency',fontsize=15)
    label_list = ['$\lambda_{1}$', '$\lambda_{2}$', '$\lambda_{3}$', 'PD', '$det(C3)$', 'Rco_X', 'Ico_X', '$I_{HH}$', '$I_{HV}$', '$I_{VV}$', 'GLCM_contrast', 'GLCM_dissimilarity', 'GLCM_homogeneity', 'GLCM_energy', 'GLCM_ASM']
    #color_list = ['C'+str(i) for i in range (1,16)]
    #print(color_list)
    #plt.subplot(1,2,1)
    for i in range(0,15):
        print(stats.shapiro(np.random.choice(pol[...,i].flatten(), size =5000, replace = False ), reta = True))
        if (i<10):
            plt.hist(pol[...,i].flatten(),bins=bin_size, rwidth=0.5,histtype='step', label=label_list[i])
        else:
            plt.hist(pol[...,i].flatten(),bins=bin_size, rwidth=0.5,histtype='step', label=label_list[i], linestyle = ':')
    
    #plt.hist(rescale(pol[...,0]).flatten(),bins=bin_size, rwidth=0.5,histtype='step', label='$\lambda_{1}$')
    #plt.hist(rescale(pol[...,1]).flatten(),bins=bin_size, rwidth=0.5,histtype='step', label='$\lambda_{2}$')
    #plt.hist(rescale(pol[...,4]).flatten(),bins=bin_size, rwidth=0.5,histtype='step', label='$det(C3)$')
    #plt.hist(rescale(pol[...,2]).flatten(),bins=bin_size, rwidth=0.5,histtype='step', label='$\lambda_{3}$')
    #plt.hist(rescale(pol[...,3]).flatten(),bins=bin_size, rwidth=0.5,histtype='step', label='PD')
    
    #plt.hist(rescale(pol[...,5]).flatten(),bins=bin_size, rwidth=0.5,histtype='step', label='Rco_X')
    #plt.hist(rescale(pol[...,6]).flatten(),bins=bin_size, rwidth=0.5,histtype='step', label='Ico_X')
    #plt.hist(rescale(pol[...,7]).flatten(),bins=bin_size, rwidth=0.5,histtype='step', label='I_hh')
    #plt.hist(rescale(pol[...,8]).flatten(),bins=bin_size, rwidth=0.5,histtype='step', label='I_hv')
    #plt.hist(rescale(pol[...,9]).flatten(),bins=bin_size, rwidth=0.5,histtype='step', label='I_vv')
    
    
    plt.xlabel('Normalized values', fontsize=font_size)
    plt.ylabel('Frequency',fontsize=font_size)
    
    plt.legend()
    plt.tight_layout()
    plt.ylim((0,200000))
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Segmentation/'+'Histogram_GLCM_stack_5_features_no_inc_corr_bin200'+'.tiff', dpi=300, box_inches='tight')
    plt.show()
    
    #============Plotting images==========
    #plt.subplots(2,3)
    #plt.subplot(2,3,1)
    #imgplot=plt.imshow(np.absolute(np.real(pol[...,0])), cmap='gray')
    #plt.colorbar()
    #plt.subplot(2,3,2)
    #imgplot=plt.imshow(np.absolute(np.real(pol[...,1])), cmap='gray')
    #plt.colorbar()
    #plt.subplot(2,3,3)
    #imgplot=plt.imshow(np.absolute(np.real(pol[...,2])), cmap='gray')
    #plt.colorbar()
    #plt.subplot(2,3,4)
    #imgplot=plt.imshow(np.absolute(np.real(pol[...,3])), cmap='gray')
    #plt.colorbar()
    ##plt.show()
    #plt.subplot(2,3,5)
    #imgplot=plt.imshow(np.absolute(np.real(pol[...,4])), cmap='gray')
    #plt.colorbar()
    #plt.show()
    #============Plotting images==========

def rescale(arr, clip_extremes = False):
    #scaler=MinMaxScaler()
    #return scaler.fit_transform(arr)
    #print(plotting.hist_stretch_all(arr,0, False))
    return plotting.hist_stretch_all(arr,0, clip_extremes)
    
def plot_scatter(window_size, correction_switch):
    pol=read_Pol_features(window_size, correction_switch)
    #X_0=pol[...,0]
    #X_0=pol[...,2]
    #X_1=pol[...,3]
    #plt.scatter(rescale(pol[...,2]), rescale(pol[...,3]), 0.1)
    plt.scatter(pol[...,2], pol[...,3], 0.1)
    plt.show()
    #rescale(pol[...,0])
    #plot_2hist(pol[...,2], pol[...,3], 'lamb_3', 'co-pol_diff', 'lamb_3 and co-pol_diff')
    
    #plt.subplots(1,2,1)
    #plt.hist(rescale(pol[...,0]).flatten(),bins=150, rwidth=0.5, density=False, histtype='step')
    ##plt.subplot(1,2,2)
    #plt.hist(rescale(pol[...,1]),bins=150, rwidth=0.5, density=False, histtype='step')
    #plt.show()

def plot_2hist(y1_array, y2_array, y1_label, y2_label, title):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title(title)
    ax1.set_xlabel('Value')
    ax1.set_ylabel(y1_label)
    ax2 = ax1.twinx()
    #ax3 = ax1.twinx()
    #ax1.plot(data['pixel_num'],np.tan(np.array(data['angle']/pi)), color='r', label='the data')
    ax1.hist(y1_array.flatten(), color='r', label=y1_label ,bins=150, rwidth=0.5, density=False, histtype='step')
    ax2.hist(y2_array.flatten(), color='g', label=y2_label,bins=150, rwidth=0.5, density=False, histtype='step')
    ax2.set_ylabel(y2_label)
    leg = ax1.legend(bbox_to_anchor=(0.15, 1))
    leg2=ax2.legend(bbox_to_anchor=(0.1, .92))
    plt.show()

def prepare_X_all(arr_dstack):
    shp = arr_dstack.shape
    return arr_dstack.reshape(shp[0]*shp[1],shp[2])

def prepare_X(arr_1, arr_2):
    return np.array([arr_1.flatten(),arr_2.flatten()]).T

def gmm_fitting_1(arr_dstack, n_comp,tolerance=0.001,num_initial=1, max_iteration=200):
    #X=prepare_X(arr_1, arr_2)
    X = prepare_X_all(arr_dstack)
    print(X.shape)
    gmm = mixture.GaussianMixture(n_components=n_comp, covariance_type='full', verbose=1, verbose_interval=1, tol=tolerance, n_init=num_initial, max_iter=max_iteration).fit(X)
    return gmm

def gmm_fitting(arr_1, arr_2, n_comp,tolerance=0.001,num_initial=1, max_iteration=200):
    X=prepare_X(arr_1, arr_2)
    #X = prepare_X_all(arr_dstack)
    print(X.shape)
    gmm = mixture.GaussianMixture(n_components=n_comp, covariance_type='full', verbose=1, verbose_interval=1, tol=tolerance, n_init=num_initial, max_iter=max_iteration).fit(X)
    return gmm

def plot_soft_probability(arr_dstack, gmm, n_comp, window_size, tolerance):
    #X=prepare_X(arr_1, arr_2)
    X = prepare_X_all(arr_dstack)
    shp=arr_1.shape
    
    
    for j in range(0, n_comp):
        proba = gmm.predict_proba(X)[:,j]
        
        prob_surface_modelling.plot_3D_prob_surface(proba.reshape(shp[0],shp[1]), shp)
        
        im=plt.imshow(proba.reshape(shp[0],shp[1]), cmap='RdYlBu')
        #im=plt.imshow(res.reshape(shp[0],shp[1]), cmap='RdYlBu')
        
        plt.xlabel('Range')
        plt.ylabel('Azimuth')
        
        plt.colorbar(label='Probability: Class_'+str(j))
        
        plt.tight_layout()
        
        plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Segmentation/Posterior/'+'Seg_proba_window_{}_inc_corr_{}_tol_{}_classes_{}_class_{}.tiff'.format(window_size, correction_switch, tolerance, n_comp, j), dpi=300,bbox_inches='tight')
        
        plt.show()
    

def plot_result(arr_dstack, gmm, n_comp):
    #X=prepare_X(arr_1, arr_2)
    X = prepare_X_all(arr_dstack)
    shp=arr_dstack.shape[:2]
    res=gmm.predict(X).reshape(shp[0],shp[1])
    

    #proba = gmm.predict_proba(X)[:,j]
    im=plt.imshow(res.reshape(shp[0],shp[1]), cmap='RdYlBu')
    #im=plt.imshow(res.reshape(shp[0],shp[1]), cmap='RdYlBu')
    
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    values = np.unique(res.ravel())
    #plot_legend(im)
    colors = [ im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label="Segment {l}".format(l=int(values[i])) ) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    #plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.15),
          #fancybox=True, shadow=True, ncol=len(values), prop={'size': 6} ) 
    #plt.colorbar()
    #plt.tight_layout()
    #plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Segmentation/Posterior/'+'Seg_proba_window_9_segment_2_inc_corr_'+str(correction_switch)+'_tol_e_4_class_'+str(j)+'.tiff', dpi=300,bbox_inches='tight')
    
    #plt.show()
    
    #plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Segmentation/'+'Segmentation_window_9_segment_2_inc_corr_no_tol_e_4'+'.tiff', dpi=300,bbox_inches='tight')
    
    #plt.show()

def plot_legend(res,im,ax):
    values = np.unique(res.ravel())
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    #plot_legend(im)
    colors = [ im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label="Segment {l}".format(l=int(values[i])) ) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.20),
          fancybox=True, shadow=True, ncol=len(values), prop={'size': 6} ) 
    #plt.legend(handles=patches, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0. )


def visualize_feature_space(X, Y_, means, covariances, title):
    splot = plt.subplot(1, 1, 1)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.001, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle,  edgecolor='k', fill=False)# color=color,
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.2)
        splot.add_artist(ell)
    plt.xlabel('$\lambda_{3}$')
    plt.ylabel('Co-Pol_diff')
    #plt.xlim(-9., 5.)
    #plt.ylim(-3., 6.)
    #plt.xticks(())
    #plt.yticks(())
    plt.title(title)
    #plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Segmentation/'+'feature_space_Segmentation_window_9_segment_2_tol_e_4_inc_corr_no'+'.tiff', dpi=300, bbox_inches='tight')
    plt.show()

def majority(arr):
    max_arr=np.bincount(arr.flatten())
    pos=np.where(max_arr==np.amax(max_arr))
    return pos[0][0]

def majority_smoothening(arr_1, arr_2, n_components, window_size):
    gmm=gmm_fitting(arr_1, arr_2, n_components, tolerance=0.0001)
    X=prepare_X(arr_1, arr_2)
    shp=arr_1.shape
    seg_image=gmm.predict(X).reshape(shp[0],shp[1])
    #print(seg_image.dtype)
    oil_class=seg_image[300,600]
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
    
    
    #ascent = misc.ascent()
    #plt.imshow(ascent, cmap='gray')
    #plt.show()
    #print(ascent.shape)

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
            if((np.amin(a)==0) & (a[1,1]==1)):
                res_bound[i+1, j+1]=1

    return res_bound

def is_Bimodal(pol, res_2,res_3):
    
    class_2_Segment_0_num = int(input('class_2_Segment_0_num '))
    class_2_Segment_1_num = int(input('class_2_Segment_1_num '))
    class_3_Segment_0_num = int(input('class_3_Segment_0_num '))
    class_3_Segment_1_num = int(input('class_3_Segment_1_num '))
    class_3_Segment_2_num = int(input('class_3_Segment_2_num '))
    
    only_oil=res_2[0]
    plt.subplot(131)
    plt.imshow(only_oil, cmap='gray_r')
    #plt.show()
    
    matplotlib.rcParams.update({'font.size': 5})
    gridspec.GridSpec(2,2)
    
    #oil_class_pos=np.where(only_oil==1)
    oil_class_pos=np.where(only_oil==class_2_Segment_1_num)
    #water_class_pos=np.where(only_oil==0)
    water_class_pos=np.where(only_oil==class_2_Segment_0_num)
    #plt.subplot(132)
    plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=1)
    plt.hist(10*np.log10(pol[...,3])[oil_class_pos].flatten(),bins=200, rwidth=0.5,histtype='step', label='PD (Segment 1)', color='b')
    plt.hist(10*np.log10(pol[...,3])[water_class_pos].flatten(),bins=200, rwidth=0.5,histtype='step', label='PD (Segment 0)', color='r')
    plt.hist(10*np.log10(pol[...,3]).flatten(),bins=200, rwidth=0.5,histtype='step', label='PD_full', color='k')
    plt.xlabel('Co-Polarization difference (dB)')
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    
    #plt.subplot(133)
    plt.subplot2grid((2,2), (1,0), colspan=1, rowspan=1)
    plt.hist(10*np.log10(pol[...,2])[oil_class_pos].flatten(),bins=200, rwidth=0.5,histtype='step', label='$\lambda_{3}$ (Segment 1)',color='b')
    plt.hist(10*np.log10(pol[...,2])[water_class_pos].flatten(),bins=200, rwidth=0.5,histtype='step', label='$\lambda_{3}$ (Segment 0)',color='r')
    plt.hist(10*np.log10(pol[...,2]).flatten(),bins=200, rwidth=0.5,histtype='step', label='$\lambda_{3}$_full', color='k')
    plt.xlabel('$\lambda_{3}$ (dB)')
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    #plt.show()
    
    only_oil=res_3[0]
    #plt.subplot(131)
    #plt.subplot2grid((2,3), (0,0), colspan=2, rowspan=2)
    
    #im=plt.imshow(only_oil, cmap='RdYlBu')
    #plt.xlabel('Range')
    #plt.ylabel('Azimuth')
    #plt.title('Segmented image; n_classes=3')
    #plot_legend(only_oil,im)
    
    #plt.show()
    
    #oil_class_pos=np.where(only_oil==0)
    #water_class_1_pos=np.where(only_oil==2)
    #water_class_2_pos=np.where(only_oil==1)
    
    oil_class_pos=np.where(only_oil==class_3_Segment_0_num)
    water_class_1_pos=np.where(only_oil==class_3_Segment_1_num)
    water_class_2_pos=np.where(only_oil==class_3_Segment_2_num)
    
    #plt.subplot(132)
    
    plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1)
    
    plt.hist(10*np.log10(pol[...,3])[oil_class_pos].flatten(),bins=200, rwidth=0.5,histtype='step', label='PD (Segment 0)', color='r')
    plt.hist(10*np.log10(pol[...,3])[water_class_1_pos].flatten(),bins=200, rwidth=0.5,histtype='step', label='PD (Segment 2)', color='b')
    plt.hist(10*np.log10(pol[...,3])[water_class_2_pos].flatten(),bins=200, rwidth=0.5,histtype='step', label='PD (Segment 1)', color='y')
    plt.hist(10*np.log10(pol[...,3]).flatten(),bins=200, rwidth=0.5,histtype='step', label='PD_full', color='k')
    plt.xlabel('Co-Polarization difference (dB)')
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    
    #plt.subplot(133)
    plt.subplot2grid((2,2), (1,1), colspan=1, rowspan=1)
    
    plt.hist(10*np.log10(pol[...,2])[oil_class_pos].flatten(),bins=200, rwidth=0.5,histtype='step', label='$\lambda_{3}$ (Segment 0)', color='r')
    plt.hist(10*np.log10(pol[...,2])[water_class_1_pos].flatten(),bins=200, rwidth=0.5,histtype='step', label='$\lambda_{3}$ (Segment 2)', color='b')
    plt.hist(10*np.log10(pol[...,2])[water_class_2_pos].flatten(),bins=200, rwidth=0.5,histtype='step', label='$\lambda_{3}$ (Segment 1)', color='y')
    plt.hist(10*np.log10(pol[...,2]).flatten(),bins=200, rwidth=0.5,histtype='step', label='$\lambda_{3}$_full', color='k')
    plt.xlabel('$\lambda_{3}$ (dB)')
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Segmentation/'+'gassian_mixtres_class_num=3_Segment_labels'+'.tiff', dpi=300, bbox_inches='tight')
    plt.show()
    #print(only_oil.shape)
    #print(pol.shape)
    
    
    
    #print(only_oil.shape)
    #print(pol.shape)

def superimpose_bound(arr_master, arr_slave, window):
    shp_master=arr_master.shape
    shp_slave=arr_slave.shape
    slave_row, slave_col=0,0
    slave_bound_addr=np.where(arr_slave==1)
    master_row=slave_bound_addr[0]+np.floor(window/2)
    master_col=slave_bound_addr[1]+np.floor(window/2)
    arr_master[master_row.astype(int), master_col.astype(int)]=10
    return arr_master

def get_gdal_object(newname, newRasterXSize, newRasterYSize, bands, fill_array, projection,geotransform):
    #def reproject_new(newname, newRasterXSize, newRasterYSize, bands, fuse_array, projection,geotransform): #makes the array into a raster image
    driver=gdal.GetDriverByName('GTiff')
    newdataset=driver.Create(newname+'.tif',newRasterXSize,newRasterYSize,bands, gdal.GDT_UInt16)
    #newdataset.SetProjection(projection)
    newdataset.SetGeoTransform(geotransform)
    newdataset.GetRasterBand(1).WriteArray(fill_array)
    #newdataset.FlushCache()
    return newdataset

def reproject_slicks(filename,slick_arr):
    reproject.save_tiff_image_1(filename, slick_arr)


def polygonize(arr, dst_filename):
    #shpp=arr.shape
    #gdal_image=get_gdal_object('test', shpp[1], shpp[0],1,arr, None, [0,1,0,0,0,1])
    gdal_image=gdal.Open('Output/slick_raster.tif')
    src_band=gdal_image.GetRasterBand(1)
    mask_band=src_band
    #mask_band=None
    format='ESRI Shapefile'
    dst_layername = 'out'
    
    #if dst_ds is None:
    drv = ogr.GetDriverByName(format)
    #if not quiet_flag:
        #print('Creating output %s of format %s.' % (dst_filename, format))
    dst_ds = drv.CreateDataSource( dst_filename )
    
    #if dst_layer is None:

    srs = None
    if gdal_image.GetProjectionRef() != '':
        srs = osr.SpatialReference()
        srs.ImportFromWkt( gdal_image.GetProjectionRef())
    
    dst_layer = dst_ds.CreateLayer(dst_layername, srs = srs )
    dst_fieldname = 'DN'
    fd = ogr.FieldDefn( dst_fieldname, ogr.OFTInteger )
    dst_layer.CreateField( fd )
    dst_field = 0
    result = gdal.Polygonize( src_band, mask_band, dst_layer, dst_field, [], callback = None)

def get_seg_mask(degree=1,ncomp=4,inci_corr=True,window_size_fea=9,window_size_smoo=43,window_size_boun=3):
    #mlc_orig=cov_arr=extract_polarimetric.extract_covariance_arr(1, True, 1)
    
    pol=read_Pol_features(window_size_fea, inci_corr, degree)
    arr_1=rescale(pol[...,2])
    arr_2=rescale(pol[...,3])
    res=majority_smoothening(arr_1, arr_2,ncomp, window_size_smoo)
    #bound_arr=boundary_oil(res[2], window_size_boun)
    return res

def plot_majority(res, pol, bound_arr, mlc_orig):
    #mpl.rcParams.update({'font.size': 5})
    fig=plt.figure(dpi = 150, tight_layout = True)
    #plt.figure(dpi = 150, tight_layout=True)

    ax = plt.subplot(1,4,1)
    
    ax.imshow(10*np.log(pol[...,2]), cmap='gray')
    #plt.colorbar(label='dB', orientation='vertical')
    #plt.title('Lowest eigen value ($\lambda_{3}$)')
    ax.set_xlabel('Range')
    ax.set_ylabel('Azimuth')
    
    ax = plt.subplot(1,4,2)
    
    im=ax.imshow(res[0], cmap='RdYlBu')
    #plt.colorbar()
    plot_legend(res[0], im,ax)
    ax.set_xlabel('Range')
    ax.set_ylabel('Azimuth')
    #plt.title('Segmented image')

    ax = plt.subplot(1,4,3)
    im=ax.imshow(res[1], cmap='RdYlBu')
    plot_legend(res[1], im, ax)
    ax.set_xlabel('Range')
    ax.set_ylabel('Azimuth')
    #plt.title('Segmented classes after smoothening')
    
    ax = plt.subplot(1,4,4)
    im=ax.imshow(res[2], cmap='gray_r')
    plot_legend(res[2], im,ax)
    ax.set_xlabel('Range')
    ax.set_ylabel('Azimuth')
    #plt.title('Oil spills')
    
    #plt.subplot(2,3,5)
    #plt.imshow(bound_arr, cmap='gray_r')
    #plt.title('Slick Boundries')
    #plt.xlabel('Range')
    #plt.ylabel('Azimuth')
    ##plt.colorbar()
    #plt.subplot(2,3,6)
    ##plt.imshow(np.log(pol[...,2]), cmap='gray', alpha=1)
    ##plt.imshow(superimpose_bound(np.zeros((pol[...,2].shape)), bound_arr, window_size_smoo), cmap='jet', alpha =0.5)
    #plt.imshow(np.pad(bound_arr,25,'constant'), cmap='jet', alpha=0.5)
    #plt.imshow(10*np.log(np.real(mlc_orig[...,2,2])), cmap='gray', alpha=1)
    
    ##plt.colorbar()
    #plt.title('Slick boundries overlaid over image')
    #plt.xlabel('Range')
    #plt.ylabel('Azimuth')
    
    plt.tight_layout()
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/EPFS_4.tiff', dpi=300, papertype='a4', bbox_inches='tight')#, bbox_inches='tight')
    plt.show()


if __name__=='__main__':
    # For the outpur sent to mam
    os.chdir('../North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01/MLC_Python_New/norway_00709_15092_000_150610_L090_CX_01_mlc')
    
    #======================GLCM======================
    #os.chdir('Output')
    #glcm = read_GLCM_C33_features()
    #next(glcm)
    #plt.imshow(next(glcm)[:-1,:-1])
    #plt.show()
    
    #os.chdir('../')
    glcm = contextual_stack()
    print(glcm.shape)
    
    #--------------------------------------------------
    
    
    degree=1
    ncomp=3
    tolerance=0.0001
    num_initialization=1
    max_iteration=200
    
    correction_switch=True
    window_size_fea=9
    window_size_smoo=35
    window_size_boun=3
    
    '''
    degree=1
    ncomp=2
    inci_corr=False
    window_size_fea=9
    window_size_smoo=50
    window_size_boun=3
    '''
    
    
    mlc_orig=cov_arr=extract_polarimetric.extract_covariance_arr(1, True, 1)
    
    arr_dstack = read_Pol_features(window_size_fea, correction_switch, degree, rescale_switch = True, log_Transform_switch = False)
    
    print(arr_dstack.shape)
    
    #================get GLCM features==================
    feature = np.dstack((arr_dstack, glcm))
    print (feature.shape)
    
    #=========Visualize_histogram============
    
    
    #visualize_hist(feature) 
    
    #sys.exit()
    
    #===========Visualize_feature_space=======
    arr_1 = arr_dstack[...,2]
    arr_2 = arr_dstack[...,3]
    res=majority_smoothening(arr_1, arr_2,ncomp, window_size_smoo)
    
    
    bound_arr=boundary_oil(res[2], window_size_boun)
    
    #=====Plotting histogram for class oil regions for PD and $\lambda_{3}$ feature for num_comp=2
    pol=read_Pol_features(window_size_fea, False, degree, rescale_switch = False, log_Transform_switch = False)
    
    #res_2=majority_smoothening(arr_1, arr_2,2, window_size_smoo)
    #plt.imshow(res_2[0], cmap = 'RdYlBu')
    #plt.show()
    
    #res_3=majority_smoothening(arr_1, arr_2,3, window_size_smoo)
    #plt.imshow(res_3[0], cmap = 'RdYlBu')
    #plt.show()
    #is_Bimodal(pol, res_2,res_3)
    
    
    ##SAVE_boundary array
    #np.save('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Segmentation/boundary_arr_from_MLC.npy', bound_arr)
    #np.save('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Segmentation/oil_segments.npy', res[2])

    #==========Segmentation=========
    #arr_seg = arr_dstack[...,[2,3]]
    #arr_seg = feature[...,[2,3]]
    
    #gmm = gmm_fitting_1(arr_seg, ncomp, tolerance=tolerance, num_initial=num_initialization, max_iteration = max_iteration)
    #gmm = gmm_fitting(arr_1,arr_2, ncomp, tolerance=tolerance, num_initial=num_initialization, max_iteration = max_iteration)
    
    #plot_result(arr_seg, gmm, ncomp)
    
    #plot_soft_probability(arr_seg, gmm, ncomp, window_size_fea, tolerance)
    
    #X=prepare_X(arr_1, arr_2)
    #X = prepare_X_all(arr_seg)
    #==========Visualize feature space=========
    
    
    
    #class_3_Segment_0_num = int(input('class_3_Segment_0_num '))
    #class_3_Segment_1_num = int(input('class_3_Segment_1_num '))
    #class_3_Segment_2_num = int(input('class_3_Segment_2_num '))
    
    #if (class_3_Segment_0_num==0 and class_3_Segment_1_num ==1 and class_3_Segment_2_num ==2):
        #color_iter = itertools.cycle(['r', 'y', 'b', 'gold', 'darkorange','navy',])
    #if (class_3_Segment_0_num==0 and class_3_Segment_1_num ==2 and class_3_Segment_2_num ==1):
        #color_iter = itertools.cycle(['r', 'b', 'y', 'gold', 'darkorange','navy',])
    #if (class_3_Segment_0_num==1 and class_3_Segment_1_num ==0 and class_3_Segment_2_num ==2):
        #color_iter = itertools.cycle(['y', 'r', 'b', 'gold', 'darkorange','navy',])
    #if (class_3_Segment_0_num==1 and class_3_Segment_1_num ==2 and class_3_Segment_2_num ==0):
        #color_iter = itertools.cycle(['y', 'b', 'r', 'gold', 'darkorange','navy',])
    #if (class_3_Segment_0_num==2 and class_3_Segment_1_num ==0 and class_3_Segment_2_num ==1):
        #color_iter = itertools.cycle(['b', 'r', 'y', 'gold', 'darkorange','navy',])
    #if (class_3_Segment_0_num==2 and class_3_Segment_1_num ==1 and class_3_Segment_2_num ==0):
        #color_iter = itertools.cycle(['b', 'y', 'r', 'gold', 'darkorange','navy',])
    
    #visualize_feature_space(X, gmm.predict(X), gmm.means_, gmm.covariances_, 'Segmentation in Feature Space')
    
    #print(gmm.means_)
    #print(gmm.covariances_)
    #print(gmm.weights_)
    
    
    
    
    #============MAJORITY SMOTTHENING Plots=========
    plot_majority(res, pol, bound_arr, mlc_orig)
    
    #===========Reproject_SLICK_Raster==========
    #filename='slick_raster'
    #slick_arr=res[2]
    #reproject_slicks(filename,np.pad(res[2],25,'constant'))
    
    #=========Polygonization============
    #polygonize(1, '/home/anurag/Documents/MScProject/SAR/OilSpill/North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01/MLC_Python_New/norway_00709_15092_000_150610_L090_CX_01_mlc/Output/slick_polygons.shp')
    
    
    #==========EXPLORATION=======
    #plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,'Gaussian Mixture')
    #a=contextual_stack()
    #print (a.shape)
    #print (next(a))
    #print (next(a))
    #visualize_hist()
    #plot_scatter(9, False)
    #polygonize(np.array([[0,0,1,0,0],[0,1,1,1,0], [0,1,1,1,0], [0,1,1,1,0], [0,0,0,0,0]]), '/home/anurag/Desktop/test_polygonize/test_prog/output_4.shp')
    