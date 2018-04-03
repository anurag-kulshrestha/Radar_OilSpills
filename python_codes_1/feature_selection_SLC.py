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
import read_binary
import feature_selection

def test_convolve(arr, kernal):
    grad = signal.convolve2d(arr, kernal, boundary='symm', mode='valid')
    return grad

def kernal(window_size_x, window_size_y):
    k=np.ones((window_size_y,window_size_x))#.reshape(window_size_x, window_size_y)
    normalize_k=k/(window_size_x*window_size_y)
    return normalize_k


'''
def get_scattering_vector(multilook_switch=True, multilook_x=3, multilook_y=12, scan_lines=86417, scan_pix=9900, cropping_list=[.11,.49,.57,.70]):
    #return read_binary.read_SLC(file_name, scan_lines, scan_pix, cropping_list, False)
    #def get_scattering_vector(multilook_switch=True, multilook_x=3, multilook_y=12, scan_lines=88086, scan_pix=9900, cropping_list=[.11,.49,.57,.70]):
    file_name_VV="norway_00709_15091_012_150610_L090VV_CX_02.slc"
    file_name_HV="norway_00709_15091_012_150610_L090HV_CX_02.slc"
    file_name_HH="norway_00709_15091_012_150610_L090HH_CX_02.slc"
    slc_HH=read_binary.read_SLC(file_name_HH, scan_lines, scan_pix, cropping_list)
    slc_VV=read_binary.read_SLC(file_name_VV, scan_lines, scan_pix, cropping_list)
    slc_HV=read_binary.read_SLC(file_name_HV, scan_lines, scan_pix, cropping_list)
    if(multilook_switch==True):
        mlc_HH=read_binary.multilooking(multilook_x, multilook_y, slc_HH)
        mlc_VV=read_binary.multilooking(multilook_x, multilook_y, slc_VV)
        mlc_HV=read_binary.multilooking(multilook_x, multilook_y, slc_HV)
        return np.dstack((mlc_HH, mlc_HV, mlc_VV))
    return np.dstack((slc_HH, np.sqrt(2)*slc_HV, slc_VV))
'''
def get_oil_slick_mask():
    #return np.load('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Segmentation/oil_segments.npy')
    return feature_selection.get_slick_wise_mask(1)

def stretch_mask_to_SLC(mlc_row_looks, mlc_col_looks, mask_id):
    mlc_mask=get_oil_slick_mask()[...,mask_id]
    row, col=np.where(mlc_mask==1)
    slc_ML_chunk=np.indices((12,3))
    
    slc_row_add=[i*mlc_row_looks for i in row]
    slc_row_add_1=[i+slc_ML_chunk[0] for i in slc_row_add]
    
    slc_col_add=[i*mlc_col_looks for i in col]
    slc_col_add_1=[i+slc_ML_chunk[1] for i in slc_col_add]
    
    return (np.array(slc_row_add_1).flatten(), np.array(slc_col_add_1).flatten())

def box_car_filtering(arr, window_size_x, window_size_y):
    return test_convolve(arr, kernal(window_size_x, window_size_y))


if __name__=='__main__':
    #cropping_list=[extent_xmin,extent_xmax, extent_ymin, extent_ymax]
    os.chdir('/home/anurag/Documents/MScProject/SAR/OilSpill/North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_02')
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
    file_name_HH="norway_00709_15092_000_150610_L090HH_CX_02.slc"
    file_name_HV="norway_00709_15092_000_150610_L090HV_CX_02.slc"
    
    pos_PO=stretch_mask_to_SLC(mlc_row_looks, mlc_col_looks,mask_id=0)
    pos_E40=stretch_mask_to_SLC(mlc_row_looks, mlc_col_looks,mask_id=1)
    pos_E60=stretch_mask_to_SLC(mlc_row_looks, mlc_col_looks,mask_id=2)
    pos_E80=stretch_mask_to_SLC(mlc_row_looks, mlc_col_looks,mask_id=3)
    
    pos_W_near=stretch_mask_to_SLC(mlc_row_looks, mlc_col_looks,mask_id=4)
    pos_W_far=stretch_mask_to_SLC(mlc_row_looks, mlc_col_looks,mask_id=6)
    pos_W_mid=stretch_mask_to_SLC(mlc_row_looks, mlc_col_looks,mask_id=5)
    
    #===========Speckle Filtering=================
    padding=((6,5),(1,1))
    #slc_VV=read_binary.read_SLC(file_name_VV, scan_lines, scan_pix, slc_cropping_list, False)
    slc_VV = np.load('S_oil_SLC.npy')[...,2]
    slc_VV_boxcar=np.pad(box_car_filtering(slc_VV, 3,12), padding, 'constant')
    
    #====SLC=====
    PO_slc=slc_VV[pos_PO]
    E40_slc=slc_VV[pos_E40]
    E60_slc=slc_VV[pos_E60]
    E80_slc=slc_VV[pos_E80]
    W_near_slc=slc_VV[pos_W_near]
    W_far_slc=slc_VV[pos_W_far]
    W_mid_slc=slc_VV[pos_W_mid]
    
    #====SLC - Speckle filtered product=====
    
    
    
    PO_fil=slc_VV_boxcar[pos_PO]
    E40_fil=slc_VV_boxcar[pos_E40]
    E60_fil=slc_VV_boxcar[pos_E60]
    E80_fil=slc_VV_boxcar[pos_E80]
    W_near_fil=slc_VV_boxcar[pos_W_near]
    W_far_fil=slc_VV_boxcar[pos_W_far]
    W_mid_fil=slc_VV_boxcar[pos_W_mid]

    #=========speckle==========
    slc_VV_spk_boxcar=np.absolute(slc_VV)-np.absolute(slc_VV_boxcar)
    
    PO_spk=np.absolute(PO_slc)-np.absolute(PO_fil)
    E40_spk=np.absolute(E40_slc)-np.absolute(E40_fil)
    E60_spk=np.absolute(E60_slc)-np.absolute(E60_fil)
    E80_spk=np.absolute(E80_slc)-np.absolute(E80_fil)
    W_near_spk=np.absolute(W_near_slc)-np.absolute(W_near_fil)
    W_far_spk=np.absolute(W_far_slc)-np.absolute(W_far_fil)
    W_mid_spk=np.absolute(W_mid_slc)-np.absolute(W_mid_fil)
    
    #creeate random samples:
    
    #print(PO_spk.shape)
    #print(E40_spk.shape)
    #print(E60_spk.shape)
    #print(E80_spk.shape)
    
    PO_spk_1=np.random.choice(PO_spk, 25000, replace=False)
    E40_spk_1=np.random.choice(E40_spk, 25000, replace=False)
    E60_spk_1=np.random.choice(E60_spk, 25000, replace=False)
    E80_spk_1=np.random.choice(E80_spk, 25000, replace=False)
    
    W_mid_spk_1 = np.random.choice(W_mid_spk, 25000, replace=False)
    
    
    #==========plotting the linear amplitude of slc and baxcar filtered product
    '''
    plt.hist(np.absolute(PO_slc),bins=300, rwidth=0.5,histtype='step', label='PO_slc')
    plt.hist(np.absolute(PO_fil),bins=300, rwidth=0.5,histtype='step', label='PO_fil')
    plt.hist(np.absolute(E40_slc),bins=300, rwidth=0.5,histtype='step', label='E40_slc')
    plt.hist(np.absolute(E40_fil),bins=300, rwidth=0.5,histtype='step', label='E40_fil')
    plt.hist(np.absolute(E60_slc),bins=300, rwidth=0.5,histtype='step', label='E60_slc')
    plt.hist(np.absolute(E60_fil),bins=300, rwidth=0.5,histtype='step', label='E60_fil')
    plt.hist(np.absolute(E80_slc),bins=300, rwidth=0.5,histtype='step', label='E80_slc')
    plt.hist(np.absolute(E80_fil),bins=300, rwidth=0.5,histtype='step', label='E80_fil')
    plt.legend()
    plt.show()
    
    #==========plotting the linear amplitude on dB scale of slc and baxcar filtered product
    plt.hist(10*np.log10(np.absolute(PO_slc)),bins=300, rwidth=0.5,histtype='step', label='PO_slc')
    plt.hist(10*np.log10(np.absolute(PO_fil)),bins=300, rwidth=0.5,histtype='step', label='PO_fil')
    plt.hist(10*np.log10(np.absolute(E40_slc)),bins=300, rwidth=0.5,histtype='step', label='E40_slc')
    plt.hist(10*np.log10(np.absolute(E40_fil)),bins=300, rwidth=0.5,histtype='step', label='E40_fil')
    plt.hist(10*np.log10(np.absolute(E60_slc)),bins=300, rwidth=0.5,histtype='step', label='E60_slc')
    plt.hist(10*np.log10(np.absolute(E60_fil)),bins=300, rwidth=0.5,histtype='step', label='E60_fil')
    plt.hist(10*np.log10(np.absolute(E80_slc)),bins=300, rwidth=0.5,histtype='step', label='E80_slc')
    plt.hist(10*np.log10(np.absolute(E80_fil)),bins=300, rwidth=0.5,histtype='step', label='E80_fil')
    plt.legend()
    plt.show()
    
    '''
    #plt.hist(PO_spk,bins=300, rwidth=0.5,histtype='step', label='PO_spk')
    #plt.hist(E40_spk,bins=300, rwidth=0.5,histtype='step', label='E40_spk')
    #plt.hist(E60_spk,bins=300, rwidth=0.5,histtype='step', label='E60_spk')
    #plt.hist(E80_spk,bins=300, rwidth=0.5,histtype='step', label='E80_spk')
    #plt.hist(W_near_spk,bins=300, rwidth=0.5,histtype='step', label='Water_near_spk')
    #plt.hist(W_far_spk,bins=300, rwidth=0.5,histtype='step', label='Water_far_spk')
    #plt.hist(W_mid_spk,bins=300, rwidth=0.5,histtype='step', label='Water_mid_spk')
    #plt.legend()
    ##plt.show()
    
    print('Slick \t Mean \t Variance')
    print('PO_spk \t'+ str(PO_spk.mean())+'\t'+str(PO_spk.var()))
    print('E40_spk \t'+ str(E40_spk.mean())+'\t'+str(E40_spk.var()))
    print('E60_spk \t'+ str(E60_spk.mean())+'\t'+str(E60_spk.var()))
    print('E80_spk \t'+ str(E80_spk.mean())+'\t'+str(E80_spk.var()))
    print('W_near_spk \t'+ str(W_near_spk.mean())+'\t'+str(W_near_spk.var()))
    print('W_far_spk \t'+ str(W_far_spk.mean())+'\t'+str(W_far_spk.var()))
    print('W_mid_spk \t'+ str(W_mid_spk.mean())+'\t'+str(W_mid_spk.var()))
    
    
    
    plt.hist(PO_spk_1,bins=500, rwidth=0.5,histtype='step', label='PO_spk')
    plt.hist(E40_spk_1,bins=500, rwidth=0.5,histtype='step', label='E40_spk')
    plt.hist(E60_spk_1,bins=500, rwidth=0.5,histtype='step', label='E60_spk')
    plt.hist(E80_spk_1,bins=500, rwidth=0.5,histtype='step', label='E80_spk')
    
    plt.hist(W_mid_spk_1,bins=500, rwidth=0.5,histtype='step', label='W_far_spk')
    plt.legend()
    plt.show()
    
    #============plot the multilooked speckle array========
    print(np.amin(slc_VV_spk_boxcar))
    print(np.amax(slc_VV_spk_boxcar))
    spk_plot=spk_multilooked=read_binary.multilooking_1(3, 12, slc_VV_spk_boxcar)
    #spk_plot = np.clip(np.real(spk_multilooked.flatten()),-0.3,0.3)
    #spk_plot_ma=ma.masked_where(spk_plot==np.nan, spk_plot)
    #print(spk_plot_ma)
    #print(np.amin(spk_plot_ma))
    #print(np.amax(spk_plot_ma))
    #import sys
    #sys.exit()
    
    
    plt.hist(np.real(spk_plot),bins=300, rwidth=0.5,histtype='step', label='spk_multilooked')
    plt.show()
    plt.imshow(np.real(spk_plot), cmap='gray')
    plt.colorbar()
    plt.show()
    #plt.subplot(121)
    #plt.imshow(10*np.log10(np.absolute(slc_VV)), cmap='gray')
    
    #plt.subplot(122)
    #plt.imshow(10*np.log10(np.absolute(slc_VV_boxcar)), cmap='gray')
    
    
    #I = np.dstack([im, im, im])
    #x = pos[1]
    #y = pos[0]
    #I[x, y, :] = [1, 0, 0]
    #plt.imshow(I)
    #plt.show()
    #plt.imshow(get_oil_slick_mask(), cmap='gray')
    #plt.show()
    #print(stretch_mask_to_SLC(mlc_row_looks, mlc_col_looks))
    
    
    #===========Co-pol Phase difference
    #slc_HH=read_binary.read_SLC(file_name_HH, scan_lines, scan_pix, slc_cropping_list, False)
    #co_pol_phase_diff=np.angle(slc_VV)-np.angle(slc_HH)
    
    