import matplotlib.pyplot as plt
#import extract_polarimetric
import numpy as np
import matplotlib.patches as patches
from PIL import Image
#import incidence_angle_corr
#from matplotlib import rc
#rc('font', **{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
from scipy import signal
import matplotlib.patches as patches
#import read_RISAT1
import math
from math import pi
#import fit_inci_model
import plotting
import read_binary
import extract_polarimetric
from numpy import linalg as LA
import os

file_name_VV="norway_00709_15091_012_150610_L090VV_CX_02.slc"
file_name_HV="norway_00709_15091_012_150610_L090HV_CX_02.slc"
file_name_HH="norway_00709_15091_012_150610_L090HH_CX_02.slc"

def get_scattering_vector(file_name_HH,file_name_HV,file_name_VV,multilook_switch=True, multilook_x=3, multilook_y=12, scan_lines=88086, scan_pix=9900, cropping_list=[.11,.49,.57,.70], is_list_ratio=True):
    #file_name_VV="norway_00709_15091_012_150610_L090VV_CX_02.slc"
    #file_name_HV="norway_00709_15091_012_150610_L090HV_CX_02.slc"
    #file_name_HH="norway_00709_15091_012_150610_L090HH_CX_02.slc"
    slc_HH=read_binary.read_SLC(file_name_HH, scan_lines, scan_pix, cropping_list, is_list_ratio)
    slc_VV=read_binary.read_SLC(file_name_VV, scan_lines, scan_pix, cropping_list, is_list_ratio)
    slc_HV=read_binary.read_SLC(file_name_HV, scan_lines, scan_pix, cropping_list, is_list_ratio)
    if(multilook_switch==True):
        mlc_HH=read_binary.multilooking(multilook_x, multilook_y, slc_HH)
        mlc_VV=read_binary.multilooking(multilook_x, multilook_y, slc_VV)
        mlc_HV=read_binary.multilooking(multilook_x, multilook_y, slc_HV)
        return np.dstack((mlc_HH, mlc_HV, mlc_VV))
    return np.dstack((slc_HH, np.sqrt(2)*slc_HV, slc_VV))

#def plot_decomp(decomp_arr):

def get_covariance_matrix(slc_arr, multilook_switch=True, multilook_x=3, multilook_y=12):
    #slc_arr=get_scattering_vector(multilook_switch=False)
    s0,s1,s2=slc_arr[...,0], slc_arr[...,1], slc_arr[...,2]
    new_shape=s0.shape
    return np.dstack((s0*np.conj(s0), s0*np.conj(s1), s0*np.conj(s2), s1*np.conj(s0), s1*np.conj(s1), s1*np.conj(s2), s2*np.conj(s0), s2*np.conj(s1), s2*np.conj(s2))).reshape(new_shape[0],new_shape[1],3,3)   

def eigen_val_decomposition(C3):
    w=LA.eigvals(C3)
    eigen_arr=np.sort(np.absolute(w), axis=2)
    ent=extract_polarimetric.entropy(eigen_arr)
    return ent

def pedestal_height(C3):
    eig=eigen_val_decomposition(C3)
    w=LA.eigvals(C3)
    eigen_arr=np.sort(np.absolute(w), axis=2)
    p_height=eigen_arr[...,0]/eigen_arr[...,2]
    #print(eigen_arr)
    return p_height

def cloude_pottier_angles(C3):
    alpha=1

def Pauli_RGB_array(S, plot_switch, multilook_switch, win_x, win_y):
    b=10*np.log10(np.absolute(S[...,0]+S[...,2])/np.sqrt(2))
    r=10*np.log10(np.absolute(S[...,0]-S[...,2])/np.sqrt(2))
    g=10*np.log10(np.absolute(S[...,1]))
    img_array_dB=np.dstack((r,g,b))
    if(multilook_switch==True):
        r=read_binary.multilooking(win_x, win_y, r)
        g=read_binary.multilooking(win_x, win_y, g)
        b=read_binary.multilooking(win_x, win_y, b)
        img_array_dB=np.dstack((r,g,b))
    if(plot_switch==True):
        b=plotting.hist_stretch_all(b,0,True)
        r=plotting.hist_stretch_all(r,0,True)
        g=plotting.hist_stretch_all(g,0,True)
        #print([r,g,b])
        #print(r.shape)
        #plt.figure(dpi = 150, tight_layout=True)
        plt.imshow(np.dstack((np.real(r),np.real(g),np.real(b))))
        #plt.colorbar(orientation='vertical', label='dB')
        plt.xlabel('Range')
        plt.ylabel('Azimuth')
        plt.title('Pauli Decomposition')
        #plt.show()
    return img_array_dB



def plot_decomposition_comp(decomp_arr, stretch_switch, comp_names, title, plot_pos_list):
    if(stretch_switch==True):
        bits=0
        clip_extremes=True
        decomp_arr[...,0]=plotting.hist_stretch_all(decomp_arr[...,0],bits, clip_extremes)
        decomp_arr[...,1]=plotting.hist_stretch_all(decomp_arr[...,1],bits, clip_extremes)
        decomp_arr[...,2]=plotting.hist_stretch_all(decomp_arr[...,2],bits, clip_extremes)
    
    #plt.figure(dpi = 150, tight_layout=True)
    #plt.suptitle(title)
    plt.subplot2grid((3,3), (0,2))
    #plt.subplot(plot_pos_list[0])
    plt.imshow(np.real(decomp_arr[...,0]), cmap='Blues')
    plt.title(comp_names[0])
    plt.colorbar(orientation='vertical', label='dB')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    
    #plt.subplot(plot_pos_list[1])
    plt.subplot2grid((3,3), (1,2))
    plt.imshow(np.real(decomp_arr[...,1]), cmap='Reds')
    plt.title(comp_names[1])
    plt.colorbar(orientation='vertical', label='dB')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    
    #plt.subplot(plot_pos_list[2])
    plt.subplot2grid((3,3), (2,2))
    plt.imshow(np.real(decomp_arr[...,2]), cmap='Greens')
    plt.title(comp_names[2])
    plt.colorbar(orientation='vertical', label='dB')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    #img.show()
    #plt.imshow(g)
    #plt.show()

def krogager_array(S, plot_switch, multilook_switch, win_x, win_y):
    iota=1j
    Shv=S[...,1]
    Shh=S[...,0]
    Svv=S[...,2]
    Srr=iota*Shv+.5*(Shh-Svv)
    Sll=iota*Shv-.5*(Shh-Svv)
    Srl=(iota/2)*(Shh+Svv)
    ks=10*np.log10(np.absolute(Srl))
    kd=10*np.log10(np.minimum(np.absolute(Srr), np.absolute(Sll)))
    kh=10*np.log10(np.maximum(np.absolute(Srr), np.absolute(Sll))-np.minimum(np.absolute(Srr), np.absolute(Sll)))
    phi_rr=np.angle(Srr)
    phi_ll=np.angle(Sll)
    phi_rl=np.angle(Srl)
    theta=(phi_rr-phi_ll-np.pi)/4
    #phi=(phi_rr+phi_ll+np.pi)/2
    phi_s=phi_rl-(phi_rr+phi_ll+np.pi)/2
    
    img_arr_dB=np.dstack((kd,kh,ks))
    
    if(multilook_switch==True):
        kd=read_binary.multilooking(win_x, win_y, kd)
        kh=read_binary.multilooking(win_x, win_y, kh)
        ks=read_binary.multilooking(win_x, win_y, ks)
        img_array_dB=np.dstack((kd,kh,ks))
    if (plot_switch==True):
        bits=0
        clip_extremes=True
        #img_arr1=img_arr
        kd=plotting.hist_stretch_all(kd,bits, clip_extremes)
        kh=plotting.hist_stretch_all(kh,bits, clip_extremes)
        ks=plotting.hist_stretch_all(ks,bits, clip_extremes)
        #plt.figure(dpi = 150, tight_layout=True)
        plt.imshow(np.dstack((np.real(kd),np.real(kh),np.real(ks))))
        plt.xlabel('Range')
        plt.ylabel('Azimuth')
        plt.title('Krogagar Decomposition ($R: |k_{d}|, G: |k_{h}|, B: |k_{s}| $)')
        #plt.show()
    return img_array_dB



def save_92_000_SLC_MLC():
    
    #print(os.getcwd())
    file_name_VV="norway_00709_15092_000_150610_L090VV_CX_02.slc"
    file_name_HV="norway_00709_15092_000_150610_L090HV_CX_02.slc"
    file_name_HH="norway_00709_15092_000_150610_L090HH_CX_02.slc"
    mlc_row_looks=12
    mlc_col_looks=3
    scan_lines=86417
    scan_pix=9900
    mlc_cropping_list=[521,1545,4049,5233]
    slc_cropping_list=[mlc_cropping_list[0]*mlc_col_looks,\
        mlc_cropping_list[1]*mlc_col_looks,\
            mlc_cropping_list[2]*mlc_row_looks,\
                mlc_cropping_list[3]*mlc_row_looks]
    
    
    S=get_scattering_vector(file_name_HH,file_name_HV,file_name_VV,multilook_switch=False, multilook_x=3, multilook_y=12, scan_lines=scan_lines, scan_pix=scan_pix, cropping_list=slc_cropping_list, is_list_ratio=False)
    
    S_mlc=get_scattering_vector(file_name_HH,file_name_HV,file_name_VV,multilook_switch=True, multilook_x=3, multilook_y=12, scan_lines=scan_lines, scan_pix=scan_pix, cropping_list=slc_cropping_list, is_list_ratio=False)
    np.save('S_oil_SLC.npy', S)
    np.save('S_mlc_only_oil.npy', S_mlc)



if __name__=='__main__':
    #============SAVING SLC and multilooked MLC================
    os.chdir('/home/anurag/Documents/MScProject/SAR/OilSpill/North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_02')
    S=np.load('S_oil_SLC.npy')
    S_mlc=np.load('S_mlc_only_oil.npy')
    
    Pauli_RGB_array(S_mlc, True)
    
    
    
    
    '''
    #==============Eigen Value decomposition========
    #C3_arr=np.load('C3_full.npy')

    C3_looked=read_binary.multilook_C3(3, 12, C3_arr)
    C3=get_covariance_matrix(S,multilook_switch=False)
    #np.save('C3_looked_12_3_stride_win_true.npy', C3_looked)
    #print(C3_looked)
    #C3_looked_arr=np.load('C3_looked_12_3_stride_win_true.npy')
    ent=eigen_val_decomposition(C3_looked_arr)
    p_height=pedestal_height(C3_looked_arr)
    #print(p_height)
    plt.imshow(ent, cmap='RdYlBu')
    #plt.imshow(10*np.log10(np.absolute(C3_looked_arr[...,2,2])), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.colorbar()
    plt.show()
    del(C3_looked_arr)
    '''
    #print(S)
    pauli_arr=Pauli_RGB_array(S, False)
    #krog_arr=krogager_array(S, True)
    plot_decomposition_comp(krog_arr, False, ['$k_{d}}$', '$k_{h}}$', '$k_{s}}$'], 'Krogager Decomposition', [131,132,133])
    #plot_decomposition_comp(pauli_arr, False, [r'|$\alpha$|', r'|$\beta$|', r'|$\gamma$|'], 'Pauli Decomposition')
    #Pauli_RGB_array(S)
    #np.save('S_oil.npy', S)