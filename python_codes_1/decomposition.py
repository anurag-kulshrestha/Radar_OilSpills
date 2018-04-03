import matplotlib.pyplot as plt
import extract_polarimetric
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
import fit_inci_model
import plotting
import os

#window_size, correction, degree

def Pauli_RGB_array(window_size, correction, degree):
    cov_arr=extract_polarimetric.extract_covariance_arr(window_size, correction, degree)
    clip_extremes=True
    ShhShh=cov_arr[:,:,0,0]
    SvvSvv=cov_arr[:,:,2,2]
    ShvShv=cov_arr[:,:,1,1]
    ShhSvv_=cov_arr[:,:,0,2]
    Shh_Svv=cov_arr[:,:,2,0]
    co_pol_sum_mag=ShhShh+SvvSvv+ShhSvv_+Shh_Svv
    co_pol_diff_mag=ShhShh+SvvSvv-ShhSvv_-Shh_Svv
    #b=plotting.hist_stretch_all(np.real(co_pol_sum_mag)/2, 0, clip_extremes)
    #r=plotting.hist_stretch_all(np.real(co_pol_diff_mag)/2, 0, clip_extremes)
    #g=plotting.hist_stretch_all(np.real(ShvShv), 0, clip_extremes)
    b=np.real(co_pol_sum_mag)/2
    r=np.real(co_pol_diff_mag)/2
    g=np.real(ShvShv)
    
    img_arr=np.dstack((r,g,b))
    #print(np.imag(co_pol_sum_mag))
    #print(np.imag(co_pol_diff_mag))
    return img_arr

def krogager_array(window_size, correction, degree):
    cov_arr=extract_polarimetric.extract_covariance_arr(window_size, correction, degree)
    clip_extremes=True
    ShhShh=cov_arr[:,:,0,0]
    SvvSvv=cov_arr[:,:,2,2]
    ShvShv=cov_arr[:,:,1,1]
    ShhSvv_=cov_arr[:,:,0,2]
    Shh_Svv=cov_arr[:,:,2,0]
    co_pol_sum_mag=ShhShh+SvvSvv+ShhSvv_+Shh_Svv
    co_pol_diff_mag=ShhShh+SvvSvv-ShhSvv_-Shh_Svv
    ShvShh_=cov_arr[:,:,1,0]/np.sqrt(2)
    ShvSvv_=cov_arr[:,:,1,2]/np.sqrt(2)
    Shv_Shh=cov_arr[:,:,0,1]/np.sqrt(2)
    Shv_Svv=cov_arr[:,:,2,1]/np.sqrt(2)
    
    iota=1j
    Srr_mag_complex=ShvShv+ 0.25*co_pol_diff_mag +iota*(ShvShh_-ShvSvv_-Shv_Shh-Shv_Svv)#jShv+0.5*(Shh-Svv))
    Srr_mod=np.absolute(Srr_mag_complex)
    Sll_mag_complex=ShvShv+ 0.25*co_pol_diff_mag -iota*(ShvShh_-ShvSvv_-Shv_Shh-Shv_Svv)#jShv-0.5*(Shh-Svv))
    Sll_mod=np.absolute(Sll_mag_complex)
    Srl_mag=ShhShh+SvvSvv+iota*(ShhSvv_+Shh_Svv)
    Srl_mod=np.absolute(Srl_mag)
    
    ks=Srl_mod
    #if(Srr_mod>Sll_mod):
    kd=np.minimum(Sll_mod, Srr_mod)
    kh=np.maximum(Sll_mod, Srr_mod)-kd
    #else:
        #kd=Srr_mod
        #kh=Sll_mod-Srr_mod
    return np.dstack((kd,kh,kh)) #mod-> k**2

def Freeman_Durdun_Decomposition_1(window_size, correction, degree):
    
    #, leaping_win=False, stride_row=1, stride_col=1
    cov_arr=extract_polarimetric.extract_covariance_arr(window_size, correction, degree)
    c11=cov_arr[:,:,0,0]
    c22=cov_arr[:,:,1,1]
    c33=cov_arr[:,:,2,2]
    c13=cov_arr[:,:,0,2]
    c13_re=np.real(c13)
    c13_im=np.imag(c13)
    shp=c11.shape
    rows=shp[0]
    cols=shp[1]
    
    alpha=beta=fv=fs=fd=np.empty((rows, cols), dtype=np.complex64)
    #=c11,c11,c11
    
    #Freeman algo
    #fv=1.5*c22
    #c11=c11-fv
    #c33=c33-fv
    #c13_re=c13_re-(fv/3)
    stride_row, stride_col=1,1
    for i in range(0, rows, stride_row):
        for j in range(0, cols, stride_col):
                fv[i,j]=1.5*c22[i,j]
                c11[i,j]=c11[i,j]-fv[i,j]
                c33[i,j]=c33[i,j]-fv[i,j]
                c13_re[i,j]=c13_re[i,j]-(fv[i,j]/3)
                #single bounce scattering dominates
                if(c13_re[i,j]>=0):
                    #print(i,j)
                    alpha[i,j]=-1
                    fd[i,j]=(c11[i,j]*c33[i,j] - c13_re[i,j]**2 - c13_im[i,j]**2)/(c11[i,j]+c33[i,j]-2*c13_re[i,j])
                    fs[i,j]=c33[i,j]-fd[i,j]
                    beta[i,j]=np.sqrt((fd[i,j]+c13_re[i,j]) * (fd[i,j]-c13_re[i,j]) + c13_im[i,j]**2)/fs[i,j]
                    #single bounce scattering dominates
                if(c13_re[i,j]<0):
                    #print(i,j)
                    beta[i,j]=1
                    fs[i,j]=(c11[i,j]*c33[i,j] - c13_re[i,j]**2 - c13_im[i,j]**2)/(c11[i,j]+c33[i,j]-2*c13_re[i,j])
                    fd[i,j]=c33[i,j]-fs[i,j]
                    alpha[i,j]=np.sqrt((fs[i,j]+c13_re[i,j]) * (fs[i,j]-c13_re[i,j]) + c13_im[i,j]**2)/fd[i,j]
        print(i)
    
    Ps=fs*(1+np.absolute(beta)**2)
    Pd=fd*(1+np.absolute(alpha)**2)
    Pv=8*fv/3
    return np.dstack((Pd,Pv,Ps))
    
def Freeman_Durdun_Decomposition(window_size, correction, degree):
    dbl_dir='norway_00709_15092_000_150610_L090_CX_01_mlc_FRE3_DBL'
    odd_dir='norway_00709_15092_000_150610_L090_CX_01_mlc_FRE3_ODD'
    vol_dir='norway_00709_15092_000_150610_L090_CX_01_mlc_FRE3_VOL'
    #print(os.getcwd())
    os.chdir('../'+dbl_dir)
    dbl=extract_polarimetric.read_Raster('C3', 'C11')+extract_polarimetric.read_Raster('C3', 'C22')+extract_polarimetric.read_Raster('C3', 'C33')
    
    os.chdir('../'+odd_dir)
    odd=extract_polarimetric.read_Raster('C3', 'C11')+extract_polarimetric.read_Raster('C3', 'C22')+extract_polarimetric.read_Raster('C3', 'C33')
    
    os.chdir('../'+vol_dir)
    vol=extract_polarimetric.read_Raster('C3', 'C11')+extract_polarimetric.read_Raster('C3', 'C22')+extract_polarimetric.read_Raster('C3', 'C33')
    
    return np.dstack((odd,dbl,vol))
    
    
    
    
if __name__=='__main__':
    a=1