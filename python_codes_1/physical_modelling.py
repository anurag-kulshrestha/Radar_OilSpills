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


#import the MLC product
def MLC_prod(window_size, correction_switch, degree):
    return extract_polarimetric.extract_covariance_arr(window_size, correction_switch, degree)
    

#import incidence angle
def get_inc_angle():
    return incidence_angle_corr.get_inc_ang_array()


#declare the Bragg coefficients
def Bragg_coeff(inc_angle, dielectric_const):
    RHH=(np.cos(inc_angle) - np.sqrt(dielectric_const - np.sin(inc_angle)**2))/\
        (np.cos(inc_angle) + np.sqrt(dielectric_const - np.sin(inc_angle)**2))
    
    RVV=(dielectric_const-1)*(np.sin(inc_angle)**2 - dielectric_const*(1+np.sin(inc_angle)**2))/\
        (dielectric_const*np.cos(inc_angle) + np.sqrt(dielectric_const - np.sin(inc_angle)**2))**2
    #return RVV
    return np.stack((RHH,RVV))

def get_dielectric_const(slick_type):
    water_dielectric_const=80+70j
    oil_dielectric_const=2.3+0.02j
    
    if(slick_type=='PO'):
        return oil_dielectric_const
    if(slick_type=='E40'):
        return .4*oil_dielectric_const + .6*water_dielectric_const
    if(slick_type=='E60'):
        return .6*oil_dielectric_const + .4*water_dielectric_const
    if(slick_type=='E80'):
        return .8*oil_dielectric_const + .2*water_dielectric_const

def get_W(kr,sigma_VV,RMS_slope, dielectric_const, inc_angle):
    
    psi=zeta=RMS_slope
    
    BRAGG=Bragg_coeff(inc_angle, dielectric_const)
    
    RHH=BRAGG[0]
    RVV=BRAGG[1]
    
    a=(np.sin(inc_angle+psi) * np.cos(zeta)/np.sin(inc_angle))**2
    
    b=(np.sin(zeta)/np.sin(inc_angle))**2
    
    #gamma_HH=np.absolute(a*RHH + b*RVV)**2
    
    gamma_VV=np.absolute(a*RVV + b*RHH)**2
    
    return sigma_VV[350:450,:]/(4*np.pi * kr**4 * np.cos(inc_angle)**4 * gamma_VV)

def main():
    os.chdir('../North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01/MLC_Python_New/norway_00709_15092_000_150610_L090_CX_01_mlc')
    
    lambd=0.24
    window_size, correction_switch, degree = 1, False, 1
    water_dielectric_const=80+70j
    oil_dielectric_const=2.3+0.02j
    kr=2*np.pi/lambd
    
    cov_arr=MLC_prod(window_size, correction_switch, degree)
    
    sigma_HH=10*np.log10(np.absolute(cov_arr[...,0,0]))
    
    sigma_VV=10*np.log10(np.absolute(cov_arr[...,2,2]))
    #sigma_VV=np.absolute(cov_arr[...,2,2])
    #co_pol_ratio=sigma_HH/sigma_VV
    
    #co_pol_w_po_e40=co_pol_ratio[350:450,:]
    
    co_pol_w_po_e40=(cov_arr[...,0,0]/cov_arr[...,2,2])[350:450,:]
    
    inc_angle=get_inc_angle()*np.pi/180 # in radians
    
    BRAGG=Bragg_coeff(inc_angle, water_dielectric_const)
    
    RVV=BRAGG[1]
    
    RHH=BRAGG[0]
    
    delta=(co_pol_w_po_e40*RVV - RHH)/(RVV-co_pol_w_po_e40*RHH)
    
    zeta=np.arcsin( np.sqrt(delta/(1+delta)) * np.sin(inc_angle) )
    
    psi=np.arcsin((np.sqrt(np.cos(zeta)**2)-np.cos(inc_angle)**2)/np.cos(zeta)) - inc_angle
    
    RMS_slope=np.sqrt((zeta**2+psi**2)/2)
    
    E40_dielectric_const = get_dielectric_const('E40')
    
    W_VV_water=get_W(kr,sigma_VV,RMS_slope.mean(), water_dielectric_const, inc_angle)
    
    
    
    #print(RMS_slope.mean())
    
    #plt.plot(np.imag(BRAGG[0]))
    #plt.plot(co_pol_w_po_e40.mean(0)**-1)
    
    plt.plot((cov_arr[...,0,0]/cov_arr[...,2,2])[350:450,:].mean(0))
    
    #plt.plot(sigma_VV[350:450,:].mean(0), label=r'$\sigma_{VV}^{0}$')
    #plt.plot(sigma_HH[350:450,:].mean(0),label=r'$\sigma_{HH}^{0}$')
    #print(np.absolute(RMS_slope).mean()*180/np.pi)
    
    
    #plt.hist(np.absolute(zeta).flatten()*180/np.pi,label=r'$\zeta(^{0})$', bins=200)
    
    #plt.hist(np.absolute(psi).flatten()*180/np.pi,label=r'$\psi(^{0})$', bins=200)
    
    #plt.plot(np.absolute(zeta).mean(0)*180/np.pi,'.-',label=r'$\zeta(^{0})$')
    
    #plt.plot(np.absolute(psi).mean(0)*180/np.pi,'.-',label=r'$\psi(^{0})$')
    
    #plt.plot(W_VV_water.mean(0), '.-', label='Spectral density')
    
    plt.legend()
    
    #plt.plot(np.absolute(BRAGG[...,1]))
    plt.show()
    '''
    #plt.imshow(np.absolute(RMS_slope)*180/np.pi, cmap='gray')
    plt.imshow(W_VV_water, cmap='gray')
    #plt.colorbar(label='degrees')
    plt.colorbar(label='spectral density')
    #plt.title(r'$\sigma_{VV}^{0}$')
    #plt.title(r'$\zeta(^{0})$')
    #plt.title('RMS_Slope')
    plt.title('W_VV')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.show()
    '''
    
if __name__=='__main__':
    main()