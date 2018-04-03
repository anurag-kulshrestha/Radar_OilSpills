import os
import matplotlib.pyplot as plt
import extract_polarimetric
import numpy as np
import matplotlib.patches as patches
from PIL import Image
import incidence_angle_corr
#from matplotlib import rc
#rc('font', **{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
from scipy import signal
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
#import read_RISAT1
import math
from math import pi
import fit_inci_model
import incidence_angle_corr
import decomposition

def plot_covariance_matrix_elements(window_size, inci_switch):
    arr=extract_polarimetric.extract_covariance_arr(window_size, inci_switch)

    fig, ax = plt.subplots(nrows=3, ncols=3)
 
    elements=['C22','C33', 'C12_real', 'C12_imag', 'C13_real', 'C13_imag', 'C23_real', 'C23_imag']

    
    plt.subplot(3,3,1)
    #plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
    plt.imshow(np.real(arr[:,:,0,0]), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('Ihh')
    plt.colorbar()
    
    plt.subplot(3,3,2)
    #plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
    plt.imshow(np.real(arr[:,:,0,1]), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('ShhShv_real')
    plt.colorbar()
    
    plt.subplot(3,3,3)
    #plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
    plt.imshow(np.real(arr[:,:,0,2]), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('Shhvv_real')
    plt.colorbar()
    
    plt.subplot(3,3,4)
    #plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
    plt.imshow(np.imag(arr[:,:,1,0]), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('Shhvv_imag')
    plt.colorbar()
    
    plt.subplot(3,3,5)
    plt.imshow(np.real(arr[:,:,1,1]), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('Ihv')
    plt.colorbar()
    
    plt.subplot(3,3,6)
    plt.imshow(np.real(arr[:,:,1,2]), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('ShvSvv_real')
    plt.colorbar()
    
    plt.subplot(3,3,7)
    plt.imshow(np.imag(arr[:,:,2,0]), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('ShhSvv_imag')
    plt.colorbar()
    
    plt.subplot(3,3,8)
    plt.imshow(np.imag(arr[:,:,2,1]), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('ShvSvv_imag')
    plt.colorbar()
    
    plt.subplot(3,3,9)
    plt.imshow(np.real(arr[:,:,2,2]), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('Ihv')
    plt.colorbar()
    
    #plt.suptitle('C3 elements - No averaging applied')
    plt.show()

def cloude_pottier(window_size):
    #fig, ax = plt.subplots(nrows=3, ncols=3)
    arr=extract_polarimetric.eigen_raster_full(window_size)
    cov_arr=extract_polarimetric.extract_covariance_arr(window_size, False)
    
    #plt.subplot(3,3,1)
    plt.imshow(arr[:,:,2], cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('lambda_1')
    plt.colorbar()
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/'+'lambda_1'+'.tiff', dpi=300)
    plt.clf()
    
    #plt.subplot(3,3,2)
    plt.imshow(arr[:,:,1], cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('lambda_2')
    plt.colorbar()
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/'+'lambda_2'+'.tiff', dpi=300)
    plt.clf()
    
    #plt.subplot(3,3,3)
    plt.imshow(arr[:,:,0], cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('lambda_3')
    plt.colorbar()
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/'+'lambda_3'+'.tiff', dpi=300)
    plt.clf()
    
    #plt.subplot(3,3,4)
    plt.imshow(extract_polarimetric.entropy(arr), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('Entropy')
    plt.colorbar()
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/'+'Entropy'+'.tiff', dpi=300)
    plt.clf()
    
    #plt.subplot(3,3,5)
    plt.imshow(extract_polarimetric.anisotropy(arr), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('Anisotropy')
    plt.colorbar()
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/'+'Anisotropy'+'.tiff', dpi=300)
    plt.clf()
    
    #plt.subplot(3,3,6)
    plt.imshow(extract_polarimetric.pol_fraction(arr), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('Pol_fraction')
    plt.colorbar()
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/'+'Pol_fraction'+'.tiff', dpi=300)
    plt.clf()
    
    #plt.subplot(3,3,7)
    plt.imshow(extract_polarimetric.pedestal_height(arr), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('pedestal_height')
    plt.colorbar()
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/'+'pedestal_height'+'.tiff', dpi=300)
    plt.clf()
    
    #plt.show()

def polarimetric_features(window_size,correction_switch):
    #fig, ax = plt.subplots(nrows=2, ncols=3)    
    
    #arr=extract_polarimetric.eigen_raster_full(window_size)
    cov_arr=extract_polarimetric.extract_covariance_arr(window_size, correction_switch)
    
    #plt.subplot(2,3,1)
    plt.imshow(np.absolute(extract_polarimetric.co_pol_power_ratio_1(cov_arr)), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('Ivv/Ihh (Co-pol power ratio)')
    plt.colorbar()
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/'+'Ivv-Ihh_(Co-pol power ratio)'+'.tiff', dpi=300)
    plt.clf()
    
    #plt.subplot(2,3,2)
    plt.imshow(incidence_angle_corr.hist_stretch(np.absolute(extract_polarimetric.determinant_cov(cov_arr)),6), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('Det(Cov)')
    plt.colorbar()
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/'+'Det(Cov)_stretched'+'.tiff', dpi=300)
    plt.clf()
    
        #plt.subplot(2,3,2)
    plt.imshow(np.absolute(extract_polarimetric.determinant_cov(cov_arr)), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('Det(Cov)')
    plt.colorbar()
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/'+'Det(Cov)'+'.tiff', dpi=300)
    plt.clf()
    
    #plt.subplot(2,3,3)
    plt.imshow(np.absolute(extract_polarimetric.co_pol_diff(cov_arr)), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('Co-pol diff (Ihh-Ivv)')
    plt.colorbar()
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/'+'Co-pol diff(Ihh-Ivv)'+'.tiff', dpi=300)
    plt.clf()

    #plt.subplot(2,3,4)
    plt.imshow(np.real(extract_polarimetric.co_pol_cross_product(cov_arr)), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('Real(ShhSvv) co-pol cross_product')
    plt.colorbar()
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/'+'Real(ShhSvv) co-pol cross_product'+'.tiff', dpi=300)
    plt.clf()
    
    #plt.subplot(2,3,5)
    plt.imshow(np.imag(extract_polarimetric.co_pol_cross_product(cov_arr)), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('Imag(ShhSvv) Co-pol cross_product')
    plt.colorbar()
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/'+'Imag(ShhSvv) Co-pol cross_product', dpi=300)
    plt.clf()
    
    #plt.show()

def plot_Pauli_comp(window_size, correction, degree):
    img_arr=decomposition.Pauli_RGB_array(window_size, correction, degree)
    #img_arr=decomposition.krogager_array(window_size, correction, degree)
    #img_arr[...,2]=hist_stretch_all(img_arr[...,2], 0, clip_extremes)
    #img_arr[...,0]=hist_stretch_all(img_arr[...,0], 0, clip_extremes)
    #img_arr[...,1]=hist_stretch_all(img_arr[...,1], 0, clip_extremes)
    plt.subplot(1,3,1)
    plt.imshow(10*np.log10(img_arr[...,0]), cmap='RdYlGn')
    plt.title('alpha')
    plt.colorbar()
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    
    plt.subplot(1,3,2)
    plt.imshow(10*np.log10(img_arr[...,1]), cmap='RdYlGn')
    plt.title('Beta')
    plt.colorbar()
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    
    plt.subplot(1,3,3)
    plt.imshow(10*np.log10(img_arr[...,2]), cmap='RdYlGn')
    plt.title('Gamma')
    plt.colorbar()
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    #img.show()
    #plt.imshow(g)
    plt.show()

def plot_Pauli_RGB(window_size, correction, degree):
    clip_extremes=True
    img_arr=decomposition.Pauli_RGB_array(window_size, correction, degree)
    #img_arr=decomposition.krogager_array(window_size, correction, degree)
    #print(img_arr.shape)
    r=hist_stretch_all(img_arr[...,0], 0, clip_extremes)
    b=hist_stretch_all(img_arr[...,2], 0, clip_extremes)
    g=hist_stretch_all(img_arr[...,1], 0, clip_extremes)
    return np.dstack((r,g,b))    

def plot_freeman_RGB(window_size, correction, degree):
    clip_extremes=True
    img_arr=decomposition.Freeman_Durdun_Decomposition(window_size, correction, degree)
    img_arr=np.absolute(img_arr)
    
    r=hist_stretch_all(img_arr[...,1], 0, clip_extremes)
    b=hist_stretch_all(img_arr[...,0], 0, clip_extremes)
    g=hist_stretch_all(img_arr[...,2], 0, clip_extremes)
    fig=plt.figure()
    gridspec.GridSpec(3,3)
    #plt.subplot(2,3,2)
    plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
    plt.imshow(np.dstack((r,g,b)))
    plt.title('Freeman RGB')
    #plt.colorbar(orientation='horizontal', label='dB')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    
    #plt.subplot(2,3,4)
    plt.subplot2grid((3,3), (0,2))
    plt.imshow(10*np.log10(img_arr[...,0]), cmap='Blues')
    plt.title('Ps')
    plt.colorbar(orientation='vertical', label='dB')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    
    #plt.subplot(2,3,5)
    plt.subplot2grid((3,3), (1,2))
    plt.imshow(10*np.log10(img_arr[...,1]), cmap='Reds')
    plt.title('Pd')
    plt.colorbar(orientation='vertical', label='dB')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    
    #plt.subplot(2,3,6)
    plt.subplot2grid((3,3), (2,2))
    plt.imshow(10*np.log10(img_arr[...,2]), cmap='Greens')
    plt.title('Pv')
    plt.colorbar(orientation='vertical', label='dB')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    #img.show()
    #plt.imshow(g)
    #plt.show()

def add_patch(image_arr, top_left_id, extent_row, extent_col):
    rect = patches.Rectangle(top_left_id, extent_row, extent_col, linewidth=1,edgecolor='r',facecolor='none')
    ax = plt.subplot(1,1,1)
    plt.imshow(image_arr)
    ax.add_artist(rect)
    plt.colorbar()
    plt.show()
    

def hist_stretch_all(arr, bits, clip_extremes):
    #bands=arr.shape[-1]
    
    n=arr.shape
    #new_arr=arr
    per=np.percentile(arr,[2.5, 97.5])
    per_max=per[1]
    per_min=per[0]
    min_arr=np.full(n, per_min)
    max_arr=np.full(n, per_max)
    if(clip_extremes==False):
        new_arr=arr
    else:
        new_arr=np.maximum(min_arr, np.minimum(max_arr, arr))
        
    #return new_arr
    if(bits==0):
        min_=np.amin(new_arr)
        max_=np.amax(new_arr)
        new_arr=(new_arr-min_)/(max_-min_)
    else:
        new_arr=np.floor((2**bits-1)*(new_arr-per_min)/(per_max-per_min))
    return new_arr

def display(arr, x_label, y_label, title):
    imgplot=plt.imshow(arr, cmap='gray')
    #plt.set_yticklabels()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.title('('+mat_ele+') '+title)
    plt.title(title)
    plt.colorbar()
    plt.show()

def plot_histogram(image_arr, xlabel, ylabel, title, bins, width):
    #plt.hist(image_arr, bins=bins)
    #print(np.histogram(hist_stretch(image_arr, 5))[0].shape, np.histogram(hist_stretch(image_arr, 5))[1].shape)
    
    #H, bins=np.histogram(image_arr, list(range(0,32)))
    H, bins=np.histogram(image_arr,bins=bins)
    #plt.hist(arr_hist[::-1], bins='auto')
    #plt.plot(arr_hist[1], np.append(arr_hist[0],256))
    #print(bins[:-1], H)
    fig, ax = plt.subplots()
    ax.bar(bins[:-1], H, width=width)# color='r')
    #ax.set_ylim(0,450)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    #ax.set_xticks(np.add(x,(width/2))) # set the position of the x ticks
    #ax.set_xticklabels(('X1', 'X2', 'X3', 'X4', 'X5'))
    plt.show()

def RISAT_features():
    S_array=read_RISAT1.img_to_array() #dim=X,Y,2
    fig, ax = plt.subplots(nrows=3, ncols=2)    
    
    plt.subplot(3,2,1)
    plt.imshow(hist_stretch_all(np.absolute(S_array[:,:,0]), 6), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('S-RH (contrast stretched)')
    plt.colorbar()
    
    plt.subplot(3,2,2)
    plt.imshow(hist_stretch_all(np.absolute(S_array[:,:,1]), 6), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('S-RV ((contrast stretched))')
    plt.colorbar()
    
    plt.subplot(3,2,3)
    H, bins=np.histogram(np.absolute(S_array[:,:,0]),bins='auto', range=(0.0,1.5))
    plt.bar(bins[:-1], H, width=0.001)# color='r')
    plt.xlabel('Amplitude')
    plt.ylabel('Frequency')
    plt.title('S-RH (original) - Histogram')
    
    plt.subplot(3,2,4)
    H, bins=np.histogram(np.absolute(S_array[:,:,1]),bins='auto', range=(0.0,1.5))
    plt.bar(bins[:-1], H, width=0.001)# color='r')
    plt.xlabel('Amplitude')
    plt.ylabel('Frequency')
    plt.title('S-RV (original) - Histogram')
    
    plt.subplot(3,2,5)
    H, bins=np.histogram(hist_stretch_all(np.absolute(S_array[:,:,0]), 6),bins='auto', range=(0.0,64))
    plt.bar(bins[:-1], H, width=1)# color='r')
    plt.xlabel('Amplitude')
    plt.ylabel('Frequency')
    plt.title('S-RH (contrast stretched) - Histogram')
    
    plt.subplot(3,2,6)
    H, bins=np.histogram(hist_stretch_all(np.absolute(S_array[:,:,1]), 6),bins='auto', range=(0.0,64))
    plt.bar(bins[:-1], H, width=1)# color='r')
    plt.xlabel('Amplitude')
    plt.ylabel('Frequency')
    plt.title('S-RV (contrast stretched) - Histogram')
    
    #plot_histogram(np.absolute(S_array[:,:,0]),'Amplitude', 'Frequency', 'Histogram of S_RH', 'auto', 0.01)
    plt.subplots_adjust(top=0.961,bottom=0.061,left=0.058,right=0.985,hspace=0.207,wspace=0.067)
    plt.savefig('Output/Srh_Srv_original.tiff', dpi=300)
    plt.show()

def plot_feature_space(arr1,arr2):
    plt.plot(arr1.flatten(), arr2.flatten(), 'ko', markersize=.1)
    plt.show()
    
def plot_feature_space_test(window_size):
    eigen_arr=extract_polarimetric.eigen_raster_full(window_size)
    cov_arr=extract_polarimetric.extract_covariance_arr(window_size, False)
    cppr=np.absolute(extract_polarimetric.co_pol_power_ratio_1(cov_arr))
    
    co_pol_diff=np.absolute(extract_polarimetric.co_pol_diff(cov_arr))
    lambda_3=np.absolute(eigen_arr[:,:,0])
    
    plot_feature_space(co_pol_diff,lambda_3)
    
    
def plot_phase_velocity(gravity,sur_ten, density, lambda_range):
    #lambda_=np.arange(lambda_range[0], lambda_range[1], lambda_range[2])
    density_oil=0.9
    density_water=1.0
    gravity=980 #cm/s
    sur_ten_oil=33 #dynes/m
    sur_ten_water=72.8 #dynes/m
    xtick_label=['$10^{-1}$', '$10^0$', '$10^1$', '$10^2$']
    x_tick_num=[-1,0,1,2]
    lambda_power=np.arange(-1,2,.01)
    k=2*pi/(10.0**lambda_power)
    #print(k)
    c_water=np.sqrt((gravity/k)+(sur_ten_water*k/density_water)) #water
    c_oil=np.sqrt((gravity/k)+(sur_ten_oil*k/density_oil))
    gravity_waves=np.sqrt(gravity/k)
    capillary_oil= np.sqrt(sur_ten_oil*k/density_oil)
    capillary_water= np.sqrt(sur_ten_water*k/density_water)
    
    c_water_plot=plt.plot(lambda_power,c_water,'b-', label='$c_{tot}^{water}$')
    c_oil_plot=plt.plot(lambda_power,c_oil,'k-',  label='$c_{tot}^{oil}$')
    cap_water_plot=plt.plot(lambda_power,capillary_water,'b--',  label='$c_{cap}^{water}$')
    cap_oil_plot=plt.plot(lambda_power,capillary_oil,'k--', label='$c_{cap}^{oil}$')
    gravity_plot=plt.plot(lambda_power,gravity_waves,'m--' , label='$c_{grav}$')
    
    plt.xlabel('Wavelength ($\lambda$) (cm)')#'$xyx_{o}$'
    plt.ylabel('Phase Velocity (cm/s)')
    plt.xticks(x_tick_num, xtick_label)
    plt.legend()
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Plot_phase_velocity.tiff', dpi=300)
    #handles=[c_water_plot, c_oil_plot, cap_water_plot, cap_oil_plot, gravity_plot], label=['a','b','c','d','e']
    #plt.xlim(xmax=2)
    #'$c_{tot}^{water}$','$c_{tot}^{oil}$','$c_{cap}^{water}$','$c_{cap}^{oil}$','$c_{grav}$'
    plt.show()
    
def plot_transect(arr, line_list, axis, name_array):#axis=0 for, 1 for column
    shp=arr.shape
    if(axis==0):
        transect_arr=arr[line_list,:,...]
        count=0
        for row in transect_arr:
            #print(row)
            #print(np.arange(shp[0]))
            plt.plot(np.arange(shp[1])+1, row.flatten(), label=name_array[count])
            count+=1
        plt.legend()
        plt.show()
    elif(axis==1):
        transect_arr=arr[:,line_list,...]
        count=0
        for col in transect_arr.T:
            plt.plot(np.arange(shp[0])+1, col.flatten(), label=name_array[count])
            count+=1
        plt.legend()
        plt.show()

def plot_transect_two_arr(arr1, arr2, line_list, name_array, element_id):
    shp=arr1.shape
    plt.subplots(len(line_list),1)
    plt.title('Intensity - VV channel incidence angle correction')
    count=1
    for line in line_list:
        transect_arr1=arr1[line:line+10,:,element_id[0],element_id[1]].mean(0)
        transect_arr2=arr2[line:line+10,:,element_id[0],element_id[1]].mean(0)
        
        missing=np.arange(521, 1538, 1)
        inc_array=fit_inci_model.extrapolate_inc_angle(missing)
        #plt.plot(np.arange(shp[1])+1, transect_arr1.flatten(), 'r-',label=name_array[0])
        #plt.plot(np.arange(shp[1])+1, transect_arr2.flatten(), 'b-',label=name_array[1])
        plt.subplot(len(line_list),1,count)
        plt.plot(inc_array, np.absolute(transect_arr1).flatten(), 'r-',label='row='+str(line)+'_inc_corr=false')
        plt.plot(inc_array, np.absolute(transect_arr2).flatten(), 'b-',label='row='+str(line)+'_inc_corr=true')
        plt.ylabel('Linear Amplitude', fontsize=15)
        plt.legend(fontsize=15)
        #plt.clf()
        count+=1
    plt.xlabel(r'Incidence angle ($\theta_{i}^{0}$) ', fontsize=15)
    plt.ylabel('Linear Amplitude')
    
    plt.tight_layout()
    
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/inci_corr.tiff', dpi=300, papertype='a4', bbox_inches='tight')
    
    plt.show()

if __name__=='__main__':
    window_size=9
    correction_switch=False
    degree=1
    #directory='../North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01/MLC_Python_New/norway_00709_15092_000_150610_L090_CX_01_mlc'
    os.chdir(directory)
    #plot_covariance_matrix_elements(1, True)
    #image_arr=incidence_angle_corr.read_Raster('C3', 'C33')
    #img_arr_dB=incidence_angle_corr.convert_to_dB(image_arr)
    #cloude_pottier(9)
    #polarimetric_features(9, True)
    
    #plot_Pauli_comp(window_size,correction_switch,degree)
    pauli_arr=plot_Pauli_RGB(window_size,correction_switch,degree)
    add_patch(pauli_arr, (104,52), 118, 100) #(1049,521), 1185, 1025)
    
    #plot_freeman_RGB(window_size, correction_switch, degree)
    #free
    
    #plot_histogram(img_arr_dB, 'sigma nought (VV) (deciBels)', 'Frequency', 'Histogram of Ivv in deciBels','auto', 0.1)
    
    #plot_histogram(image_arr, 'sigma nought (VV) (Linear Units)', 'Frequency', 'Histogram of Ivv linear units','auto', 0.0005)
    #RISAT_features()
    #plot_feature_space_test(30)
    
    #plot_phase_velocity(980,33,0.9,[0.1,50, 0.01])