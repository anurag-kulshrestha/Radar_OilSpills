#UAVSAR, TSX, RISAT-1

import os
import struct
import numpy as np  
from matplotlib import pyplot as plt
#import reproject
from scipy import ndimage
from scipy import misc
import numpy.ma as ma
from numpy import pi
from scipy import signal
#import plotting
import read_binary
import extract_polarimetric
import matplotlib.pyplot as plt
import read_TerraSARX
import read_RISAT1
import scipy
import matplotlib.patches as patches
import matplotlib
import plotting

#matplotlib.rcParams.update({'font.size': 5})
sub_plot_id=iter([1,2,3,4])


def extract_UAVSAR(directory, window_size, correction_switch, degree):
    os.chdir(directory)
    
    C3=extract_polarimetric.extract_covariance_arr(window_size, correction_switch, degree)
    return C3

def extract_UAVSAR_T3(directory, window_size, correction_switch, degree):
    os.chdir(directory)
    T3=extract_polarimetric.extract_covariance_arr(window_size, correction_switch, degree)
    return T3

def extract_TerraSAR(directory,window_size_y, window_size_x):
    amp=read_TerraSARX.multilook_TSX(directory,window_size_y, window_size_x)
    return amp

def extract_RISAT_1(directory, oil_subset=True):
    arr=read_RISAT1.img_to_array(RISAT_1_dir)
    
    Srh_arr=arr[:,:,0]#Srh
    Srv_arr=arr[:,:,1]#Srv
    if(oil_subset==True):
        #Srh_arr=read_RISAT1.oil_subset(Srh_arr)
        #Srv_arr=read_RISAT1.oil_subset(Srv_arr)
        Srh_arr=read_RISAT1.oil_subset_1(Srh_arr)
        Srv_arr=read_RISAT1.oil_subset_1(Srv_arr)
    return np.dstack((Srh_arr, Srv_arr))

def get_UAVSAR(UAVSAR_directory_TSX, cropping=False, croppingList=[],window_size=1,correction_switch=False,degree=1,UAVSAR_heading_ang=7, plotting=False, time='0626'):
    #window_size,correction_switch,degree=1,False,1
    #UAVSAR_heading_ang=7
    
    C3=extract_UAVSAR(UAVSAR_directory_TSX,window_size, correction_switch, degree)
    
    Ivv_db=10*np.log10(np.absolute(C3[...,2,2]))
    if (cropping==True):
        C3=C3[croppingList[0]:croppingList[1],croppingList[2]:croppingList[3],...]
        Ivv_db=10*np.log10(np.absolute(C3[...,2,2]))
    
    if (plotting==True):
        sub_plot=next(sub_plot_id)
        print (sub_plot)
        #print(sub_plot)
        ax=plt.subplot(2,2,sub_plot)
        #ax = plt.subplot(111)
        im = plt.imshow(np.flip(np.flip(Ivv_db,0),1), cmap='gray')#,aspect=1.26)
        plt.colorbar(label='dB', orientation='vertical')
        #plt.title('UAVSAR: dType:MLC (Look_az=12, Look_range=5), I_vv; Time: '+str(time)+'; \n Resolution(az*slant range):7.2*5')
        #plt.show()
        
        plt.xlabel('Range')
        plt.ylabel('Azimuth')
        x,y,dx,dy=50,750,-7,7*np.tan(97*np.pi/180)
        x2,y2=x+dx,y+dy
        ax.add_patch(
        patches.Arrow(x,y,dx,dy,width=50,facecolor='k'))
        
        ax.text(x2,y2,'N')
        
        
        
        #ax1=ax.twinx()
        
    return C3
    #return Ivv_db

def get_TSx(TSX_dir, window_size_x=9,window_size_y=9, cropping=True, croppingList=[65,2265,824,2074], plotting=False):
    sub_plot=next(sub_plot_id)
    print (sub_plot)
    
    TSx_head_ang=192
    TSX_amp=extract_TerraSAR(TSX_dir,window_size_x, window_size_y)
    #amp_HH=TSX_amp[...,0]
    #amp_VV=TSX_amp[...,1]
    #I_HH=np.absolute(amp_HH)**2
    I_VV=np.absolute(TSX_amp[...,1])**2
    Ivv_dB=10*np.log10(I_VV)
    if(cropping==True):
        TSX_amp=TSX_amp[croppingList[0]:croppingList[1],croppingList[2]:croppingList[3],...]
        I_VV=np.absolute(TSX_amp[...,1])**2
    if (plotting==True):
        ax=plt.subplot(2,2,sub_plot)
        #ax = plt.subplot(111)
        
        Ivv_dB=10*np.log10(I_VV)
        plt.imshow(np.flip(Ivv_dB, axis=1), cmap='gray')#, aspect=1.26)
        #plt.imshow(10*np.log10(np.absolute(amp_VV)), cmap='gray')
        plt.colorbar(label='dB', orientation='vertical')
        #plt.show()
        
        plt.xlabel('Range')
        plt.ylabel('Azimuth')
        #plt.title('TS-x: dType:SSC (Look_az='+str(window_size_y)+', Look_range='+str(window_size_x)+'), I_vv; Time: 0624; \n Resolution(az*slant range): 6.6*1.17')
        x,y,dx,dy=127,2131,-30,30*np.tan(102*np.pi/180)
        x2,y2=x+dx,y+dy
        ax.add_patch(
        patches.Arrow(x,y,dx,dy,width=150,facecolor='k'))
        
        ax.text(x2,y2,'N')
    #return TSX_amp
    #return Ivv_dB
    return I_VV
    
def get_RISAT1(RISAT_1_dir, rotation_angle=-25, plotting=False, window_x=1, window_y=1):
    sub_plot=next(sub_plot_id)
    print (sub_plot)
    
    ax=plt.subplot(2,2,sub_plot)
    #ax = plt.subplot(111)
    
    #RS1_arr=extract_RISAT_1(RISAT_1_dir)
    #np.save('/home/anurag/Documents/MScProject/SAR/OilSpill/RISAT-1/RI1_SAR_L1SLC_FRS1_CR_20150610T071918_20150610T071923_17197_1515551004/oil_subset_SrhSrv_large_extent.npy',RS1_arr)
    #RS1_arr=np.load('/home/anurag/Documents/MScProject/SAR/OilSpill/RISAT-1/RI1_SAR_L1SLC_FRS1_CR_20150610T071918_20150610T071923_17197_1515551004/oil_subsetSrhSrv.npy')
    
    RS1_arr=np.load('/home/anurag/Documents/MScProject/SAR/OilSpill/RISAT-1/RI1_SAR_L1SLC_FRS1_CR_20150610T071918_20150610T071923_17197_1515551004/oil_subset_SrhSrv_large_extent.npy')
    
    RS1_arr_cropped=RS1_arr[1750:3900,1100:2800,...]
    arr_s11=RS1_arr[:,:,0]#Srh
    arr_s21=RS1_arr[:,:,1]
    arr_s21_ML=read_RISAT1.averaging_arr_1(np.absolute(arr_s21),window_x,window_y)
        
    arr_s21_ML_rot=scipy.ndimage.interpolation.rotate(arr_s21_ML,rotation_angle)
        
    arr_s21_ML_rot_ma=ma.masked_where(arr_s21_ML_rot==0, arr_s21_ML_rot)
    
    return_var=0
    
    if(plotting==True):
        
        if(rotation_angle==0):
            
            #real aur imag alag alag karke rotate karna hoga
            arr_s21_ML_rot_ma_db=10*np.log10(arr_s21_ML_rot_ma)
            arr_s21_ML_rot_ma_db_crop=arr_s21_ML_rot_ma_db[1750:3900,1300:2650]
            plt.imshow(arr_s21_ML_rot_ma_db_crop, cmap='gray')#, aspect="auto")
            return_var=arr_s21_ML_rot_ma_db_crop
        #=========crop from rot (x,y) y-> row==========
        
        else:
            UL=(2095,2258)
            LR=(3502,4526)
            
            arr_s21_ML_rot_oil_subset=arr_s21_ML_rot[2258:4526,2095:3452]
            
            arr_s21_ML_rot_oil_subset_db=10*np.log10(arr_s21_ML_rot_oil_subset)
            
            plt.imshow(arr_s21_ML_rot_oil_subset_db, cmap='gray')#, aspect=1.26)
            
            return_var=arr_s21_ML_rot_oil_subset_db
        #plt.imshow(10*np.log10(np.absolute(arr_s21_ML_rot_ma)), cmap='gray')
        plt.colorbar(label='dB', orientation='vertical')
        plt.xlabel('Range')
        plt.ylabel('Azimuth')
        #plt.title('RISAT-1: dType:SLC (Look_az='+str(window_y)+', Look_range='+str(window_x)+'), I_rv; Time: 0719; \n Resolution(az*slant range):3.33*2.34')
        
        x,y,dx,dy=1250,2070,-70,70*np.tan((132+rotation_angle)*np.pi/180)
        x2,y2=x+dx,y+dy
        ax.add_patch(
        patches.Arrow(x,y,dx,dy,width=150,facecolor='k'))
        
        ax.text(x2,y2,'N')
        #return arr_s21_ML_rot_oil_subset_db
    return arr_s21_ML
    return RS1_arr_cropped
    #return return_var

#def linear_stretching(arr,val_range):
    ##a=1
    ##per=np.percentile(arr,[2.5, 97.5])
    ##per_max=per[1]
    ##per_min=per[0]
    
    ##new_arr=(new_arr-per_min)/(per_max-per_min)
    #new_arr=arr
    #min_=np.amin(new_arr)
    #max_=np.amax(new_arr)
    #new_arr=(((new_arr-min_)/(max_-min_)))#+ val_range[0])*val_range[1]) 
    #plt.imshow(new_arr, cmap='gray')
    #plt.colorbar()
    #plt.show()

def image_squeeze(arr,new_shp):
    orig_shp=arr.shape
    win_y=orig_shp[0]//new_shp[0]
    win_x=orig_shp[1]//new_shp[1]
    #win_y,win_x=np.floor(row_ratio),np.floor(col_ratio)
    #for i in range(orig_shp[0]):
        #for j in range(orig_shp[0])
    
    
    #dim=np.ndim(slc_arr)
    #print(win_x, win_y)
    stride_row=win_y
    stride_col=win_x
    rows=orig_shp[0]
    cols=orig_shp[1]
    #print(rows)
    mod_shp=new_shp
    #print(mod_shp)
    res=np.empty((mod_shp[0]+50,mod_shp[1]+50), dtype=np.float64)
    res_row=0
    res_col=0
    for i in range(0, rows-win_y, stride_row):
        for j in range(0, cols-win_x, stride_col):
            a=arr[i:i+win_y, j:j+win_x, ...]
            #res[res_row,res_col]=np.mean(a*np.conj(a))
            #print(a)
            res[res_row,res_col]=a.mean(1).mean(0)
            #res[res_row,res_col]=a.mean()
            #print((i,j))
            res_col+=1
        res_col=0
        res_row+=1
        print(i)
    #incidence_angle_corr.display(res, 'Range(pixel#)', 'Azimuth (pixel #)', 'Contrast GLCM feature(dir=0, window_size=15, row_stride=2, col_stride=2)')
    return res[0:new_shp[0],0:new_shp[1]]
    
    
    

def multi_freq_ratio(arr_UAV, arr_TSx):
    arr_UAV=arr_UAV[135:700,244:500]
    arr_TSx=arr_TSx[366:2080,214:1000]
    UAV_shp=arr_UAV.shape
    
    arr_TSx_min_max=[np.amin(arr_TSx),np.amax(arr_TSx)]
    #TS_x_resized= scipy.misc.imresize(arr_TSx, (UAV_shp[0],UAV_shp[1]))
    
    #linear stretching to original values
    TS_x_resized=image_squeeze(arr_TSx,UAV_shp)
    
    #TS_x_resized=np.clip(TS_x_resized,-26,0.001)
    #print(TS_x_resized.shape)
    #plt.imshow(plotting.hist_stretch_all(TS_x_resized, 6, True), cmap='gray')
    #plt.imshow(TS_x_resized, cmap='gray')
    #plt.colorbar()
    #plt.show()
    
    #arr_UAV=scipy.misc.imresize(arr_UAV, (UAV_shp[0],UAV_shp[1]))
    
    plt.subplot(221)
    plt.imshow(arr_UAV, cmap='gray')
    plt.colorbar(label='dB')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('UAVSAR_cropped')
    
    plt.subplot(222)
    
    plt.imshow(TS_x_resized, cmap='gray')
    plt.colorbar(label='dB')
    plt.title('TerraSAR-X_cropped')
    plt.subplot(223)
    freq_ratio= arr_UAV/TS_x_resized
    #freq_ratio=plotting.hist_stretch_all(freq_ratio, 0, True)
    
    plt.imshow(freq_ratio, cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    
    plt.title('UAV/TS_x')
    plt.colorbar()
    
    plt.subplot(224)
    freq_ratio= TS_x_resized/arr_UAV
    #freq_ratio=plotting.hist_stretch_all(freq_ratio, 0, True)
    #TS_x/UA
    plt.imshow(freq_ratio, cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('TS_x/UAV')
    plt.colorbar()
    
    plt.tight_layout()
    #plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results//Multifrequency/Dual_freq_ratio_UAV_TSX_no_stretch.tiff', dpi=300, papertype='a4', bbox_inches='tight')
    
    plt.show()
    

if __name__=='__main__':
    #UAVSAR_directory_name_prep
    
    UAVSAR_directory_TSX = '/home/anurag/Documents/MScProject/SAR/OilSpill/North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15091_004_150610_L090_CX_01/norway_00709_15091_004_150610_L090_CX_01_mlc'#/MLC_Python_New/norway_00709_15092_000_150610_L090_CX_01_mlc'
    
    UAVSAR_directory_RS_1='/home/anurag/Documents/MScProject/SAR/OilSpill/North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15091_008_150610_L090_CX_01/norway_00709_15091_008_150610_L090_CX_01_mlc'
    
    TSX_dir = '/home/anurag/Documents/MScProject/SAR/OilSpill/TerraSAR_X/dims_op_oc_dfd2_567023152_2/TSX-1.SAR.L1B/TSX1_SAR__SSC______SM_D_SRA_20150610T062401_20150610T062409/sybset/subset_0_of_TSX1_SAR_SSC_SM_D_SRA_20150610T062401_20150610T062409_Cal.data'
    
    RISAT_1_dir='/home/anurag/Documents/MScProject/SAR/OilSpill/RISAT-1/RI1_SAR_L1SLC_FRS1_CR_20150610T071918_20150610T071923_17197_1515551004'
    
    #==========UAVSAR_ TS-x=========
    C3_UAV_TS=get_UAVSAR(UAVSAR_directory_TSX,cropping=True, croppingList=[50,788,87,673], plotting=True, window_size=1, time='0626')
    
    #plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results//Multifrequency/UAVSAR_TS.tiff', dpi=300, papertype='a4', bbox_inches='tight')
    
    #plt.close()
    
    #==========TerraSAR_X==========
    TS_x=get_TSx(TSX_dir, plotting=True,window_size_x=8,window_size_y=8)
    
    #plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results//Multifrequency/TSx.tiff', dpi=300, papertype='a4', bbox_inches='tight')
    
    #plt.close()
    #==========UAVSAR_ RS-1=========
    
    C3_UAV_RS1=get_UAVSAR(UAVSAR_directory_RS_1, plotting=True, time = '0717' )
    
    #plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results//Multifrequency/UAVSAR_RS1.tiff', dpi=300, papertype='a4', bbox_inches='tight')
   
    #plt.close()
    #==========RISAT-1==========
    RS_1=get_RISAT1(RISAT_1_dir, plotting=True, rotation_angle=0, window_x=8, window_y=8)
    
    #plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results//Multifrequency/RISAT1.tiff', dpi=300, papertype='a4', bbox_inches='tight')
    
    #plt.close()
    #===========plotting switch===========
    
    #plt.tight_layout()
    #plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Multifrequency/Datasets.tiff', dpi=300, papertype='a4', bbox_inches='tight')
    
    #plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results//Multifrequency/IVV_plots_2.tiff', dpi=300, papertype='a4', bbox_inches='tight')
    #plt.show()
    
    
    
    plt.close()
    
    #make hisrograms
    #plt.hist(np.real(C3_UAV_TS[...,2,2]).flatten(),bins=300, rwidth=0.5,histtype='step', label='UAVSAR ($I_{VV}$)', range=[0,1], linestyle = '-', color='k')
    
    plt.hist(np.random.choice(np.real(C3_UAV_TS[...,2,2]).flatten(), size=50000, replace=False),bins=300, rwidth=0.5,histtype='step', label='UAVSAR ($I_{VV}$)', range=[0,1], linestyle = ':', color='k')
    
    
    
    #plt.hist(TS_x.flatten(),bins=500, rwidth=0.5,histtype='step', label='$ TerraSAR-X (I_{VV})$', range=[0,1], linestyle = '-.', color='k')
    
    plt.hist(np.random.choice(TS_x.flatten(), size=50000, replace=False),bins=500, rwidth=0.5,histtype='step', label='$ TerraSAR-X (I_{VV})$', range=[0,1], linestyle = '-', color='k')
    
    
    
    #plt.hist(RS_1.flatten(),bins=500, rwidth=0.5,histtype='step', label='RS-1 ($I_{RV})$', range=[0,1], linestyle = '--', color='k')
    
    plt.hist(np.random.choice(RS_1.flatten(), size=50000, replace=False),bins=500, rwidth=0.5,histtype='step', label='RS-1 ($I_{RV})$', range=[0,1], linestyle = '--', color='k')
    
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.xlabel('Linear Intensity',fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Multifrequency/IntesityHistograms.tiff', dpi=300, papertype='a4', bbox_inches='tight')
    
    
    
    plt.show()
    
    
    
    #========Make transect plots==============
    C3_UAV_TS=np.flip(np.flip(C3_UAV_TS,0),1)
    
    TS_x=np.flip(TS_x, axis=1)
    '''
    E80_UAV_TSx=C3_UAV_TS[150:160,...].mean(0)
    E60_UAV_TSx=C3_UAV_TS[300:320,...].mean(0)
    E40_UAV_TSx=C3_UAV_TS[470:490,...].mean(0)
    PO_UAV_TSx=C3_UAV_TS[650:670,...].mean(0)
    
    plt.subplot(241)
    plt.plot(10*np.log10(E80_UAV_TSx[...,2,2]), label='E80_UAV_TSx')
    plt.xlabel('Range')
    plt.ylabel('dB')
    plt.title('E80_UAV_TSx')
    
    plt.subplot(242)
    plt.plot(10*np.log10(E60_UAV_TSx[...,2,2]), label='E60_UAV_TSx')
    plt.xlabel('Range')
    plt.ylabel('dB')
    plt.title('E60_UAV_TSx')
    
    plt.subplot(243)
    plt.plot(10*np.log10(E40_UAV_TSx[...,2,2]), label='E40_UAV_TSx')
    plt.xlabel('Range')
    plt.ylabel('dB')
    plt.title('E40_UAV_TSx')
    
    plt.subplot(244)
    plt.plot(10*np.log10(PO_UAV_TSx[...,2,2]), label='PO_UAV_TSx')
    plt.xlabel('Range')
    plt.ylabel('dB')
    plt.title('PO_UAV_TSx')
    #plt.legend()
    #plt.show()
    
    
    
    
    
    E80_TSx=TS_x[500:520,...].mean(0)
    E60_TSx=TS_x[1000:1020,...].mean(0)
    E40_TSx=TS_x[1480:1500,...].mean(0)
    PO_TSx=TS_x[2000:2020,...].mean(0)
    
    plt.subplot(245)
    plt.plot(10*np.log10(E80_TSx[...,1]**2), label='PO_UAV_TSx')
    plt.xlabel('Range')
    plt.ylabel('dB')
    plt.title('E80_TSx')
    
    plt.subplot(246)
    plt.plot(10*np.log10(E60_TSx[...,1]**2), label='E60_TSx')
    plt.xlabel('Range')
    plt.ylabel('dB')
    plt.title('E60_TSx')
    
    plt.subplot(247)
    plt.plot(10*np.log10(E40_TSx[...,1]**2), label='E40_TSx')
    plt.xlabel('Range')
    plt.ylabel('dB')
    plt.title('E40_TSx')
    
    plt.subplot(248)
    plt.plot(10*np.log10(PO_TSx[...,1]**2), label='PO_TSx')
    plt.xlabel('Range')
    plt.ylabel('dB')
    plt.title('PO_TSx')
    
    plt.tight_layout()
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results//Multifrequency/Detectability.tiff', dpi=300, papertype='a4', bbox_inches='tight')
    
    plt.show()
    
    
    '''
    #=======multi_freq_ratio===========
    #IVV_UAV=10*np.log10(np.absolute(C3_UAV_TS[...,2,2]))
    
    #I_VV=np.absolute(TS_x[...,1])**2
    #I_VV_TSX=10*np.log10(I_VV)
    
    #multi_freq_ratio(IVV_UAV, I_VV_TSX)
    
    
    #===========H-alpha decomposition==============
    os.chdir(UAVSAR_directory_TSX)
    eigen_raster_full=extract_polarimetric.eigen_raster_full(window_size=9, correction_switch=False, degree=0)[35:815,87:724,...]
    
    H=extract_polarimetric.entropy(eigen_raster_full)
    A=extract_polarimetric.anisotropy(eigen_raster_full)
    
    plt.imshow(A,cmap='jet')
    plt.colorbar()
    plt.show()
    
    