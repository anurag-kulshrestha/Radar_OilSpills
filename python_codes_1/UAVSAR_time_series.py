#UAVSAR, TSX, RISAT-1
from osgeo import gdal, ogr, osr
import os
import sys
#import struct
#import argparse

import matplotlib
from matplotlib import pyplot as plt

from scipy import ndimage
from scipy import signal

import numpy as np  
import numpy.ma as ma
from numpy import pi
from numpy import linalg as LA

#import plotting
import read_binary
import reproject
import extract_polarimetric
import EPFS


def test_convolve(arr, kern):
    grad = signal.convolve2d(arr, kern , boundary='symm', mode='valid')
    return grad

def kernal(window_size):
    k=np.ones(window_size*window_size).reshape(window_size, window_size)
    normalize_k=k/(window_size**2)
    return normalize_k


def pixel_id_to_lat(GT, Y):
    lat=GT[5]*Y + GT[3]
    return round(lat,2)

def pixel_id_to_lon(GT, X):
    lon=GT[1]*X + GT[0]
    return round(lon,2)

def extract_archives(directory):
    f=open('unzip_data.sh','w')
    f.write('if [ ! -d {} ]; then \n'.format(directory))
    f.write('mkdir {}\n'.format(directory))
    #f.write('cd {}\n'.format(directory))
    f.write('unzip {}.zip -d . ; \n'.format(directory))
    
    f.write('mv *.{} {}\n'.format(directory[-3:],directory))
    #f.write('mv *.grd\n')
    
    f.write('fi \n')
    f.close()
    os.system('sudo bash unzip_data.sh')
    os.system('rm unzip_data.sh')

def get_base_file_name(region='norway',heading='007',counter_num='09',year='15',num_flights_year='092',data_take='000',day='10',month='06',band='L',steering_angle='090',cross_talk='CX',processing_version='02'):
    
    file_name=base_file_name=region+'_'+heading+counter_num+'_'+year+num_flights_year+'_'+data_take+'_'+year+month+day+'_'+band+steering_angle
    
    base_file_name=file_name+'_'+cross_talk+'_'+processing_version
    return base_file_name

def getdir(region='norway',heading='007',counter_num='09',year='15',num_flights_year='092',data_take='000',day='10',month='06',band='L',steering_angle='090',cross_talk='CX',processing_version='02'):
    
    base_dir='/home/anurag/Documents/MScProject/SAR/OilSpill//North_Sea_UAVSAR/UAV_norway'
    file_ext=['.ann','.dat','.gif','.hgt','.inc','.kmz','.slope','_hgt.tif','_pauli.tif']
    folders=['_mlc','_grd']
    
    base_file_name=get_base_file_name(region,heading,counter_num,year,num_flights_year,data_take,day,month,band,steering_angle,cross_talk,processing_version)
    
    #../North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01
    
    folder_name='UA_'+base_file_name
    wd=os.path.join(base_dir,folder_name)
    return wd

def metadata_dict(base_file_name):
    meta = open(base_file_name+'.ann', 'r') 
    #print(meta.readline())
    meta_dict={}
    for i in meta.readlines():
        if (i[0]!=';'):
            #print(i)
            j=i[:-1].split('=') #removing \n's from end of line
            #print(j)
            if(len(j)>1):
                #print(len(j))
                if(j[1].find(';')>0):
                    #print(j)
                    j[1]=j[1][:j[1].index(';')] #trim further after ;
                j=[str.strip(k) for k in j] #trimming
                if(j[0][-1]==')'):
                    j[0]=j[0].strip()[:j[0].strip()[::-1].find(' ')*-1]
                j=[str.strip(k) for k in j] #trimming
            #if()
                meta_dict[j[0]]=j[1]
                #print (j)
    #print(meta_dict)
    return meta_dict


#def extract_UAVSAR_(directory, window_size, correction_switch, degree):
    #os.chdir(directory)

def get_inc_angle(lines, samples, base_file_name,cropping_list, cropping_switch = True, is_List_Ratio=False):
    #read_binary.plot_inc_ang_arr(lines, samples, base_file_name)
    return read_binary.read_inc_file(lines, samples, base_file_name, cropping_list, cropping_switch = cropping_switch, is_List_Ratio=is_List_Ratio)

def get_SLC(meta,base_file_name, polarization='VV', mlc_cropping_list=[521,1545,4049,5233], cropping_List_MLC=True):

    mlc_row_looks=int(meta['Number of Azimuth Looks in MLC'])
    mlc_col_looks=int(meta['Number of Range Looks in MLC'])
    scan_lines=int(meta['slc_mag.set_rows'])
    scan_pix=int(meta['slc_mag.set_cols'])
    
    
    slc_cropping_list=[mlc_cropping_list[0]*mlc_col_looks,\
        mlc_cropping_list[1]*mlc_col_looks,\
            mlc_cropping_list[2]*mlc_row_looks,\
                mlc_cropping_list[3]*mlc_row_looks]
    #file_name="norway_00709_15092_000_150610_L090VV_CX_02.slc"
    slc_file_name=file_name+polarization+'_'+cross_talk+'_'+processing_version+'.slc'
    
    slc_arr=read_binary.read_SLC(slc_file_name, scan_lines, scan_pix, slc_cropping_list, False)
    return slc_arr

def get_GRD_MLC(dType, meta, base_file_name, convolution_kernal, inc_ang_arr,inc_correction_sin_degree=2, component='VVVV', grd_cropping_list=[2500,3700,2500,3700], is_List_Ratio=False, cropping_switch=True, reproject_switch=False, plotting_switch=True, convolution_switch = False, inc_correction_switch=False, save_plot_switch=False):
    grd_UL_lat=float(meta['grd_pwr.row_addr'])
    grd_UL_lon=float(meta['grd_pwr.col_addr'])
    
    grd_lat_spacing=float(meta['grd_pwr.row_mult'])
    grd_lon_spacing=float(meta['grd_pwr.col_mult'])
    
    scan_lines=int(meta[dType+'_mag.set_rows'])
    scan_pix=int(meta[dType+'_mag.set_cols'])
    
    #print(base_file_name)
    
    grd_dir=base_file_name+'_'+dType
    grd_filename=base_file_name[:-6]+component+base_file_name[-6:]+'.'+dType
    print (grd_filename)
    
    #============Unzip the file if the directory foesn't exist. Caution: only works on Ubuntu
    extract_archives(grd_dir)
    #===========Change the file's directory============
    if (os.getcwd()!=grd_dir):
        os.chdir(grd_dir)
    
    
    #===========Get the extracted file=================
    
    print(os.getcwd())
    
    grd_arr=read_binary.read_GRD(grd_filename, scan_lines, scan_pix, grd_cropping_list, is_List_Ratio,meta,cropping_switch)
    
    #===========Calculation and  of Geo-Transform===========
    
    
    grd_gt=[grd_UL_lon+grd_cropping_list[0]*grd_lon_spacing,\
        grd_lon_spacing,\
            0,\
                grd_UL_lat+(grd_cropping_list[2]-1)*grd_lat_spacing,\
                    0,\
                        grd_lat_spacing]# subtracted 1 to avoid the effect of 0-indexing in python
    
    #===========Calculation of Geo-Transform===========
    proj = osr.SpatialReference()
    #proj.SetUTM(zone, is_North)
    proj.SetWellKnownGeogCS( "WGS84" )
    grd_Proj = proj.ExportToWkt()
    
    #===========Reproject=============
    if reproject_switch==True:
        tmp_dir=os.getcwd()
        os.chdir('/home/anurag/Documents/MScProject/Meetings_ITC/Results/TimeSeries')
        reproject.reproject_image(grd_filename, grd_arr.shape[1], grd_arr.shape[0], 1, 10*np.log10(np.absolute(grd_arr)), grd_Proj, grd_gt)
        os.chdir(tmp_dir)
    
    
    
    #=============Incidence Angle Correction===========
    if inc_correction_switch==True:
        inc_shp = inc_ang_arr.shape
        grd_shp = grd_arr.shape
        #if (inc_shp[0] - grd_shp[0]) !=0:
            #print(inc_ang_arr.shape)
            #buffer_delete = (inc_shp[0] - grd_shp[0])//2
            #inc_ang_arr = inc_ang_arr[buffer_delete:inc_shp[0]-buffer_delete, buffer_delete:inc_shp[1]-buffer_delete]
            
        grd_arr = grd_arr * np.sin(inc_ang_arr)**inc_correction_sin_degree/(np.sin(inc_ang_arr.mean())**inc_correction_sin_degree)
        
            
    
    #==============Spatial Averaging===============
    if convolution_switch==True:
        print('Aaya yaha?')
        grd_arr = test_convolve(grd_arr, convolution_kernal)
        #grd_arr = np.pad(grd_arr, convolution_kernal.shape[0]//2, 'constant', constant_values=-10000)
        #grd_arr = ma.masked_where(grd_arr==-10000,grd_arr)
    #=================Plotting==================
    if plotting_switch==True:
        lat_pixels=list(range(1000,scan_lines,1000))
        lon_pixels=list(range(1000,scan_pix,1000))
        
        slick_lat_pixels=list(range(100,grd_cropping_list[3]-grd_cropping_list[2],200))
        slick_lon_pixels=list(range(100,grd_cropping_list[1]-grd_cropping_list[0],200))
        
        plt.imshow(10*np.log10(np.absolute(grd_arr)), cmap='gray')
        #plt.xlabel('Longitude ($^{0}$)')
        #plt.ylabel('Latitude ($^{0}$)')
        if (cropping_switch==True):
            plt.xticks(slick_lon_pixels, [str(pixel_id_to_lon(grd_gt,i)) for i in slick_lon_pixels])
            plt.yticks(slick_lat_pixels, [str(pixel_id_to_lat(grd_gt,i)) for i in slick_lat_pixels])
            
        else:
            plt.xticks(lon_pixels, [str(pixel_id_to_lon(grd_gt,i)) for i in lon_pixels])
            plt.yticks(lat_pixels, [str(pixel_id_to_lat(grd_gt,i)) for i in lat_pixels])
        #plt.colorbar(label='dB')
        plt.tight_layout()
        if save_plot_switch==True:
            plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/'+base_file_name+'_plot', dpi=300,bbox_inches='tight')
        
        
        #plt.show()
        
    os.chdir('../')
    
    return grd_arr

def get_covariance_matrix_grd(dType, meta, base_file_name, convolution_kernal, inc_ang_arr,inc_correction_sin_degree=2, cropping_list_GRD=[2500,3700,2500,3700], is_List_Ratio=False, cropping_switch=True, convolution_switch = False, inc_correction_switch=False):
    
    grd_rows = int(meta['grd_pwr.set_rows'])
    grd_cols = int(meta['grd_pwr.set_cols'])
    
    convolution_window_size = convolution_kernal.shape[0]
    
    if (cropping_switch == True):
        if (is_List_Ratio == True):
            oil_rows_start = np.floor(cropping_list_GRD[2]*grd_rows) #ratios gotten from estimates
            oil_rows_end = np.floor(cropping_list_GRD[3]*grd_rows)
            oil_cols_start = np.floor(cropping_list_GRD[0]*grd_cols)
            oil_cols_end = np.floor(cropping_list_GRD[1]*grd_cols)
        else:
            oil_rows_start = cropping_list_GRD[2]
            oil_rows_end = cropping_list_GRD[3]
            oil_cols_start = cropping_list_GRD[0]
            oil_cols_end = cropping_list_GRD[1]
    else:
        oil_rows_start=1
        oil_rows_end=grd_rows
        oil_cols_start=1
        oil_cols_end=grd_cols
    
    tot_oil_rows=int(oil_rows_end-oil_rows_start) - convolution_window_size +1 
    tot_oil_cols=int(oil_cols_end-oil_cols_start) - convolution_window_size +1 
    #print(tot_oil_rows)
    components=['HHHH','HHHV','HHVV','HVHV','HVVV','VVVV']
    comp_cov_loc=[(0,0),(0,1),(0,2),(1,1), (1,2), (2,2)]
    multiplying_factor = [1,np.sqrt(2),1,2,np.sqrt(2),1]
    cov_arr=np.empty((tot_oil_rows,tot_oil_cols,3,3), dtype=np.complex64())
    
    for i in range(0,6):
        print(cov_arr.shape)
        grd_comp = get_GRD_MLC(dType, meta, base_file_name,convolution_kernal, inc_ang_arr,inc_correction_sin_degree=inc_correction_sin_degree, component=components[i],grd_cropping_list=cropping_list_GRD, is_List_Ratio=is_List_Ratio, cropping_switch=cropping_switch, reproject_switch=False, plotting_switch=False, convolution_switch = convolution_switch, inc_correction_switch=inc_correction_switch) * multiplying_factor[i]
        print (grd_comp.shape)
        
        cov_arr[...,comp_cov_loc[i][0],comp_cov_loc[i][1]] = grd_comp
        
        
    cov_arr[...,1,0] = np.conj(cov_arr[...,0,1])
    cov_arr[...,2,0] = np.conj(cov_arr[...,0,2])
    cov_arr[...,2,1] = np.conj(cov_arr[...,1,2])
    
    return cov_arr

def get_coherecy_matrix_grd(cov_arr):
    C11=cov_arr[...,0,0]
    C12=cov_arr[...,0,1]
    C13=cov_arr[...,0,2]
    C21=cov_arr[...,1,0]
    C22=cov_arr[...,1,1]
    C23=cov_arr[...,1,2]
    C31=cov_arr[...,2,0]
    C32=cov_arr[...,2,1]
    C33=cov_arr[...,2,2]
    
    cov_arr[...,0,0] = (C11 + C33 + C13 + C31)/2
    cov_arr[...,0,1] = (C11 - C13 + C31 - C33)/2
    cov_arr[...,0,2] = (C12 + C32) /np.sqrt(2)
    
    cov_arr[...,1,1] = (C11 + C33 - C13 - C31)/2
    cov_arr[...,1,2] = (C12 - C32) /np.sqrt(2)
    
    cov_arr[...,2,2] = C22
    
    
    cov_arr[...,1,0] = np.conj(cov_arr[...,0,1])
    cov_arr[...,2,0] = np.conj(cov_arr[...,0,2])
    cov_arr[...,2,1] = np.conj(cov_arr[...,1,2])
    
    return cov_arr

def get_eigen_values_vectors(arr):
    w,v=LA.eig(arr)
    return [w,v]

def eigen_raster_full(coh_arr):
    eigen_arr=np.sort(np.absolute(get_eigen_values_vectors(coh_arr)[0]), axis=2) # this method does indeed give real eigen vaues of the hermitian covariance and coherency matrices
    #eigen_arr=np.sort(get_eigen_values(arr), axis=2)
    return eigen_arr

def plot_all_UAVSAR_data_takes():
    matplotlib.rcParams.update({'font.size': 3})
    #files = ['UA_norway_00709_15091_000_150610_L090_CX_01', 'UA_norway_18709_15091_001_150610_L090_CX_01',
             #'UA_norway_00710_15091_002_150610_L090_CX_01', 'UA_norway_18710_15091_003_150610_L090_CX_01',
             #'UA_norway_00709_15091_004_150610_L090_CX_01', 'UA_norway_18709_15091_005_150610_L090_CX_01',
             #'UA_norway_00710_15091_006_150610_L090_CX_01', 'UA_norway_18709_15091_007_150610_L090_CX_01',
             #'UA_norway_00709_15091_008_150610_L090_CX_01', 'UA_norway_18709_15091_009_150610_L090_CX_01',
             #'UA_norway_00709_15091_010_150610_L090_CX_01', 'UA_norway_18709_15091_011_150610_L090_CX_01',
             #'UA_norway_00709_15091_012_150610_L090_CX_01', 'UA_norway_18709_15091_013_150610_L090_CX_01',
             #'UA_norway_00709_15091_014_150610_L090_CX_01', 'UA_norway_14203_15091_015_150610_L090_CX_01',
             #'UA_norway_00709_15092_000_150610_L090_CX_01', 'UA_norway_18709_15092_001_150610_L090_CX_01',
             #'UA_norway_00709_15092_002_150610_L090_CX_01', 'UA_norway_18709_15092_003_150610_L090_CX_01',
             #'UA_norway_00709_15092_004_150610_L090_CX_01', 'UA_norway_18709_15092_005_150610_L090_CX_01']
    
    files = ['UA_norway_00709_15091_000_150610_L090_CX_01',
             'UA_norway_00709_15091_004_150610_L090_CX_01',
             'UA_norway_00709_15091_008_150610_L090_CX_01',
             'UA_norway_00709_15091_010_150610_L090_CX_01',
             'UA_norway_00709_15091_012_150610_L090_CX_01',
             'UA_norway_00709_15091_014_150610_L090_CX_01',
             'UA_norway_00709_15092_000_150610_L090_CX_01',
             'UA_norway_00709_15092_002_150610_L090_CX_01',
             'UA_norway_00709_15092_004_150610_L090_CX_01'
             ]

    
    for i, dir_base_file_name in enumerate(files):
        #dType = 'grd'
        dType = 'mlc'
        Region = 'norway'
        Heading=dir_base_file_name[10:13]
        Counter_num = dir_base_file_name[13:15]
        Year='15'
        Num_flights_year=dir_base_file_name[18:21]
        Data_take=dir_base_file_name[22:25]
        Day='10'
        Month='06'
        Band='L'
        Steering_angle='090'
        Cross_talk='CX'
        Processing_version='01'
        Polarization='VVVV'
        #====================================================
        
        Is_List_Ratio = False
        Cropping_switch = True
        Reproject_switch = False
        Plotting_switch = True
        Convolution_switch = True
        Inc_correction_switch = True
        Window_size_fea = 9
        Inc_correction_sin_degree = 2
        
        #====================================================
        
        ncomp=6
        tolerance=0.0001
        num_initialization=1
        max_iteration=200
        
        window_size_smoo = 35
        
        bin_size = 400
        
        #=====================================================
        
        wd=getdir(region=Region,heading=Heading,counter_num=Counter_num,year=Year,num_flights_year=Num_flights_year,data_take=Data_take,day=Day,month=Month,band=Band,steering_angle=Steering_angle,cross_talk=Cross_talk,processing_version=Processing_version)
        
        base_file_name=get_base_file_name(region=Region,heading=Heading,counter_num=Counter_num,year=Year,num_flights_year=Num_flights_year,data_take=Data_take,day=Day,month=Month,band=Band,steering_angle=Steering_angle,cross_talk=Cross_talk,processing_version=Processing_version)
        
        os.chdir(wd)
        #print(os.getcwd())
        
        #=======+Get Matadata========
        meta=metadata_dict(base_file_name)
        
        #=============Defining the subset=====================
        
        if dType == 'mlc':
            grd_cropping_list_00709=[521,1543,4049,5233]#[2000,3500,2500,4000]
            grd_cropping_list_18709=[1000,2300,3250,4450]
        elif dType == 'grd':
            grd_cropping_list_00709=[2000,3500,2500,4000]
            grd_cropping_list_18709=[900,2400,3150, 4650]
        
        if Heading=='007':
            cropping_list_GRD = grd_cropping_list_00709
            if (Counter_num == '10'):
                cropping_list_GRD = [2500,4000,2500,4000]
        elif Heading=='187':
            cropping_list_GRD = grd_cropping_list_18709
            if (Counter_num == '10'):
                cropping_list_GRD = [1400,2900,3150,4650]
        
        elif Heading == '142':
            cropping_list_GRD = [2500,4000,2500,4000]
        
        #===========Defining Convolution keranl===========
        
        Convolution_kernal = kernal(Window_size_fea)
        
        #============incidence angle=================
        
        grd_lines = int(meta['grd_pwr.set_rows'])
        grd_samples = int(meta['grd_pwr.set_cols'])
        
        inc_ang_arr = get_inc_angle(grd_lines, grd_samples, base_file_name, cropping_list_GRD, cropping_switch = Cropping_switch, is_List_Ratio=Is_List_Ratio)
        #inc_ang_arr = 0
        #plt.imshow(inc_ang_arr)
        #plt.show()
        
        #=============Extract GRD Component and plotting=====================
        plt.subplot(4,6,i+1)
        
        #grd_data = get_GRD_MLC(dType,meta, base_file_name,Convolution_kernal, inc_ang_arr, component = Polarization,grd_cropping_list=cropping_list_GRD, is_List_Ratio=False, cropping_switch=True, reproject_switch=False, plotting_switch=True, convolution_switch = True, inc_correction_switch=False, save_plot_switch = False)
        #time_stamp = meta['Date of Acquisition'][12:17]
        #plt.title(time_stamp + ['AM' if int(time_stamp[:2])<12 else ' PM' ][0], fontsize = 4)
        
        #plt.ylabel('Longitude' if((i)%6==0) else '', fontsize = 7)
        #plt.xlabel('Latitude' if (i>=18) else '', fontsize = 7)
    
    #plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/All_UAVSAR_plot_grd.tiff', dpi=300,bbox_inches='tight')
    #plt.show()
        cov_arr = get_covariance_matrix_grd(dType,meta, base_file_name, Convolution_kernal, inc_ang_arr,inc_correction_sin_degree=Inc_correction_sin_degree, cropping_list_GRD=cropping_list_GRD, is_List_Ratio=Is_List_Ratio, cropping_switch=Cropping_switch,convolution_switch = Convolution_switch,inc_correction_switch=Inc_correction_switch)
        
        
        #=============Convert to T3=========
        coh_arr = get_coherecy_matrix_grd(cov_arr)
        
        #########################################
        #============Extraction of Pol. Features===========
        #########################################
        
        eigen_raster = eigen_raster_full(coh_arr)
        
        arr_lamb1=eigen_raster[:,:,2]
        arr_lamb2=eigen_raster[:,:,1]
        arr_lamb3=eigen_raster[:,:,0]
        
        #============get_co_pol_diff============
        co_pol_diff = extract_polarimetric.co_pol_diff(cov_arr)
        arr_1 = 10*np.log10(np.absolute(arr_lamb3))
        arr_2 = 10*np.log10(np.absolute(co_pol_diff))
            
        feature_stack = np.dstack((arr_1, arr_2))#, arr_3, arr_4, cov_arr[...,2,2]))
    
    #-------Perform EPFS-----------
        gmm = EPFS.gmm_fitting_1(feature_stack, ncomp, tolerance=tolerance, num_initial=num_initialization, max_iteration=max_iteration)
        X = EPFS.prepare_X_all(feature_stack)
        shp=feature_stack.shape[:2]
        res=gmm.predict(X).reshape(shp[0],shp[1])
        im=plt.imshow(res.reshape(shp[0],shp[1]), cmap='RdYlBu')
        plt.ylabel('Azimuth' if((i)%6==0) else '', fontsize = 7)
        plt.xlabel('Range' if (i>=18) else '', fontsize = 7)
        time_stamp = meta['Date of Acquisition'][12:17]
        plt.title(time_stamp + ['AM' if int(time_stamp[:2])<12 else ' PM' ][0], fontsize = 4)
    plt.tight_layout()
    #plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Segmentation/Multi_temporal/Segmentation_ncomp_{}_inc_corr_degree_{}_tol10E-4.tiff'.format(str(ncomp),str(Inc_correction_sin_degree)), dpi=300,bbox_inches='tight')
    plt.show()
    

    
    


def main():
    
    plot_all_UAVSAR_data_takes()
    
    sys.exit()
    
    #UAVSAR_directory_name_prep
    
    base_dir='../North_Sea_UAVSAR/UAV_norway'
    file_ext=['.ann','.dat','.gif','.hgt','.inc','.kmz','.slope','_hgt.tif','_pauli.tif']
    #folders=['_mlc','_grd']
    dType = 'grd'
    Region='norway'
    Heading='007'
    Counter_num='09'
    Year='15'
    Num_flights_year='092'
    Data_take='000'
    Day='10'
    Month='06'
    Band='L'
    Steering_angle='090'
    Cross_talk='CX'
    Processing_version='01'
    Polarization='HHHV'
    #====================================================
    
    Is_List_Ratio=False
    Cropping_switch=True
    Reproject_switch=False
    Plotting_switch=True
    Convolution_switch = False
    Inc_correction_switch=False
    Window_size_fea = 9
    Inc_correction_sin_degree = 3
    
    #====================================================
    
    wd=getdir(region=Region,heading=Heading,counter_num=Counter_num,year=Year,num_flights_year=Num_flights_year,data_take=Data_take,day=Day,month=Month,band=Band,steering_angle=Steering_angle,cross_talk=Cross_talk,processing_version=Processing_version)
    
    base_file_name=get_base_file_name(region=Region,heading=Heading,counter_num=Counter_num,year=Year,num_flights_year=Num_flights_year,data_take=Data_take,day=Day,month=Month,band=Band,steering_angle=Steering_angle,cross_talk=Cross_talk,processing_version=Processing_version)
    
    os.chdir(wd)
    #print(os.getcwd())
    
    #=======+Get Matadata========
    meta=metadata_dict(base_file_name)
    
    
    
    #===============SLC========================
    #directory_SLC='/home/anurag/Documents/MScProject/SAR/OilSpill/North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_02'
    #print(slc_file_name)
    #print(os.getcwd())
    #print(meta)
    #print(os.getcwd())
    #os.chdir(directory_SLC)
    #slc_VV=get_SLC(meta,base_file_name, polarization='VV', dType='slc')
    #slc_HH=get_SLC(meta,base_file_name, polarization='HH', dType='slc')
    
    #=============Defining the subset=====================
    
    if dType == 'mlc':
            grd_cropping_list_00709=[521,1543,4049,5233]#[2000,3500,2500,4000]
            grd_cropping_list_18709=[1000,2300,3250,4450]
    elif dType == 'grd':
            grd_cropping_list_00709=[2000,3500,2500,4000]
            grd_cropping_list_18709=[900,2400,3150, 4650]
        
    if Heading=='007':
        cropping_list_GRD = grd_cropping_list_00709
        if (Counter_num == '10'):
            cropping_list_GRD = [2500,4000,2500,4000]
    elif Heading=='187':
        cropping_list_GRD = grd_cropping_list_18709
        if (Counter_num == '10'):
            cropping_list_GRD = [1400,2900,3150,4650]
    
    elif Heading == '142':
        cropping_list_GRD = [2500,4000,2500,4000]
    
    #===========Defining Convolution keranl===========
    
    Convolution_kernal = kernal(Window_size_fea)
    
    #============incidence angle=================
    
    grd_lines = int(meta['grd_pwr.set_rows'])
    grd_samples = int(meta['grd_pwr.set_cols'])
    
    inc_ang_arr = get_inc_angle(grd_lines, grd_samples, base_file_name, cropping_list_GRD, cropping_switch = Cropping_switch, is_List_Ratio=Is_List_Ratio)
    #plt.imshow(inc_ang_arr)
    #plt.show()
    
    #=============Extract GRD Component=====================
    
    get_GRD_MLC(dType,meta, base_file_name,Convolution_kernal, inc_ang_arr, component = Polarization,grd_cropping_list=cropping_list_GRD, is_List_Ratio=False, cropping_switch=True, reproject_switch=False, plotting_switch=True, convolution_switch = True, inc_correction_switch=True)
    
    
    #=============Plot all data takes=======================
    

    
    
    #==================Get covariance matrix===============
    cov_arr = get_covariance_matrix_grd(dType,meta, base_file_name, Convolution_kernal, inc_ang_arr,inc_correction_sin_degree=Inc_correction_sin_degree, cropping_list_GRD=cropping_list_GRD, is_List_Ratio=Is_List_Ratio, cropping_switch=Cropping_switch,convolution_switch = Convolution_switch,inc_correction_switch=Inc_correction_switch)
    
    
    #=============Convert to T3=========
    coh_arr = get_coherecy_matrix_grd(cov_arr)
    
    #########################################
    #============Extraction of Pol. Features===========
    #########################################
    
    eigen_raster = eigen_raster_full(coh_arr)
    
    arr_lamb1=eigen_raster[:,:,2]
    arr_lamb2=eigen_raster[:,:,1]
    arr_lamb3=eigen_raster[:,:,0]
    
    #============get_co_pol_diff============
    co_pol_diff = extract_polarimetric.co_pol_diff(cov_arr)
    
    #================Plotting lambda 3 and co_pol diff===============
    #import sys
    #sys.exit()
    plt.subplot(221)
    plt.imshow(10*np.log10(np.absolute(arr_lamb1)))
    plt.colorbar()
    
    plt.subplot(222)
    plt.imshow(10*np.log10(np.absolute(arr_lamb2)))
    plt.colorbar()
    
    plt.subplot(223)
    plt.imshow(10*np.log10(np.absolute(arr_lamb3)))
    plt.colorbar()
    
    plt.subplot(224)
    plt.imshow(np.log10(np.absolute(co_pol_diff)))
    plt.colorbar()
    
    plt.show()
    
    ######################################
    #=================EPFS===============
    ######################################
    ncomp=3
    tolerance=0.0001
    num_initialization=1
    max_iteration=200
    
    correction_switch=True
    window_size_fea=9
    window_size_smoo=1#35
    window_size_boun=3
    
    bin_size=400
    font_size = 15
    
    #-------Plot histogram--------
    arr_1 = EPFS.rescale(np.log10(np.absolute(arr_lamb3)), clip_extremes = False)
    arr_3 = EPFS.rescale(np.log10(np.absolute(arr_lamb2)), clip_extremes = False)
    arr_4 = EPFS.rescale(np.log10(np.absolute(arr_lamb1)), clip_extremes = False)
    arr_2 = EPFS.rescale(np.log10(np.absolute(co_pol_diff)), clip_extremes = False)
    
    plt.hist(arr_1.flatten(),bins=bin_size, rwidth=0.5,histtype='step', label='$\lambda_{3}$')
    plt.hist(arr_3.flatten(),bins=bin_size, rwidth=0.5,histtype='step', label='$\lambda_{2}$')
    plt.hist(arr_4.flatten(),bins=bin_size, rwidth=0.5,histtype='step', label='$\lambda_{1}$')
    plt.hist(arr_2.flatten(),bins=bin_size, rwidth=0.5,histtype='step', label='PD')
    
    plt.xlabel('Normalized values', fontsize=font_size)
    plt.ylabel('Frequency',fontsize=font_size)
    plt.legend()
    
    plt.show()
    
    #==========Stack of features===============
    feature_stack = np.dstack((arr_1, arr_2, arr_3, arr_4, cov_arr[...,2,2]))
    
    #-------Perform EPFS-----------
    gmm = EPFS.gmm_fitting_1(feature_stack, ncomp, tolerance=tolerance,num_initial=num_initialization, max_iteration=max_iteration)
    
    EPFS.plot_result(feature_stack, gmm, ncomp)
    #plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #=========Co-pol_phase_diff===========
    '''
    
    co_pol_phase_diff=np.angle(slc_VV)-np.angle(slc_HH)
    
    plt.subplot(121)
    plt.imshow(co_pol_phase_diff/np.pi*180)
    plt.colorbar()
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    
    plt.subplot(122)
    plt.hist(np.angle(co_pol_phase_diff).flatten()*180/np.pi, bins=300, rwidth=0.5,histtype='step', label='Copol_phase_diff')
    plt.xlabel('Copol_phase_diff')
    plt.ylabel('Frequency')
    
    
    
    
    
    #plot phase histogram
    #plt.hist(np.angle(slc_VV).flatten()*180/np.pi, bins=300, rwidth=0.5,histtype='step', label='SlC_VV phase')
    
    #plt.imshow(10*np.log10(np.absolute(slc_VV)), cmap='gray')
    plt.show()
    '''

if __name__=='__main__':
    main()
    