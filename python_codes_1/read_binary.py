import os
import struct
import numpy as np  
from matplotlib import pyplot as plt
import reproject
from scipy import ndimage
import numpy.ma as ma
from numpy import pi
from scipy import signal
import plotting
#os.chdir('/home/anurag/Documents/MScProject/SAR/OilSpill/North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01')

#os.chdir('/home/anurag/Documents/MScProject/SAR/OilSpill/North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15091_012_150610_L090_CX_02')

def bin_to_dec(string):
    print(struct.unpack("<f", string)[0])

def read_slope_file():
    slope_arr=np.empty((8836,4501,2))
    with open("norway_00709_15092_000_150610_L090_CX_01.slope", "rb") as f:
        #while(True):
        for j in range(0,8836):
            slope_list_east=[]
            slope_list_north=[]
            for i in range(0, 4501):
                total_east=f.read(4)
                total_north=f.read(4)
                slope_list_east.append(struct.unpack("<f", total_east)[0])
                slope_list_north.append(struct.unpack("<f", total_north)[0])
            slope_arr[j]=np.array([slope_list_east, slope_list_north]).T
    return slope_arr


def read_inc_file(lines, samples, base_file_name, cropping_list, cropping_switch = True, is_List_Ratio=False):    
    if (cropping_switch== True):
        if (is_List_Ratio==True):
            oil_rows_start=np.floor(cropping_list[2]*lines) #ratios gotten from estimates
            oil_rows_end=np.floor(cropping_list[3]*lines)
            oil_cols_start=np.floor(cropping_list[0]*samples)
            oil_cols_end=np.floor(cropping_list[1]*samples)
        else:
            oil_rows_start=cropping_list[2]
            oil_rows_end=cropping_list[3]
            oil_cols_start=cropping_list[0]
            oil_cols_end=cropping_list[1]
    else:
        oil_rows_start = 1
        oil_rows_end = lines
        oil_cols_start = 1
        oil_cols_end = samples
    
    tot_oil_rows=int(oil_rows_end-oil_rows_start)
    tot_oil_cols=int(oil_cols_end-oil_cols_start)
        
    inc_arr=np.empty((tot_oil_rows,tot_oil_cols))
    with open(base_file_name+".inc", "rb") as f:
        jump_to_oil=samples*(oil_rows_start-1)+oil_cols_start
        row_jump=samples-tot_oil_cols
        f.seek(int(jump_to_oil*4))
        
        for j in range(0,tot_oil_rows):
            inc_list=[]
            for i in range(0, tot_oil_cols):
                total=f.read(4)
                #print(total)
                inc_list.append(struct.unpack("<f", total)[0])
            f.read(int(row_jump*4))
            inc_arr[j]=np.array([inc_list])
            if(total==''):
                break
    return inc_arr
    
    
def reproject_inc_file(lines, samples, base_file_name):
    img_str=base_file_name+'_pauli.tif'
    reproject.save_inc_ang_tiff('inc_rad',img_str, samples,lines,read_inc_file())
    
def reproject_inc_file_rot():
    #img_str='norway_00709_15092_000_150610_L090_CX_01_pauli.tif'
    rot=2*pi-6.97008742*pi/180
    reproject.save_inc_ang_tiff_1('slope_rad',rot, 4501,8836,read_inc_file())
    
def plot_inc_ang_arr(lines, samples, base_file_name, cropping_list, cropping_switch = True, is_List_Ratio=False):
    inc_arr=read_inc_file(lines, samples, base_file_name)
    inc_arr_ma=ma.masked_values(inc_arr,-10000)
    #inc_arr_ma_rot=ndimage.interpolation.rotate(inc_arr,6.97008742)
    #inc_arr_ma_rot[np.where(inc_arr_ma_rot==0)]=-10000
    #inc_arr_ma_rot_ma=ma.masked_where(inc_arr_ma_rot==-10000,inc_arr_ma_rot)
    #print(inc_arr_ma_rot_ma)
    #inc_arr_flip=np.flip(np.flip(inc_arr_ma, axis=1), axis=0)
    #plt.subplot(1,2,1)
    #plt.imshow(inc_arr_flip, cmap='RdYlGn')
    #plt.subplot(1,2,2)
    plt.imshow(inc_arr_ma, cmap='RdYlGn')
    plt.colorbar()
    plt.show()
    
    
def plot_slope_arr(direction):
    slope_arr=read_slope_file()[...,direction]
    plt.imshow(slope_arr, cmap='hsv')
    plt.show()

def read_SLC(file_name, scan_lines, scan_pix, cropping_list, is_List_Ratio):
    
    '''
    cropping_list=[extent_xmin,extent_xmax, extent_ymin, extent_ymax]
    
    scan_lines=88086
    scan_pix=9900
    cropping_list=[.1,.55,.57,.72]
    '''
    
    slc_rows=scan_lines# norway_00709_15091_012_150610_L090_CX_02
    slc_cols=scan_pix
    iota=1j
    if (is_List_Ratio==True):
        oil_rows_start=np.floor(cropping_list[2]*slc_rows) #ratios gotten from estimates
        oil_rows_end=np.floor(cropping_list[3]*slc_rows)
        oil_cols_start=np.floor(cropping_list[0]*slc_cols)
        oil_cols_end=np.floor(cropping_list[1]*slc_cols)
    else:
        oil_rows_start=cropping_list[2]
        oil_rows_end=cropping_list[3]
        oil_cols_start=cropping_list[0]
        oil_cols_end=cropping_list[1]
    
    tot_oil_rows=int(oil_rows_end-oil_rows_start)
    tot_oil_cols=int(oil_cols_end-oil_cols_start)
        
    #tot_oil_rows=13000
    
    slc_oil_array=np.empty((tot_oil_rows,tot_oil_cols), dtype=np.complex64)
    
    
    with open(file_name, "rb") as f:

        #print(struct.unpack("<d", f.read(8))[0])
        
        jump_to_oil=slc_cols*(oil_rows_start-1)+oil_cols_start
        row_jump=slc_cols-tot_oil_cols
        #print(row_jump)
        f.seek(int(jump_to_oil*8)) #jumpinf to oil spill area
        #while(1==1):
            #a=struct.unpack("<f", f.read(4))[0]
            #b=struct.unpack("<f", f.read(4))[0]
            #print(np.sqrt(a**2+b**2))
        
        for j in range(0,tot_oil_rows):
            oil_list=[]
            for i in range(0, tot_oil_cols):
                DN_real=f.read(4)
                DN_imag=f.read(4)
                oil_list.append(struct.unpack("<f", DN_real)[0]+iota*struct.unpack("<f", DN_imag)[0])
            f.read(int(row_jump*8))
            slc_oil_array[j]=oil_list
    return slc_oil_array


def read_GRD(file_name, scan_lines, scan_pix, cropping_list, is_List_Ratio, meta, cropping_List_GRD):
    grd_rows=scan_lines# norway_00709_15091_012_150610_L090_CX_02
    grd_cols=scan_pix
    grd_pwr_val_size = int(meta['grd_pwr.val_size'])
    grd_mag_val_size = int(meta['grd_mag.val_size']) #bytes per pixel
    print(grd_pwr_val_size)
    iota=1j
    component=file_name[-14:-10]
    if (cropping_List_GRD== True):
        if (is_List_Ratio==True):
            oil_rows_start=np.floor(cropping_list[2]*grd_rows) #ratios gotten from estimates
            oil_rows_end=np.floor(cropping_list[3]*grd_rows)
            oil_cols_start=np.floor(cropping_list[0]*grd_cols)
            oil_cols_end=np.floor(cropping_list[1]*grd_cols)
        else:
            oil_rows_start=cropping_list[2]
            oil_rows_end=cropping_list[3]
            oil_cols_start=cropping_list[0]
            oil_cols_end=cropping_list[1]
    else:
        oil_rows_start=1
        oil_rows_end=grd_rows
        oil_cols_start=1
        oil_cols_end=grd_cols
    
    tot_oil_rows=int(oil_rows_end-oil_rows_start)
    tot_oil_cols=int(oil_cols_end-oil_cols_start)
    
    if (component=='VVVV' or component=='HHHH' or component=='HVHV'):
        bytes_per_pixel=grd_pwr_val_size
    elif(component=='HVVV' or component=='HHHV' or component=='HHVV'):
        bytes_per_pixel=grd_mag_val_size
    else:
        bytes_per_pixel=0
    print(component)
    print('bytes_per_pixel= '+str(bytes_per_pixel))
        
    grd_oil_array=np.empty((tot_oil_rows,tot_oil_cols), dtype=np.complex64)
    
    with open(file_name, "rb") as f:
        jump_to_oil=grd_cols*(oil_rows_start-1)+oil_cols_start
        row_jump=grd_cols-tot_oil_cols
        f.seek(int(jump_to_oil*bytes_per_pixel)) #jumpinf to oil spill area
        for j in range(0,tot_oil_rows):
            oil_list=[]
            for i in range(0, tot_oil_cols):
                if (component=='VVVV' or component=='HHHH' or component=='HVHV'):
                    
                    DN_real=f.read(bytes_per_pixel)
                    oil_list.append(struct.unpack("<f", DN_real)[0])
                elif(component=='HVVV' or component=='HHHV' or component=='HHVV'):

                    DN_real=f.read(bytes_per_pixel//2)
                    DN_imag=f.read(bytes_per_pixel//2)
                    
                    oil_list.append(struct.unpack("<f", DN_real)[0]+iota*struct.unpack("<f", DN_imag)[0])
            f.read(int(row_jump*bytes_per_pixel))
            grd_oil_array[j]=oil_list
    
    return grd_oil_array

def multilooking(win_x, win_y, slc_arr):
    #kern=kernal(win_x,win_y)
    #m_look_array=test_convolve(slc_arr, kern)
    #return m_look_array
    shp=slc_arr.shape
    #shp=fullshp[0:2].shape
    #def occurance_kernel(direction, method, window_size, stride_row, stride_col):
    #img_arr_hist_inci_corr=incidence_angle_corr.hist_stretch(incidence_angle_corr.inci_correction('C3', 'C33'), 5)
    #ndim=4
    dim=np.ndim(slc_arr)
    stride_row=win_y
    stride_col=win_x
    rows=shp[0]
    cols=shp[1]
    #print(rows)
    mod_shp=((int(rows/stride_row),)+(int(cols/stride_col)-1,))#-1,))
    #print(mod_shp)
    res=np.empty(mod_shp, dtype=np.complex64)
    res_row=0
    res_col=0
    for i in range(0, rows-win_y, stride_row):
        for j in range(0, cols-win_x, stride_col):
            a=slc_arr[i:i+win_y, j:j+win_x]
            #res[res_row,res_col]=np.mean(a*np.conj(a))
            #print(a)
            res[res_row,res_col]=(a*np.conj(a)).mean(1).mean(0)
            #res[res_row,res_col]=a.mean()#1).mean(0)
            #print((i,j))
            res_col+=1
        res_col=0
        res_row+=1
        print(i)
    #incidence_angle_corr.display(res, 'Range(pixel#)', 'Azimuth (pixel #)', 'Contrast GLCM feature(dir=0, window_size=15, row_stride=2, col_stride=2)')
    return res

def multilooking_1(win_x, win_y, slc_arr):
    #kern=kernal(win_x,win_y)
    #m_look_array=test_convolve(slc_arr, kern)
    #return m_look_array
    shp=slc_arr.shape
    #shp=fullshp[0:2].shape
    #def occurance_kernel(direction, method, window_size, stride_row, stride_col):
    #img_arr_hist_inci_corr=incidence_angle_corr.hist_stretch(incidence_angle_corr.inci_correction('C3', 'C33'), 5)
    #ndim=4
    dim=np.ndim(slc_arr)
    stride_row=win_y
    stride_col=win_x
    rows=shp[0]
    cols=shp[1]
    #print(rows)
    mod_shp=((int(rows/stride_row),)+(int(cols/stride_col)-1,)+shp[2:])
    print(mod_shp)
    res=np.empty(mod_shp, dtype=np.complex64)
    res_row=0
    res_col=0
    for i in range(0, rows-win_y, stride_row):
        for j in range(0, cols-win_x, stride_col):
            a=slc_arr[i:i+win_y, j:j+win_x, ...]
            #res[res_row,res_col]=np.mean(a*np.conj(a))
            print(a)
            res[res_row,res_col]=a.mean(1).mean(0)
            #res[res_row,res_col]=a.mean()
            #print((i,j))
            res_col+=1
        res_col=0
        res_row+=1
        print(i)
    #incidence_angle_corr.display(res, 'Range(pixel#)', 'Azimuth (pixel #)', 'Contrast GLCM feature(dir=0, window_size=15, row_stride=2, col_stride=2)')
    return res

def multilook_C3_1(win_x, win_y, C3_arr, leaping_win=True,stride_row=1, stride_col=1):
    shp=C3_arr.shape
    if(leaping_win==True):
        stride_row=win_y
        stride_col=win_x
    
    rows=shp[0]
    cols=shp[1]
    #res=np.zeros((math.floor(rows/(stride_row-1))-window_size, math.floor(cols/(stride_col-1))-window_size))
    #res=np.empty((int(rows/stride_row), int(cols/stride_col),3,3), dtype=np.complex64)
    res_row=0
    res_col=0
    for i in range(0, rows-win_y, stride_row):
        for j in range(0, cols-win_x, stride_col):
            a=C3_arr[i:i+win_y, j:j+win_x,...]
            C3_arr[i:i+win_y, j:j+win_x,...]=np.mean(np.mean(a,axis=0), axis=0)
            #return res
            res_col=j+win_x
        #res_col=0
        res_row=i+win_y
        print(i)
        
    if(res_row<rows):
        for j in range(0, cols-win_x, stride_col):
            a=C3_arr[res_row:rows, j:j+win_x,...]
            C3_arr[res_row:rows, j:j+win_x,...]=np.mean(np.mean(a,axis=0), axis=0)
            
        #for the Right-Lower patch
        a=C3_arr[res_row:rows, res_col:cols,...]
        C3_arr[res_row:rows, res_col:cols,...]=np.mean(np.mean(a,axis=0), axis=0)
    if(res_col<cols):
        for i in range(0, rows-win_y, stride_row):
            a=C3_arr[i:i+win_y, res_col:cols,...]
            C3_arr[i:i+win_y, res_col:cols,...]=np.mean(np.mean(a,axis=0), axis=0)
    
    #plot_arr=np.array(C3_arr[...,2,2], dtype=np.float64)
    #plt.imshow(plot_arr, cmap='gray')
    #plt.show()
    
    return C3_arr


def multilook_C3(win_x, win_y, C3_arr, leaping_win=True,stride_row=1, stride_col=1):
    shp=C3_arr.shape
    if(leaping_win==True):
        stride_row=win_y
        stride_col=win_x
    
    rows=shp[0]
    cols=shp[1]
    #res=np.zeros((math.floor(rows/(stride_row-1))-window_size, math.floor(cols/(stride_col-1))-window_size))
    #res=np.empty((int(rows/stride_row), int(cols/stride_col),3,3), dtype=np.complex64)
    res=np.empty((int(rows/stride_row), int(cols/stride_col)-1,*shp[2:]), dtype=np.complex64)
    
    res_row=0
    res_col=0
    for i in range(0, rows-win_y, stride_row):
        for j in range(0, cols-win_x, stride_col):
            a=C3_arr[i:i+win_y, j:j+win_x,...]
            res[res_row,res_col,...]=np.mean(np.mean(a,axis=0), axis=0)
            #return res
            res_col+=1
        res_col=0
        res_row+=1
        print(i)
    return res

if __name__=='__main__':
    #bin_to_dec(b'V?\x11?')
    plot_inc_ang_arr()
    #plot_slope_arr(1)
    #reproject_inc_file(8836, 4501, 'norway_00709_15092_000_150610_L090_CX_01')
    #reproject_inc_file_rot()
    #subsetting_ratios=[]
    
    
    '''
    scan_lines=88086
    scan_pix=9900
    cropping_list=[.11,.49,.57,.70]
    slc_VV=read_SLC("norway_00709_15091_012_150610_L090VV_CX_02.slc", scan_lines, scan_pix, cropping_list, True)
    #print(slc)
    multi_looked_arr_VV=multilooking(3,12,slc_VV)
    plt.figure(dpi = 150, tight_layout=True)
    #plt.imshow(10*np.log10(np.absolute(slc_VV)), cmap='gray')
    #plt.imshow(plotting.hist_stretch_all(10*np.log10(np.absolute(multi_looked_arr_VV)),0,True), cmap='gray')
    plt.imshow(10*np.log10(np.absolute(multi_looked_arr_VV)), cmap='gray')
    plt.colorbar(label='dB')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.show()
    '''
    #print(np.absolute(multilooking(1,1,np.array([[[1,2],[2,3], [3,4]], [[1,2],[2,3], [3,4]]]))))
    