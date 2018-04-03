from osgeo import gdal
import plotting
import read_binary
import extract_polarimetric
import os
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import numpy as np
import decomposition_SLC
import decomposition
import read_RISAT1
import numpy.ma as ma
from skimage.draw import circle
import matplotlib.patches as patches


def data_dake(directory_grd):
    os.chdir(directory_grd)
    C33=gdal.Open('C3/C33.bin')
    C33_gt=C33.GetGeoTransform()
    C33_arr=C33.ReadAsArray()
    
    inc_angle=gdal.Open('../inc_rad.tif')
    inc_arr=inc_angle.ReadAsArray()
    
    lat_pixels=list(range(1000,9000,1000))
    lon_pixels=list(range(1000,6000,1000))
    
    slick_lat_pixels=list(range(100,1100,200))
    slick_lon_pixels=list(range(100,1000,200))
    
    flight_centre=coord_to_pixel_id(C33_gt,59.950769642, 2.584468640)
    
    plot_arr=ma.masked_where(np.clip(10*np.log10(C33_arr),-25,0)==-30,np.clip(10*np.log10(C33_arr),-25,0))
    
    inc_arr=ma.masked_where(inc_arr==-10000, inc_arr)
    
    #adding flight center point, flight direction
    #add_patch(plot_arr,inc_arr, flight_centre, 20, C33_gt,lat_pixels, lon_pixels)
    
    
    
    add_oil_patch(plot_arr, slick_lon_pixels, slick_lat_pixels, C33_gt)
    
    
    
    plt.xlabel('Longitude $(^{0}$)')
    plt.ylabel('Latitude $(^{0}$)')
    
    plt.tight_layout()
    
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Data_take_oil_only_black_hori.tiff', dpi=300, papertype='a4', bbox_inches='tight')
    
    
    plt.show()
    


def pixel_id_to_lat(GT, Y):
    lat=GT[5]*Y + GT[3]
    return round(lat,2)

def pixel_id_to_lon(GT, X):
    lon=GT[1]*X + GT[0]
    return round(lon,2)
    
    
def coord_to_pixel_id(GT, lat, lon ):
    Y=(lat-GT[3])//GT[5]
    X=(lon-GT[0])//GT[1]
    return (X,Y)

def add_oil_patch(plot_arr,slick_lon_pixels,slick_lat_pixels, C33_gt):
    only_slicks=plot_arr[2616:3683,2284:3227]
    
    ax = plt.subplot(1,1,1)
    plt.imshow(only_slicks, cmap='gray')
    plt.colorbar(label='$I_{VV}(dB)$', orientation = 'horizontal')
    
    plt.text(150,226, 'E80', color='w', fontsize='20')
    plt.text(150,450, 'E60', color='w', fontsize='20')
    plt.text(150,692, 'E40', color='w', fontsize='20')
    plt.text(150,957, 'PO', color='w', fontsize='20')
    
    #rect = patches.Rectangle((1,1), 943-5,1067-5, linewidth=1,edgecolor='m',facecolor='none')
    #ax.add_artist(rect)
    
    plt.xticks(slick_lon_pixels, [str(pixel_id_to_lon(C33_gt,i+2616)) for i in slick_lon_pixels])
    plt.yticks(slick_lat_pixels, [str(pixel_id_to_lat(C33_gt,i+2284)) for i in slick_lat_pixels])
    

def add_patch(image_arr,inc_arr, top_left_id, radius, C33_gt, lat_pixels, lon_pixels):
    
    #lat_pixels=list(range(1000,5000,1000))
    #lon_pixels=list(range(1000,5000,1000))
    
    #adding flight center point
    circ = patches.Circle(top_left_id, radius, linewidth=1,edgecolor='k',facecolor='k', fill=True)
    ax = plt.subplot(1,1,1)
    
    inc_ang_clip_poly = patches.Polygon(np.array([[950,273],[4483,461],[4483-19,461+217],[931,490]]), closed=True, transform=ax.transData)
    
    C33_img_clip_poly = patches.Polygon(np.array([[931-19, 490+217],[4483-38,461+434],[3504,8579],[23,8373]]), closed=True, transform=ax.transData)
    
    #ax.add_artist(inc_ang_clip_poly)
    
    inc_im=plt.imshow(inc_arr*180/np.pi, cmap='gray_r', alpha=1)
    
    inc_im.set_clip_path(inc_ang_clip_poly)
    
    plt.colorbar(orientation='vertical', label='Incidence angle $(^{0})$', ticks=[20.5,30,40,50,60,67.5])
    
    C33_im = plt.imshow(image_arr, cmap='gray', alpha=0.5)
    
    C33_im.set_clip_path(C33_img_clip_poly)
    
    plt.colorbar(label='$I_{VV}(dB)$')
    
    plt.xlim(0,6000)
    
    ax.add_artist(circ)
    
    
    x,y,dx,dy=top_left_id[0],top_left_id[1],100,100*np.tan(97*np.pi/180)
    x2,y2=x+dx,y+dy
    
    ax.add_patch(patches.Arrow(x,y,dx,dy,width=50,facecolor='k'))
    
    ax.add_patch(patches.Arrow(x,y,-700,-700*np.tan(7*np.pi/180),width=50,facecolor='k', linestyle='dashed'))
    
    ax.add_patch(patches.Arrow(x,y,-510,-510*np.tan(97*np.pi/180),width=50,facecolor='k', linestyle='dashed'))
    
    ax.add_patch(patches.Arrow(5000,1000,0,-500,width=50,facecolor='k', linestyle='dashed'))  #north Arrow
    
    # Add swath width
    plt.annotate('', xy=(287, 6041), xycoords='data',xytext=(3533+287, 6041+188), textcoords='data',arrowprops={'arrowstyle': '<->'})
    
    #plt.annotate('Swath = 20 km', xy=(3533//2+287, 94+9041), xycoords='data',xytext=(5, 0), textcoords='offset points')
    ax.text(3533//2+287, 94+6041, 'Swath = 20 km', fontsize=7)
    
    
    ax.text(x2,y2,r'$\vec{v}$')
    ax.text(x,y,'O')
    ax.text(x-700,y-700*np.tan(7*np.pi/180),'A') #offset_row
    ax.text(x-510,y-510*np.tan(97*np.pi/180),'B') #offset_col
    
    ax.text(5000,500, 'N') 
    
    #To indicate the oil spill region
    rect = patches.Rectangle((2284,2616), 943,1067, linewidth=1,edgecolor='k',facecolor='none')
    ax.add_artist(rect)
    '''
    rect = patches.Rectangle((4651,6811), 1200, 1900, linewidth=1,edgecolor='r',facecolor='none')
    write_offset = 20
    y_iter=iter([20,40,60,80,100,120])
    
    write_x=4651+write_offset
    write_y=6811+next(y_iter)
    
    ax.text(write_x,write_y,r'$\vec{v}$: Velocity of airplane = 220 m/s')
    
    ax.add_artist(rect)
    '''
    
    #ax.text(1753,466, str((inc_arr[466,4459]*180/np.pi).round(2)))
    
    ax.text(2361,271, 'Incidence Angle', fontsize=7)
    
    plt.xticks(lon_pixels, [str(pixel_id_to_lon(C33_gt,i)) for i in lon_pixels])
    plt.yticks(lat_pixels, [str(pixel_id_to_lat(C33_gt,i)) for i in lat_pixels])
    
    

def plot_MLC_component(cov_arr, cov_add): #cov_add: list of 2 values containing row and col value of cov_matrix
    plt.imshow(10*np.log10(np.absolute(cov_arr[...,cov_add[0], cov_add[1]])), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.colorbar(label='dB', orientation='vertical', ticks=[-25, -20, -15, -10])
    #plt.title(title)
    plt.show()

def plot_SLC_component(arr, title):
    #plt.imshow(10*np.log10(np.absolute(arr[...,slc_add])), cmap='gray')
    plt.imshow(10*np.log10(np.absolute(arr)), cmap='gray')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.colorbar(label='dB', orientation='vertical')
    #plt.title(title)
    #plt.show()
    
def plot_int_VV_MLC_SLC(mlc_arr,mlc_add, slc_arr, slc_add): # here MLC - is the C3 MLC
    plt.subplot(121)
    plot_MLC_component(mlc_arr,mlc_add)
    plt.subplot(122)
    plot_SLC_component(slc_arr, slc_add)

def plot_int_VV_MLC_SLC_1(mlc_arr, slc_arr, mlc_title, slc_title):
    plt.subplot(111)
    plot_SLC_component(mlc_arr, mlc_title)
    
    #plt.subplot(122)
    #plot_SLC_component(slc_arr, slc_title)
    

def plot_SLC_MLC(file_name, directory_SLC):

    os.chdir(directory_SLC)
    slc_VV=read_binary.read_SLC(file_name, scan_lines, scan_pix, slc_cropping_list, False)
    mlc_VV=read_binary.multilooking(multilook_x, multilook_y, slc_VV)
    matplotlib.rcParams.update({'font.size': 8})
    fig=plt.figure()
    #os.chdir('../'+directory_MLC)
    #mlc_arr=extract_polarimetric.extract_covariance_arr(window_size, correction_switch, degree)
    plt.tight_layout()
    plot_int_VV_MLC_SLC_1(mlc_VV,slc_VV, r'$MLC: |S_{VV}S_{VV}^{*}|$', '$SLC: |S_{VV}|$')
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/1_MLC.tiff', dpi=300, papertype='a4', bbox_inches='tight')
    plt.show()

def plot_PauliRGB_Components(S_arr, plot_pos_list):
    matplotlib.rcParams.update({'font.size': 5})
    fig=plt.figure()
    #fig, ax = plt.subplots()
    #plt.subplot(2,3,2)
    gridspec.GridSpec(3,3)
    
    plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
    #plt.locator_params(axis='x', nbins=5)
    #plt.locator_params(axis='y', nbins=5)
    pauli_arr=decomposition_SLC.Pauli_RGB_array(S_arr, True, True, 3,12)
    #plt.subplot(plot_pos_list)
    decomposition_SLC.plot_decomposition_comp(pauli_arr, False, [r'|$\alpha$|', r'|$\beta$|', r'|$\gamma$|'], '', plot_pos_list)
    #print(1)
    plt.tight_layout()
    #plt.suptitle('Pauli Decomposition')
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/3_Pauli_ML_corrected.tiff', dpi=300, papertype='a4', bbox_inches='tight')#, bbox_inches='tight')
    plt.show()

def plot_KrogagerRGB_Components(S_arr, plot_pos_list):
    matplotlib.rcParams.update({'font.size': 5})
    fig=plt.figure()
    #fig, ax = plt.subplots()
    #plt.subplot(2,3,2)
    gridspec.GridSpec(3,3)
    
    plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
    krogager_arr=decomposition_SLC.krogager_array(S_arr, True, True, 3,12)
    #plt.subplot(plot_pos_list)
    decomposition_SLC.plot_decomposition_comp(krogager_arr, False, [r'|$k_{d}$|', r'|$k_{h}$|', r'|$k_{s}|$'], '', plot_pos_list)
    #print(1)
    plt.tight_layout()
    #plt.suptitle('Pauli Decomposition')
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Krogager_ML.tiff', dpi=300, papertype='a4', bbox_inches='tight')#, bbox_inches='tight')
    plt.show()

def plot_Freeman(directory_MLC, window_size, correction, degree):
    os.chdir(directory_MLC)
    matplotlib.rcParams.update({'font.size': 5})
    #fig=plt.figure()
    #gridspec.GridSpec(3,3)
    
    #plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
    plotting.plot_freeman_RGB(window_size, correction, degree)
    plt.tight_layout()
    #plt.suptitle('Pauli Decomposition')
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Freeman_ML_new.tiff', dpi=300, papertype='a4', bbox_inches='tight')#, bbox_inches='tight')
    plt.show()
    
def plot_RISAT_m_chi_delta_alpha(directory_RISAT1):
    #os.chdir(directory_RISAT1)
    #window_size=10
    arr=read_RISAT1.img_to_array(directory_RISAT1)
    Srh_arr=read_RISAT1.oil_subset(arr[...,0])
    Srv_arr=read_RISAT1.oil_subset(arr[...,1])
    
    #Srh_arr=read_RISAT1.averaging_arr(Srh_arr,9)
    #Srv_arr=read_RISAT1.averaging_arr(Srv_arr,9)
    
    matplotlib.rcParams.update({'font.size': 5})
    fig=plt.figure()
    plt.tight_layout()
    plt.subplot(1,3,1)
    read_RISAT1.m_chi_decomposition(Srh_arr,Srv_arr, window_size)
    
    plt.subplot(1,3,2)
    read_RISAT1.m_delta_decomposition(Srh_arr,Srv_arr, window_size)
    
    plt.subplot(1,3,3)
    read_RISAT1.m_alpha_decomposition(Srh_arr,Srv_arr, window_size)
    
    plt.tight_layout()
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/m_chi_delta_alpha_decomposition_win_9.tiff', dpi=300, papertype='a4', bbox_inches='tight')#, bbox_inches='tight')
    plt.show()

def plot_all_UAVSAR_features(directory_MLC,window_size, correction_switch, degree):
    os.chdir(directory_MLC)
    #ax = plt.subplots(3,3)
    matplotlib.rcParams.update({'font.size': 3})
    
    cov_arr=extract_polarimetric.extract_covariance_arr(window_size, correction_switch, degree)
    
    
    coh_arr=extract_polarimetric.extract_coherency_arr(window_size, correction_switch, degree)
    
    #===============1=============
    plt.subplot(5,4,1)
    plt.imshow(10*np.log10(np.absolute(cov_arr[...,0,0])), cmap='gray')
    plt.colorbar(label = 'dB')
    plt.ylabel('Azimuth')
    plt.title('$I_{HH}$')
    
    plt.subplot(5,4,2)
    plt.imshow(10*np.log10(np.absolute(cov_arr[...,1,1])), cmap='gray')
    plt.colorbar(label = 'dB')
    #plt.ylabel('Azimuth')
    plt.title('$I_{HV}$')
    
    plt.subplot(5,4,3)
    plt.imshow(10*np.log10(np.absolute(cov_arr[...,2,2])), cmap='gray')
    plt.colorbar(label = 'dB')
    #plt.ylabel('Azimuth')
    plt.title('$I_{VV}$')
    
    
    #===============1=============
    arr_co_pol_pow_ratio = extract_polarimetric.co_pol_power_ratio_1(cov_arr)
    plt.subplot(5,4,4)
    plt.imshow(np.absolute(arr_co_pol_pow_ratio), cmap='gray')
    plt.colorbar()
    #plt.ylabel('Azimuth')
    plt.title('$\gamma_{CO}$')
    
    #===============2=============
    Rco_X = np.real(cov_arr[...,0,2])
    Ico_X = np.imag(cov_arr[...,0,2])
    plt.subplot(5,4,5)
    plt.imshow(10*np.log10(Rco_X), cmap='gray')
    plt.colorbar(label = 'dB')
    plt.title('$r_{CO}$')
    plt.ylabel('Azimuth')
    
    
    #===============6=============
    plt.subplot(5,4,6)
    plt.imshow(10*np.log10(abs(Ico_X)), cmap='gray')
    plt.colorbar(label = 'dB')
    plt.title('$|i_{CO}|$')
    
    
    #===============4=============
    plt.subplot(5,4,7)
    slc_dir='/home/anurag/Documents/MScProject/SAR/OilSpill/North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_02'
    sd_phase_diff=extract_polarimetric.sd_co_pol_phase_diff(slc_dir)
    plt.imshow(np.absolute(sd_phase_diff), cmap='gray')
    plt.colorbar()
    plt.title('$\phi_{CO}$')
    
    #===============5=============
    plt.subplot(5,4,8)
    plt.imshow(10*np.log10(extract_polarimetric.determinant_cov(cov_arr)), cmap='gray')
    plt.colorbar(label = 'dB')
    plt.title('$det (C_{FP})$')
    #plt.ylabel('Azimuth')
    
    #===============6=============
    plt.subplot(5,4,9)
    plt.imshow(10*np.log10(np.absolute(cov_arr[...,0,0] - cov_arr[...,2,2])), cmap='gray')
    plt.colorbar(label = 'dB')
    plt.title('PD')
    plt.ylabel('Azimuth')
    
    #===============7=============
    plt.subplot(5,4,10)
    plt.imshow(np.absolute(cov_arr[...,1,1])/np.absolute((cov_arr[...,0,0]+cov_arr[...,2,2])), cmap='gray')
    plt.colorbar()
    plt.title('$P_{X}$')
    
    #===============8=============
    
    eigen_full=extract_polarimetric.eigen_raster_full( window_size, correction_switch, degree)
    arr_lamb1=eigen_full[:,:,2]
    arr_lamb2=eigen_full[:,:,1]
    arr_lamb3=eigen_full[:,:,0]
    
    #===============8=============
    plt.subplot(5,4,11)
    plt.imshow(10*np.log10(np.absolute(arr_lamb1)), cmap='gray')
    plt.colorbar(label = 'dB')
    plt.title('$\lambda_{1}$')
    
    #===============9=============
    plt.subplot(5,4,12)
    plt.imshow(10*np.log10(np.absolute(arr_lamb2)), cmap='gray')
    plt.colorbar(label = 'dB')
    #plt.ylabel('Azimuth')
    plt.title('$\lambda_{2}$')
    
    #===============10=============
    plt.subplot(5,4,13)
    plt.imshow(10*np.log10(np.absolute(arr_lamb3)), cmap='gray')
    plt.colorbar(label = 'dB')
    plt.title('$\lambda_{3}$')
    plt.ylabel('Azimuth')
    #===============11=============
    plt.subplot(5,4,14)
    arr_ent=extract_polarimetric.entropy(eigen_full)
    plt.imshow(10*np.log10(np.absolute(arr_ent)), cmap='gray')
    plt.colorbar()
    plt.title('$H$')
    
    #===============12=============
    plt.subplot(5,4,15)
    arr_pol_frac=extract_polarimetric.pol_fraction(eigen_full)
    plt.imshow(arr_pol_frac, cmap='gray')
    plt.colorbar()
    plt.title('$PF$')
    
    #===============13=============
    plt.subplot(5,4,16)
    arr_anisotropy=extract_polarimetric.anisotropy(eigen_full)
    plt.imshow(arr_anisotropy, cmap='gray')
    plt.colorbar()
    #plt.xlabel('Range')
    #plt.ylabel('Azimuth')
    plt.title('$A$')
    
    #================14==============
    plt.subplot(5,4,17)
    co_pol_corr_arr=extract_polarimetric.co_pol_correlation(cov_arr)
    plt.imshow(np.absolute(co_pol_corr_arr), cmap='gray')
    plt.colorbar()
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('$Co-pol Correlation$')
    
    #================15==============
    plt.subplot(5,4,18)
    arr_conform_coeff=extract_polarimetric.conformity_coeff(cov_arr)
    plt.imshow(np.absolute(arr_conform_coeff), cmap='gray')
    plt.colorbar()
    plt.xlabel('Range')
    plt.title('Conformity Coefficient')
    
    #================16==============
    plt.subplot(5,4,19)
    alpha = extract_polarimetric.mean_alpha_angle(eigen_full,window_size, correction_switch, degree)
    plt.imshow(alpha, cmap='gray')
    plt.colorbar(label = 'degrees')
    plt.title('Mean alpha angle')
    plt.xlabel('Range')
    
    #================16==============
    plt.subplot(5,4,20)
    PH = extract_polarimetric.pedestal_height(eigen_full)
    plt.imshow(PH, cmap='gray')
    plt.colorbar()
    plt.title('PH')
    plt.xlabel('Range')
    
    plt.tight_layout()
    
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Features/Polarimetric_Features/Inc_corr_applied/Det_cov_conj.tiff', dpi=300, box_inches = 'tight',papertype = 'a4',orientation = 'portrait')
    
    plt.show()

def plot_all_RISAT1_features():
    a=1




if __name__=='__main__':
    #MLC argument initialization
    window_size=1
    correction_switch=False
    degree=2
    mlc_add,slc_add=[2,2], 0
    multilook_x=3
    multilook_y=12
    #SLC argument initialization
    #scan_lines=88086
    #scan_pix=9900
    #cropping_list=[.11,.49,.57,.70]
    mlc_row_looks=12
    mlc_col_looks=3
    scan_lines=86417
    scan_pix=9900
    mlc_cropping_list=[521,1545,4049,5233]
    slc_cropping_list=[mlc_cropping_list[0]*mlc_col_looks,\
        mlc_cropping_list[1]*mlc_col_looks,\
            mlc_cropping_list[2]*mlc_row_looks,\
                mlc_cropping_list[3]*mlc_row_looks]
    file_name="norway_00709_15092_000_150610_L090VV_CX_02.slc"
    directory_SLC='/home/anurag/Documents/MScProject/SAR/OilSpill/North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_02'
    directory_MLC='/home/anurag/Documents/MScProject/SAR/OilSpill/North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01/MLC_Python_New/norway_00709_15092_000_150610_L090_CX_01_mlc'
    
    directory_grd = '/home/anurag/Documents/MScProject/SAR/OilSpill/North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01/norway_00709_15092_000_150610_L090_CX_01_grd'
    
    directory_RISAT1='../RISAT-1/RI1_SAR_L1SLC_FRS1_CR_20150610T071918_20150610T071923_17197_1515551004'
    
    plot_pos_list=[234,235,236]
    
    os.chdir(directory_SLC)
    #S = np.load('S_oil_SLC.npy')
    S_mlc=np.load('S_mlc_only_oil.npy')
    
    
    #============Plotting SLC_MLC======side by side
    #print(S_mlc.shape)
    #plot_SLC_component(np.absolute(S_mlc[...,2]), 'test')
    plot_SLC_MLC(file_name, directory_SLC)
    #==========Plotting Pauli MLC==============
    #plot_PauliRGB_Components(S, plot_pos_list)
    #plot_KrogagerRGB_Components(S, plot_pos_list)
    #============Plotting Freeman============
    #plot_Freeman(directory_MLC,window_size, correction_switch, degree)
    
    #=============Plotting m-chi and m-delta decomposition============
    #plot_RISAT_m_chi_delta_alpha(directory_RISAT1)
    
    #============Plotting the Data take==========
    #data_dake(directory_grd)
    
    #==============Plotting all extracted features==========
    #plot_all_UAVSAR_features(directory_MLC,9,False,0)
    
    
    