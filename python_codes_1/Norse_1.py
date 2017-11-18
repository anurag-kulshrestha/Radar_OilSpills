#python for NORSE-2015

from osgeo import gdal
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#top_left=gdal.Open('subset_0_of_S1A_IW_GRDH_1SDV_20170129T003132_20170129T003157_015039_01892E_6D04_Amplitude_VV_top_left_new.tif')

os.chdir('../North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01/norway_00709_15092_000_150610_L090_CX_01_grd_1/C3')
print(os.getcwd())
#/home/anurag/Documents/MScProject/SAR/OilSpill/North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01/norway_00709_15092_000_150610_L090_CX_01_grd_1

spills_c11=gdal.Open('C11.bin')
spills_gt=spills_c11.GetGeoTransform()
C3=spills_c11.ReadAsArray()

def getGT():
    spills_c11=gdal.Open('C11.bin')
    spills_gt=spills_c11.GetGeoTransform()
    return spills_gt

def create_C3():
    spills_c11=gdal.Open('C11.bin')
    spills_gt=spills_c11.GetGeoTransform()
    C3=spills_c11.ReadAsArray()
    
    spills_c22=gdal.Open('C22.bin')
    
    

#def create_T3():
    


def dimension_data(data):
    return data.shape


if __name__=='__main__':
    #print(dimension_data(C3[0,0]))
    print(C3[,,0])