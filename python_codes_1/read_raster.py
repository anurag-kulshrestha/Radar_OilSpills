from osgeo import gdal
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

os.chdir('../North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01/MLC_Python_New/norway_00709_15092_000_150610_L090_CX_01_mlc')
#print(os.getcwd())

C33=gdal.Open('C3/C33.bin')
C33_gt=C33.GetGeoTransform()
arr_C33=C33.ReadAsArray()
#print(C33.RasterXSize, C33.RasterYSize, C33.RasterCount)
print(arr_C33.shape)