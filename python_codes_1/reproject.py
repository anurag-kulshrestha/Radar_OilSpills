from osgeo import gdal
import numpy as np
import os

#os.chdir('../North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01/MLC_Python_New/norway_00709_15092_000_150610_L090_CX_01_mlc')

def read_image(matrix, element):
    mat_ele=matrix+'/'+element
    img=gdal.Open(mat_ele+'.bin')
    return img

def get_image_geo_details(matrix='C3', element='C33'):
    img=read_image(matrix, element)
    img_gt=img.GetGeoTransform()
    img_proj = img.GetProjection()
    img_dim=[img.RasterXSize, img.RasterYSize, img.RasterCount]
    return [img_dim, img_proj, img_gt]


def reproject_image(newname, newRasterXSize, newRasterYSize, bands, output_array, projection, geotransform): #makes the array into a raster image
    driver=gdal.GetDriverByName('GTiff')
    newdataset=driver.Create(newname+'.tif',newRasterXSize,newRasterYSize,bands, gdal.GDT_Float32)
    newdataset.SetProjection(projection)
    newdataset.SetGeoTransform(geotransform)
    if bands>1:
        for i in range(1,bands+1):
            newdataset.GetRasterBand(i).WriteArray(output_array[i-1])
    else:
        newdataset.GetRasterBand(1).WriteArray(output_array)
    newdataset.FlushCache()
    
def save_tiff_image(name, output_array, window_size):
    geo=get_image_geo_details()
    
    output_dir=os.getcwd()+'/Output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.chdir('Output')
    reproject_image(name, geo[0][0]-window_size+1, geo[0][1]-window_size+1, geo[0][2], output_array,geo[1], geo[2])

if __name__=='__main__':
    a=1
    #save_tiff_image('test', [0,1])
    #reproject_image()