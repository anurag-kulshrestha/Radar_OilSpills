from osgeo import gdal
import numpy as np
import os
from pyproj import Proj, transform

#os.chdir('../North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01/MLC_Python_New/norway_00709_15092_000_150610_L090_CX_01_mlc')

def read_image(matrix, element):
    mat_ele=matrix+'/'+element
    img=gdal.Open(mat_ele+'.bin')
    return img

def get_image_geo_details(matrix='C3', element='C33' ):
    img=read_image(matrix, element)
    img_gt=img.GetGeoTransform()
    img_proj = img.GetProjection()
    img_dim=[img.RasterXSize, img.RasterYSize, img.RasterCount]
    return [img_dim, img_proj, img_gt]

def convert_proj_sys(x1, y1, in_epsg, out_epsg):
    inProj = Proj(init='epsg:'+str(in_epsg))
    #print(inProj)
    outProj = Proj(init='epsg:'+str(out_epsg))
    #x1,y1 = -1170reproject_image_complex5274.6374,4826473.6922
    x2,y2 = transform(inProj,outProj,x1,y1)
    return (x2,y2)


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

def reproject_image_complex(newname, newRasterXSize, newRasterYSize, bands, output_array, projection, geotransform): #makes the array into a raster image
    driver=gdal.GetDriverByName('GTiff')
    newdataset=driver.Create(newname+'.tif',newRasterXSize,newRasterYSize,bands, gdal.GDT_CFloat32)
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
    #shp=output_array.shape
    output_dir=os.getcwd()+'/Output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.chdir('Output')
    reproject_image(name, geo[0][0]-window_size+1, geo[0][1]-window_size+1, geo[0][2], output_array,geo[1], geo[2])
    
def save_tiff_image_1(name, output_array):
    
    shp=output_array.shape
    print(shp)
    geo=get_image_geo_details()
    img_dim=geo[0]
    print(img_dim)
    img_gt=list(geo[2])
    #img_gt[1]=img_gt[1]*img_dim[0]/shp[0]
    #img_gt[4]=img_gt[4]*img_dim[1]/shp[1]
    
    output_dir=os.getcwd()+'/Output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.chdir('Output')
    reproject_image(name, shp[1], shp[0], 1, output_array, geo[1], img_gt)
    
    
def save_inc_ang_tiff(name,img_str, newRasterXSize,newRasterYSize,output_arr):
    img=gdal.Open(img_str)
    img_gt=img.GetGeoTransform()
    img_proj = img.GetProjection()
    img_dim=[img.RasterXSize, img.RasterYSize, img.RasterCount]
    #print(img_dim)
    reproject_image(name, newRasterXSize, newRasterYSize, 1, output_arr, img_proj, img_gt)

def save_inc_ang_tiff_1(name,rot_angle, newRasterXSize,newRasterYSize,output_arr):
    #img=gdal.Open(img_str)
    img_gt=[0,np.cos(rot_angle),-1*np.sin(rot_angle),0,np.sin(rot_angle),np.cos(rot_angle)]
    img_proj = 1
    #img_dim=[img.RasterXSize, img.RasterYSize, img.RasterCount]
    #print(img_dim)
    reproject_image(name, newRasterXSize, newRasterYSize, 1, output_arr, img_proj, img_gt)

if __name__=='__main__':
    print(get_image_geo_details()[2])
    a=1
    #save_tiff_image('test', [0,1])
    #reproject_image()