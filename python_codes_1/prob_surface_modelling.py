import plotting 
from osgeo import gdal, ogr, osr
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import pandas as pd
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
import classification
import W_MRF
from scipy import signal

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import feature_selection
import scipy.optimize as opt
from matplotlib import cm


'''
## Create a GL View widget to display data
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('Plot 3D')
w.setCameraPosition(distance=50)

## Add a grid to the view
g = gl.GLGridItem()
g.scale(1,1,20)
g.setDepthValue(10)  # draw grid after surfaces since they may be translucent
w.addItem(g)

## Simple surface plot example
## x, y values are not specified, so assumed to be 0:50
z = np.random.normal(size=(50,50))
p1 = gl.GLSurfacePlotItem(z=z, shader='shaded', color=(0.5, 0.5, 1, 1))
#p1.scale(1., 1, 1.0)
#p1.translate(-18, 2, 0)
w.addItem(p1)

index = 0
def update():
    global p4, z, index
    index -= 1
    p4.setData(z=z[index%z.shape[0]])
    
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(30)
'''

def plot_3D_prob_surface(df, shp_cov):
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #print(np.array(df['Oil'].as_matrix().reshape(shp_cov[0],shp_cov[1])).astype(np.complex64))
    
    #Z=np.array(np.absolute(df['Oil'].as_matrix()).reshape(shp_cov[0],shp_cov[1]), dtype=np.float64)
    Z=df
    #Z=Z/np.amax(Z)
    
    
    #Z=np.absolute(df['Oil'].as_matrix().reshape(shp_cov[0],shp_cov[1]))

    X=np.arange(0,shp_cov[0])
    Y=np.arange(0,shp_cov[1])
    
    X,Y=np.meshgrid(Y,X)
    #print(X.shape, Y.shape,Z.shape)
    
    
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    #surf = ax.plot_surface(X, Y, Z,  linewidth=0, antialiased=True)

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def gauss_2D(xy,x_0,y_0,I,theta,sigma_x, sigma_y):
#def gauss_2D(xy,I,theta,sigma_x, sigma_y):
    #x_0,y_0 = 640,320
    
    #x_0,y_0 = next(X_NOTS),next(Y_NOTS)
    
    x,y=xy
    x_dash=(x-x_0)*np.cos(theta) - (y-y_0)*np.sin(theta)
    y_dash=(x-x_0)*np.sin(theta) + (y-y_0)*np.cos(theta)
    
    z=I*np.exp(-0.5*((x_dash/sigma_x)**2 + (y_dash/sigma_y)**2))
    return z
    

def gaussian_Fitting(df, shp_cov, slick_mask):
    
    YX=np.where(slick_mask==1)
    #YX=np.where(df > 0.000000000000001)
    print('num_of_slick_pixels={}'.format(YX[0].size))
    Z=df[YX]
    
    XY=np.array([YX[1],YX[0]])
    
    yx_not = np.where(df==df.max())
    y_not,x_not=yx_not[0][0],yx_not[1][0]
    print('Max_prob_point_id={} , {}'.format(y_not,x_not))
    
    guess = [x_not, y_not, 0.5, 0.04, 1,1]
    #guess = [1, 1, 1, 0.01, 1,1]
    #guess = [0.5, 5, 10,10]
    pred_params, uncert_cov = opt.curve_fit(gauss_2D, XY, Z, p0=guess)
    #print (uncert_cov)
    #plot_fitted_gauss_model(pred_params, shp_cov)
    return pred_params

def plot_fitted_gauss_model(pred_params, shp_cov, window_size_avg):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    X=np.arange(0,shp_cov[0])
    Y=np.arange(0,shp_cov[1])
    
    XY=np.meshgrid(Y,X)
    
    pred_params_PO = pred_params[0]
    pred_params_E40 = pred_params[1]
    pred_params_E60 = pred_params[2]
    pred_params_E80 = pred_params[3]
    #print(pred_params_PO)
    Z_pred_PO = gauss_2D(XY,*pred_params_PO)
    Z_pred_E40 = gauss_2D(XY,*pred_params_E40)
    Z_pred_E60 = gauss_2D(XY,*pred_params_E60)
    Z_pred_E80 = gauss_2D(XY,*pred_params_E80)
    
    Z_pred_all = Z_pred_PO + Z_pred_E40 + Z_pred_E60 + Z_pred_E80
    #print(Z_pred_PO)
    
    X,Y=XY
    surf = ax.plot_surface(X, Y, Z_pred_all , cmap=cm.coolwarm, linewidth=0, antialiased=True)
    #plt.imshow(Z_pred_all, cmap='jet')
    #ax.colorbar(label='Modelled Oil Probability')
    ax.set_xlabel('Range')
    ax.set_ylabel('Azimuth')
    ax.set_zlabel('Probability')
    
    
    plt.tight_layout()
    
    #plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Classification/Prob_Surface_Modelling/Probability_surface_3D_win_size_'+str(window_size_avg)+'1.tiff', dpi=300, papertype='a4', bbox_inches='tight')
    plt.show()
    
    
    #ax = fig.gca(projection='2d')
    plt.imshow(Z_pred_all, cmap='jet')
    plt.colorbar(label='Modelled Oil Probability')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.tight_layout()
    
    #plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Classification/Prob_Surface_Modelling/Probability_surface_coutour_win_size_'+str(window_size_avg)+'.tiff', dpi=300, papertype='a4', bbox_inches='tight')
    
    plt.show()
    
#def main():

def gaussian_fitting_all():
    a=1

def test_convolve(arr, kernal):
    grad = signal.convolve2d(arr, kernal, boundary='symm', mode='valid')
    return grad

def kernal(window_size):
    k=np.ones(window_size*window_size).reshape(window_size, window_size)
    normalize_k=k/(window_size**2)
    return normalize_k

if __name__=='__main__':
    #import sys
    #if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        #QtGui.QApplication.instance().exec_()
    
    #==========initilizing variables===========
    
    window_size_cov, correction_switch, degree=1,False,1
    num_features=8
    num_classes=2
    m=5
    Niter=10
    Looks=12*3
    #Looks=1
    beta=1.4
    
    window_size_avg = 1
    
    os.chdir('../North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01/MLC_Python_New/norway_00709_15092_000_150610_L090_CX_01_mlc')
    
    #==========Getting Slick Masks===========
    slicks = feature_selection.get_slick_wise_mask()
    #print (slicks.shape)
    #plt.imshow(slicks[...,0], cmap='gray', alpha=0.1)
    #plt.imshow(slicks[...,1], cmap='gray', alpha=0.1)
    #plt.imshow(slicks[...,2], cmap='gray', alpha=0.1)
    #plt.imshow(slicks[...,3], cmap='gray', alpha=0.1)
    #plt.show()
    
        
    #============Get W_classification results=========
    #============Loading Wishart Image==========
    f='/home/anurag/Documents/MScProject/Meetings_ITC/Results/Classification/Wishart_Looks_36_m_5_absolute_val.npy'
    
    
    Wishart_oil=np.load(f)
    shp_cov = Wishart_oil.shape
    window_size_avg_list= [1,5,11,21,31,41,51] 
    
    for window_size_avg in window_size_avg_list:
        print ('Window_size ={}x{}'.format(window_size_avg,window_size_avg))
        #=========Averaging and padding to get the same extent============
        kern=kernal(window_size_avg)
        Wishart_oil=np.pad(test_convolve(Wishart_oil, kern), window_size_avg//2,'constant')
        
        #===========Normalization========
        #Wishart_oil=Wishart_oil/np.amax(Wishart_oil)
        
        #==========Plotting Probability Surface==========
        #plot_3D_prob_surface(Wishart_oil, shp_cov)
        
        #Plotting the image
        print(Wishart_oil.max())
        plt.imshow(Wishart_oil,cmap='gray_r', alpha=0.5)
        plt.imshow(classification.get_slick_boundry(), alpha=0.5, cmap='gray_r')
        
        plt.colorbar(label='Oil Probability')
        plt.xlabel('Range')
        plt.ylabel('Azimuth')
        
        plt.tight_layout()
        #plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Classification/Prob_Surface_Modelling/Slick_and_probability_Win_size_'+str(window_size_avg),dpi=300, papertype='a4', bbox_inches='tight')
        plt.show()
        
        
        
        
        
        #Wishart=pd.read_csv(f, delimiter=',', index_col=0, names=['a', 'b'], converters={1: parse_pair, 2: parse_pair,3:parse_pair})
        
        #print(Wishart)
        
        #Wishart = W_MRF.Wishart_Likelihood(num_classes,m,beta, cov_arr,window_size_cov, Niter, Looks,cov_df,TR,q=3)
        
        #===============Get masked slicks and their values=============
        PO_loc=np.where(slicks[...,0]==1)
        E40_loc=np.where(slicks[...,1]==1)
        E60_loc=np.where(slicks[...,2]==1)
        E80_loc=np.where(slicks[...,3]==1)
        
        
        prob_PO=Wishart_oil[PO_loc]
        prob_E40=Wishart_oil[E40_loc]
        prob_E60=Wishart_oil[E60_loc]
        prob_E80=Wishart_oil[E80_loc]
        
        print('PO_prob: '+str(prob_PO.max()))
        print('E40_prob: '+str(prob_E40.max()))
        print('E60_prob: '+str(prob_E60.max()))
        print('E80_prob: '+str(prob_E80.max()))
        
        prob_PO_img = Wishart_oil.copy()
        prob_E40_img = Wishart_oil.copy()
        prob_E60_img = Wishart_oil.copy()
        prob_E80_img = Wishart_oil.copy()
        #prob_PO_img = prob_E40_img = prob_E60_img = prob_E80_img = Wishart_oil.copy()
        
        prob_PO_img[np.where(slicks[...,0]==0)]=0
        prob_E40_img[np.where(slicks[...,1]==0)]=0
        prob_E60_img[np.where(slicks[...,2]==0)]=0
        prob_E80_img[np.where(slicks[...,3]==0)]=0
            
        
        #prob_PO_img[np.where(np.logical_and\
            #(np.logical_and(slicks[...,0]==0,slicks[...,1]==0),\
                #(np.logical_and(slicks[...,2]==0,slicks[...,3]==0))))]=0
        
        #yx_not_PO = np.where(prob_PO_img==prob_PO_img.max())
        #y_not_PO,x_not_PO=yx_not_PO[0][0],yx_not_PO[1][0]
        
        #yx_not_E40 = np.where(prob_E40_img==prob_E40_img.max())
        #y_not_E40,x_not_E40=yx_not_E40[0][0],yx_not_E40[1][0]
        
        #yx_not_E60 = np.where(prob_E60_img==prob_E60_img.max())
        #y_not_E60,x_not_E60=yx_not_E60[0][0],yx_not_E60[1][0]
        
        #yx_not_E80 = np.where(prob_E80_img==prob_E80_img.max())
        #y_not_E80,x_not_E80=yx_not_E80[0][0],yx_not_E80[1][0]
        #print(y_not,x_not)
        
        #Y_NOTS=iter([y_not_PO, y_not_E40, y_not_E60, y_not_E80])
        #X_NOTS=iter([x_not_PO, x_not_E40, x_not_E60, x_not_E80])
        
        #=============Plotting the raw probalilities===============
        #plot_3D_prob_surface(prob_PO_img, slicks.shape)
        

        
        #===========Gaussian Fitting=============
        PO_params = gaussian_Fitting(prob_PO_img, shp_cov,slicks[...,0])
        print(PO_params)
        E40_params = gaussian_Fitting(prob_E40_img, shp_cov,slicks[...,1])
        print(E40_params)
        
        E60_params = gaussian_Fitting(prob_E60_img, shp_cov,slicks[...,2])
        print(E60_params)
        E80_params = gaussian_Fitting(prob_E80_img, shp_cov,slicks[...,3])
        print(E80_params)
        
        plot_fitted_gauss_model([PO_params, E40_params, E60_params, E80_params], shp_cov, window_size_avg)
        
        