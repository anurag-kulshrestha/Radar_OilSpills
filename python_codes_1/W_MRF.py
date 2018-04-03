import plotting 
from osgeo import gdal, ogr, osr
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import incidence_angle_corr
from math import pi
import itertools
#from scipy import linalg
from sklearn import mixture

import extract_polarimetric
import glcm_sklearn
import fit_inci_model
from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage,misc
import reproject
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


import EPFS
from skimage.draw import circle
import numpy.ma as ma
import feature_selection
#plt.style.use('ggplot')
import pandas as pd
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from scipy.special import factorial
import read_binary
import classification

def Wishart_Likelihood(num_classes,m,beta, cov_arr, cov_window_size, Niter, L,cov_df,TR,q=3):
    #getting ensemble average of covariance matrix
    shp_cov=cov_arr.shape
    #L=L*(cov_window_size**2)
    npix=shp_cov[0]*shp_cov[1]
    '''
    plt.imshow(10*np.log10(np.absolute(cov_arr[...,2,2])),cmap='gray')#, origin='lower')
    plt.colorbar(label='dB')
    plt.title('Ivv*')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.show()
    '''
    
    
    Wishart=pd.DataFrame(np.zeros((npix,num_classes+1)), columns=['Oil', 'Water','class_id'])
    
    cov_arr_cols=['Ihh','ShhShv','ShhSvv','ShvShh','Ihv','ShvSvv','SvvShh','SvvShv', 'Ivv']
    C_a_cols=['C_a_Ihh','C_a_ShhShv','C_a_ShhSvv','C_a_ShvShh','C_a_Ihv','C_a_ShvSvv','C_a_SvvShh','C_a_SvvShv', 'C_a_Ivv']
    
    C_a= read_binary.multilook_C3_1(win_x=m, win_y=m, C3_arr=cov_arr, leaping_win=True,stride_row=m, stride_col=m)
    '''
    plt.imshow(10*np.log10(np.absolute(C_a[...,2,2])),cmap='gray')#, origin='lower')
    plt.colorbar(label='dB')
    plt.title('Ivv, m=5*')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.show()
    '''
    
    cov_a_df=pd.DataFrame(np.reshape(C_a,(np.prod(shp_cov[:2]),shp_cov[2]*shp_cov[3])), columns=C_a_cols)
    
    #cov_df=pd.concat([cov_df,cov_a_df], axis=1)
    cov_TR_mean=np.empty((num_classes,3,3), dtype=np.complex64)
    cov_TR_mean_inv=np.empty((num_classes,3,3), dtype=np.complex64)
    # average covarance matrix for each class
    for k in range(num_classes):
        #cov_arr_TR=TR
        ind = TR.where(TR['class_id']==k+1)[cov_arr_cols]
        cov_TR_mean[k]=np.array([ind.mean(axis=0)[i] for i in range(0,len(cov_arr_cols))]).reshape(3,3)
        cov_TR_mean_inv[k]=LA.inv(cov_TR_mean[k])
    
    #print(cov_TR_mean_inv)
    #return cov_TR_mean[0]
    
    # Calculating the det of C_a of full image
    linear_C_a=np.reshape(C_a,(np.prod(shp_cov[:2]),shp_cov[2],shp_cov[3]))
    C_a_det_arr=LA.det(linear_C_a)
    C_a_det_arr=C_a_det_arr.reshape(shp_cov[0]*shp_cov[1])
    
    
    cov_a_df['det_C_a']=C_a_det_arr
    C_a = cov_a_df[C_a_cols].as_matrix().reshape(npix, 3,3)

    #print(C_a.shape)
    
    #return cov_a_df
    
    class_leg={1:'Oil',2:'Water'}
    
    class_leg_1={'Oil':1,'Water':2}

    for k in range(num_classes):
        #return dot_product_trace(cov_TR_mean_inv[k],C_a)
        Wishart[class_leg[k+1]] = (L**(q*L)) * (cov_a_df['det_C_a'].as_matrix()**(L-q)) \
            * np.exp(-L*   dot_product_trace(cov_TR_mean_inv[k],C_a))\
            /(R(L,q) * LA.det(cov_TR_mean[k])**L)
    
    #print(np.absolute(Wishart['class_id'].as_matrix()).reshape(shp_cov[0],shp_cov[1]))
    #print(Wishart.iloc[:, [1,0]])
    #print(Wishart)
    
    #Wishart=Wishart/max(abs(Wishart.iloc[:, [1,0]]).max(axis=0))
    
    #Wishart_soft_prob=
    
    Wishart.iloc[:, [2]]=np.array([class_leg_1[i] for i in abs(Wishart.iloc[:, [1,0]]).idxmax(axis=1)]).reshape(npix,1)
    #print(Wishart['Oil'].abs())
    #Wishart.iloc[Wishart.where(Wishart['Oil'].abs().astype('float') >= 0.4 ), [2]] = 1
    
    #W_HARD(Wishart, p_thresh=0.4)
    #Wishart=W_HARD(Wishart, p_thresh=0.4)
    
    #Wishart.to_csv('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Classification/Wishart_Looks_36_m_5.csv', sep=',')
    
    Wishart_prob=Wishart.copy()
    
    #Wishart_prob.iloc[:,[0,1]] = Wishart.iloc[:,[0,1]].divide(Wishart.sum(axis=1), axis=0)
    
    
    plt.subplot(1,3,1)
    
    plot_arr=np.array(np.absolute(Wishart_prob['Oil'].as_matrix()).reshape(shp_cov[0],shp_cov[1]), dtype=np.float64)
    
    #plot_arr=np.array(np.absolute(Wishart['Oil'].as_matrix()).reshape(shp_cov[0],shp_cov[1]), dtype=np.float64)
    
    print(plot_arr.max(), plot_arr.min())
    
    plot_arr_1=plot_arr.copy()
    #np.save('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Classification/Wishart_Looks_36_m_5_absolute_val.npy',plot_arr)
    
    plot_arr=plot_arr/np.amax(plot_arr)
    plt.imshow(np.clip(plot_arr,0,0.5), cmap='gray_r')
    plt.title('Probablity - Oil')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    #plt.colorbar(orientation='horizontal')
    
    plt.subplot(1,3,2)
    
    plot_arr=np.array(np.absolute(Wishart_prob['Water'].as_matrix()).reshape(shp_cov[0],shp_cov[1]), dtype=np.float64)
    plot_arr=plot_arr/np.amax(plot_arr)
    plt.imshow(plot_arr, cmap='gray_r')
    plt.title('Probablity - Water')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    #plt.colorbar(orientation='horizontal')
    
    
    plt.subplot(1,3,3)
    #plt.imshow(np.c3ip(plot_arr, a_max=10**(-10), a_min=0), cmap='gray')
    im=plt.imshow(Wishart['class_id'].as_matrix().reshape(shp_cov[0],shp_cov[1]),cmap='gray')
    plt.title('Wishart - MLC')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    
    values = np.unique(Wishart['class_id'].as_matrix().ravel())
    #plot_legend(im)
    colors = [ im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label=class_leg[values[i]] ) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    
    #plt.colorbar()
    plt.show()
    #return abs(Wishart).avg(axis=0)
    '''
    
    #==============3D plotting===================
    
    plot_3D_prob_surface(Wishart, shp_cov)
    
    
    
    
    
    #calculating the inverse of the ensemble average matrix
    linear_C_a_arr=np.reshape(C_a,(np.prod(shp_cov[:2]),shp_cov[2],shp_cov[3]))
    inv_arr=LA.inv(linear_C_a_arr)
    inv_arr=inv_arr.reshape(shp_cov[0],shp_cov[1],shp_cov[2],shp_cov[3])
    
    
    
    #Calculating the det of covariance matrix of full image
    linear_cov_arr=np.reshape(cov_arr,(np.prod(shp_cov[:2]),shp_cov[2],shp_cov[3]))
    det_arr=LA.det(linear_cov_arr)
    det_arr=det_arr.reshape(shp_cov[0],shp_cov[1])
    
    #Calculating the det of C_a of full image
    linear_C_a=np.reshape(C_a,(np.prod(shp_cov[:2]),shp_cov[2],shp_cov[3]))
    C_a_det_arr=LA.det(linear_C_a)
    C_a_det_arr=C_a_det_arr.reshape(shp_cov[0],shp_cov[1])
    
    #calculating the dot product between inv_arr and cov_arr
    dot_inv_cov=dot_product(inv_arr,cov_arr)
    '''
    
    #p=(L**(q*L)) * (det_arr**(L-q)) * np.exp(-L* np.trace(dot_inv_cov))/\
        #(R(L,q) * C_a_det_arr**L)
    
    return Wishart

def dot_product_trace(C_class_avg,C_a):
    shp=C_a.shape
    res_dot=np.empty(shp[0], dtype=np.complex64)
    for i in range(shp[0]):
        #print(C_a[i], C_class_avg)
        #return 0
        res_dot[i]=np.trace(np.dot(C_class_avg,C_a[i]))
    return res_dot


def dot_product(arr1,arr2):
    shp=arr2.shape
    res_dot=np.empty(shp, dtype=np.complex64)
    if len(shp)>1:
        for i in range(shp[0]):
            for j in range(shp[1]):
                res_dot[i,j]=np.dot(arr1[i,j],arr2[i,j])
    elif len(shp)==1:
        for i in range(shp[0]):
            res_dot[i]=np.dot(arr1[i],arr2[i])
                
    return res_dot

def R(L,q):
    #gamma=[(factorial]
    gamma=1
    for i in range(1,q+1):
        gamma=gamma*factorial(L-i)
            
    return (np.pi**(q*(q-1)/2))* gamma


def plot_3D_prob_surface(df, shp_cov):
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    Z=np.array(np.absolute(df['Oil'].as_matrix()).reshape(shp_cov[0],shp_cov[1]), dtype=np.float64)
    Z=Z/np.amax(Z)
    
    #Z=np.absolute(df['Oil'].as_matrix().reshape(shp_cov[0],shp_cov[1]))

    X=np.arange(0,shp_cov[0])
    Y=np.arange(0,shp_cov[1])
    
    X,Y=np.meshgrid(Y,X)
    #print(X.shape, Y.shape,Z.shape)
    
    
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)

    #fig.colorbar(surf)#, shrink=0.5, aspect=5)

    plt.show()


def W_HARD(Wishart, p_thresh):
    
    #plot_arr=np.array(np.absolute(Wishart['Oil'].as_matrix()).reshape(shp_cov[0],shp_cov[1]), dtype=np.float64)
    Wishart=Wishart/max(abs(Wishart.iloc[:, [1,0]]).max(axis=0))
    #print( Wishart)
    #val=abs(Wishart.iloc[:, [1,0,2]])
    
    #return (val)
    
    #print(Wishart.count(axis=0)['Oil'])
    
    for i in range(Wishart.count(axis=0)['Oil']):
        if(abs(Wishart.iloc[i,[0]])[0]>p_thresh):
            Wishart.iloc[i,[2]]=1
        
        elif(abs(Wishart.iloc[i,[0]])[0]>abs(Wishart.iloc[i,[1]])[0]):
            Wishart.iloc[i,[2]]=1
        
        elif(abs(Wishart.iloc[i,[0]])[0]<abs(Wishart.iloc[i,[1]])[0]):
            Wishart.iloc[i,[2]]=2
    
    return Wishart
    
    Wishart.iloc[:, [2]]=np.array([class_leg_1[i] \
        for i in abs(Wishart.iloc[:, [1,0]]).idxmax(axis=1)]).reshape(npix,1)






def WMRF(m,beta, cov_arr, shp, Niter, L,q=3):
    
    #dividing the image equally into regions of m*m
    
    K_M=int(np.ceil(shp[0]/m)) #rows in each region
    K_N=int(np.ceil(shp[1]/m)) #cols in each region
    
    #print number of rows and columns
    print(K_M,K_N)
    #Region labels
    X=np.arange(K_M*K_N) #region labels
    
    f=np.floor(0.5+np.random.uniform(low=0.0, high=1.0, size=shp[0]*shp[1]))+1
    f_new=np.floor(0.5+np.random.uniform(low=0.0, high=1.0, size=shp[0]*shp[1]))+1
    
    
    for j in range(0,Niter):
        
        #calculating averave covariance matrix of each region 
        C_a= read_binary.multilook_C3_1(win_x=m, win_y=m, C3_arr=cov_arr, leaping_win=True,stride_row=m, stride_col=m)
        #for a in X:
        #Clipping to the extent of regions
        #C_a=C_a[:K_M,:K_N,...]
        
        #Calculating the det of covariance matrix of full image
        shp_cov=cov_arr.shape
        linear_cov_arr=np.reshape(cov_arr,(np.prod(shp_cov[:2]),shp_cov[2],shp_cov[3]))
        det_arr=LA.det(linear_cov_arr)
        #inv_arr=LA.inv(linear_cov_arr)
        det_arr=det_arr.reshape(shp_cov[0],shp_cov[1])
        #inv_arr=inv_arr.reshape(shp_scov[0],shp_cov[1],shp_cov[2],shp_cov[3])
        
        #beta u(s)
        
        linear_C_a_arr=np.reshape(C_a,(np.prod(shp_cov[:2]),shp_cov[2],shp_cov[3]))
        inv_arr=LA.inv(linear_C_a_arr)
        inv_arr=inv_arr.reshape(shp_cov[0],shp_cov[1],shp_cov[2],shp_cov[3])
        
        
        #p=L**(q*L)*LA.det(cov_arr)
        
        
        #print(C_a[:K_M,:K_N....])
        print(dot_product(inv_arr,cov_arr))
        #plt.imshow(np.absolute(C_a[...,2,2]), cmap='gray')
        #plt.imshow(10*np.log10(np.absolute(np.absolute(det_arr))), cmap='gray')
        
        #plt.show()
        return 0

def Wishart_Classification(cov_arr):
    #cov_arr=extract_polarimetric.extract_covariance_arr(window_size, correction_switch, degree)
    a=1
    
    
def make_cov_arr_df(window_size, correction_switch, degree):
    
    cov_arr=extract_polarimetric.extract_covariance_arr(window_size, correction_switch, degree)
    shp_cov=cov_arr.shape
    linear_cov_arr=np.reshape(cov_arr,(np.prod(shp_cov[:2]),shp_cov[2]*shp_cov[3]))
    #print(linear_cov_arr[1])
    cov_df=pd.DataFrame(linear_cov_arr, columns=['Ihh','ShhShv','ShhSvv','ShvShh','Ihv','ShvSvv','SvvShh','SvvShv', 'Ivv'])
    #cov_df=cov_df.drop(['ShvShh','SvvShh','SvvShv'], axis=1)
    return cov_df

#def probability_surface_model():
    #Find the local maxima,
    
    #Compute a region using region growing algorithm
    
    



if __name__=='__main__':
    os.chdir('../North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01/MLC_Python_New/norway_00709_15092_000_150610_L090_CX_01_mlc')
    
    
    window_size_cov, correction_switch, degree=1,False,1
    num_features=8
    num_classes=2
    m=5
    Niter=10
    Looks=12*3
    #Looks=1
    beta=1.4 # smoothening parameter
    
    
    cov_arr=extract_polarimetric.extract_covariance_arr(window_size_cov, correction_switch, degree)
    
    cov_df = make_cov_arr_df(window_size_cov, correction_switch, degree)
    #print(cov_df)
    #print(cov_df)
    #
    TR=classification.get_training_set(cov_df)
    TS = classification.get_test_set(cov_df, padding=25)
    #print(TR.shape)
    Wishart = Wishart_Likelihood(num_classes,m,beta, cov_arr,window_size_cov, Niter, Looks,cov_df,TR,q=3)
    
    #print(Wishart)
    Wishart_pred_ts=Wishart.iloc[TS['pix_id']]
    
    #WMRF(m, beta, cov_arr, shp, Niter, Looks)
    #print(shp)
    acc_ass=classification.accurcy_Assessment(TS['class_id'].as_matrix(), Wishart_pred_ts['class_id'].as_matrix())
    
    print(acc_ass)
    
    
    
    
    
    [[ 1257.69616699  -27.35939026j,    16.56762505  -32.37125778j,-530.17950439 +235.97766113j],[  768.25396729-2456.34277344j,  2616.02783203 -263.84240723j,
     82.83757019+1421.60266113j],[ -338.83160400  -99.28826904j,   -35.64578629 +246.82649231j,
    247.31065369  -24.94279099j]]
