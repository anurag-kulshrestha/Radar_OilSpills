import matplotlib
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
from scipy import linalg
from sklearn import mixture

import extract_polarimetric
import glcm_sklearn
import fit_inci_model
from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage,misc
import reproject
import matplotlib.patches as mpatches
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





def get_feature_stack(window_size, correction_switch, degree):
    return feature_selection.get_padded_feature_stack(window_size, correction_switch, degree, pad=0)
    #return EPFS.read_Pol_features(window_size, correction_switch, degree)

def get_slick_boundry(padding=25):
    directory='/home/anurag/Documents/MScProject/Meetings_ITC/Results/Segmentation'
    #slick=np.load(directory+'/oil_segments.npy')#
    slick=np.load(directory+'/boundary_arr_from_MLC.npy')
    img=np.pad(slick, padding, 'constant')
    shp=img.shape
    img=np.insert(img, shp[0],0, axis=0)
    img=np.insert(img, shp[1],0, axis=1)
    #plt.imshow(slick, cmap='gray_r')
    #plt.colorbar()
    #plt.show()
    return img

def plot_training_area(padding=25):
    img=get_slick_boundry(padding)
    #img_1=get_slick_boundry(padding)
    shp=img.shape
    #print(img.shape)
    #print(pol.shape)
    PO = circle(300+padding,600+padding,17)
    E40 = circle(473+padding,578+padding,17)
    E60 = circle(724+padding,557+padding,17)
    E80 = circle(918+padding,458+padding,17)
    W_LU = circle(1025+padding,452+padding,27)
    W_LB = circle(840+padding,475+padding,27)
    W_RU = circle(390+padding,600+padding,27)
    W_RB = circle(600+padding,600+padding,27)
    img[PO]=img[E40] =img[E60] =img[E80] = 2
    img[W_LU]=img[W_LB]=img[W_RU]=img[W_RB]=3
    
    #img = ma.masked_array(img==0, img)
    
    class_leg={2:'Oil',3:'Water', 0:'Unlabelled', 1:'Slick Boundaries'}
    
    im=plt.imshow(img, alpha=0.9, cmap='gray_r')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    
    
    values = np.unique(img.ravel())
    values=values[1:]
    colors = [ im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label=class_leg[values[i]])  for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(0.525   , 1), loc=2, borderaxespad=0. )
    plt.tight_layout()
    
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Classification/Training_set_b_w_1.tiff', dpi=300, box_inches='tight')
    
def plot_test_set(padding=25):
    img=get_slick_boundry(padding)
    #
    shp=img.shape
    #print(img.shape)
    #print(pol.shape)
    PO = circle(264+padding, 605+padding, 15)
    E40= circle(519+padding,496+padding,15)
    E60= circle(759+padding,457+padding ,15)
    E80= circle(944+padding,368+padding,15)
    W_LU=circle(139+padding,232+padding,25)
    W_LB=circle(832+padding,152+padding,25)
    W_RU=circle(171+padding,846+padding,25)
    W_RB=circle(902+padding,768+padding,25)
    img[PO]=img[E40] =img[E60] =img[E80] = 2.1
    img[W_LU]=img[W_LB]=img[W_RU]=img[W_RB]=3.1
    plt.imshow(img, alpha = 0.5, cmap = 'gray_r')
    
    
    
def get_test_set(df, padding=25):
    
    img=get_slick_boundry(padding)
    #
    shp=img.shape
    #print(img.shape)
    #print(pol.shape)
    PO = circle(264+padding,605+padding,15)
    E40= circle(519+padding,496+padding,15)
    E60= circle(759+padding,457+padding,15)
    E80= circle(944+padding,368+padding,15)
    W_LU=circle(139+padding,232+padding,25)
    W_LB=circle(832+padding,152+padding,25)
    W_RU=circle(171+padding,846+padding,25)
    W_RB=circle(902+padding,768+padding,25)
    img[PO]=img[E40] =img[E60] =img[E80] = 2
    img[W_LU]=img[W_LB]=img[W_RU]=img[W_RB]=3
    '''
    pol_oil=ma.masked_where(np.repeat(img!=2,num_features).reshape(shp[0],shp[1],num_features), pol)
    pol_water=ma.masked_where(np.repeat(img!=3,num_features).reshape(shp[0],shp[1],num_features), pol)
    plt.imshow(get_slick_boundry(padding), cmap='gray_r', alpha=1)
    plt.imshow(pol_water[...,0], cmap='gray',alpha=1)
    plt.show()
    '''
    #df=make_data_frame(window_size, correction_switch, degree)
    oil_pos=np.where(img==2)
    water_pos=np.where(img==3)
    #print(oil_pos[0].shape)
    #print(water_pos[0].shape)
    oil_pixels=[[i*shp[1] for i in oil_pos[0]][j]+oil_pos[1][j] for j in range(0,oil_pos[0].size)]
    water_pixels=[[i*shp[1] for i in water_pos[0]][j]+water_pos[1][j] for j in range(0,water_pos[0].size)]
    
    #==========testing_pixels are in correct locations. They indeed are===========
    '''
    img_1=get_slick_boundry(padding)
    a=img_1.flatten()
    a[water_pixels]=2
    a=a.reshape(shp[0],shp[1])
    print(a.shape)
    plt.imshow(a, cmap='gray',alpha=1)
    plt.show()
    '''
    
    df_oil=df.loc[oil_pixels,]
    df_water=df.loc[water_pixels,]
    df_oil['class_id']=np.repeat(1,len(oil_pixels))
    df_oil['pix_id']=np.array(oil_pixels)
    df_water['class_id']=np.repeat(2,len(water_pixels))
    df_water['pix_id']=np.array(water_pixels)
    return pd.concat([df_oil, df_water])

#def get_test_set_1(df, padding=25):
    #img=get_slick_boundry(padding)
    #shp=img.shape
    #PO = circle(264+padding,605+padding,15)
    #E40= circle(519+padding,496+padding,15)
    #E60= circle(759+padding,457+padding,15)
    #E80= circle(944+padding,368+padding,15)
    #W_LU=circle(139+padding,232+padding,25)
    #W_LB=circle(832+padding,152+padding,25)
    #W_RU=circle(171+padding,846+padding,25)
    #W_RB=circle(902+padding,768+padding,25)
    #img[PO]=img[E40] =img[E60] =img[E80] = 2
    #img[W_LU]=img[W_LB]=img[W_RU]=img[W_RB]=3
    
    

def get_training_set(df, padding=25):
    #img = np.zeros((10, 10), dtype=np.uint8)
    #pol=get_feature_stack(window_size, correction_switch, degree)
    #pol_shape=pol.shape
    #pol_shape=(1,2,8)
    #num_features=pol_shape[-1]
    img=get_slick_boundry(padding)
    #
    shp=img.shape
    #print(img.shape)
    #print(pol.shape)
    PO = circle(300+padding, 600+padding, 17)
    E40= circle(473+padding,578+padding,17)
    E60= circle(724+padding,557+padding ,17)
    E80= circle(918+padding,458+padding,17)
    W_LU=circle(390+padding,200+padding,27)
    W_LB=circle(670+padding,200+padding,27)
    W_RU=circle(390+padding,780+padding,27)
    W_RB=circle(670+padding,780+padding,27)
    #PO = circle(264+padding, 605+padding, 15)
    #E40= circle(519+padding,496+padding,15)
    #E60= circle(759+padding,457+padding ,15)
    #E80= circle(944+padding,368+padding,15)
    #W_LU=circle(139+padding,232+padding,25)
    #W_LB=circle(832+padding,152+padding,25)
    #W_RU=circle(171+padding,846+padding,25)
    #W_RB=circle(902+padding,768+padding,25)
    img[PO]=img[E40] =img[E60] =img[E80] = 2
    img[W_LU]=img[W_LB]=img[W_RU]=img[W_RB]=3
    '''
    pol_oil=ma.masked_where(np.repeat(img!=2,num_features).reshape(shp[0],shp[1],num_features), pol)
    pol_water=ma.masked_where(np.repeat(img!=3,num_features).reshape(shp[0],shp[1],num_features), pol)
    plt.imshow(get_slick_boundry(padding), cmap='gray_r', alpha=1)
    plt.imshow(pol_water[...,0], cmap='gray',alpha=1)
    plt.show()
    '''
    #df=make_data_frame(window_size, correction_switch, degree)
    oil_pos=np.where(img==2)
    water_pos=np.where(img==3)
    #print(oil_pos[0].shape)
    #print(water_pos[0].shape)
    oil_pixels=[[i*shp[1] for i in oil_pos[0]][j]+oil_pos[1][j] for j in range(0,oil_pos[0].size)]
    water_pixels=[[i*shp[1] for i in water_pos[0]][j]+water_pos[1][j] for j in range(0,water_pos[0].size)]
    
    #==========testing_pixels are in correct locations. They indeed are===========
    '''
    img_1=get_slick_boundry(padding)
    a=img_1.flatten()
    a[water_pixels]=2
    a=a.reshape(shp[0],shp[1])
    print(a.shape)
    plt.imshow(a, cmap='gray',alpha=1)
    plt.show()
    '''
    
    df_oil=df.loc[oil_pixels,]
    df_water=df.loc[water_pixels,]
    df_oil['class_id']=np.repeat(1,len(oil_pixels))
    df_oil['pix_id']=np.array(oil_pixels)
    df_water['class_id']=np.repeat(2,len(water_pixels))
    df_water['pix_id']=np.array(water_pixels)
    return pd.concat([df_oil, df_water])
    
def get_col_names():
    #return ['det(C3)']
    return ['$I_{hh}$','$I_{hv}$','$I_{vv}$','$\lambda_{1}$','$\lambda_{2}$','$\lambda_{3}$','PD', 'det(C3)']#, '$R_{CO}X$','$I_{CO}X$','det(C3)']

def make_data_frame(window_size, correction_switch, degree):
    pol=get_feature_stack(window_size, correction_switch, degree)
    feature_names=get_col_names()
    shp=pol.shape
    print(shp)
    if len(shp)>2:
        data=pol.flatten().reshape(shp[0]*shp[1], shp[2])
    else:
        data = pol.flatten()
    df=pd.DataFrame(data, columns=feature_names)
    return df
    #print(data.shape)
    #print(data)
    #plt.imshow(data[:,2].reshape(shp[0], shp[1]))
    #plt.show()

def estimate_stats(num_class, num_features, TR):
    mu=np.zeros((num_class, num_features), dtype=np.float64)
    Cov=np.zeros((num_class, num_features,num_features), dtype=np.float64)
    C_inv=np.zeros((num_class, num_features,num_features), dtype=np.float64)
    
    for k in range(num_class):
        ind=(TR.where(TR['class_id']==k+1))[get_col_names()]
        mu[k]=[ind.mean(axis=0)[i] for i in range(0,num_features)]
        Cov[k]=ind.cov().as_matrix()
        C_inv[k]=LA.inv(ind.cov().as_matrix())
    return [mu,Cov,C_inv]

def draw3D_feature_space(nR,nG,nB, plot_All_switch):
    col_names=get_col_names()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for class_id,c, m in [(1,'g', 'o'), (2,'b', 'o'),(0, 'k', 'o')]:
        TR_only_class=TR['class_id']==class_id
        xs=TR.where(TR_only_class)[col_names[nR]]
        ys=TR.where(TR_only_class)[col_names[nG]]
        zs=TR.where(TR_only_class)[col_names[nB]]
        ax.scatter(xs, ys, zs, c=c, marker=m)
    if(plot_All_switch==True):
        pol=get_feature_stack(window_size, correction_switch, degree)
        sample_size=100000
        xs=np.random.choice(pol[...,nR].flatten(),sample_size)
        ys=np.random.choice(pol[...,nG].flatten(),sample_size)
        zs=np.random.choice(pol[...,nB].flatten(),sample_size)
        ax.scatter(xs, ys, zs, c='k', marker='o', s=0.1)
    #all_val=
    
    ax.set_xlabel(col_names[nR])
    ax.set_ylabel(col_names[nG])
    ax.set_zlabel(col_names[nB])
    plt.show()

def get_class_sep(num_class, num_features, TR):
    
    stats=estimate_stats(num_class, num_features, TR)
    mu,cov,c_inv=stats[0],stats[1], stats[2]
    
    ED = np.zeros((num_class, num_class), dtype=np.float64)

    Div = np.zeros((num_class, num_class), dtype=np.float64)
    TD = np.zeros((num_class, num_class), dtype=np.float64)

    B = np.zeros((num_class, num_class), dtype=np.float64)
    JM =np.zeros((num_class, num_class), dtype=np.float64)

    I0 =np.zeros((num_features, num_features), dtype=np.float64)
    np.fill_diagonal(I0,1)
    
    for k in range(0,num_class):
        for l in range(0,num_class):
            if(k!=l):
                #print((mu[k,...]-mu[l,...])**2)
                ED[k,l]=np.sum((mu[k,...]-mu[l,...])**2)
                Div[k,l]=(np.sum(np.diag((c_inv[k,...].dot(cov[l,...])+ c_inv[l,...].dot(cov[k,...]) -2*I0 )))+\
                    (mu[k,...]-mu[l,...]).T.dot(c_inv[k,...]+c_inv[l,...]).dot(mu[k,...]-mu[l,...])  )/2
                
                TD[k,l]=2*(1-np.exp(-Div[k,l]/8))
                
                B[k,l]=((mu[k,...]-mu[l,...]).dot(LA.inv(cov[k,...]+cov[l,...])).dot(mu[k,...]-mu[l,...]))/4 + \
                    0.5*np.log(np.absolute(LA.det(0.5*(cov[k,...]+cov[l,...])))/np.sqrt(np.absolute(LA.det(cov[k,...])*LA.det(cov[l,...]))))
                JM[k,l]=2*(1-np.exp(-B[k,l]))
                #print(LA.det(cov[k,...])*LA.det(cov[l,...]))
                #print(LA.det(0.5*(cov[k,...]+cov[l,...])))
    print (TD)
    #print(Div)
    #print(B)
    print(JM)
    
def MLC_classification(image_df, num_classes, num_features, npix, shp,mu,cov,c_inv, plotting_switch=False, class_leg={'Oil':1,'Water':2}):
    
    class_leg_plot={1:'Oil',2:'Water'}
    
    MLC=pd.DataFrame(np.zeros((npix,num_classes+1)), columns=['Oil', 'Water','class_id'])
    MLC_dist=MLC.copy()
    
    P=np.zeros((npix, num_classes))
    P_dist=np.zeros((npix, num_classes))
    x=image_df.as_matrix()
    
    for k in range(0, num_classes):
        dx=x-mu[k]
        temp=0.5*np.sum((dx*(c_inv[k,...].dot(dx.T)).T), axis=1)
        P_dist[...,k]=(np.exp(-temp))
        P[...,k]=P_dist[...,k]/((2*np.pi)**(num_features/2)*np.sqrt(np.absolute(LA.det(cov[k,...]))))
        MLC.iloc[:, [k]]=P[...,k].reshape(npix,1)
        
        MLC_dist.iloc[:, [k]]=P_dist[...,k].reshape(npix,1)
    
    #print(MLC.idxmax(axis=1))
    
    MLC.iloc[:, [2]]=np.array([class_leg[i] for i in MLC.idxmax(axis=1)]).reshape(npix,1)
    MLC_dist.iloc[:, [2]]=np.array([class_leg[i] for i in MLC_dist.idxmax(axis=1)]).reshape(npix,1)
    
    output=MLC['class_id'].as_matrix().reshape(shp[0],shp[1])
    output_dist=MLC_dist['class_id'].as_matrix().reshape(shp[0],shp[1])
    
    oil_prob=MLC['Oil'].as_matrix().reshape(shp[0],shp[1])
    oil_dist=MLC_dist['Oil'].as_matrix().reshape(shp[0],shp[1])
    
    water_prob=MLC['Water'].as_matrix().reshape(shp[0],shp[1])
    water_dist=MLC_dist['Water'].as_matrix().reshape(shp[0],shp[1])
    
    if(plotting_switch==True):
        '''
        plt.subplot(133)
        im=plt.imshow(output,cmap='gray')
        plt.xlabel('Range')
        plt.ylabel('Azimuth')
        plt.title('MLC_prob - Hard')
        values = np.unique(output.ravel())
        
        #plot_legend(im)
        colors = [ im.cmap(im.norm(value)) for value in values]
        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color=colors[i], label=class_leg_plot[values[i]])  for i in range(len(values)) ]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        #plt.colorbar()
        '''
        matplotlib.rcParams.update({'font.size': 7})
        plt.subplot(133)
        im=plt.imshow(output_dist,cmap='gray')
        #im=plt.imshow(output,cmap='gray')
        plt.xlabel('Range')
        plt.ylabel('Azimuth')
        plt.title('MLC_distance - Hard')
        values = np.unique(output_dist.ravel())
        
        #plot_legend(im)
        colors = [ im.cmap(im.norm(value)) for value in values]
        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color=colors[i], label=class_leg_plot[values[i]])  for i in range(len(values)) ]
        # put those patched as legend-handles into the legend
        #plt.legend(handles=patches, bbox_to_anchor=(0.75, 0), loc='lower center', borderaxespad=0, ncol = 2)
        #plt.colorbar()
        
        plt.subplot(131)
        plt.imshow(oil_dist,cmap='gray_r')
        #plt.imshow(oil_prob,cmap='gray_r')
        plt.xlabel('Range')
        plt.ylabel('Azimuth')
        plt.title('Oil_prob - Soft')
        plt.colorbar(orientation='horizontal')
        
        plt.subplot(132)
        plt.imshow(water_dist,cmap='gray_r')
        #plt.imshow(water_prob,cmap='gray_r')
        plt.xlabel('Range')
        plt.ylabel('Azimuth')
        plt.title('Water_prob- Soft')
        plt.colorbar(orientation='horizontal')
        
        plt.tight_layout()
        plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Classification/MRF/'+'MLC_hard_soft'+'.tiff', dpi=150, box_inches='tight')
        
        plt.show()
    
    
    return MLC_dist
    
def accurcy_Assessment(arr_true, arr_pred):
    error_mat= confusion_matrix(arr_true, arr_pred)
    cohen_kappa=cohen_kappa_score(arr_true, arr_pred)
    return [error_mat, cohen_kappa]

def U(f1,f2,Likelihood, lamb,dxy_clique, npix,shp,n_cliques,beta_clique,num_classes):
    #return Uprior(f1,f2,npix,shp,n_cliques,dxy_clique,beta_clique)
    #return Ulikelihood(f1,Likelihood,npix,num_classes)
    return lamb*Uprior(f1,f2,npix,shp,n_cliques,dxy_clique,beta_clique)+\
        (1-lamb)*Ulikelihood(f1,Likelihood,npix,num_classes)

def Uprior(f1,f2,npix, shp, n_cliques, dxy_clique,beta_clique):
    img_row,img_col=shp[0],shp[1]
    val=np.zeros(npix)
    g1=f1.reshape(img_row,img_col)
    g2=f2.reshape(img_row,img_col)
    
    
    
    for i in range(0,n_cliques):
        dcol, drow=int(dxy_clique[i,0]),int(dxy_clique[i,1])
       
        ij=g1.copy()
        ij.fill(0)
        if(drow>=0):
            ij[0:(img_row-drow),0:(img_col-dcol)]=\
                g1[0:(img_row-drow),0:(img_col-dcol)] !=\
                    g2[drow:img_row,dcol:img_col]
        else:
            ij[abs(drow):img_row,0:(img_col-dcol)] = \
                g1[abs(drow):img_row,0:(img_col-dcol)]!= \
                    g2[0:(img_row-abs(drow)),dcol:img_col]
        
        val_add=ij.flatten()
        ij.fill(0)
        
        if(drow>=0):
            ij[drow:img_row,dcol:img_col]=\
                g1[drow:img_row,dcol:img_col] !=\
                    g2[0:(img_row-drow),0:(img_col-dcol)]
        else:
            ij[0:(img_row-abs(drow)),0:img_col-dcol] = \
                g1[0:(img_row-abs(drow)),0:img_col-dcol] != \
                    g2[abs(drow):img_row,dcol:img_col]
        
        val_add=val_add+ij.flatten()
        
        val=val+val_add*beta_clique[i]
        #print(val_add)
        #plt.imshow(val_add.reshape(img_row,img_col), cmap='gray')
        #plt.colorbar()
        #plt.show()
        return val/2

def Ulikelihood(f,Likelihood, npix,num_classes):
    val=np.zeros(npix)
    
    for i in range(1,num_classes+1):
        pos=np.where(f==i)
        #print(pos)
        if(len(pos[0])>1):
            val[pos]=Likelihood.iloc[pos[0],i-1]
    return val

def MAP_SA_MRF_real(MRF,Likelihood,T_0,T_upd,Niter,lamb, dxy_clique,beta_clique, shp,num_classes,Class_legend, TS_1, display_results=True,AssessAccuracy=True):
    
    f=MRF['class_id'].as_matrix()
    x_out=MRF.copy()
    
    npix=MRF.count()[0]
    n_cliques=dxy_clique.shape[0]
    min_thresh=0.1*10**-2
    
    max_stop_crit=3
    
    stop_crit=0
    
    kappa_evol=np.empty((Niter)) if(AssessAccuracy==True) else None
    E_evol=np.empty((Niter))
    T_evol=np.empty((Niter))
    
    T=T_0
    
    plt.ion()
    
    for i in range(0,Niter):
        f_new=np.floor(0.5+np.random.uniform(low=0.0, high=1.0, size=shp[0]*shp[1]))+1
        #return U(f,f,Likelihood, lamb,dxy_clique, npix,shp,n_cliques, beta_clique,num_classes)
        u1 = U(f,f,Likelihood, lamb,dxy_clique, npix,shp,n_cliques, beta_clique,num_classes)
        u2 = U(f_new,f,Likelihood, lamb,dxy_clique, npix,shp,n_cliques, beta_clique,num_classes)
        
        if(T>0):
            du=np.exp(-(u2-u1)/T)
            xi=np.random.uniform(low=0.0, high=1.0, size=npix)
            
            pos=np.where((xi<du) & (f_new!=f))
            if(len(pos[0])>0):
                f[pos]=f_new[pos]
        else:
            du=u2-u1
            pos=np.where((du<0) & (f_new!=f))
            if(len(pos[0])>0):
                f[pos]=f_new[pos]
        
        upd_count=len(pos[0])
        
        x_out['class_id']=f
        #return x_out
        
        T_evol[i]=T
        
        if(AssessAccuracy==True):
            acc_ass=accurcy_Assessment(TS_1['class_id'].as_matrix(), x_out.iloc[TS_1['pix_id']]['class_id'].as_matrix())
            kappa_evol[i]=acc_ass[1]
        
        E_evol[i]=np.mean(u1)
        
        if(upd_count<=min_thresh*npix):
            stop_crit=stop_crit+1
        else:
            stop_crit=0
            
        if(stop_crit >= max_stop_crit):
            break
        T=T*T_upd
        #T = starting_temperature/np.log(2+i)
        
        if(display_results==True):
            plt.imshow(x_out['class_id'].as_matrix().reshape(shp[0],shp[1]), cmap='gray')
            plt.title('MRF: Iteration '+str(i))
            #plt.show()
            #plt.close()
            plt.tight_layout()
            
            #for saving plots
            #plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Classification/MRF/'+'MLC_MRF_Iteration_'+str(i)+'.tiff', dpi=150, box_inches='tight')
            
            #for visualizing plots
            #plt.pause(0.05)
        
    E_evol = E_evol[0:i]
    T_evol = T_evol[0:i]
    kappa_evol = kappa_evol[0:i]
        
    #plt.close()
    
    plt.subplot(1,3,1)
    plt.plot(kappa_evol)
    plt.xlabel('iter')
    plt.ylabel('kappa')
    plt.title('kappa')
    
    plt.subplot(1,3,2)
    plt.plot(T_evol)
    plt.xlabel('iter')
    plt.ylabel('Temperature')
    plt.title('Temperature')
    
    plt.subplot(1,3,3)
    plt.plot(E_evol)
    plt.xlabel('iter')
    plt.ylabel('Energy')
    plt.title('Energy minimisation')
    
    plt.tight_layout()
    plt.savefig('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Classification/MRF/'+'MLC_MRF_kappa_U_Temp'+'.tiff', dpi=300, box_inches='tight')
    plt.pause(1000)
    #return [kappa_evol,T_evol,E_evol]        

#def WMRF(m=5, cov_arr, shp):
    #K_M=shp[0]//m #rows in each region
    #K_N=shp[1]//m #cols in each region
    
    #K
    
    #X=[1,2] #region labels
    
    
    

#def Wishart_Classification(cov_arr):
    ##cov_arr=extract_polarimetric.extract_covariance_arr(window_size, correction_switch, degree)
    #a=1

def main():
    #=========DATA IMPORT===========
    window_size, correction_switch, degree=9,True,1
    
    num_classes=2
    
    shp=(1177, 1017, 8)
    
    #print(get_feature_stack(window_size, correction_switch, degree).shape)
    #
    #get_slick_boundry()
    
    #===========Making Dataframe================
    os.chdir('/home/anurag/Documents/MScProject/SAR/OilSpill/North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01/MLC_Python_New/norway_00709_15092_000_150610_L090_CX_01_mlc')
    image_df=make_data_frame(window_size, correction_switch, degree)
    image_csv_dir='/home/anurag/Documents/MScProject/Meetings_ITC/Results/Classification/image_df_Win_9_corr_False.csv'
    
    #image_df.to_csv(image_csv_dir, sep=',')
    #return 0
    #image_df=pd.DataFrame.from_csv(image_csv_dir, sep=',')
    #shp=image_df.shape
    num_features=shp[-1]
    
    npix=image_df.count()[0]
    print(npix)
    
    class_leg={'Oil':1,'Water':2}
    
    #print(df.head())
    #print(df.loc[[0,5],])
    
    #=============pixels+training set + Test Set===============
    TR=get_training_set(image_df, degree)
    #TS=get_test_set(window_size, correction_switch, degree)
    #TR.to_csv('/home/anurag/Documents/MScProject/Meetings_ITC/Results/Classification/TR_Win_9_corr_False.csv', sep=',')
    
    #return 0
    TR_csv_dir='/home/anurag/Documents/MScProject/Meetings_ITC/Results/Classification/TR_Win_9_corr_False.csv'
    
    TS_csv_dir='/home/anurag/Documents/MScProject/Meetings_ITC/Results/Classification/TS_Win_9_corr_False.csv'
    
    
    
    #TR=pd.DataFrame.from_csv(TR_csv_dir, sep=',')
    TS=pd.DataFrame.from_csv(TS_csv_dir, sep=',')
    #print(TR)
    #plot_training_area(25)
    #plot_test_set(25)
    #plt.show()
    
    
    #======== How many pixels=========
    
    count_class_oil=TR.where(TR['class_id']==1).count()[0]
    count_class_water=TR.where(TR['class_id']==2).count()[0]
    print('Oil training pixels = '+str(count_class_oil))
    print('Water training pixels = '+ str(count_class_water))
    
    #==================Estimate stats==============
    #print((TR.where(TR['class_id']==1))[get_col_names()].mean(axis=0)[0])#.iloc[0:5]['Ihh'])
    #print((TR.where(TR['class_id']==1))[get_col_names()].cov().as_matrix())#[0])#.iloc[0:5]['Ihh'])
    mu,cov,c_inv=estimate_stats(num_classes, num_features, TR)
    
    #===============Draw 3D feature space ===============
    '''
    nR = 5
    nG = 6
    nB = 7
    draw3D_feature_space(nR,nG,nB, plot_All_switch=False)
    '''
    
    #========Class_separability=======
    #get_class_sep(num_classes, num_features, TR)
    
    
    #========MLC Classification==========
    
    #print(image_df.head())
    #tmp=image_df.copy()
    #MLC=image_df.copy()
    
    mlc_pred=MLC_classification(image_df, num_classes, num_features, npix, shp,mu,cov,c_inv,plotting_switch=True, class_leg={'Oil':1,'Water':2})
    print(mlc_pred)
    mlc_pred_ts=mlc_pred.iloc[TS['pix_id']]
    #print(TS['class_id'].as_matrix())
    
    
    acc_ass=accurcy_Assessment(TS['class_id'].as_matrix(), mlc_pred_ts['class_id'].as_matrix())
    print(acc_ass)
    #print(mlc_pred_ts)
    
    #plt.imshow(MRF['class_id'].as_matrix().reshape(shp[0],shp[1]), cmap='gray')
    #plt.hist(MRF['class_id'].as_matrix(),bins=500, rwidth=0.5,histtype='step', range=[-1,2])
    #plt.show()
    
    
    #========MLC-MRF Classification : Energy optimisation with simulated annealing==========
    
    #=======================================================================
    # Subset the image: not done
    #=======================================================================
    #image_df #same hai 
    #print(mlc_pred) #same hai
    
    tmp=mlc_pred.iloc[:,0:2]#['Oil', 'Water']
    #print(tmp[0:500])
    
    tmp=tmp/(max(tmp.max()))
    print(tmp)
    eps=10**-100
    tmp[tmp<eps]=eps
    #Likelihood=np.zeros(tmp.shape)
    Likelihood=-np.log(tmp)
    
    #print(Likelihood)
    #print(Likelihood.iloc[[1,2,3],0])
    #================================================================================================
    # Block 5:  Energy optimisation with simulated annealing
    #=========================================================
    
    lamb=0.6
    T_0=3.5
    T_upd=0.95
    
    n_cliques=4
    dxy_clique=np.zeros((n_cliques,2))
    beta_clique=np.zeros((n_cliques))
    
    dxy_clique[0]=[1,0]
    dxy_clique[1]=[0,1]
    dxy_clique[2]=[1,1]
    dxy_clique[3]=[1,-1]
    
    beta_clique=1/(n_cliques*np.sqrt(np.sum(dxy_clique**2, axis=0)))
    
    Niter=10000
    
    #========================================================
    # Define the starting image class labels (f)
    #========================================================
    f=np.floor(0.5+np.random.uniform(low=0.0, high=1.0, size=shp[0]*shp[1]))+1
    #print (f[1:100])
    
    #MRF=pd.DataFrame(np.zeros((npix,num_classes+1)), columns=['Oil', 'Water','class_id'])
    MRF=pd.DataFrame(np.zeros((npix,1)), columns=['class_id'])
    
    MRF['class_id']=f
    
    
    #MAP_SA_MRF_real(MRF,Likelihood,T_0,T_upd,Niter,lamb, dxy_clique,beta_clique, shp,num_classes,Class_legend=0,TS_1=TS,display_results=True,AssessAccuracy=True)
    
    
    
    
    

if __name__=='__main__':
    main()