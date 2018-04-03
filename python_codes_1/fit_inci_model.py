import math
from math import pi
import matplotlib.pyplot as plt
import numpy as np
#from sklearn import datasets, linear_model
#from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import csv
import os
from scipy import signal
from scipy import misc


#os.chdir('../North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01/MLC_Python_New/norway_00709_15092_000_150610_L090_CX_01_mlc')

#os.chdir('../North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01/MLC_Python_New_Nov/norway_00709_15092_000_150610_L090_CX_01_mlc_full_extent')

#print(os.getcwd())

#with open('inci_angle.csv') as csvfile:
#    readCSV=csv.reader(csvfile, delimiter=',')
#    #print(readCSV)
#    for row in readCSV:
#        print(row)
    


def test_convolve(arr, kernal):
    grad = signal.convolve2d(arr, kernal, boundary='symm', mode='valid')
    return grad

def kernal(window_size_x, window_size_y):
    k=np.ones(window_size_x*window_size_y).reshape(window_size_x, window_size_y)
    normalize_k=k/(window_size_x*window_size_y)
    return normalize_k

##plotting

def plot_inc_angle():
    data = np.genfromtxt('inci_angle.csv', delimiter=',', skip_header=1, skip_footer=0, names=['pixel_num', 'y', 'angle'])
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title("Incidence angle vs range")
    ax1.set_xlabel('Range-pixelID')
    ax1.set_ylabel('Incidence angle')
    #ax1.plot(data['pixel_num'],np.tan(np.array(data['angle']/pi)), color='r', label='the data')
    ax1.plot(data['pixel_num'],np.tan(data['angle']/180*pi), color='r', label='tan(inc_angle)')
    ax1.plot(data['pixel_num'],data['angle'], color='g', label='inc_angle')
    
    leg = ax1.legend()
    plt.show()

def regr_fitting():
    #os.chdir()
    data = np.genfromtxt('inci_angle.csv', delimiter=',', skip_header=1, skip_footer=0, names=['pixel_num', 'y', 'angle'])
    data_2d_arr=np.array([data['pixel_num'], data['angle']/180*pi])
    #regr = linear_model.LinearRegression()
    #regr.fit(data_2d_arr)
    pl=np.polyfit(data_2d_arr[0], np.tan(data_2d_arr[1]),1)
    return pl
    #print(data_2d_arr[1])

def extrapolate_inc_angle(test_array):
    #print(os.getcwd())
    parameters=regr_fitting()
    slope=parameters[0]
    intercept=parameters[1]
    result_array=test_array*slope+intercept
    return np.arctan(result_array)*180/pi

def convolove_inc_angle(angle_arr, window_size_x):
    tan_array=np.tan(angle_arr*pi/180)
    kern=kernal(window_size_x,1)
    return test_convolve(tan_array, kern)

if __name__=='__main__':
    directory='../North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01/MLC_Python_New/norway_00709_15092_000_150610_L090_CX_01_mlc'
    os.chdir(directory)
    #print(os.getcwd())
    data = np.genfromtxt('inci_angle.csv', delimiter=',', skip_header=1, skip_footer=0, names=['pixel_num', 'y', 'angle'])
    #plot_inc_angle()
    #print(np.tan(data['angle']/180*pi))
    missing=np.arange(521, 1546, 1)
    angle_arr=extrapolate_inc_angle(missing)
    print(angle_arr)
    conv_angle_array=convolove_inc_angle(angle_arr, 15)
    print(conv_angle_array)
    #print(regr_fitting())
    #a=np.tan(np.array(data['angle']/pi))
    #print(a)