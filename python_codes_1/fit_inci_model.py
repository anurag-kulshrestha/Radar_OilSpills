import math
from math import pi
import matplotlib.pyplot as plt
import numpy as np
#from sklearn import datasets, linear_model
#from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import csv
import os

os.chdir('../North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01/MLC_Python_New/norway_00709_15092_000_150610_L090_CX_01_mlc')
#os.chdir('../North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01/MLC_Python_New/norway_00709_15092_000_150610_L090_CX_01_mlc')

#os.chdir('../North_Sea_UAVSAR/UAV_norway/UA_norway_00709_15092_000_150610_L090_CX_01/MLC_Python_New_Nov/norway_00709_15092_000_150610_L090_CX_01_mlc_full_extent')

#print(os.getcwd())

#with open('inci_angle.csv') as csvfile:
#    readCSV=csv.reader(csvfile, delimiter=',')
#    #print(readCSV)
#    for row in readCSV:
#        print(row)
    
data = np.genfromtxt('inci_angle.csv', delimiter=',', skip_header=1, skip_footer=0, names=['pixel_num', 'y', 'angle'])

##plotting

def plot_inc_angle():
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
    data_2d_arr=np.array([data['pixel_num'], data['angle']/180*pi])
    #regr = linear_model.LinearRegression()
    #regr.fit(data_2d_arr)
    pl=np.polyfit(data_2d_arr[0], np.tan(data_2d_arr[1]),1)
    return pl
    #print(data_2d_arr[1])

def extrapolate_inc_angle(test_array):
    parameters=regr_fitting()
    slope=parameters[0]
    intercept=parameters[1]
    result_array=test_array*slope+intercept
    return np.arctan(result_array)*180/pi


if __name__=='__main__':
    #plot_inc_angle()
    #print(np.tan(data['angle']/180*pi))
    missing=np.arange(521, 1546, 1)
    print(extrapolate_inc_angle(missing))
    #print(regr_fitting())
    #a=np.tan(np.array(data['angle']/pi))
    #print(a)