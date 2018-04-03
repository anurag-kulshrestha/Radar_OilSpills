import os
import sys
from PyQt5 import QtGui, QtCore
from PyQt5 import QtWidgets
import UAVSAR_time_series
import pyqtgraph as pg
#app = QtWidgets.QApplication(sys.argv)

#window=QtWidgets.QWidget()

#window.setGeometry(50,50,500,300)

#window.show()

class Window(QtWidgets.QMainWindow):
    
    def __init__(self):
        super(Window,self).__init__()
        self.setGeometry(50,50,500,300)
        self.setWindowTitle('Test')
        self.setWindowIcon(QtGui.QIcon('/home/anurag/Downloads/penguin-oil-spill-mythology.jpg'))
        self.load_img()
    
    def load_img(self):
        print(1)
        base_dir='../North_Sea_UAVSAR/UAV_norway'
        file_ext=['.ann','.dat','.gif','.hgt','.inc','.kmz','.slope','_hgt.tif','_pauli.tif']
        folders=['_mlc','_grd']
        Region='norway'
        Heading='007'
        Counter_num='09'
        Year='15'
        Num_flights_year='092'
        Data_take='000'
        Day='10'
        Month='06'
        Band='L'
        Steering_angle='090'
        Cross_talk='CX'
        Processing_version='01'
        Polarization='VV'
        #====================================================
        
        is_List_Ratio=False
        cropping_switch=True
        reproject_switch=False
        plotting_switch=True
        convolution_switch = True
        inc_correction_switch=False
        window_size_fea = 9
        inc_correction_sin_degree = 1
        
        #====================================================
        
        wd=UAVSAR_time_series.getdir(region=Region,heading=Heading,counter_num=Counter_num,year=Year,num_flights_year=Num_flights_year,data_take=Data_take,day=Day,month=Month,band=Band,steering_angle=Steering_angle,cross_talk=Cross_talk,processing_version=Processing_version)
        
        base_file_name=UAVSAR_time_series.get_base_file_name(region=Region,heading=Heading,counter_num=Counter_num,year=Year,num_flights_year=Num_flights_year,data_take=Data_take,day=Day,month=Month,band=Band,steering_angle=Steering_angle,cross_talk=Cross_talk,processing_version=Processing_version)
        
        os.chdir(wd)
        #print(os.getcwd())
        
        #=======+Get Matadata========
        meta=UAVSAR_time_series.metadata_dict(base_file_name)
        
        #=============Defining the subset=====================
        grd_cropping_list_00709=[2500,4000,2500,4000]
        grd_cropping_list_18709=[1000,2300,3250,4450]
        
        if Heading=='007':
            cropping_list_GRD=grd_cropping_list_00709
        elif Heading=='187':
            cropping_list_GRD=grd_cropping_list_18709
        
        #===========Defining Convolution keranl===========
        
        convolution_kernal = UAVSAR_time_series.kernal(window_size_fea)
        
        #============incidence angle=================
        
        grd_lines = int(meta['grd_pwr.set_rows'])
        grd_samples = int(meta['grd_pwr.set_cols'])
        
        inc_ang_arr = UAVSAR_time_series.get_inc_angle(grd_lines, grd_samples, base_file_name, cropping_list_GRD, cropping_switch = cropping_switch, is_List_Ratio=is_List_Ratio)
        
        
        
        btn = QtWidgets.QPushButton('press me', self)
        btn.clicked.connect(self.load_grd_image(meta, base_file_name, convolution_kernal, inc_ang_arr,inc_correction_sin_degree=inc_correction_sin_degree, cropping_list_GRD=cropping_list_GRD, is_List_Ratio=is_List_Ratio, cropping_switch=cropping_switch,convolution_switch = convolution_switch,inc_correction_switch=inc_correction_switch))
        btn.resize(110,100)
        btn.move(0,50)
        
        self.show()
        
    def load_grd_image(self, meta, base_file_name, convolution_kernal, inc_ang_arr,inc_correction_sin_degree, cropping_list_GRD, is_List_Ratio, cropping_switch,convolution_switch ,inc_correction_switch):
        UAVSAR_time_series.get_GRD(meta, base_file_name,convolution_kernal, inc_ang_arr,inc_correction_sin_degree=inc_correction_sin_degree, component='VVVV',grd_cropping_list=cropping_list_GRD, is_List_Ratio=is_List_Ratio, cropping_switch=cropping_switch, reproject_switch=False, plotting_switch=True, convolution_switch = convolution_switch, inc_correction_switch=inc_correction_switch)
        return 0
        

def run():
    app = QtWidgets.QApplication(sys.argv)
    GUI=Window()
    sys.exit(app.exec_())
    

if __name__=='__main__':
    run()