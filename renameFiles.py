import os
from netCDF4 import Dataset, date2num, num2date
import datetime
import numpy as np
import glob

lats = ['N9', 'N8', 'N5', 'N2', 'N0', 'S9', 'S8', 'S5', 'S2']
lons = ['110W' , '125W' , '140W' , '155W'  ,'165E' , '170W' , '180W'  ,'95W']

folder = f'/home/shikhar.rai/WHOI/SatelliteVsBuoy/downloads/Buoy/TAO_Bouy2000/'
fileList = glob.glob(folder + '*.nc') 
for file in fileList:
    fname = file.rsplit("/")[-1]
    print(fname)
    prefix = fname.rsplit("_")[1]
    print('prefix', prefix)
    ds = Dataset(file)
    timeList = np.array(ds.variables['TIME'])
    units = ds.variables['TIME'].units
    timeStart = num2date(timeList[0], units)
    timeEnd = num2date(timeList[-1], units)

    strStartTime = '{0:04d}_{1:02d}_{2:02d}_{3:02d}:{4:02d}:{5:02d}'.format(timeStart.year, 
                                                                        timeStart.month, 
                                                                        timeStart.day, 
                                                                        timeStart.hour, 
                                                                        timeStart.minute, 
                                                                        timeStart.second)
    
    strEndTime = '{0:04d}_{1:02d}_{2:02d}_{3:02d}:{4:02d}:{5:02d}'.format(timeEnd.year, 
                                                                        timeEnd.month, 
                                                                        timeEnd.day, 
                                                                        timeEnd.hour, 
                                                                        timeEnd.minute, 
                                                                        timeEnd.second)
    newFName = prefix+ f'_{strStartTime}___{strEndTime}.nc'
    cmd = 'cp '+ file + ' '+ folder + newFName
    print(cmd)
    os.system(cmd)