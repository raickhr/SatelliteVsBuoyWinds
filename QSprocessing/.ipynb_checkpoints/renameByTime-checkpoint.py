import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import glob
import os

latList = [-8]
lonList = [-95, -125, -140, -155, -170, -180, 165]

# latList = [-8, -9, -5, -2, 0, 2, 5, 8, 9]
# lonList = [-110, -95, -125, -140, -155, -170, -180, 165]

writeDir = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/QS_data/'

for thisLat in latList:
    for thisLon in lonList:
        if thisLat < 0:
            latUnit = 'S'
        else:
            latUnit = 'N'
        
        if thisLon < 0:
            lonUnit = 'W'
        else:
            lonUnit = 'E'

        posFolder = f'TAOpos_{abs(thisLat):03.0f}{latUnit:s}_{abs(thisLon):03.0f}{lonUnit:s}'
        fileList = glob.glob(writeDir + posFolder + '/*.nc')
        print(f'path {writeDir + posFolder}')
        #print(f'fileList {fileList}')
        for fileName in fileList:
            ds = xr.open_dataset(fileName)
            timeArr = ds['time']
            iniTime = timeArr[0].dt
            endTime = timeArr[-1].dt
            strStartTime = f'{iniTime.year:04d}{iniTime.month:02d}{iniTime.day:02d}_{iniTime.hour:02d}{iniTime.minute:02d}{iniTime.second:02d}'
            strEndTime = f'{endTime.year:04d}{endTime.month:02d}{endTime.day:02d}_{endTime.hour:02d}{endTime.minute:02d}{endTime.second:02d}'
            newFileName = writeDir + posFolder + '/' + f'TAOpos_{abs(thisLat):03.0f}{latUnit:s}_{abs(thisLon):03.0f}{lonUnit:s}_' + strStartTime +'_'+strEndTime +'.nc'
            cmd = f'mv {fileName} {newFileName}'
            print(cmd)
            os.system(cmd)
            