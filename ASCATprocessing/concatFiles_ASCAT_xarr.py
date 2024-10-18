import numpy as np
import os

# latList = [-8, -9, -5, -2, 0, 2, 5, 8, 9]
# lonList = [-110, -95, -125, -140, -155, -170, -180, 165]

# latList = [-8]
# lonList = [-95, -125, -140, -155, -170, -180, 165]

# latList = [-8]
# lonList = [-110] #, -125, -140, -155, -170, -180, 165]

latList = [-9, -5, -2, 0, 2, 5, 8, 9]
lonList = [-110, -95, -125, -140, -155, -170, -180, 165]

writeDir = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/ASCAT_data/'

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
        file = f'T_{abs(thisLat):03.0f}{latUnit:s}_{abs(thisLon):03.0f}{lonUnit:s}_ASCAT_fileNumber*.nc'
        fileFormat = writeDir + posFolder +'/' + file

        concatFileName = f'TAOpos_{abs(thisLat):03.0f}{latUnit:s}_{abs(thisLon):03.0f}{lonUnit:s}_ASCAT.nc'
        dst = writeDir + concatFileName
        
        cmd = f'ncrcat -h {fileFormat} {dst}'
        print(cmd)
        os.system(cmd)
        