import numpy as np
import os

latList = [-8, -9, -5, -2, 0, 2, 5, 8, 9]
lonList = [-110, -95, -125, -140, -155, -170, -180, 165]

writeDir = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/TRMM_data/TRMM_byTAOpos/'

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
        cmd = f'rm -rf {writeDir}{posFolder}'
        print(cmd)
        os.system(cmd)

        cmd = f'mkdir -p {writeDir}{posFolder}'
        print(cmd)
        os.system(cmd)
        
        wFile = f'T_{abs(thisLat):03.0f}{latUnit:s}_{abs(thisLon):03.0f}{lonUnit:s}_TRMM_fileNumber*_rank*.nc'

        fileFormat = writeDir + wFile
        dst = writeDir + posFolder
        cmd = f'mv {fileFormat}  {dst}'
        print(cmd)
        os.system(cmd)
        
