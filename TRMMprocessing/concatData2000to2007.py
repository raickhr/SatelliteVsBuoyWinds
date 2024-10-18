import os
import numpy as np

latList = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
lonList = [-95, -110, -125, -140, -155, -170, -180, 165]
dirLoc = '/srv/seolab/srai/observation/SatelliteVsBuoy/downloads/TRMM_data/downloaded/'

for lat in latList:
    for lon in lonList:
        if lat < 0:
            latUnit = 'S'
        else:
            latUnit = 'N'

        if lon < 0:
            lonUnit = 'W'
        else:
            lonUnit = 'E'

        latLonVal = f'T_{abs(lat):.0f}{latUnit:s}_{abs(lon):.0f}{lonUnit:s}_'

        lon =abs(lon)

        cmd = f'ncrcat -h {dirLoc}TRMM_nc_*/{latLonVal}*.nc {dirLoc}/{latLonVal}_TRMM_2000to2007.nc'
        print(cmd)
        os.system(cmd)

