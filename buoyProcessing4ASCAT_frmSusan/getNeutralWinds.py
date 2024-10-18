import sys
import xarray as xr
sys.path.append("../COARE3p5/COAREalgorithm/Python/COARE3p5")
from coare35vn import *
import warnings
warnings.filterwarnings("ignore")

from netCDF4 import Dataset, num2date, date2num
from datetime import datetime, timedelta
import os

import numpy as np
from mpi4py import MPI


comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

def calcNeutralWinds(fldLoc, lat, lon):
    if lat < 0:
        latUnits = 's'
    else:
        latUnits = 'n'

    if lon < 0:
        lonUnits = 'w'
    else:
        lonUnits = 'e'
    lat = abs(lat)
    lon = abs(lon)
    
    fileName = f'met{lat}{latUnits}{lon}{lonUnits}_10m.cdf'

    if os.path.isfile(fldLoc + '/' + fileName):
        writeFileName = fldLoc + f'/T_{lat:02d}{latUnits.upper()}_{lon:03d}{lonUnits.upper()}_COARE3p5_2007.nc'

        xds = xr.open_dataset(fldLoc + '/' + fileName)
        forAS_ds = xds.sel(time=slice('2007-01-01', '2014-12-31'))
        time = forAS_ds['time']
        lat = forAS_ds['lat']
        lon = forAS_ds['lon']
        rh = forAS_ds['RH_910'].to_numpy()[:,0,0,0]
        airt = forAS_ds['AT_21'].to_numpy()[:,0,0,0]
        sst = forAS_ds['T_25'].to_numpy()[:,0,0,0]
        speed = forAS_ds['WS_401'].to_numpy()[:,0,0,0]
        coareOutPutArr = coare35vn(speed, airt, rh, sst, zu=4.0, zt = 3, zq = 3)

        U10N = np.zeros((len(time),1,1,1), dtype=np.float64)
        u10 = np.zeros((len(time),1,1,1), dtype=np.float64)

        U10N[:,0,0,0] = coareOutPutArr[0,:]
        u10[:,0,0,0] = coareOutPutArr[1,:]

        
        height = [10]
        ds = xr.Dataset(
        {
            'U10N': (['time', 'height', 'lat', 'lon'], U10N),
            'U10': (['time', 'height', 'lat', 'lon'], u10),
        },
        coords={
            'time': time,
            'height': height,
            'lat': lat,
            'lon': lon,
        })

        ds['U10N'].attrs['units'] = 'm/s'
        ds['U10N'].attrs['long_name'] = 'neutral wind speed from coare3.5'
        ds['U10'].attrs['units'] = 'm/s'
        ds['U10'].attrs['long_name'] = 'wind speed from coare3.5'

        result_ds = xr.merge([ds, forAS_ds], join='outer')

        result_ds = result_ds.rename({'RH_910': 'RH',
                        'AT_21' : 'AIRT',
                        'T_25'  : 'SST',
                        'WS_401' : 'wspd',
                        'WD_410': 'wdir',
                        })

        result_ds.to_netcdf(writeFileName)
    else:
        print(f'{fldLoc}/{fileName} not found')


print('nprocs = ', nprocs)
latList = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
lonList = [-95, -110, -125, -140, -155, -170, -180, 165]


ylen = len(latList)
xlen = len(lonList)

taskList = []

for latId  in range(ylen):
    for lonId in range(xlen):
        taskList.append([latList[latId], lonList[lonId]])

ntasks = len(taskList)

remainder = ntasks%nprocs
ntasksForMe = int(ntasks//nprocs)

# startIndex in each processor fileIndex start from 1
taskListInMe = [rank]  # the list goes on as [0,5,10] for 5 processors

if rank < remainder:
    ntasksForMe += 1

for i in range(1, ntasksForMe):
    taskListInMe.append(taskListInMe[-1]+nprocs)

#folder = "../../downloads/QS_data/"


for task in taskListInMe:
    lat = taskList[task][0]
    lon = taskList[task][1]
    fldLoc = '../../downloads/SusanDownload/TMA_HR/MET'
    calcNeutralWinds(fldLoc, lat, lon)

    

        
