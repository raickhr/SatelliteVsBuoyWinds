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

def sub_selByQuality(ds, qualVarName, goodQualityIndexList):
    mask = ds[qualVarName].isin(goodQualityIndexList).to_numpy()[:,0]
    ds= ds.sel(TIME=mask)
    return ds

def make1D(speed):
    shape = speed.shape
    if len(shape) > 1:
        largerDim = np.argmax(shape)
        nanArr = np.isnan(speed)
        numOfNanData = np.sum(nanArr, axis=largerDim)
        index = np.argmin(numOfNanData)

        speed = speed[index,:] if shape[1] > shape[0] else speed[:,index]      
    return speed

def add_dimension(ds, dimName, dimVal):
    if not dimName in ds.dims:
        for var in list(ds.data_vars.keys()):
            ds[var] = ds[var].expand_dims({dimName: 1}).assign_coords({ dimName: [dimVal]})
    return ds

def drop_dimension(ds, dimName):
    if dimName in ds.dims:
        ds = ds.drop_dims(dimName)
    return ds

def calcNeutralWinds(fldLoc, lat, lon):
    if lat < 0:
        latUnits = 'S'
    else:
        latUnits = 'N'

    if lon < 0:
        lonUnits = 'W'
    else:
        lonUnits = 'E'
    lat = abs(lat)
    lon = abs(lon)
    
    bWinds = f'WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_WINDS_2007.nc'
    bAirT = f'AIRT/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_AIRT_2007.nc'
    bSST = f'SST/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_SST_2007.nc'
    bRH = f'RH/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_RH_2007.nc'

    if os.path.isfile(fldLoc + '/' + bWinds):
        writeFileName = fldLoc + f'/WINDS/T_{lat:02d}{latUnits.upper()}_{lon:03d}{lonUnits.upper()}_COARE3p5_2007.nc'

        winds_ds = xr.open_dataset(fldLoc + '/' + bWinds)
        winds_ds = winds_ds.drop_duplicates(dim="TIME")
        winds_ds = winds_ds.sortby("TIME")
        winds_ds = add_dimension(winds_ds, 'HEIGHT', 4)
        #winds_ds = drop_dimension(winds_ds, 'HEIGHT')
        winds_ds = sub_selByQuality(winds_ds, 'WSPD_QC', [1,2])
        winds_ds = sub_selByQuality(winds_ds, 'WDIR_QC', [1,2])
        
        airt_ds = xr.open_dataset(fldLoc + '/' + bAirT)
        airt_ds = airt_ds.drop_duplicates(dim="TIME")
        airt_ds = airt_ds.sortby("TIME")
        airt_ds = add_dimension(airt_ds, 'HEIGHT', 3)
        #airt_ds = drop_dimension(airt_ds, 'HEIGHT')
        airt_ds = sub_selByQuality(airt_ds, 'AIRT_QC', [1,2])
        
        sst_ds = xr.open_dataset(fldLoc + '/' + bSST)
        sst_ds = sst_ds.drop_duplicates(dim="TIME")
        sst_ds = sst_ds.sortby("TIME")
        sst_ds = add_dimension(sst_ds, 'DEPTH', 1)
        #sst_ds = drop_dimension(sst_ds, 'DEPTH')
        sst_ds = sub_selByQuality(sst_ds, 'SST_QC', [1,2])
        
        rh_ds = xr.open_dataset(fldLoc + '/' + bRH)
        rh_ds = rh_ds.drop_duplicates(dim="TIME")
        rh_ds = rh_ds.sortby("TIME")
        rh_ds = add_dimension(rh_ds, 'HEIGHT', 3)
        #rh_ds = drop_dimension(rh_ds, 'HEIGHT')
        rh_ds = sub_selByQuality(rh_ds, 'RELH_QC', [1,2])

        xds = xr.merge([winds_ds, airt_ds, sst_ds, rh_ds], join='outer')

        forAS_ds = xds.sel(TIME=slice('2007-01-01', '2014-12-31'))
        time = forAS_ds['TIME']
        
        rh = forAS_ds['RELH'].to_numpy()
        rh = make1D(rh)
        airt = forAS_ds['AIRT'].to_numpy()
        airt = make1D(airt)
        sst = forAS_ds['SST'].to_numpy()
        sst = make1D(sst)
        speed = forAS_ds['WSPD'].to_numpy()
        speed = make1D(speed)
        coareOutPutArr = coare35vn(speed, airt, rh, sst, zu=4.0, zt = 3, zq = 3)

        U10N = np.zeros((len(time),1), dtype=np.float64)
        u10 = np.zeros((len(time),1), dtype=np.float64)

        U10N[:,0] = coareOutPutArr[0,:]
        u10[:,0] = coareOutPutArr[1,:]

        
        height = [10]
        ds = xr.Dataset(
        {
            'U10N': (['TIME','HEIGHT'], U10N),
            'U10': (['TIME','HEIGHT'], u10),
        },
        coords={
            'TIME': time,
            'HEIGHT': height
        })

        ds['U10N'].attrs['units'] = 'm/s'
        ds['U10N'].attrs['long_name'] = 'neutral wind speed from coare3.5'
        ds['U10'].attrs['units'] = 'm/s'
        ds['U10'].attrs['long_name'] = 'wind speed from coare3.5'

        result_ds = xr.merge([ds, forAS_ds], join='outer')

        result_ds = result_ds.rename({'TIME': 'time',
                                      'HEIGHT': 'height',
                                      'DEPTH': 'depth',
                                      'RELH': 'RH',
                                      'WSPD': 'wspd',
                                      'UWND': 'uwnd',
                                      'VWND': 'vwnd',
                                      'WDIR': 'U10_direction'})

        result_ds.to_netcdf(writeFileName)
    else:
        print(f'{fldLoc}/{bWinds} not found')


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
    fldLoc = '../../downloads/Buoy/extractedGZ2'
    calcNeutralWinds(fldLoc, lat, lon)

    

        
