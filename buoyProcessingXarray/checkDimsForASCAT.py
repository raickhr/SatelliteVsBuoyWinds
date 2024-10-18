import sys
sys.path.append("../COARE3p5/COAREalgorithm/Python/COARE3p5")
from coare35vn import *
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import xarray as xr

from netCDF4 import Dataset, num2date, date2num
from datetime import datetime, timedelta
import os

import numpy as np
from mpi4py import MPI


comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

def converttoDatetimeList(ds, timeVar='TIME'):
    timeArr = ds[timeVar].to_numpy()
    tval = pd.to_datetime(timeArr)
    timeSeries = np.array([datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) for dtm in tval])
    ds['TIME'] = timeSeries
    return ds


def divideTask():
    if rank == 0:
        print(f'Running with {nprocs} processors with mpi4py')
    latList = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
    lonList = [-95, -110, -125, -140, -155, -170, -180, 165]

    ylen = len(latList)
    xlen = len(lonList)

    taskList = []

    for latId  in range(ylen):
        for lonId in range(xlen):
            taskList.append([latList[latId], lonList[lonId]])

    taskList = np.array(taskList)
    ntasks = len(taskList)

    remainder = ntasks%nprocs
    ntasksForMe = int(ntasks//nprocs)

    # startIndex in each processor fileIndex start from 1
    taskListInMe = [rank]  # the list goes on as [0,5,10] for 5 processors

    if rank < remainder:
        ntasksForMe += 1

    for i in range(1, ntasksForMe):
        taskListInMe.append(taskListInMe[-1]+nprocs)
        
    return taskList[np.array(taskListInMe, dtype=int)]

def selectMatchingTime(ds1, ds2, ds3, ds4, timeVar1 = 'TIME', timeVar2='TIME', timeVar3 = 'TIME', timeVar4='TIME'):
    time1 = ds1[timeVar1].to_numpy()
    time2 = ds2[timeVar2].to_numpy()
    time3 = ds3[timeVar3].to_numpy()
    time4 = ds4[timeVar4].to_numpy()

    i = 0 
    j = 0
    k = 0
    l = 0 
    loop = True
    indices = []
    while loop:
        if time1[i] == time2[j] and time1[i] == time3[k] and time1[i] == time4[l]:
            indices.append([i,j,k,l])
            i+=1
            j+=1
            k+=1
            l+=1
        else:
            lowest = np.argmin([time1[i], time2[j], time3[k], time4[l]])
            #print(time1[i], time2[j], time3[k], time4[l])
            if lowest == 0:
                i += 1
            elif lowest == 1:
                j += 1
            elif lowest == 2:
                k += 1
            elif lowest == 3:
                l += 1
        
        if i == len(time1) or j==len(time2) or k == len(time3) or l == len(time4):
            loop = False
            
    indices = np.array(indices, dtype=int)
    sel_ds1 = ds1.isel(TIME = indices[:,0])
    sel_ds2 = ds2.isel(TIME = indices[:,1])
    sel_ds3 = ds3.isel(TIME = indices[:,2])
    sel_ds4 = ds4.isel(TIME = indices[:,3])

    return sel_ds1, sel_ds2, sel_ds3, sel_ds4

def main():
    taskListInMe = divideTask()
    for latLon in taskListInMe:
        lat = latLon[0]
        lon = latLon[1]

        if lat < 0:
            latUnits = 'S'
        else:
            latUnits = 'N'

        if lon < 0:
            lonUnits = 'W'
        else:
            lonUnits = 'E'
        
        lat=abs(lat)
        lon=abs(lon)

        bWinds = f'../../downloads/Buoy/extractedGZ2/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_WINDS_2007.nc'
        bAirT = f'../../downloads/Buoy/extractedGZ2/AIRT/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_AIRT_2007.nc'
        bSST = f'../../downloads/Buoy/extractedGZ2/SST/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_SST_2007.nc'
        bRH = f'../../downloads/Buoy/extractedGZ2/RH/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_RH_2007.nc'

        if os.path.isfile(bWinds):
            #print(f'T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}')
            sys.stdout.flush()
            ds_WIND = xr.open_dataset(bWinds)
            ds_SST = xr.open_dataset(bSST)
            ds_AIRT = xr.open_dataset(bAirT)
            ds_RH = xr.open_dataset(bRH)

            if not 'HEIGHT' in list(ds_WIND.dims):
                print(f'HEIGHT dim not present in WINDS T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}')
            elif not 4.0 in ds_WIND['HEIGHT']:
                print(f'HEIGHT 4.0 not present in WINDS T_{lat:02d}{latUnits}_{lon:03d}{lonUnits} present val is {ds_WIND.HEIGHT.to_numpy()}')


            if not 'HEIGHT' in list(ds_AIRT.dims):
                print(f'HEIGHT dim not present in AIRT T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}')
            elif not 3.0 in ds_AIRT['HEIGHT']:
                print(f'HEIGHT 3.0 not present in WINDS T_{lat:02d}{latUnits}_{lon:03d}{lonUnits} present val is {ds_AIRT.HEIGHT.to_numpy()}')

            if not 'HEIGHT' in list(ds_RH.dims):
                print(f'HEIGHT dim not present in RH T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}')
            elif not 3.0 in ds_RH['HEIGHT']:
                print(f'HEIGHT 3.0 not present in WINDS T_{lat:02d}{latUnits}_{lon:03d}{lonUnits} present val is {ds_RH.HEIGHT.to_numpy()}')
            
            if not 'DEPTH' in list(ds_SST.dims):
                print(f'DEPTH dim not present in SST T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}')
            elif not 1.0 in ds_SST['DEPTH']:
                print(f'DEPTH 1.0 not present in WINDS T_{lat:02d}{latUnits}_{lon:03d}{lonUnits} present val is {ds_SST.DEPTH.to_numpy()}')
            


if __name__ == "__main__":
    main()






