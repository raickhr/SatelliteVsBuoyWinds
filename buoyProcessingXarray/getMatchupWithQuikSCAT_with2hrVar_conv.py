import sys
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import xarray as xr

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

def selectMatchingTime(ds_QS, ds_TAO, timeVar1 = 'TIME', timeVar2='TIME'):

    ds_QS = ds_QS.rename({timeVar1: 'QS_TIME'})
    ds_TAO = ds_TAO.rename({timeVar2: 'TAO_TIME'})

    ds_TAO = ds_TAO.drop('LATITUDE')
    ds_TAO = ds_TAO.drop('LONGITUDE')
    
    time1 = ds_QS['QS_TIME'].to_numpy()
    time2 = ds_TAO['TAO_TIME'].to_numpy()

    tlen1 = len(time1)
    tlen2 = len(time2)

    i = 0 
    j = 0
    loop = True
    indices = []
    while loop:
        if np.array(abs(time1[i]- time2[j]), dtype='timedelta64[s]') < 600:
            indices.append([i,j])
            i+=1
            j+=1
        elif time1[i] < time2[j]:
            i+=1
        else:
            j+=1
        
        if i == len(time1) or j==len(time2):
            loop = False
            
    indices = np.array(indices, dtype=int)
    #print(indices.shape)
    sel_ds_QS = ds_QS.isel(QS_TIME = indices[:,0])
    sel_ds_TAO = ds_TAO.isel(TAO_TIME = indices[:,1])
    TAO_TIME = sel_ds_TAO['TAO_TIME'].to_numpy()
    sel_ds_TAO = sel_ds_TAO.rename({'TAO_TIME':'QS_TIME'})
    sel_ds_TAO['QS_TIME'] = sel_ds_QS['QS_TIME'].to_numpy()
    sel_ds_TAO['TAO_TIME'] = xr.DataArray(TAO_TIME, dims=['QS_TIME'])
    
    return sel_ds_QS, sel_ds_TAO

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

        bFile = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_xrr_COARE3p5_2000_2hrMeanVar.nc'
        satFile = f'../../downloads/QS_data/TAOpos_{lat:03d}{latUnits}_{lon:03d}{lonUnits}_QS.nc'
        #print(bFile, rank)
        if os.path.isfile(bFile):
            writeFname = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}COARE3p5_2000_2hrMeanVar_QS_Matchup.nc'
            print(f'T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}')
            sys.stdout.flush()
            

            ds_Buoy = xr.open_dataset(bFile)
            ds_Sat = xr.open_dataset(satFile)

            ds_Buoy = converttoDatetimeList(ds_Buoy)#, timeVar='TIME')
            ds_Sat = converttoDatetimeList(ds_Sat, timeVar='time')

            ds_QS, ds_TAO = selectMatchingTime(ds_Sat, ds_Buoy, timeVar1='time')
            
            allDS = xr.merge((ds_QS, ds_TAO))

            allDS.to_netcdf(writeFname, unlimited_dims='QS_TIME')


if __name__ == "__main__":
    main()
