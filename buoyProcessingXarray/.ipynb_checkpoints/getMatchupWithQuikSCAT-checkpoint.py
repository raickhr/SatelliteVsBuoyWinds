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

def selectMatchingTime(ds1, ds2, timeVar1 = 'TIME', timeVar2='TIME'):

    ds1 = ds1.rename({timeVar1: 'QS_TIME'})
    ds2 = ds2.rename({timeVar2: 'TAO_TIME'})

    # ds1.drop(timeVar1)
    # ds2.drop(timeVar2)
    
    time1 = ds1['QS_TIME'].to_numpy()
    time2 = ds2['TAO_TIME'].to_numpy()

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
    sel_ds1 = ds1.isel(QS_TIME = indices[:,0])
    
    # sel_ds2 = ds2.isel(TAO_TIME = indices[:,1])
    # TAO_TIME = sel_ds2['TAO_TIME'].to_numpy()
    # sel_ds2 = sel_ds2.rename({'TAO_TIME':'QS_TIME'})
    # sel_ds2['QS_TIME'] = sel_ds1['QS_TIME'].to_numpy()
    # sel_ds2['TAO_TIME'] = xr.DataArray(TAO_TIME, dims=['QS_TIME'])

    sel_ds2 = xr.Dataset()
    
    buoyIndices = indices[:,1]
    satIndices = indices[:,0]
    selBuoyDS = []
    for indxCount in range(len(buoyIndices)):
        if indxCount%100 == 0: 
            print(f'In rank {rank}: {indxCount/len(buoyIndices) * 100:5.2f} complete')
        buoyIndex = buoyIndices[indxCount]
        satIndex = satIndices[indxCount]
        #print(buoyIndex, satIndex)
        startIndex = buoyIndex - 6
        endIndex = buoyIndex + 6
        if startIndex < 0:
            startIndex = 0
        if endIndex > tlen2:
            endIndex = tlen2
        dummyDS = ds2.isel(TAO_TIME = slice(startIndex,endIndex))
        dummyLen = endIndex - startIndex

        TAO_time = dummyDS['TAO_TIME'].to_numpy()
        dummyDS['TAO_TIME'] = np.arange(dummyLen)
        dummyDS = dummyDS.rename({'TAO_TIME':'TAO_TIME_INDEX'})
        dummyDS['TAO_TIME'] = xr.DataArray(TAO_time, dims=['TAO_TIME_INDEX'])
        #print(ds1.QS_TIME.shape)
        dummyTime = ds1.isel(QS_TIME = satIndex)['QS_TIME'].to_numpy()
        tval = pd.to_datetime(dummyTime)
        dummyTime = datetime(tval.year, tval.month, tval.day, tval.hour, tval.minute, tval.second)
        dummyDS = dummyDS.expand_dims({'QS_TIME' : pd.Series([dummyTime])}) 
        selBuoyDS.append(dummyDS)
        # if indxCount == 0 :
        #     sel_ds2 = dummyDS
        # else:
        #     sel_ds2 = xr.concat([sel_ds2, dummyDS], dim='QS_TIME' )
    sel_ds2 = xr.concat(selBuoyDS, dim='QS_TIME')
    
    return sel_ds1, sel_ds2

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

        bFile = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_xrr_COARE3p5_2000.nc'
        satFile = f'../../downloads/QS_data/TAOpos_{lat:03d}{latUnits}_{lon:03d}{lonUnits}_QS.nc'
        
        if os.path.isfile(bFile):
            writeFname = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_xrr_MatchUp_2000.nc'
            print(f'T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}')
            sys.stdout.flush()
            

            ds_Buoy = xr.open_dataset(bFile)
            ds_Sat = xr.open_dataset(satFile)

            ds_Buoy = converttoDatetimeList(ds_Buoy)#, timeVar='TIME')
            ds_Sat = converttoDatetimeList(ds_Sat, timeVar='time')

            ds1, ds2 = selectMatchingTime(ds_Buoy, ds_Sat, timeVar2='time')
            
            allDS = xr.merge((ds1, ds2))

            allDS.to_netcdf(writeFname, unlimited_dims='QS_TIME')


if __name__ == "__main__":
    main()
