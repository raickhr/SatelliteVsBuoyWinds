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

def selectMatchingTime(ds1, ds2, ds3, ds4, ds5,  timeVar1 = 'TIME', timeVar2='TIME', timeVar3 = 'TIME', timeVar4='TIME', timeVar5='TIME'):
    time1 = ds1[timeVar1].to_numpy()
    time2 = ds2[timeVar2].to_numpy()
    time3 = ds3[timeVar3].to_numpy()
    time4 = ds4[timeVar4].to_numpy()
    time5 = ds5[timeVar5].to_numpy()

    i = 0 
    j = 0
    k = 0
    l = 0 
    m = 0

    loop = True
    indices = []
    while loop:
        timeDiff1 = np.array(abs(time1[i] -  time2[j]), dtype='timedelta64[s]')
        timeDiff2 = np.array(abs(time1[i] -  time3[k]), dtype='timedelta64[s]')
        timeDiff3 = np.array(abs(time1[i] -  time4[l]), dtype='timedelta64[s]')
        timeDiff4 = np.array(abs(time1[i] -  time5[m]), dtype='timedelta64[s]')

        if timeDiff1 < 1 and timeDiff2 < 1 and timeDiff3 < 1 and timeDiff4 < 1:
            indices.append([i,j,k,l,m])
            i+=1
            j+=1
            k+=1
            l+=1
            m+=1
        else:
            lowest = np.argmin([time1[i], time2[j], time3[k], time4[l], time5[m]])
            #print(time1[i], time2[j], time3[k], time4[l])
            if lowest == 0:
                i += 1
            elif lowest == 1:
                j += 1
            elif lowest == 2:
                k += 1
            elif lowest == 3:
                l += 1
            elif lowest == 4:
                m += 1
        
        if i == len(time1) or j==len(time2) or k == len(time3) or l == len(time4) or m == len(time5):
            loop = False
    print(f'shape Indices at lon {ds5.LONGITUDE.to_numpy()}, lat {ds5.LATITUDE.to_numpy()} is {np.array(indices).shape}')   
    indices = np.array(indices, dtype=int)
    sel_ds1 = ds1.isel(TIME = indices[:,0])
    sel_ds2 = ds2.isel(TIME = indices[:,1])
    sel_ds3 = ds3.isel(TIME = indices[:,2])
    sel_ds4 = ds4.isel(TIME = indices[:,3])
    sel_ds5 = ds5.isel(TIME = indices[:,4])

    return sel_ds1, sel_ds2, sel_ds3, sel_ds4, sel_ds5

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

        bWinds = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_WINDS_2000.nc'
        bAirT = f'../../downloads/Buoy/extractedGZ/AIRT/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_AIRT_2000.nc'
        bSST = f'../../downloads/Buoy/extractedGZ/SST/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_SST_2000.nc'
        bRH = f'../../downloads/Buoy/extractedGZ/RH/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_RH_2000.nc'
        bRAIN = f'../../downloads/Buoy/extractedGZ/RAIN/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_RAIN_2000.nc'


        if os.path.isfile(bRAIN):
            writeFname = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_xrr_COARE3p5_2000_withRAIN.nc'
            print(f'T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}')
            sys.stdout.flush()
            ds_WIND = xr.open_dataset(bWinds)
            ds_SST = xr.open_dataset(bSST)
            ds_AIRT = xr.open_dataset(bAirT)
            ds_RH = xr.open_dataset(bRH)
            ds_RAIN = xr.open_dataset(bRAIN)

            ds_WIND = converttoDatetimeList(ds_WIND, timeVar='TIME')
            ds_SST = converttoDatetimeList(ds_SST, timeVar='TIME')
            ds_AIRT = converttoDatetimeList(ds_AIRT, timeVar='TIME')
            ds_RH = converttoDatetimeList(ds_RH, timeVar='TIME')
            ds_RAIN = converttoDatetimeList(ds_RAIN, timeVar='TIME')


            ds_WIND = ds_WIND.sortby('TIME')
            ds_SST = ds_SST.sortby('TIME')
            ds_AIRT = ds_AIRT.sortby('TIME')
            ds_RH = ds_RH.sortby('TIME')
            ds_RAIN = ds_RAIN.sortby('TIME')


            ds1, ds2 , ds3, ds4, ds5 = selectMatchingTime(ds_WIND, ds_SST, ds_AIRT, ds_RH, ds_RAIN)

            allDS = xr.merge((ds1, ds2, ds3, ds4, ds5))

            

            speed = allDS['WSPD'].sel(HEIGHT=4.0).to_numpy()
            rh = allDS['RELH'].sel(HEIGHT=3.0).to_numpy()
            sst = allDS['SST'].sel(DEPTH=1.0).to_numpy()
            airt = allDS['AIRT'].sel(HEIGHT=3.0).to_numpy()

            coareOutPutArr = coare35vn(speed, airt, rh, sst, zu=4.0, zt = 3, zq = 3)

            U10N = coareOutPutArr[0,:]
            u10 = coareOutPutArr[1,:]

            WSPD_10N = xr.DataArray(np.array([U10N]).T, dims = ['TIME','HEIGHT'],
                                coords = {'TIME': allDS['TIME'],
                                         'HEIGHT': [10.0]},
                                attrs = {'units': 'meters/second',
                                         'long_name': '10 m neutral winds from COARE3.5',
                                         'vars_used_to_calculate': 'SST RH AIRT WSPD'})
            WSPD_10 = xr.DataArray(np.array([u10]).T, dims = ['TIME','HEIGHT'],
                                coords = {'TIME': allDS['TIME'],
                                         'HEIGHT': [10.0]},
                                attrs = {'units': 'meters/second',
                                         'long_name': '10 m winds from COARE3.5',
                                         'vars_used_to_calculate': 'SST RH AIRT WSPD'})
            
            nds = xr.Dataset()
            nds['WSPD_10N'] = WSPD_10N
            nds['WSPD_10'] = WSPD_10

            allDS = xr.merge((allDS, nds))

            allDS.to_netcdf(writeFname, unlimited_dims='TIME')


if __name__ == "__main__":
    main()






