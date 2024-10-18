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

def getMeanAndStdDev(ds, timeWindow):
    TAO_TIME = np.array(ds['TAO_TIME'].to_numpy(), dtype='datetime64[ns]')[0,:]
    QS_TIME = np.array(ds['QS_TIME'].to_numpy(), dtype='datetime64[ns]')
    mask = np.isnat(TAO_TIME)
    ds = ds.isel(QS_TIME=0)
    ds = ds.isel(TAO_TIME_INDEX = ~mask)
    TAO_TIME = np.array(ds['TAO_TIME'].to_numpy(), dtype='datetime64[ns]')
    timeDiffInMins = np.array(TAO_TIME - QS_TIME, dtype='timedelta64[s]')/60
    absTimeDiffInMins = np.array(abs(timeDiffInMins), dtype=np.float32)
    mask = absTimeDiffInMins <= timeWindow/2
    mask = np.logical_and(mask, ds.sel(HEIGHT=4)['WSPD_QC'].isin([1,2]).to_numpy())
    mask = np.logical_and(mask, ds.sel(HEIGHT=4)['WDIR_QC'].isin([1,2]).to_numpy())
    mask = np.logical_and(mask, ds.sel(DEPTH=1)['SST_QC'].isin([1,2]).to_numpy())
    mask = np.logical_and(mask, ds.sel(HEIGHT=3)['AIRT_QC'].isin([1,2]).to_numpy())
    mask = np.logical_and(mask, ds.sel(HEIGHT=3)['RELH_QC'].isin([1,2]).to_numpy())
    
    sel_WSPD = ds.sel(HEIGHT = 4.0)['WSPD'].to_numpy()[mask]
    sel_WSPD_10N = ds.sel(HEIGHT = 10.0)['WSPD_10N'].to_numpy()[mask]
    sel_WDIR = ds.sel(HEIGHT = 4.0)['WDIR'].to_numpy()[mask]

    sel_Ux_10N = sel_WSPD_10N * np.cos(np.deg2rad((-(sel_WDIR - 90.0) + 360)%360))
    sel_Vy_10N = sel_WSPD_10N * np.sin(np.deg2rad((-(sel_WDIR - 90.0) + 360)%360))

    sel_cosWDIR = np.cos(np.deg2rad((-(sel_WDIR - 90.0) + 360)%360))
    sel_sinWDIR = np.sin(np.deg2rad((-(sel_WDIR - 90.0) + 360)%360))

    if np.sum(mask) > 0:
        stdWSPD = np.nanstd(sel_WSPD)
        meanWSPD = np.nanmean(sel_WSPD)

        stdWSPD_10N = np.nanstd(sel_WSPD_10N)
        meanWSPD_10N = np.nanmean(sel_WSPD_10N)

        stdUx_10N = np.nanstd(sel_Ux_10N)
        meanUx_10N = np.nanmean(sel_Ux_10N)

        stdVy_10N = np.nanstd(sel_Vy_10N)
        meanVy_10N = np.nanmean(sel_Vy_10N)
    
        stdWDIR = np.nanstd(sel_WDIR)
        meanWDIR = np.nanmean(sel_WDIR)

        stdCosWDIR = np.nanstd(sel_cosWDIR)
        meanCosWDIR = np.nanmean(sel_cosWDIR)

        stdSinWDIR = np.nanstd(sel_sinWDIR)
        meanSinWDIR = np.nanmean(sel_sinWDIR)
    else:
        stdWSPD = np.nan
        meanWSPD = np.nan

        stdWSPD_10N = np.nan
        meanWSPD_10N = np.nan

        stdUx_10N = np.nan
        meanUx_10N = np.nan

        stdVy_10N = np.nan
        meanVy_10N = np.nan
    
        stdWDIR = np.nan
        meanWDIR = np.nan

        stdCosWDIR = np.nan
        meanCosWDIR = np.nan

        stdSinWDIR = np.nan
        meanSinWDIR = np.nan
    
    return meanWSPD, stdWSPD, meanWSPD_10N, stdWSPD_10N, stdUx_10N, meanUx_10N, stdVy_10N, meanVy_10N, meanWDIR, stdWDIR, meanCosWDIR, stdCosWDIR, meanSinWDIR, stdSinWDIR

def makeMeanAndStdXarrVars(dummyDS, timeWindowInMins):
    meanWSPD, stdWSPD, meanWSPD_10N, stdWSPD_10N, stdUx_10N, meanUx_10N, stdVy_10N, meanVy_10N, meanWDIR, stdWDIR, meanCosWDIR, stdCosWDIR, meanSinWDIR, stdSinWDIR = getMeanAndStdDev(dummyDS, timeWindowInMins)

    dummyDS[f'mean WSPD {timeWindowInMins:d}min'] = xr.DataArray([meanWSPD], dims = ['QS_TIME'], 
                                                attrs = {'units' : 'm/s', 'long_name': f'{timeWindowInMins} min centered mean'})
    dummyDS[f'std. dev. WSPD {timeWindowInMins:d}min'] = xr.DataArray([stdWSPD], dims = ['QS_TIME'], 
                                                attrs = {'units' : 'm/s', 'long_name': f'{timeWindowInMins} min centered std. dev.'})
    
    dummyDS[f'mean WSPD_10N {timeWindowInMins:d}min'] = xr.DataArray([meanWSPD_10N], dims = ['QS_TIME'], 
                                                attrs = {'units' : 'm/s', 'long_name': f'{timeWindowInMins} min centered mean'})
    dummyDS[f'std. dev. WSPD_10N {timeWindowInMins:d}min'] = xr.DataArray([stdWSPD_10N], dims = ['QS_TIME'], 
                                                attrs = {'units' : 'm/s', 'long_name': f'{timeWindowInMins} min centered std. dev.'})
    
    dummyDS[f'mean Ux_10N {timeWindowInMins:d}min'] = xr.DataArray([meanUx_10N], dims = ['QS_TIME'], 
                                                attrs = {'units' : 'm/s', 'long_name': f'{timeWindowInMins} min centered mean'})
    dummyDS[f'std. dev. Ux_10N {timeWindowInMins:d}min'] = xr.DataArray([stdUx_10N], dims = ['QS_TIME'], 
                                                attrs = {'units' : 'm/s', 'long_name': f'{timeWindowInMins} min centered std. dev.'})
    
    dummyDS[f'mean Vy_10N {timeWindowInMins:d}min'] = xr.DataArray([meanVy_10N], dims = ['QS_TIME'], 
                                                attrs = {'units' : 'm/s', 'long_name': f'{timeWindowInMins} min centered mean'})
    dummyDS[f'std. dev. Vy_10N {timeWindowInMins:d}min'] = xr.DataArray([stdVy_10N], dims = ['QS_TIME'], 
                                                attrs = {'units' : 'm/s', 'long_name': f'{timeWindowInMins} min centered std. dev.'})
    
    dummyDS[f'mean WDIR {timeWindowInMins:d}min'] = xr.DataArray([meanWDIR], dims = ['QS_TIME'], 
                                                attrs = {'units' : 'degrees (True)', 'long_name': f'{timeWindowInMins} min centered mean'})
    dummyDS[f'std. dev. WDIR {timeWindowInMins:d}min'] = xr.DataArray([stdWDIR], dims = ['QS_TIME'], 
                                                attrs = {'units' : 'degrees (True)', 'long_name': f'{timeWindowInMins} min centered std. dev'})
    
    dummyDS[f'mean cosWDIR {timeWindowInMins:d}min'] = xr.DataArray([meanCosWDIR], dims = ['QS_TIME'], 
                                                attrs = {'units' : '', 'long_name': f'{timeWindowInMins} min centered mean'})
    dummyDS[f'std. dev. cosWDIR {timeWindowInMins:d}min'] = xr.DataArray([stdCosWDIR], dims = ['QS_TIME'], 
                                                attrs = {'units' : '', 'long_name': f'{timeWindowInMins} min centered std. dev'})
    
    dummyDS[f'mean sinWDIR {timeWindowInMins:d}min'] = xr.DataArray([meanSinWDIR], dims = ['QS_TIME'], 
                                                attrs = {'units' : '', 'long_name': f'{timeWindowInMins} min centered mean'})
    dummyDS[f'std. dev. sinWDIR {timeWindowInMins:d}min'] = xr.DataArray([stdSinWDIR], dims = ['QS_TIME'], 
                                                attrs = {'units' : '', 'long_name': f'{timeWindowInMins} min centered std. dev'})
    return dummyDS


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
            writeFname = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_xrr_MatchUp_2hrMeanVar.nc'
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
