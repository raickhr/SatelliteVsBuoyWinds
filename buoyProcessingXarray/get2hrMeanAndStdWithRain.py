import numpy as np
from scipy import fft
import xarray as xr
import pandas as pd
import os
from scipy import signal
from glob import glob

from mpi4py import MPI

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()


fileList = glob('/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/Buoy/extractedGZ/WINDS/*_xrr_COARE3p5_2000_withRAIN.nc')

nfiles = len(fileList)

avg, rem = divmod(nfiles, nprocs)

nfilesInAllProcs = np.ones((nprocs), dtype=int) * avg
nfilesInAllProcs[0:rem] += 1

endIndices = np.cumsum(nfilesInAllProcs)
startIndices = np.roll(endIndices, 1)
startIndices[0] = 0

for i in range(startIndices[rank], endIndices[rank]):
    bFileName = fileList[i]
    wFileName = bFileName.rstrip('.nc') + '_2hrMeanVar.nc'
    print('at rank', rank, bFileName.lstrip('/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/Buoy/extractedGZ/WINDS/').rstrip('_xrr_COARE3p5_2000_withRAIN.nc'))
    ds = xr.open_dataset(bFileName)
    mask1 = ds.sel(HEIGHT=4)['WSPD_QC'].isin([1,2]).to_numpy()
    mask2 = ds.sel(HEIGHT=4)['WDIR_QC'].isin([1,2]).to_numpy()
    mask3 = ds.sel(DEPTH=1)['SST_QC'].isin([1,2]).to_numpy()
    mask4 = ds.sel(HEIGHT=3)['RELH_QC'].isin([1,2]).to_numpy()
    mask5 = ds.sel(HEIGHT=3)['AIRT_QC'].isin([1,2]).to_numpy()
    mask6 = ds.sel(HEIGHT=3)['RAIN_QC'].isin([1,2]).to_numpy()
    
    selectMask = np.logical_and(mask1, mask2)
    selectMask = np.logical_and(selectMask, mask3)
    selectMask = np.logical_and(selectMask, mask4)
    selectMask = np.logical_and(selectMask, mask5)
    selectMask = np.logical_and(selectMask, mask6)
    
    np_WSPD = ds.sel(HEIGHT=4)['WSPD'].to_numpy()
    np_WDIR = ds.sel(HEIGHT=4)['WDIR'].to_numpy()
    np_SST = ds.sel(DEPTH=1)['SST'].to_numpy()
    np_RELH = ds.sel(HEIGHT=3)['RELH'].to_numpy()
    np_AIRT = ds.sel(HEIGHT=3)['AIRT'].to_numpy()
    np_RAIN = ds.sel(HEIGHT=3)['RAIN'].to_numpy()
    
    
    mask1 = ~np.isnan(np_WSPD)
    mask2 = ~np.isnan(np_WDIR)
    mask3 = ~np.isnan(np_SST)
    mask4 = ~np.isnan(np_AIRT)
    mask5 = ~np.isnan(np_RELH)
    mask6 = ~np.isnan(np_RAIN)
    
    selectMask = np.logical_and(selectMask, mask1)
    selectMask = np.logical_and(selectMask, mask2)
    selectMask = np.logical_and(selectMask, mask3)
    selectMask = np.logical_and(selectMask, mask4)
    selectMask = np.logical_and(selectMask, mask5)
    selectMask = np.logical_and(selectMask, mask6)
    
    mask1 = ~(abs(np_WSPD) > 1000)
    mask2 = ~(abs(np_WDIR) > 1000)
    mask3 = ~(abs(np_SST) > 1000)
    mask4 = ~(abs(np_AIRT) > 1000)
    mask5 = ~(abs(np_RELH) > 1000)
    mask6 = ~(abs(np_RAIN) > 1000)
    
    selectMask = np.logical_and(selectMask, mask1)
    selectMask = np.logical_and(selectMask, mask2)
    selectMask = np.logical_and(selectMask, mask3)
    selectMask = np.logical_and(selectMask, mask4)
    selectMask = np.logical_and(selectMask, mask5)
    selectMask = np.logical_and(selectMask, mask6)
    
    indices = selectMask.nonzero()[0]
    
    ds = ds.isel(TIME=indices)

    ds['SST - AIRT'] = ds.sel(DEPTH=1)['SST'] - ds.sel(HEIGHT=3)['AIRT']
    
    ## this for finding the time differences greater or less than 10 minutes 
    ## and fill the values nan for when the time is discontinious
    time = pd.to_datetime(ds['TIME'].to_numpy())
    deltaTime = np.roll(time, -1) - time
    #deltaTime = deltaTime
    deltaTime = np.array(deltaTime, dtype='timedelta64[s]')
    mask = np.logical_or(abs(deltaTime) > np.array([602], dtype='timedelta64[s]') , 
                         abs(deltaTime) < np.array([508], dtype='timedelta64[s]'))
    stopIndices = mask.nonzero()[0]+1
    stopIndices = np.concatenate(([0], stopIndices))
    
    
    ds['WDIR'] = ((-ds['WDIR'] + 90)+360)%360
    ds['cosWDIR'] = np.cos(np.deg2rad(ds['WDIR']))
    ds['sinWDIR'] = np.sin(np.deg2rad(ds['WDIR']))
    ds['U10N_x'] = ds.WSPD_10N.sel(HEIGHT=[10]) * ds.cosWDIR.sel(HEIGHT=4)
    ds['U10N_y'] = ds.WSPD_10N.sel(HEIGHT=[10]) * ds.sinWDIR.sel(HEIGHT=4)
    
    
    vars = ['WSPD', 'WSPD_10N', 'SST', 'AIRT', 'RELH', 'U10N_x', 'U10N_y', 'cosWDIR', 'sinWDIR', 'RAIN', 'SST - AIRT']
    
    for var in vars:
        print(f'at rank {rank} : {var}')
        if var in ['WSPD', 'cosWDIR', 'sinWDIR']:
            xarr = ds[var].sel(HEIGHT=4)
            xarr = xarr.drop_vars('HEIGHT')
        elif var in ['WSPD_10N', 'U10N_x', 'U10N_y', 'WVEL_10N']:
            xarr = ds[var].sel(HEIGHT=10)
            xarr = xarr.drop_vars('HEIGHT')
        elif var == 'SST':
            xarr = ds[var].sel(DEPTH=1)
            xarr = xarr.drop_vars('DEPTH')
        else:
            xarr = ds[var].sel(HEIGHT=3)
            xarr = xarr.drop_vars('HEIGHT')
        ksize = 240
        np_arr = xarr.to_numpy()
        kernel = np.ones((ksize), dtype=float)
        kernel /= sum(kernel)
        arr_bar = signal.convolve(np_arr, kernel, mode='same')
        arrSq_bar = signal.convolve((np_arr)**2, kernel, mode='same') #
        arrVar = arrSq_bar - (arr_bar)**2
        arrStd = np.sqrt(arrVar)
        hksize = int(ksize//2)
        arrLen = len(np_arr)
        for index in stopIndices:
            start = index - hksize-1
            end = index + hksize + 2
            if start < 0:
                start = 0
            if end>arrLen:
                end =arrLen
            
            ### filling nan values to half kernel size to left and right of time discontinuity

            arr_bar[start:end] = np.nan
            arrStd[start:end] = np.nan
            
        std = xr.DataArray(arrStd, dims = ['TIME'],
                           coords = {'TIME': xarr['TIME']},
                           attrs = {'statistic': f'running std. dev with {ksize}*10 minutes using convolution'})
    
        mean = xr.DataArray(arr_bar, dims = ['TIME'],
                           coords = {'TIME': xarr['TIME']},
                           attrs = {'statistic': f'running mean with {ksize}*10 minutes using convolution'})
        
        ds[f'mean_{var}'] = mean
        ds[f'std_{var}'] = std
    
    ds.to_netcdf(wFileName, unlimited_dims='TIME')
    ds.close()
    

