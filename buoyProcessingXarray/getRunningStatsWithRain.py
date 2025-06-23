import numpy as np
import xarray as xr
import pandas as pd
import os
from scipy import signal
from glob import glob
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

# Input file list
fileList = glob('/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/Buoy/extractedGZ/WINDS/*_xrr_COARE3p5_2000_withRAIN.nc')
nfiles = len(fileList)

# Distribute files across MPI ranks
avg, rem = divmod(nfiles, nprocs)
nfilesInAllProcs = np.ones(nprocs, dtype=int) * avg
nfilesInAllProcs[:rem] += 1
endIndices = np.cumsum(nfilesInAllProcs)
startIndices = np.roll(endIndices, 1)
startIndices[0] = 0

# Function for running mean and std
def running_stats(arr, kernel_size, stop_indices):
    kernel = np.ones(kernel_size) / kernel_size
    mean = signal.convolve(arr, kernel, mode='same')
    sq_mean = signal.convolve(arr ** 2, kernel, mode='same')
    std = np.sqrt(sq_mean - mean ** 2)
    hksize = kernel_size // 2
    for idx in stop_indices:
        start = max(0, idx - hksize)
        end = min(len(arr), idx + hksize + 1)
        mean[start:end] = np.nan
        std[start:end] = np.nan
    return mean, std

# Variable to dimension mapping
dim_map = {
    'WSPD': ('HEIGHT', 4.0), 
    'cosWDIR': ('HEIGHT', 4.0), 
    'sinWDIR': ('HEIGHT', 4.0),
    'WSPD_10N': ('HEIGHT', 10.0), 
    'U10N_x': ('HEIGHT', 10.0), 
    'U10N_y': ('HEIGHT', 10.0),
    'SST': ('DEPTH', 1.0), 
    'AIRT': ('HEIGHT', 3.0), 
    'RELH': ('HEIGHT', 3.0), 
    'RAIN': ('HEIGHT', 3.0),
    'SST - AIRT': None
}

# Main processing loop
for i in range(startIndices[rank], endIndices[rank]):
    bFileName = fileList[i]
    wFileName = bFileName.replace('.nc', '_2hrMeanVar.nc')
    print(f'Rank {rank} processing: {os.path.basename(bFileName)}')

    ds = xr.open_dataset(bFileName)

    # Quality control mask
    qc_mask = (
        ds.sel(HEIGHT=4)['WSPD_QC'].isin([1, 2]).values &
        ds.sel(HEIGHT=4)['WDIR_QC'].isin([1, 2]).values &
        ds.sel(DEPTH=1)['SST_QC'].isin([1, 2]).values &
        ds.sel(HEIGHT=3)['RELH_QC'].isin([1, 2]).values &
        ds.sel(HEIGHT=3)['AIRT_QC'].isin([1, 2]).values &
        ds.sel(HEIGHT=3)['RAIN_QC'].isin([1, 2]).values
    )

    # NaN and range filtering
    valid_mask = (
        ~np.isnan(ds.sel(HEIGHT=4)['WSPD'].values) & (np.abs(ds.sel(HEIGHT=4)['WSPD'].values) < 1000) &
        ~np.isnan(ds.sel(HEIGHT=4)['WDIR'].values) & (np.abs(ds.sel(HEIGHT=4)['WDIR'].values) < 1000) &
        ~np.isnan(ds.sel(DEPTH=1)['SST'].values) & (np.abs(ds.sel(DEPTH=1)['SST'].values) < 1000) &
        ~np.isnan(ds.sel(HEIGHT=3)['RELH'].values) & (np.abs(ds.sel(HEIGHT=3)['RELH'].values) < 1000) &
        ~np.isnan(ds.sel(HEIGHT=3)['AIRT'].values) & (np.abs(ds.sel(HEIGHT=3)['AIRT'].values) < 1000) &
        ~np.isnan(ds.sel(HEIGHT=3)['RAIN'].values) & (np.abs(ds.sel(HEIGHT=3)['RAIN'].values) < 1000)
    )

    total_mask = qc_mask & valid_mask
    ds = ds.isel(TIME=total_mask)

    # Compute SST - AIRT
    ds['SST - AIRT'] = ds.sel(DEPTH=1)['SST'] - ds.sel(HEIGHT=3)['AIRT']
    # Wind direction conversion and components
    ds['WDIR'] = ((-ds['WDIR'] + 90) + 360) % 360
    ds['cosWDIR'] = np.cos(np.deg2rad(ds['WDIR']))
    ds['sinWDIR'] = np.sin(np.deg2rad(ds['WDIR']))
    ds['U10N_x'] = ds['WSPD_10N'] * ds.cosWDIR.sel(HEIGHT=4)
    ds['U10N_y'] = ds['WSPD_10N'] * ds.sinWDIR.sel(HEIGHT=4)


    # Time discontinuity detection
    time = pd.to_datetime(ds['TIME'].to_numpy())
    deltaTime = np.array(np.roll(time, -1) - time, dtype='timedelta64[s]')
    stopIndices = np.concatenate(([0], np.nonzero((deltaTime > np.timedelta64(602, 's')) | (deltaTime < np.timedelta64(508, 's')))[0] + 1))

    # List of variables to process
    vars_to_process = list(dim_map.keys())
    ksize = 12

    for var in vars_to_process:
        print(f'Rank {rank}: processing {var}')
        if var == 'SST - AIRT':
            xarr = ds[var]
        else:
            dim, val = dim_map[var]
            xarr = ds[var].sel({dim: val}).drop_vars(dim)

        arr = xarr.to_numpy()
        mean, std = running_stats(arr, ksize, stopIndices)

        ds[f'mean_{var}'] = xr.DataArray(mean, dims=['TIME'], coords={'TIME': xarr['TIME']},
                                         attrs={'statistic': f'Running mean over {ksize*10} min'})
        ds[f'std_{var}'] = xr.DataArray(std, dims=['TIME'], coords={'TIME': xarr['TIME']},
                                        attrs={'statistic': f'Running std dev over {ksize*10} min'})

    # Save output
    ds.to_netcdf(wFileName, unlimited_dims='TIME')
    ds.close()
