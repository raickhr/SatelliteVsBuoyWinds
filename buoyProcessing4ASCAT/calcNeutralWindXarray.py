# ================== Import Required Libraries ==================
import sys
import os
import re
import warnings
import glob
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from mpi4py import MPI

# Add COARE3p5 algorithm path
sys.path.append("../COARE3p5/COAREalgorithm/Python/COARE3p5")
from coare35vn import coare35vn  # Import COARE 3.5 bulk flux algorithm

# Suppress warnings
warnings.filterwarnings("ignore")

# ================== MPI Setup ==================
comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

# ================== Utility Functions ==================

def convert_to_datetime_list(ds, time_var='TIME'):
    """
    Convert a dataset's time variable to a list of Python datetime objects.
    """
    time_array = ds[time_var].to_numpy()
    tval = pd.to_datetime(time_array)
    time_series = np.array([datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in tval])
    ds[time_var] = time_series
    return ds

def get_latlon_for_allVarsInAllPos():
    variables = ['WINDS', 'SST', 'RH', 'AIRT', 'BARO', 'RAD', 'LWR', 'RAIN']
    base_path = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/Buoy/TAO_data'
    locs = []
    for var in variables:
        var_folder = os.path.join(base_path, var)
        allLocFolders = glob.glob(f'{var_folder}/*')
        print (f'{var} : {len(allLocFolders)}')
        for thisLocFolder in allLocFolders:
            locString = os.path.basename(thisLocFolder).lstrip('T_')
            locs.append(locString)
    ## retun only unique locations
    return list(set(locs))
            

def divide_tasks():
    """
    Divide (latitude, longitude) grid points among available processors.
    """
    if rank == 0:
        print(f'Running with {nprocs} processors with mpi4py')

    lat_list = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
    lon_list = [-95, -110, -125, -140, -155, -170, -180, 165]

    # Create all latitude-longitude combinations
    task_list = np.array([[lat, lon] for lat in lat_list for lon in lon_list])
    ntasks = len(task_list)

    base_tasks = ntasks // nprocs
    extra = ntasks % nprocs

    # Initialize each processor's task list
    my_task_indices = [rank]
    my_ntasks = base_tasks + (1 if rank < extra else 0)

    for _ in range(1, my_ntasks):
        my_task_indices.append(my_task_indices[-1] + nprocs)

    return task_list[np.array(my_task_indices)]

def select_matching_times_old(ds1, ds2, ds3, ds4, time_var='TIME'):
    """
    Find and align matching time indices across four datasets.
    """
    time1, time2, time3, time4 = [ds[time_var].to_numpy() for ds in (ds1, ds2, ds3, ds4)]

    i = j = k = l = 0
    indices = []

    while i < len(time1) and j < len(time2) and k < len(time3) and l < len(time4):
        if time1[i] == time2[j] == time3[k] == time4[l]:
            indices.append([i, j, k, l])
            i += 1; j += 1; k += 1; l += 1
        else:
            # Advance the pointer with the smallest time
            min_idx = np.argmin([time1[i], time2[j], time3[k], time4[l]])
            if min_idx == 0: i += 1
            elif min_idx == 1: j += 1
            elif min_idx == 2: k += 1
            else: l += 1

    indices = np.array(indices, dtype=int)

    # Select aligned times from each dataset
    return (ds1.isel(TIME=indices[:,0]),
            ds2.isel(TIME=indices[:,1]),
            ds3.isel(TIME=indices[:,2]),
            ds4.isel(TIME=indices[:,3]))


def select_matching_times(ds1, ds2, ds3, ds4, time_var='TIME'):
    """
    Vectorized version: Find matching timestamps across four datasets.
    """
    time1, time2, time3, time4 = [ds[time_var].to_numpy() for ds in (ds1, ds2, ds3, ds4)]

    # Step 1: Find the common times
    common_times = np.intersect1d(np.intersect1d(time1, time2), 
                                  np.intersect1d(time3, time4))

    if len(common_times) == 0:
        raise ValueError("No matching times found across the four datasets.")

    # Step 2: For each dataset, find the indices where time matches common_times
    idx1 = np.nonzero(np.isin(time1, common_times))[0]
    idx2 = np.nonzero(np.isin(time2, common_times))[0]
    idx3 = np.nonzero(np.isin(time3, common_times))[0]
    idx4 = np.nonzero(np.isin(time4, common_times))[0]

    # Step 3: Return selected datasets
    return (ds1.isel({time_var: idx1}),
            ds2.isel({time_var: idx2}),
            ds3.isel({time_var: idx3}),
            ds4.isel({time_var: idx4}))

def open_and_sort(file_list):
    """
    Open a list of NetCDF files, sort by TIME, and merge them into one dataset.
    """
    listDS = []
    for fileName in file_list:
        ds = xr.open_dataset(fileName)
        ds = ds.sortby('TIME')
        for var in ds.data_vars:
            if var not in ['LATITUDE', 'LONGITUDE', 'TIME', 'HEIGHT', 'DEPTH']:
                ndims = ds[var].ndim
                if ndims < 2:
                    print(var + ' has only one dims in file ' + fileName)

        listDS.append(ds)
    mergedDS = xr.concat(listDS, dim ='TIME')
    mergedDS = mergedDS.sortby('TIME')
    return mergedDS
# ================== Main Processing Function ==================

def main():
    """
    Main function to calculate COARE3.5 10m winds for each lat-lon site.
    """
    task_list = divide_tasks()

    for lat, lon in task_list:
        # Determine hemisphere indicators
        lat_unit = 'S' if lat < 0 else 'N'
        lon_unit = 'W' if lon < 0 else 'E'

        # Use absolute values for file naming
        lat_abs = abs(lat)
        lon_abs = abs(lon)

        # Define file paths for Buoy data
        pattern = f'T_{lat_abs:03d}{lat_unit}_{lon_abs:03d}{lon_unit}'
        bWinds = f'../../downloads/Buoy/TAO_data/WINDS/{pattern}/*10min*.nc'
        bAirT = f'../../downloads/Buoy/TAO_data/AIRT/{pattern}/*10min*.nc'
        bSST = f'../../downloads/Buoy/TAO_data/SST/{pattern}/*10min*.nc'
        bRH = f'../../downloads/Buoy/TAO_data/RH/{pattern}/*10min*.nc'

        wind_files = glob.glob(bWinds)
        airt_files = glob.glob(bAirT)
        sst_files = glob.glob(bSST)
        rh_files = glob.glob(bRH)

        # Check if Wind files exist; if not, skip
        if not (wind_files and airt_files and sst_files and rh_files):
            print(f"Missing files for {pattern}")
            continue

        # Output filename
        write_fname = f'../../downloads/Buoy/TAO_data/WINDS/{pattern}_xrr_COARE3p5_2013_2020.nc'
        
        print(f'Processing {pattern}')
        sys.stdout.flush()

        # Open and sort datasets by TIME
        ds_WIND = open_and_sort(wind_files) #xr.open_mfdataset(bWinds).sortby('TIME')
        ds_WIND = convert_to_datetime_list(ds_WIND) # rewrite time to ignore time difference less than a second

        ds_SST = open_and_sort(sst_files) #xr.open_mfdataset(bSST).sortby('TIME')
        ds_SST = convert_to_datetime_list(ds_SST) # rewrite time to ignore time difference less than a second

        ds_AIRT = open_and_sort(airt_files) #xr.open_mfdataset(bAirT).sortby('TIME')
        ds_AIRT = convert_to_datetime_list(ds_AIRT) # rewrite time to ignore time difference less than a second

        ds_RH = open_and_sort(rh_files) #xr.open_mfdataset(bRH).sortby('TIME')
        ds_RH = convert_to_datetime_list(ds_RH) # rewrite time to ignore time difference less than a second

        # Find matching times across datasets
        ds1, ds2, ds3, ds4 = select_matching_times_old(ds_WIND, ds_SST, ds_AIRT, ds_RH)

        # Merge selected datasets
        all_ds = xr.merge((ds1, ds2, ds3, ds4))

        # Extract input variables for COARE
        speed = all_ds['WSPD'].sel(HEIGHT=4.0).to_numpy()
        rh = all_ds['RELH'].sel(HEIGHT=3.0).to_numpy()
        sst = all_ds['SST'].sel(DEPTH=1.0).to_numpy()
        airt = all_ds['AIRT'].sel(HEIGHT=3.0).to_numpy()

        # Run COARE3.5 algorithm
        coare_output = coare35vn(speed, airt, rh, sst, zu=4.0, zt=3, zq=3)

        U10N = coare_output[0, :]
        U10 = coare_output[1, :]

        # Create new DataArrays for U10N and U10
        time_coords = all_ds['TIME']
        WSPD_10N = xr.DataArray(U10N, dims=['TIME'], coords={'TIME': time_coords},
                                attrs={'units': 'm/s',
                                       'long_name': '10 m neutral winds from COARE3.5',
                                       'vars_used_to_calculate': 'SST RH AIRT WSPD'})
        
        WSPD_10N = WSPD_10N.expand_dims(dim='HEIGHT')  # Add new dimension
        WSPD_10N = WSPD_10N.assign_coords(HEIGHT=[10.0])  # Set HEIGHT value

        WSPD_10 = xr.DataArray(U10, dims=['TIME'], coords={'TIME': time_coords},
                               attrs={'units': 'm/s',
                                      'long_name': '10 m winds from COARE3.5',
                                      'vars_used_to_calculate': 'SST RH AIRT WSPD'})
        
        WSPD_10 = WSPD_10.expand_dims(dim='HEIGHT')
        WSPD_10 = WSPD_10.assign_coords(HEIGHT=[10.0])

        # Create new dataset and merge
        new_ds = xr.Dataset({'WSPD_10N': WSPD_10N, 'WSPD_10': WSPD_10})
        final_ds = xr.merge((all_ds, new_ds))

        # Save merged dataset
        final_ds.to_netcdf(write_fname, unlimited_dims='TIME')

# ================== Execute Main ==================
if __name__ == "__main__":
    main()
