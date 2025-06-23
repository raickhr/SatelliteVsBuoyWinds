import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from mpi4py import MPI

# Suppress warnings
warnings.filterwarnings("ignore")

# MPI initialization
comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()


def convert_to_datetime(ds, time_var='TIME'):
    """
    Convert time variable to datetime64 and reassign to dataset.
    """
    dt_array = pd.to_datetime(ds[time_var].to_numpy())
    ds[time_var] = np.array([datetime(*dt.timetuple()[:6]) for dt in dt_array])
    return ds


def generate_task_list():
    """
    Generate and distribute tasks evenly across processors.
    """
    if rank == 0:
        print(f"Running with {nprocs} processors using mpi4py")

    lat_list = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
    lon_list = [-95, -110, -125, -140, -155, -170, -180, 165]

    tasks = np.array([(lat, lon) for lat in lat_list for lon in lon_list])
    total_tasks = len(tasks)

    tasks_per_proc = total_tasks // nprocs
    if rank < total_tasks % nprocs:
        tasks_per_proc += 1

    indices = list(range(rank, total_tasks, nprocs))[:tasks_per_proc]
    return tasks[indices]


def find_matching_times(ds_sat, ds_buoy, time_var_sat='time', time_var_buoy='TIME'):
    """
    Find matched timestamps between satellite and buoy datasets within 10-minute tolerance.
    """
    ds_sat = ds_sat.rename({time_var_sat: 'CC_TIME'})
    ds_buoy = ds_buoy.rename({time_var_buoy: 'TAO_TIME'}).drop_vars(['LATITUDE', 'LONGITUDE'], errors='ignore')

    t_sat = ds_sat['CC_TIME'].to_numpy()
    t_buoy = ds_buoy['TAO_TIME'].to_numpy()

    matches = []
    i = j = 0
    while i < len(t_sat) and j < len(t_buoy):
        delta = abs(t_sat[i] - t_buoy[j]).astype('timedelta64[s]')
        if delta < 600:
            matches.append((i, j))
            i += 1
            j += 1
        elif t_sat[i] < t_buoy[j]:
            i += 1
        else:
            j += 1

    indices = np.array(matches, dtype=int)
    sel_sat = ds_sat.isel(CC_TIME=indices[:, 0])
    sel_buoy = ds_buoy.isel(TAO_TIME=indices[:, 1])

    # Rename dimensions and keep original TAO_TIME
    sel_buoy = sel_buoy.rename_dims({'TAO_TIME': 'CC_TIME'}).drop_vars(['TAO_TIME'])
    sel_buoy['CC_TIME'] = xr.DataArray(sel_sat['CC_TIME'].values, 
                                       dims = ['CC_TIME'],
                                       attrs = {'long_name':'satellite time'})
    sel_buoy['TAO_TIME'] = xr.DataArray(
        ds_buoy['TAO_TIME'].values[indices[:, 1]],
        dims=['CC_TIME'],
        attrs={'long_name': 'original TAO time'}
    )
    return sel_sat, sel_buoy


def format_lat_lon(lat, lon):
    """
    Format lat/lon strings for filenames.
    """
    lat_dir = 'N' if lat >= 0 else 'S'
    lon_dir = 'E' if lon >= 0 else 'W'
    return abs(lat), lat_dir, abs(lon), lon_dir


def process_matchup(lat, lon):
    """
    Process a single matchup between satellite and buoy data.
    """
    lat_val, lat_dir, lon_val, lon_dir = format_lat_lon(lat, lon)
    lon_360 = (lon + 360) % 360

    buoy_file = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat_val:02d}{lat_dir}_{lon_val:03d}{lon_dir}_xrr_COARE3p5_2000_2hrMeanVar.nc'
    sat_file = f'../../downloads/CCMP/atBuoyLocs/CCMP_tseries_{lon_360:03d}_{lat_val:02d}{lat_dir}.nc'
    output_file = f'../../downloads/CCMP/atBuoyLocs/CCMP_and_Buoy_{lat_val:02d}{lat_dir}_{lon_val:03d}{lon_dir}_xrr_MatchUp_120_mins_2000.nc'

    if not os.path.isfile(buoy_file):
        return

    print(f'Processing: T_{lat_val:02d}{lat_dir}_{lon_val:03d}{lon_dir}')
    sys.stdout.flush()

    ds_buoy = convert_to_datetime(xr.open_dataset(buoy_file))
    ds_sat = convert_to_datetime(xr.open_dataset(sat_file), time_var='time')

    matched_sat, matched_buoy = find_matching_times(ds_sat, ds_buoy)
    merged = xr.merge([matched_sat, matched_buoy])
    merged.to_netcdf(output_file, unlimited_dims='CC_TIME')


def main():
    task_list = generate_task_list()
    for lat, lon in task_list:
        process_matchup(lat, lon)


if __name__ == "__main__":
    main()
