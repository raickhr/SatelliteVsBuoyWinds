import sys
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from mpi4py import MPI

# ============== CONFIGURATION =====================
LAT_LIST = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
LON_LIST = [-95, -110, -125, -140, -155, -170, -180, 165]
TASKS = [(lat, lon) for lat in LAT_LIST for lon in LON_LIST]

ROOT_DIR = Path('../../downloads')
BUOY_DIR = ROOT_DIR / 'Buoy/TAO_data/WINDS'
METOP_DIR = 'MetOP_B'
SAT_DIR = ROOT_DIR / 'QuikSCAT_data'/ METOP_DIR


# ============== MPI SETUP ========================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ============== LOGGING ==========================
def log(msg, level='INFO', location=''):
    colors = {'INFO': '\033[94m', 
              'OK': '\033[92m', 
              'WARN': '\033[93m', 
              'ERR': '\033[91m'}
    prefix = f"[Rank {rank}] [{location}] {colors.get(level, '')}[{level}]"
    endc = '\033[0m'
    print(f"{prefix} {msg}{endc}")
    sys.stdout.flush()

# ============== HELPERS ==========================

def distribute_tasks(tasks):
    """Distribute tasks evenly across MPI ranks."""
    avg, rem = divmod(len(tasks), size)
    counts = [avg + 1 if r < rem else avg for r in range(size)]
    starts = [sum(counts[:r]) for r in range(size)]
    ends = [start + count for start, count in zip(starts, counts)]
    return tasks[starts[rank]:ends[rank]]

def convert_to_datetime(ds, time_var='TIME'):
    """Convert time variable in dataset to Python datetime objects."""
    try:
        times = pd.to_datetime(ds[time_var].to_numpy())
        ds[time_var] = np.array([datetime(*dt.timetuple()[:6]) for dt in times])
        return ds
    except Exception as e:
        log(f"Failed to convert {time_var} to datetime: {e}", level='ERR')
        raise

def format_coord(lat, lon):
    """Format latitude and longitude into a string like '009S_095W'."""
    lat_unit = 'S' if lat < 0 else 'N'
    lon_unit = 'W' if lon < 0 else 'E'
    return f"{abs(lat):03d}{lat_unit}_{abs(lon):03d}{lon_unit}"

def match_times(ds_sat, ds_buoy, 
                coord_str, 
                sat_time_var='time', 
                buoy_time_var='TIME', 
                tolerance_sec=600):
    """Match timestamps within a specified tolerance."""
    ds_sat = ds_sat.rename({sat_time_var: 'QS_TIME'})
    ds_buoy = ds_buoy.rename({buoy_time_var: 'TAO_TIME'}).drop_vars(['LATITUDE', 'LONGITUDE'], errors='ignore')

    time_sat = ds_sat['QS_TIME'].to_numpy()
    time_buoy = ds_buoy['TAO_TIME'].to_numpy()

    if len(time_sat) == 0 or len(time_buoy) == 0:
        log("Empty satellite or buoy time array", level='ERR', location=coord_str)
        return None, None

    indices = []
    i = j = 0
    while i < len(time_sat) and j < len(time_buoy):
        diff = abs((time_sat[i] - time_buoy[j]).astype('timedelta64[s]').astype(int))
        if diff < tolerance_sec:
            indices.append((i, j))
            i += 1
            j += 1
        elif time_sat[i] < time_buoy[j]:
            i += 1
        else:
            j += 1

    if not indices:
        log("No time matches found", level='WARN', location=coord_str)
        return None, None
    else:
        log(f"{len(indices)} matches found", level='INFO', location=coord_str)

    indices = np.array(indices, dtype=int)

    idx_sat, idx_buoy = indices[:,0], indices[:,1]
    ds_sat_sel = ds_sat.isel(QS_TIME=list(idx_sat))
    ds_buoy_sel = ds_buoy.isel(TAO_TIME=list(idx_buoy))

    # Replace time coordinate in buoy data and store original
    TAO_TIME = ds_buoy_sel['TAO_TIME'].to_numpy()
    ds_buoy_sel = ds_buoy_sel.rename({'TAO_TIME': 'QS_TIME'})
    ds_buoy_sel['QS_TIME'] = ds_sat_sel['QS_TIME'].to_numpy()
    ds_buoy_sel['TAO_TIME'] = xr.DataArray(TAO_TIME, dims=['QS_TIME'])

    return ds_sat_sel, ds_buoy_sel

# ============== MAIN PROCESSING ==================
def process_matchup(lat, lon):
    coord_str = format_coord(lat, lon)

    buoy_file = BUOY_DIR / f'T_{coord_str}_xrr_COARE3p5_2013_2020_2hrMeanVar.nc'
    sat_file = SAT_DIR / f'TAOpos_{coord_str}_AS.nc'
    output_filename = f'T_{coord_str}_COARE3p5_2013_2020_2hrMeanVar_QS_Matchup.nc'
    out_file = BUOY_DIR / METOP_DIR/ output_filename

    if out_file.exists():
        try:
            # Try opening the NetCDF file to check if it's valid
            with xr.open_dataset(out_file, engine='h5netcdf') as ds:
                ds.load()  # force read to ensure it's not corrupt
            log(f"{output_filename} already exists and is valid, skipping.", level='INFO')
            return
        except Exception as e:
            log(f"{output_filename} is corrupt or unreadable. Deleting and regenerating. Error: {e}", level='WARN')
            out_file.unlink()  # delete the corrupt file

    if not buoy_file.exists():
        log(f"Missing buoy file: {buoy_file}", level='WARN', location=coord_str)
        return
    if not sat_file.exists():
        log(f"Missing satellite file: {sat_file}", level='WARN', location=coord_str)
        return

    log("Processing", level='INFO', location=coord_str)

    try:
        ds_buoy = xr.open_dataset(buoy_file)
        ds_sat = xr.open_dataset(sat_file)

        ds_buoy = convert_to_datetime(ds_buoy, 'TIME')
        ds_sat = convert_to_datetime(ds_sat, 'time')

        ds_sat_matched, ds_buoy_matched = match_times(ds_sat, ds_buoy, coord_str=coord_str)
        if ds_sat_matched is None:
            return

        merged = xr.merge([ds_sat_matched, ds_buoy_matched])
        merged.to_netcdf(out_file, unlimited_dims='QS_TIME')
        log(f"Saved to {out_file}", level='OK', location=coord_str)

    except Exception as e:
        log(f"Exception during processing: {e}", level='ERR', location=coord_str)

# ============== MAIN =============================
def main():
    my_tasks = distribute_tasks(TASKS)
    for lat, lon in my_tasks:
        process_matchup(lat, lon)

if __name__ == "__main__":
    main()
