import sys
import os
import warnings
from datetime import datetime, timedelta
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
BUOY_DIR = ROOT_DIR / 'Buoy/TAO_data/WINDS/fivePointAvg'
METOP_DIR = ''
SAT_DIR = ROOT_DIR / 'QuikSCAT_data'/ METOP_DIR
WAVE_DIR = ROOT_DIR / 'WaveReanalysis'
GLORYS_DIR = ROOT_DIR / 'oceanReanalysis'


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
        dummy = np.array([datetime(*dt.timetuple()[:6]) for dt in times], dtype='datetime64[ns]')
        ds[time_var] = xr.DataArray(dummy, dims = [time_var])
        return ds
    except Exception as e:
        log(f"Failed to convert {time_var} to datetime: {e}", level='ERR')
        raise

def format_coord(lat, lon):
    """Format latitude and longitude into a string like '009S_095W'."""
    lat_unit = 'S' if lat < 0 else 'N'
    lon_unit = 'W' if lon < 0 else 'E'
    return f"{abs(lat):03d}{lat_unit}_{abs(lon):03d}{lon_unit}"

def match_times(ds_matchups, 
                ds_wave, 
                ds_glorys, 
                coord_str, 
                wave_time_var='time', 
                glorys_time_var='time', 
                tolerance_wave_insecs = 10800, ## three hours
                tolerance_glorys_insecs = 86400): ## one day
    """Match timestamps within a specified tolerance."""
    
    ds_wave = ds_wave.rename({wave_time_var: 'WAVE_TIME'})
    ds_glorys = ds_glorys.rename({glorys_time_var: 'GLORYS_TIME'})

    time_matchups = ds_matchups['QS_TIME'].to_numpy()
    time_wave = ds_wave['WAVE_TIME'].to_numpy()
    time_glorys = ds_glorys['GLORYS_TIME'].to_numpy() + np.timedelta64(12, 'h')

    ds_matchups['fillNaN_wave'] = xr.DataArray(np.zeros(len(time_matchups), dtype=int), dims = ['QS_TIME'])
    ds_matchups['fillNaN_glorys'] = xr.DataArray(np.zeros(len(time_matchups), dtype=int), dims = ['QS_TIME'])

    if len(time_matchups) == 0 or len(time_wave) == 0 or len(time_glorys) == 0:
        log("Empty time array in sat buoy matchups or wave data or glorys data", level='ERR', location=coord_str)
        return None, None


    wave_indices = []
    glorys_indices = []

    wave_indx = 0 
    glorys_indx = 0

    for i in range(len(time_matchups)):
        diff_wave = (time_wave[wave_indx] - time_matchups[i]).astype('timedelta64[s]').astype(int)
        while diff_wave <= -(tolerance_wave_insecs/2) and (wave_indx < len(time_wave)-1):
            wave_indx += 1
            diff_wave = (time_wave[wave_indx] - time_matchups[i]).astype('timedelta64[s]').astype(int)

        wave_indices.append(wave_indx)
        if diff_wave > tolerance_wave_insecs/2:
            ds_matchups['fillNaN_wave'].values[i] = 1
        else:
            ds_matchups['fillNaN_wave'].values[i] = 0

        diff_glorys = (time_glorys[glorys_indx] - time_matchups[i]).astype('timedelta64[s]').astype(int)
        while diff_glorys <= -(tolerance_glorys_insecs/2) and (glorys_indx < len(time_glorys)-1):
            glorys_indx += 1
            diff_glorys = (time_glorys[glorys_indx] - time_matchups[i]).astype('timedelta64[s]').astype(int)

        glorys_indices.append(glorys_indx)
        if diff_glorys > tolerance_glorys_insecs/2:
            ds_matchups['fillNaN_glorys'].values[i] = 1
        else:
            ds_matchups['fillNaN_glorys'].values[i] = 0

    if not wave_indices:
        log("No time matches found for wave", level='WARN', location=coord_str)
        return None, None
    else:
        log(f"{len(wave_indices)} matches found for wave for {len(time_matchups)} points", level='INFO', location=coord_str)
        log(f"{np.sum(ds_matchups['fillNaN_wave'].to_numpy())} wave nan points", level='INFO', location=coord_str)
    
    if not glorys_indices:
        log("No time matches found for GLORYS data", level='WARN', location=coord_str)
        return None, None
    else:
        log(f"{len(glorys_indices)} matches found for GLORYS {len(time_matchups)} points", level='INFO', location=coord_str)
        log(f"{np.sum(ds_matchups['fillNaN_glorys'].to_numpy())} glorys nan points", level='INFO', location=coord_str)

    
    ds_glorys_sel = ds_glorys.isel(GLORYS_TIME=glorys_indices)
    ds_wave_sel = ds_wave.isel(WAVE_TIME=wave_indices)

    # Replace time coordinate in buoy data and store original
    WAVE_TIME = ds_wave_sel['WAVE_TIME'].to_numpy()
    GLORYS_TIME = ds_glorys_sel['GLORYS_TIME'].to_numpy()

    ds_wave_sel = ds_wave_sel.drop_vars('WAVE_TIME')
    ds_wave_sel = ds_wave_sel.rename_dims({'WAVE_TIME': 'QS_TIME'})

    ds_glorys_sel = ds_glorys_sel.drop_vars('GLORYS_TIME')
    ds_glorys_sel = ds_glorys_sel.rename_dims({'GLORYS_TIME': 'QS_TIME'})

    ds_wave_sel['WAVE_TIME'] = xr.DataArray(WAVE_TIME, dims = ['QS_TIME'])
    ds_glorys_sel['GLORYS_TIME'] = xr.DataArray(GLORYS_TIME, dims = ['QS_TIME'])

    return ds_wave_sel,  ds_glorys_sel

# ============== MAIN PROCESSING ==================
def process_matchup(lat, lon):
    coord_str = format_coord(lat, lon)

    buoySatMatchup_filename = f'T_{coord_str}_COARE3p5_1999_2009_2hrMeanVar_QS_Matchup_fz.nc'
    buoySatMatchup_file = BUOY_DIR / METOP_DIR/ buoySatMatchup_filename

    output_filename = f'T_{coord_str}_COARE3p5_1999_2009_2hrMeanVar_QS_Buoy_fz_Glorys_Wave_Matchup.nc'
    out_file = BUOY_DIR / METOP_DIR/ output_filename

    wave_filename = f'T_{coord_str}_waveReanalysis_xarr.nc'
    wave_file = WAVE_DIR / wave_filename

    glorys_filename = f'T_{coord_str}_oceanReanalysis_xarr.nc'
    glorys_file = GLORYS_DIR / glorys_filename

    if out_file.exists():
        try:
            # Try opening the NetCDF file to check if it's valid
            with xr.open_dataset(out_file, engine='h5netcdf') as ds:
                ds.load()  # force read to ensure it's not corrupt
            log(f"{out_file} already exists and is valid, skipping.", level='INFO')
            return
        except Exception as e:
            log(f"{output_filename} is corrupt or unreadable. Deleting and regenerating. Error: {e}", level='WARN')
            out_file.unlink()  # delete the corrupt file

    if not buoySatMatchup_file.exists():
        log(f"Missing matcup file: {buoySatMatchup_file}", level='WARN', location=coord_str)
        return
    if not wave_file.exists():
        log(f"Missing wave file: {wave_file}", level='WARN', location=coord_str)
        return
    if not glorys_file.exists():
        log(f"Missing glorys file: {glorys_file}", level='WARN', location=coord_str)
        return

    log("Processing", level='INFO', location=coord_str)

    try:
        ds_matchup = xr.open_dataset(buoySatMatchup_file)
        ds_wave = xr.open_dataset(wave_file)
        ds_glorys = xr.open_dataset(glorys_file)

        ds_matchup = ds_matchup.sortby('QS_TIME')
        ds_wave = ds_wave.sortby('time')
        ds_glorys = ds_glorys.sortby('time')

        ds_matchup = convert_to_datetime(ds_matchup, 'QS_TIME')
        ds_wave = convert_to_datetime(ds_wave, 'time')
        ds_glorys = convert_to_datetime(ds_glorys, 'time')

        ds_wave_matched, ds_glorys_matched = match_times(ds_matchup, 
                                                        ds_wave, 
                                                        ds_glorys, 
                                                        coord_str=coord_str)
        if ds_wave_matched is None or ds_glorys_matched is None:
            return

        merged = xr.merge([ds_matchup, ds_wave_matched, ds_glorys_matched])
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



