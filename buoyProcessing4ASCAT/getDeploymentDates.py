import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from mpi4py import MPI

# ============== CONFIGURATION =====================
LAT_LIST = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
LON_LIST = [-95, -110, -125, -140, -155, -170, -180, 165]
TASKS = [(lat, lon) for lat in LAT_LIST for lon in LON_LIST]

ROOT_DIR = Path('../../downloads')
BUOY_DIR = ROOT_DIR / 'Buoy/TAO_data/WINDS'

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


def format_coord(lat, lon):
    """Format latitude and longitude into a string like '009S_095W'."""
    lat_unit = 'S' if lat < 0 else 'N'
    lon_unit = 'W' if lon < 0 else 'E'
    return f"{abs(lat):03d}{lat_unit}_{abs(lon):03d}{lon_unit}"


def get_deployment_dates(lat, lon):
    """Extract deployment start and end times for each buoy file."""
    coord_str = format_coord(lat, lon)
    buoy_folder = BUOY_DIR / f"T_{coord_str}"
    output_file = buoy_folder / f"T_{coord_str}_DeploymentDates.nc"

    start_times = []
    end_times = []

    if not buoy_folder.exists():
        log(f"Directory not found: {buoy_folder}", level='WARN', location=coord_str)
        return

    files = list(buoy_folder.glob("*10min*.nc"))
    if not files:
        log(f"No 10min files found in {buoy_folder}", level='WARN', location=coord_str)
        return

    for file_path in sorted(files):
        try:
            ds = xr.open_dataset(file_path)
            ds = ds.sortby('TIME')
            start_times.append(ds.TIME.isel(TIME=0).values)
            end_times.append(ds.TIME.isel(TIME=-1).values)
        except Exception as e:
            log(f"Error reading {file_path.name}: {e}", level='ERR', location=coord_str)
            continue

    # Save deployment range to new file
    try:
        ds_out = xr.Dataset({
            'startDate': xr.DataArray(start_times, dims=['TIME']),
            'endDate': xr.DataArray(end_times, dims=['TIME'])
        })
        ds_out.to_netcdf(output_file, unlimited_dims='TIME')
        log(f"Saved: {output_file}", level='OK', location=coord_str)
    except Exception as e:
        log(f"Failed to save {output_file}: {e}", level='ERR', location=coord_str)


# ============== MAIN =============================
def main():
    my_tasks = distribute_tasks(TASKS)
    for lat, lon in my_tasks:
        get_deployment_dates(lat, lon)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
