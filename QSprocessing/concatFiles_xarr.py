import numpy as np
import os
from mpi4py import MPI
import xarray as xr
import glob

# =================== CONFIGURATION ===================
LAT_LIST = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
LON_LIST = [-95, -110, -125, -140, -155, -170, -180, 165]
TASKS = [(lat, lon) for lat in LAT_LIST for lon in LON_LIST]

WRITE_DIR = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/QS_data/'

# ================ MPI SETUP ==========================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def distribute_tasks():
    avg, rem = divmod(len(TASKS), size)
    task_counts = np.full(size, avg)
    task_counts[:rem] += 1
    end_indices = np.cumsum(task_counts)
    start_indices = np.roll(end_indices, 1)
    start_indices[0] = 0
    return start_indices, end_indices

def format_coord(lat, lon):
    lat_unit = 'S' if lat < 0 else 'N'
    lon_unit = 'W' if lon < 0 else 'E'
    return f"{abs(lat):03d}{lat_unit}_{abs(lon):03d}{lon_unit}"

def process_task(lat, lon):
    coord_str = format_coord(lat, lon)
    pos_folder = f'TAOpos_{coord_str}'
    file_pattern = f'T_{coord_str}_QS_fileNumber??????_rank??.nc'
    #print(file_pattern)
    file_path_pattern = os.path.join(WRITE_DIR, pos_folder, file_pattern)

    # Destination file
    output_filename = f'{pos_folder}_QS.nc'
    output_path = os.path.join(WRITE_DIR, output_filename)

    if os.path.isfile(output_path):
        print(output_filename,' present')
        return
    else:
        # Find matching files
        file_list = sorted(glob.glob(file_path_pattern))
        if not file_list:
            print(f'[Rank {rank}] No files found for {pos_folder}')
            return

        try:
            ds = xr.open_mfdataset(file_list, combine='by_coords')
            ds = ds.sortby('time')
            ds.to_netcdf(output_path)
            ds.close()
            print(f'[Rank {rank}] Saved: {output_path}')
        except Exception as e:
            print(f'[Rank {rank}] Failed processing {pos_folder}: {e}')

def main():
    start_indices, end_indices = distribute_tasks()

    for i in range(start_indices[rank], end_indices[rank]):
        lat, lon = TASKS[i]
        process_task(lat, lon)

if __name__ == "__main__":
    main()
