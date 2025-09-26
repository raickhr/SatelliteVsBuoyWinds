import xarray as xr
import numpy as np
from numpy import sin, cos
import sys
from mpi4py import MPI
import glob
import os

# =================== CONFIGURATION ===================
EARTH_RADIUS_KM = 6371.0
LAT_LIST = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
LON_LIST = [-95, -110, -125, -140, -155, -170, -180, 165]
TOL = 0.1

#READ_DIR = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/QS_data/qs_l2b_v4p1/'
#WRITE_DIR = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/QS_data/'

READ_DIR = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/QuikSCAT_data/ncfiles/'
WRITE_DIR = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/QuikSCAT_data/fivePointAverage/'

# ================ MPI SETUP ==========================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ============== FILE DISTRIBUTION ====================
file_list = sorted(glob.glob(os.path.join(READ_DIR, 'qs_l2b_?????_v4.1_*.nc')))
total_files = len(file_list)

if rank == 0:
    # Determine file chunks for each process
    base_count = total_files // size
    counts = np.full(size, base_count)
    counts[:total_files % size] += 1
    starts = np.cumsum(np.insert(counts[:-1], 0, 0))
    ends = np.cumsum(counts)
else:
    starts = ends = np.zeros(size, dtype=int)

start_idx = np.zeros(1, dtype=int)
end_idx = np.zeros(1, dtype=int)
comm.Scatter(starts, start_idx, root=0)
comm.Scatter(ends, end_idx, root=0)

print(f'[Rank {rank}] Processing files {start_idx[0]} to {end_idx[0]}')
sys.stdout.flush()

# ==================== MAIN LOOP ======================
for i in range(start_idx[0], end_idx[0]):
    file_path = file_list[i]
    file_name = os.path.basename(file_path)
    # Clean previous outputs
    for file_name in glob.glob(os.path.join(WRITE_DIR, file_name)):
        os.remove(file_name)
        continue

    print(f'[Rank {rank:02d}] File {i-start_idx[0]+1}/{end_idx[0]-start_idx[0]}: {file_path}')
    sys.stdout.flush()

    ds = xr.open_dataset(file_path)

    wind_direction = ((-ds['retrieved_wind_direction'] - 90)+360)%360
    wind_speed = ds['retrieved_wind_speed']
    uwind = wind_speed * cos(np.deg2rad(wind_direction))
    vwind = wind_speed * sin(np.deg2rad(wind_direction))
    uwind = 0.2 *(uwind + uwind.roll(along_track=1) + uwind.roll(along_track=-1) + uwind.roll(cross_track=1) + uwind.roll(cross_track=-1))
    vwind = 0.2 *(vwind + vwind.roll(along_track=1) + vwind.roll(along_track=-1) + vwind.roll(cross_track=1) + vwind.roll(cross_track=-1))
    wind_speed = np.sqrt(uwind**2 + vwind**2)
    wind_direction = (np.rad2deg(np.arctan2(-vwind, -uwind)) + 90) % 360

    # First and last along both dimensions to NaN
    wind_speed[{ "cross_track": 0 }]  = np.nan
    wind_speed[{ "cross_track": -1 }] = np.nan

    wind_speed[{ "along_track": 0 }]  = np.nan
    wind_speed[{ "along_track": -1 }] = np.nan

    wind_direction[{ "cross_track": 0 }]  = np.nan
    wind_direction[{ "cross_track": -1 }] = np.nan
    wind_direction[{ "along_track": 0 }]  = np.nan
    wind_direction[{ "along_track": -1 }] = np.nan

    ds['retrieved_wind_speed'] = wind_speed
    ds['retrieved_wind_direction'] = wind_direction
    
    flags = ds['flags'].astype(np.int16)
    new_flags = flags.roll(along_track = 1) | flags.roll(along_track = -1) | flags.roll(cross_track = 1) | flags.roll(cross_track = -1) | flags

    eflags = ds['eflags'].astype(np.int16)
    new_eflags = eflags.roll(along_track = 1) | eflags.roll(along_track = -1) | eflags.roll(cross_track = 1) | eflags.roll(cross_track = -1) | eflags

    ds['flags'] = new_flags
    ds['eflags'] = new_eflags    
    
    ds.to_netcdf(os.path.join(WRITE_DIR, file_name), unlimited_dims='time')

    ds.close()

