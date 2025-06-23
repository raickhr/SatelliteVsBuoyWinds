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

READ_DIR = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/QS_data/qs_l2b_v4p1/'
WRITE_DIR = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/QS_data/'

# ================ MPI SETUP ==========================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ================ DISTANCE FUNCTION ==================
def get_great_circle_distance(lat1, lon1, lat2, lon2):
    d_lambda = np.deg2rad(np.abs(lon2 - lon1))
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)

    num = (cos(phi2) * sin(d_lambda))**2 + (cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(d_lambda))**2
    num = np.sqrt(num)
    denom = sin(phi1) * sin(phi2) + cos(phi1) * cos(phi2) * cos(d_lambda)
    dsigma = np.arctan2(num, denom)

    return EARTH_RADIUS_KM * dsigma

# ============== FILE DISTRIBUTION ====================
file_list = sorted(glob.glob(os.path.join(READ_DIR, 'qs_l2b_?????_v4.1_*.nc')))
total_files = len(file_list)

if rank == 0:
    # Clean previous outputs
    for f in glob.glob(os.path.join(WRITE_DIR, 'T_????_????_QS_*.nc')):
        os.remove(f)

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
    print(f'[Rank {rank:02d}] File {i-start_idx[0]+1}/{end_idx[0]-start_idx[0]}: {file_path}')
    sys.stdout.flush()

    ds = xr.open_dataset(file_path).set_coords('time').swap_dims({'along_track': 'time'})

    for lat in LAT_LIST:
        lat_unit = 'S' if lat < 0 else 'N'
        for lon in LON_LIST:
            lon_unit = 'W' if lon < 0 else 'E'
            lon_mod = (lon + 360) % 360

            mask = (np.abs(ds['lat'] - lat) < TOL) & (np.abs(ds['lon'] - lon_mod) < TOL)
            if mask.sum() == 0:
                continue

            sub_ds = ds.where(mask, drop=True).reset_coords(['lat', 'lon']).rename({'lat': 'QS_LAT', 'lon': 'QS_LON'})
            QS_lat, QS_lon = sub_ds['QS_LAT'].to_numpy(), sub_ds['QS_LON'].to_numpy()
            distance = get_great_circle_distance(lat, lon, QS_lat, QS_lon)

            sub_ds['dist_from_TAO_pos'] = xr.DataArray(
                distance,
                dims=['time', 'cross_track'],
                attrs={'units': 'kilometers', 'long_name': 'great circle distance from TAO position'}
            )

            tlen = len(sub_ds['time'])
            out_ds = xr.Dataset()

            for var in sub_ds.data_vars:
                data = sub_ds[var]
                if var not in ['ambiguity_speed', 
                               'ambiguity_direction', 
                               'ambiguity_obj']:
                    arr = np.full((tlen, 3), np.nan)
                    arr[:, :data.shape[1]] = data[:, :3]
                    out_ds[var] = xr.DataArray(arr, 
                                               dims=['time', 'cross_track'], 
                                               coords={'time': sub_ds['time'], 
                                                       'cross_track': np.arange(3)}, 
                                               attrs=data.attrs)
                else:
                    arr = np.full((tlen, 3, 4), np.nan)
                    arr[:, :data.shape[1], :] = data[:, :3, :]
                    out_ds[var] = xr.DataArray(arr, dims=['time', 
                                                          'cross_track', 
                                                          'ambiguities'],
                                                    coords={'time': sub_ds['time'], 
                                                            'cross_track': np.arange(3), 
                                                            'ambiguities': np.arange(4)}, 
                                                            attrs=data.attrs)

            fname = f'T_{abs(lat):03.0f}{lat_unit}_{abs(lon):03.0f}{lon_unit}_QS_fileNumber{i:06d}_rank{rank:02d}.nc'
            out_ds.to_netcdf(os.path.join(WRITE_DIR, fname), unlimited_dims='time')

    ds.close()
