import xarray as xr
import numpy as np
from numpy import sin, cos
import sys
from mpi4py import MPI
import glob
import os

import warnings
warnings.filterwarnings('ignore')

# =================== CONFIGURATION ===================
EARTH_RADIUS_KM = 6371.0
LAT_LIST = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
LON_LIST = [-95, -110, -125, -140, -155, -170, -180, 165]
TOL = 0.1
METOP = 'MetOP_A'
metop = 'metopa'

READ_DIR = f'/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/ASCAT_data/{METOP}/{METOP}'
WRITE_DIR = f'/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/ASCAT_data/{METOP}'

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
file_list = sorted(glob.glob(os.path.join(READ_DIR, f'ascat_*_{metop}_*.nc')))
total_files = len(file_list)

if rank == 0:
    # Clean previous outputs
    for f in glob.glob(os.path.join(WRITE_DIR, 'T_????_????_AS_*.nc')):
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

    ds = xr.open_dataset(file_path)
    ds['time'] = ds.time.isel(NUMCELLS = 0)
    ds = ds.set_coords('time').swap_dims({'NUMROWS': 'time'})
    ds['lon'] = (ds['lon'] + 360)%360

    for lat in LAT_LIST:
        lat_unit = 'S' if lat < 0 else 'N'
        for lon in LON_LIST:
            lon_unit = 'W' if lon < 0 else 'E'
            lon_mod = (lon + 360) % 360

            ### masking the values that are outsite range as nan values
            mask = (np.abs(ds['lat'] - lat) < TOL) & (np.abs(ds['lon'] - lon_mod) < TOL)
            if mask.sum() == 0:
                continue

            masked_ds = ds.where(mask, np.nan)
            ds.close()

            # selecting the time indices when the satellite crosses the position withing tolerance
            selTimeIndices = []
            for t in range(ds.dims['time']):
                notNans = np.sum(mask.isel(time=t).to_numpy())
                if notNans > 0:
                    selTimeIndices.append(t)


            selTimeIndices = np.array(selTimeIndices)
            sub_ds = ds.isel(time = selTimeIndices)
            mask = mask.isel(time = selTimeIndices)

            ## calculate distance from the target lat lon to the lat lon position of the satellite
            sub_ds = sub_ds.rename({'lat': 'AS_LAT', 'lon': 'AS_LON'})
            AS_LAT, AS_LON = sub_ds['AS_LAT'].to_numpy(), sub_ds['AS_LON'].to_numpy()
            distance = get_great_circle_distance(lat, lon, AS_LAT, AS_LON)

            sub_ds['dist_from_TAO_pos'] = xr.DataArray(
                distance,
                dims=['time', 'NUMCELLS'],
                attrs={'units': 'kilometers', 
                       'long_name': 'great circle distance from TAO position'}
            )


            ## within the selected the times sort by the distance in a row
            tlen = len(sub_ds['time'])
            out_ds = xr.Dataset()

            max_numcells = 3  ## max three values
            for var in sub_ds.data_vars:
                data = sub_ds[var]
                arr = np.full((tlen, max_numcells), np.nan)
                for t in range(tlen):
                    thisDist = distance[t,:]
                    varArr = sub_ds[var].isel(time = t).to_numpy()
                    varArr = varArr[thisDist.argsort()] # sort by distance
                    arr[t, :] = varArr[0:max_numcells]
                    
                out_ds[var] = xr.DataArray(arr, 
                                        dims=['time', 'NUMCELLS'], 
                                        coords={'time': sub_ds['time'], 
                                                'NUMCELLS': np.arange(max_numcells)}, 
                                        attrs=data.attrs)

            fname = f'T_{abs(lat):03.0f}{lat_unit}_{abs(lon):03.0f}{lon_unit}_AS_fileNumber{i:06d}_rank{rank:02d}.nc'
            out_ds.to_netcdf(os.path.join(WRITE_DIR, fname), unlimited_dims='time')
            out_ds.close()

    ds.close()
