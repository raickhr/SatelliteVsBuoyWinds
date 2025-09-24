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
METOP_DIR = ''
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


def match_times_withFrozenField(
    ds_sat, ds_buoy, 
    coord_str, 
    sat_time_var='time', 
    buoy_time_var='TIME', 
    tolerance_sec=600,
    path_length_m=25000,    # 25 km
    dt_sec=600,              # 10-min buoy cadence (used for distance integral)
    max_window_minutes=400,  # ±time window around center
):
    """Match timestamps within a specified tolerance and compute frozen-field stats.
    If one side has data gaps or we are near the ends of the record, the window
    will extend on the available side by filling in the edge values in the ghost nodes until path length is reached
    """
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

    try:
        # --- pre-bind variables ---
        u_da    = ds_buoy['U10N_x'].sel(HEIGHT=10.0)
        v_da    = ds_buoy['U10N_y'].sel(HEIGHT=10.0)

        u_da = u_da.interpolate_na(dim='TAO_TIME', method='nearest')
        v_da = v_da.interpolate_na(dim='TAO_TIME', method='nearest')

        tao_t   = ds_buoy['TAO_TIME']
        nt      = ds_buoy.sizes['TAO_TIME']

        # --- outputs aligned to matches ---
        ndata = len(idx_buoy)
        frozenTimeAvgWSPD      = np.full(ndata, np.nan)
        frozenTimeAvgUWND      = np.full(ndata, np.nan)
        frozenTimeAvgVWND      = np.full(ndata, np.nan)
        frozenTimeAvgWDIR      = np.full(ndata, np.nan)
        frozenTimeStdWSPD      = np.full(ndata, np.nan)
        frozenTimeStdCosWDIR   = np.full(ndata, np.nan)
        frozenTimeStdSinWDIR   = np.full(ndata, np.nan)
        frozenIntegrationTime  = np.full(ndata, np.nan)
        
    except Exception as e:
        log(f"initialization error {e}", level='ERR', location=coord_str)


    # def _get_wspd_time(k):
    #     return float(np.sqrt(u_da.isel(TAO_TIME=k).to_numpy()**2 + v_da.isel(TAO_TIME=k).to_numpy()**2)), tao_t.isel(TAO_TIME=k).to_numpy()

    def _get_wspd_time(k):
        return float(u_da.isel(TAO_TIME=k).to_numpy()), tao_t.isel(TAO_TIME=k).to_numpy()

    for ii in range(ndata):
        try:
            center_idx = int(idx_buoy[ii])
            t_center   = tao_t.isel(TAO_TIME=center_idx).to_numpy()

            w0, tl = _get_wspd_time(center_idx)
            if np.isnan(w0):
                continue

            # window contents and accumulated distance
            wspd_list = [w0]
            u_list    = [float(u_da.isel(TAO_TIME=center_idx).values)]
            v_list    = [float(v_da.isel(TAO_TIME=center_idx).values)]
            deltaX    = w0 * dt_sec
            deltaT    = dt_sec

            left  = center_idx
            right = center_idx
        except Exception as e:
            log(f"Center Value error {e}", level='ERR', location=coord_str)



        # keep growing until we hit target distance 

        try:
            while abs(deltaX) < path_length_m and deltaT < 86400:
                left -= 1
                if left < 0:
                    left = 0 
                wl, tl = _get_wspd_time(left)
                tdiff = abs((tl - t_center).astype('timedelta64[s]').astype(int))

                ### if there is gap in the time series sometimes tdiff > max_window_minutes
                ### if gap is found repeat the edge, repeat the index, assuming that the gappy values are values at the edge;
                if tdiff >= max_window_minutes*60:
                    left += 1

                wl, tl = _get_wspd_time(left)
                wspd_list.append(wl)
                u_list.append(float(u_da.isel(TAO_TIME=left).values))
                v_list.append(float(v_da.isel(TAO_TIME=left).values))
                deltaX += wl * dt_sec
                deltaT += dt_sec

                right += 1
                if right > nt - 1:
                    right = nt - 1 
                wl, tl = _get_wspd_time(right)
                tdiff = abs((tl - t_center).astype('timedelta64[s]').astype(int))

                ### if there is gap in the time series sometimes tdiff > max_window_minutes
                ### if gap is found repeat the edge, repeat the index, assuming that the gappy values are values at the edge;
                if tdiff >= max_window_minutes*60:
                    right -= 1

                wl, tl = _get_wspd_time(right)
                wspd_list.append(wl)
                u_list.append(float(u_da.isel(TAO_TIME=right).values))
                v_list.append(float(v_da.isel(TAO_TIME=right).values))
                deltaX += wl * dt_sec
                deltaT += dt_sec

        except Exception as e:
            log(f"Time integration Error{e}", level='ERR', location=coord_str)
            

        try:
            wspd_win = np.array(wspd_list, dtype=float)
            u_win    = np.array(u_list, dtype=float)
            v_win    = np.array(v_list, dtype=float)

            theta  = np.arctan2(v_win, u_win)
            cosT   = np.cos(theta)
            sinT   = np.sin(theta)

            DeltaT = float(len(wspd_win) * dt_sec)  # seconds; keep constant-cadence assumption
            Ubar   = float(np.mean(u_win))
            Vbar   = float(np.mean(v_win))
            dir_deg = (np.rad2deg(np.arctan2(Vbar, Ubar)) + 360.0) % 360.0

            frozenTimeAvgWSPD[ii]      = np.sqrt(Ubar**2 + Vbar**2)
            frozenTimeAvgUWND[ii]      = Ubar
            frozenTimeAvgVWND[ii]      = Vbar
            frozenTimeAvgWDIR[ii]      = dir_deg
            frozenTimeStdWSPD[ii]      = float(np.std(wspd_win, ddof=0))
            frozenTimeStdCosWDIR[ii]   = float(np.std(cosT,   ddof=0))
            frozenTimeStdSinWDIR[ii]   = float(np.std(sinT,   ddof=0))
            frozenIntegrationTime[ii]  = DeltaT
        except Exception as e:
            log(f"Array-float error {e}", level='ERR', location=coord_str)


    try:
        # --- subset matched pairs and attach results ---
        ds_sat_sel  = ds_sat.isel(QS_TIME=idx_sat)
        ds_buoy_sel = ds_buoy.isel(TAO_TIME=idx_buoy)

        ds_buoy_sel['fz_mean_WSPD_10N'] = xr.DataArray(
            frozenTimeAvgWSPD, dims=['TAO_TIME'],
            attrs={'long_name': 'average neutral wind speed (frozen-field)', 'units':'m s-1'}
        )

        ds_buoy_sel['fz_mean_UWND_10N'] = xr.DataArray(
            frozenTimeAvgUWND, dims=['TAO_TIME'],
            attrs={'long_name': 'average neutral zonal wind speed (frozen-field)', 'units':'m s-1'}
        )

        ds_buoy_sel['fz_mean_VWND_10N'] = xr.DataArray(
            frozenTimeAvgVWND, dims=['TAO_TIME'],
            attrs={'long_name': 'average neutral meridional wind speed (frozen-field)', 'units':'m s-1'}
        )

        ds_buoy_sel['fz_mean_WDIR'] = xr.DataArray(
            frozenTimeAvgWDIR, dims=['TAO_TIME'],
            attrs={'long_name': 'average wind direction (frozen-field)', 'units':'degrees'}
        )
        ds_buoy_sel['fz_std_WSPD_10N'] = xr.DataArray(
            frozenTimeStdWSPD, dims=['TAO_TIME'],
            attrs={'long_name': 'std of neutral wind speed (frozen-field)', 'units':'m s-1'}
        )
        ds_buoy_sel['fz_std_cosWDIR'] = xr.DataArray(
            frozenTimeStdCosWDIR, dims=['TAO_TIME'],
            attrs={'long_name': 'std of cos(theta) (frozen-field)'}
        )
        ds_buoy_sel['fz_std_sinWDIR'] = xr.DataArray(
            frozenTimeStdSinWDIR, dims=['TAO_TIME'],
            attrs={'long_name': 'std of sin(theta) (frozen-field)'}
        )
        ds_buoy_sel['fz_integrationTime'] = xr.DataArray(
            frozenIntegrationTime/60.0, dims=['TAO_TIME'],
            attrs={'long_name': 'integration time (frozen-field)', 'units':'minutes'}
        )
    except Exception as e:
        log(f"Creation of xarray dataset {e}", level='ERR', location=coord_str)


    # keep both time axes (align rows to QS_TIME; stash original TAO time)
    orig_tao_time = ds_buoy_sel['TAO_TIME'].to_numpy()
    ds_buoy_sel = ds_buoy_sel.rename({'TAO_TIME':'QS_TIME'})
    ds_buoy_sel = ds_buoy_sel.assign_coords(QS_TIME=ds_sat_sel['QS_TIME'])
    ds_buoy_sel['TAO_TIME_at_QS_TIME'] = xr.DataArray(orig_tao_time, dims=['QS_TIME'])

    return ds_sat_sel, ds_buoy_sel


# ============== MAIN PROCESSING ==================
def process_matchup(lat, lon):
    coord_str = format_coord(lat, lon)

    buoy_file = BUOY_DIR / f'T_{coord_str}_xrr_COARE3p5_2013_2020_2hrMeanVar.nc'
    sat_file = SAT_DIR / f'TAOpos_{coord_str}_QS.nc'
    output_filename = f'T_{coord_str}_COARE3p5_1999_2009_2hrMeanVar_QS_Matchup_fz_zonal.nc'
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

        ds_buoy = ds_buoy.sortby("TIME")
        ds_sat = ds_sat.sortby('time')

        ds_buoy = ds_buoy.drop_duplicates(dim="TIME", keep="first")
        ds_sat = ds_sat.drop_duplicates(dim="time", keep="first")

        ds_buoy = convert_to_datetime(ds_buoy, 'TIME')
        ds_sat = convert_to_datetime(ds_sat, 'time')

        ds_sat_matched, ds_buoy_matched = match_times_withFrozenField(ds_sat, ds_buoy, coord_str=coord_str)
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

