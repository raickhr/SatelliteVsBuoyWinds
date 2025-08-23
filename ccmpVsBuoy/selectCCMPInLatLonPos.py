#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import gc
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Iterable

import numpy as np
import pandas as pd
import xarray as xr
from mpi4py import MPI

# ==================== CONFIGURATION ====================
LAT_LIST: List[int] = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
LON_LIST: List[int] = [-95, -110, -125, -140, -155, -170, -180, 165]
TASKS: List[Tuple[int, int]] = [(lat, lon) for lat in LAT_LIST for lon in LON_LIST]

ROOT_DIR = Path("../../downloads").resolve()
BUOY_DIR = ROOT_DIR / "Buoy/TAO_data/WINDS"          # (not used below, kept for context)
SAT_DWNDIR = ROOT_DIR / "CCMP/data.remss.com/ccmp/v03.1"
SAT_DIR = ROOT_DIR / "CCMP/atBuoyLocs"               # this is the location where the time series will be written
SAT_DIR.mkdir(parents=True, exist_ok=True)

# xarray open options for stability (tuned to avoid dask-parallel issues on some systems)
XR_OPEN_KW = dict(engine="h5netcdf", parallel=False)

# ==================== MPI SETUP ====================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ==================== LOGGING ======================
def log(msg: str, level: str = "INFO", where: str = "") -> None:
    """Pretty printer with MPI rank and ANSI colors."""
    colors = {
        "INFO": "\033[94m",
        "OK": "\033[92m",
        "WARN": "\033[93m",
        "ERR": "\033[91m",
        "DEBUG": "\033[90m",
    }
    prefix = f"[Rank {rank:02d}]"
    if where:
        prefix += f" [{where}]"
    color = colors.get(level, "")
    endc = "\033[0m"
    print(f"{prefix} {color}[{level}] {msg}{endc}")
    sys.stdout.flush()

# ==================== HELPERS ======================
def distribute_tasks(tasks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Distribute tasks evenly across MPI ranks."""
    n = len(tasks)
    avg, rem = divmod(n, size)
    counts = [avg + 1 if r < rem else avg for r in range(size)]
    starts = [sum(counts[:r]) for r in range(size)]
    ends = [start + count for start, count in zip(starts, counts)]
    mine = tasks[starts[rank]:ends[rank]]
    log(f"Assigned {len(mine)} / {n} tasks.", level="DEBUG", where="distribute_tasks")
    return mine

def format_coord(lat: int, lon: int) -> str:
    """Format latitude/longitude into strings like '009S_095W'."""
    lat_unit = "S" if lat < 0 else "N"
    lon_unit = "W" if lon < 0 else "E"
    return f"{abs(int(lat)):03d}{lat_unit}_{abs(int(lon)):03d}{lon_unit}"

def ensure_time_sorted(ds: xr.Dataset, time_var: str = "time") -> xr.Dataset:
    """Sort by time if present; warn if missing."""
    if time_var in ds:
        return ds.sortby(time_var)
    warnings.warn(f"Dataset missing time variable '{time_var}'. Skipping time sort.")
    log(f"Missing '{time_var}' in dataset.", level="WARN", where="ensure_time_sorted")
    return ds

def normalize_longitude(ds: xr.Dataset, lon_var: str = "longitude") -> xr.Dataset:
    """Normalize longitude to [0, 360) and sort."""
    if lon_var not in ds:
        warnings.warn(f"Dataset missing longitude variable '{lon_var}'.")
        log(f"Missing '{lon_var}' in dataset.", level="WARN", where="normalize_longitude")
        return ds
    ds = ds.assign_coords({lon_var: (ds[lon_var] + 360) % 360})
    return ds.sortby(lon_var)

def find_nearest_index(coord_vals: np.ndarray, target: float) -> int:
    """Index of nearest coordinate value to target."""
    return int(np.nanargmin(np.abs(coord_vals - target)))

def monthly_ccmp_paths(year: int, month: int) -> List[Path]:
    """Return sorted list of daily CCMP files for a given year/month."""
    pattern = f"Y{year:04d}/M{month:02d}/CCMP_Wind_Analysis_{year:04d}{month:02d}??_V03.1_L4.nc"
    files = sorted((SAT_DWNDIR / f"Y{year:04d}" / f"M{month:02d}").glob(f"CCMP_Wind_Analysis_{year:04d}{month:02d}??_V03.1_L4.nc"))
    if not files:
        warnings.warn(f"No CCMP files found for {year}-{month:02d} with pattern {pattern}")
        log(f"No files for {year}-{month:02d}.", level="WARN", where="monthly_ccmp_paths")
    return files

def write_time_series_at_pos(lon: int, lat: int,
                             years: Iterable[int] = range(2000, 2008),
                             months: Iterable[int] = range(1, 13)) -> None:
    """Extract monthly time series at nearest grid point and save per-month NetCDFs."""
    coord_str = format_coord(lat, lon)
    lon360 = (lon + 360) % 360

    for year in years:
        for month in months:
            wfname = f"CCMP_tseries_{coord_str}_{year:04d}_{month:02d}.nc"
            out_path = SAT_DIR / wfname

            if out_path.exists():
                log(f"Exists -> {out_path.name} (skip)", level="DEBUG", where="write_time_series_at_pos")
                continue

            files = monthly_ccmp_paths(year, month)
            if not files:
                # Nothing to do this month
                continue

            try:
                ds = xr.open_mfdataset(files, combine="by_coords", **XR_OPEN_KW)
            except Exception as e:
                log(f"open_mfdataset failed for {year}-{month:02d}: {e}", level="ERR", where="write_time_series_at_pos")
                continue

            # Standardize & sort
            ds = ensure_time_sorted(ds, time_var="time")
            ds = normalize_longitude(ds, lon_var="longitude")

            # Sanity checks
            for needed in ["latitude", "longitude"]:
                if needed not in ds.coords and needed not in ds.variables:
                    log(f"Missing coordinate '{needed}' in dataset.", level="ERR", where="write_time_series_at_pos")
                    ds.close()
                    del ds
                    gc.collect()
                    continue

            lat_vals = ds["latitude"].to_numpy()
            lon_vals = ds["longitude"].to_numpy()

            lat_idx = find_nearest_index(lat_vals, float(lat))
            lon_idx = find_nearest_index(lon_vals, float(lon360))

            try:
                selds = ds.isel(latitude=lat_idx, longitude=lon_idx)
            except Exception as e:
                log(f"Indexing failed at lat_idx={lat_idx}, lon_idx={lon_idx}: {e}", level="ERR", where="write_time_series_at_pos")
                ds.close()
                del ds
                gc.collect()
                continue

            # Save compactly, keep unlimited time if present
            enc = {}
            time_dim = "time" if "time" in selds.dims else None
            try:
                selds.to_netcdf(out_path,
                                unlimited_dims=(time_dim if time_dim else None),
                                encoding=enc)
                log(f"Wrote {out_path.name}", level="OK", where="write_time_series_at_pos")
            except Exception as e:
                log(f"Failed writing {out_path.name}: {e}", level="ERR", where="write_time_series_at_pos")
            finally:
                # Clean up
                try:
                    selds.close()
                except Exception:
                    pass
                try:
                    ds.close()
                except Exception:
                    pass
                del selds, ds
                gc.collect()

def concat_time_series_at_pos(lon: int, lat: int) -> None:
    """Concatenate all monthly files for a (lat, lon) into a single NetCDF."""
    coord_str = format_coord(lat, lon)
    final_name = f"CCMP_tseries_{coord_str}.nc"
    final_path = SAT_DIR / final_name

    # Gather all monthly files for this coordinate
    monthly_files = sorted(SAT_DIR.glob(f"CCMP_tseries_{coord_str}_????_??.nc"))
    if not monthly_files:
        warnings.warn(f"No monthly files found to concatenate for {coord_str}.")
        log(f"No monthly pieces for {coord_str}.", level="WARN", where="concat_time_series_at_pos")
        return

    try:
        ds = xr.open_mfdataset(monthly_files, combine="by_coords", **XR_OPEN_KW)
        ds = ensure_time_sorted(ds, time_var="time")
    except Exception as e:
        log(f"Concatenation open failed for {coord_str}: {e}", level="ERR", where="concat_time_series_at_pos")
        return

    try:
        ds.to_netcdf(final_path)
        log(f"Concatenated -> {final_path.name}", level="OK", where="concat_time_series_at_pos")
    except Exception as e:
        log(f"Failed writing {final_path.name}: {e}", level="ERR", where="concat_time_series_at_pos")
    finally:
        try:
            ds.close()
        except Exception:
            pass
        del ds
        gc.collect()

def main() -> None:
    # Distribute unique (lat, lon) tasks per rank
    my_tasks = distribute_tasks(TASKS)

    # Step 1: Create monthly time series files
    for lat, lon in my_tasks:
        log(f"Start monthly extraction for {format_coord(lat, lon)}", level="INFO", where="main")
        write_time_series_at_pos(lon=lon, lat=lat)

    # Ensure all ranks finish writing before concatenation
    comm.Barrier()

    # Step 2: Concatenate per-location files
    for lat, lon in my_tasks:
        log(f"Start concatenation for {format_coord(lat, lon)}", level="INFO", where="main")
        concat_time_series_at_pos(lon=lon, lat=lat)

    # Optional final barrier and graceful finalize
    comm.Barrier()
    log("All done.", level="OK", where="main")
    MPI.Finalize()

if __name__ == "__main__":
    # Quiet some noisy warnings (optional)
    warnings.filterwarnings("once", category=UserWarning)
    main()

