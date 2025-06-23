from mpi4py import MPI
import xarray as xr
import numpy as np
from pathlib import Path
import glob

# =================== CONFIGURATION ===================
LAT_LIST = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
LON_LIST = [-95, -110, -125, -140, -155, -170, -180, 165]

METOP = 'MetOP_A'
metop = 'metopa'

# LAT_LIST = [-2]
# LON_LIST = [-155]
TASKS = [(lat, lon) for lat in LAT_LIST for lon in LON_LIST]

WRITE_DIR = Path(f'/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/ASCAT_data/{METOP}')

# ================ MPI SETUP ==========================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# =================== HELPERS ===================

def distribute_tasks():
    """Divide tasks among MPI ranks."""
    avg, rem = divmod(len(TASKS), size)
    counts = np.full(size, avg)
    counts[:rem] += 1
    ends = np.cumsum(counts)
    starts = np.roll(ends, 1)
    starts[0] = 0
    return starts, ends

def format_coord(lat, lon):
    """Format latitude and longitude into a string like '009S_095W'."""
    lat_unit = 'S' if lat < 0 else 'N'
    lon_unit = 'W' if lon < 0 else 'E'
    return f"{abs(lat):03d}{lat_unit}_{abs(lon):03d}{lon_unit}"

def log(msg, level='INFO'):
    """Log messages with rank and status label."""
    levels = {'INFO': '\033[94m[INFO]', 
              'OK': '\033[92m[OK]', 
              'WARN': '\033[93m[WARN]', 
              'ERR': '\033[91m[ERR]'}
    endc = '\033[0m'
    prefix = f"[Rank {rank}] {levels.get(level, '[INFO]')}"
    print(f"{prefix} {msg}{endc}")

# =================== TASK PROCESSING ===================

def process_task(lat, lon):
    coord_str = format_coord(lat, lon)
    pos_folder = f'TAOpos_{coord_str}'
    folder_path = WRITE_DIR / pos_folder

    file_pattern = f'T_{coord_str}_AS_fileNumber??????_rank??.nc'
    file_list = sorted(folder_path.glob(file_pattern))

    output_filename = f'{pos_folder}_AS.nc'
    output_path = WRITE_DIR / output_filename

    if output_path.exists():
        try:
            # Try opening the NetCDF file to check if it's valid
            with xr.open_dataset(output_path, engine='h5netcdf') as ds:
                ds.load()  # force read to ensure it's not corrupt
                satVars = ['wind_speed',
                            'wind_dir',
                            'wvc_quality_flag',
                            'model_speed',
                            'model_dir',
                            'bs_distance',
                            'dist_from_TAO_pos']
                
                missing_var = False

                for var in satVars:
                    if var not in ds.data_vars:
                        log(f"{output_filename} does not contain variable {var}. Deleting and regenerating.", level='WARN')
                        missing_var = True
                        break
            if missing_var:
                output_path.unlink()
            else:
                log(f"{output_filename} already exists and is valid, skipping.", level='INFO')
                return
        except Exception as e:
            log(f"{output_filename} is corrupt or unreadable. Deleting and regenerating. Error: {e}", level='WARN')
            output_path.unlink()  # delete the corrupt file

    if not file_list:
        log(f"No files found in {folder_path}", level='WARN')
        return

    try:
        ds = xr.open_mfdataset(file_list, combine='by_coords')
        ds = ds.sortby('time')
        ds.to_netcdf(output_path)
        ds.close()
        log(f"Saved merged NetCDF: {output_path.name}", level='OK')
    except Exception as e:
        log(f"Failed to process {coord_str}: {e}", level='ERR')

# =================== MAIN ===================

def main():
    starts, ends = distribute_tasks()
    for i in range(starts[rank], ends[rank]):
        lat, lon = TASKS[i]
        process_task(lat, lon)

if __name__ == "__main__":
    main()
