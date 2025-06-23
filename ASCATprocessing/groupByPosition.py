from pathlib import Path
import shutil

# Constants
LAT_LIST = sorted([-8, -9, -5, -2, 0, 2, 5, 8, 9])
LON_LIST = sorted([-110, -95, -125, -140, -155, -170, -180, 165])

METOP = 'MetOP_A'
metop = 'metopa'

WRITE_DIR = Path(f'/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/ASCAT_data/{METOP}')

def format_coord(lat, lon):
    """Return formatted coordinate string like '008S_095W'."""
    lat_unit = 'S' if lat < 0 else 'N'
    lon_unit = 'W' if lon < 0 else 'E'
    return f'{abs(lat):03d}{lat_unit}_{abs(lon):03d}{lon_unit}'

def move_matching_files(lat, lon):
    """Move files matching the lat/lon pattern into corresponding directory."""
    coord_str = format_coord(lat, lon)
    pos_folder = f'TAOpos_{coord_str}'
    dst_dir = WRITE_DIR / pos_folder
    dst_dir.mkdir(parents=True, exist_ok=True)

    pattern = f'T_{coord_str}_AS_fileNumber*_rank*.nc'
    matching_files = list(WRITE_DIR.glob(pattern))

    if not matching_files:
        print(f"[INFO] No files found for {coord_str}")
        return

    for file_path in matching_files:
        try:
            shutil.move(str(file_path), str(dst_dir))
            print(f"[OK] Moved: {file_path.name} -> {dst_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to move {file_path.name}: {e}")

def main():
    for lat in LAT_LIST:
        for lon in LON_LIST:
            move_matching_files(lat, lon)

if __name__ == "__main__":
    main()
