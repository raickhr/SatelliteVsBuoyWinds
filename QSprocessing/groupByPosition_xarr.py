import os
import glob
import shutil

LAT_LIST = sorted([-8, -9, -5, -2, 0, 2, 5, 8, 9])
LON_LIST = sorted([-110, -95, -125, -140, -155, -170, -180, 165])
WRITE_DIR = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/QuikSCAT_data/'

def format_coord(lat, lon):
    lat_unit = 'S' if lat < 0 else 'N'
    lon_unit = 'W' if lon < 0 else 'E'
    return f'{abs(lat):03d}{lat_unit}_{abs(lon):03d}{lon_unit}'

def main():
    for lat in LAT_LIST:
        for lon in LON_LIST:
            coord_str = format_coord(lat, lon)
            pos_folder = f'TAOpos_{coord_str}'
            dst_dir = os.path.join(WRITE_DIR, pos_folder)
            os.makedirs(dst_dir, exist_ok=True)

            wildcard_filename = f'T_{coord_str}_QS_fileNumber*_rank*.nc'
            file_pattern = os.path.join(WRITE_DIR, wildcard_filename)
            files_to_move = glob.glob(file_pattern)

            if not files_to_move:
                print(f"No files found for {coord_str}")
                continue

            for file_path in files_to_move:
                try:
                    shutil.move(file_path, dst_dir)
                    print(f"Moved: {file_path} -> {dst_dir}")
                except Exception as e:
                    print(f"Error moving {file_path}: {e}")

if __name__ == "__main__":
    main()
