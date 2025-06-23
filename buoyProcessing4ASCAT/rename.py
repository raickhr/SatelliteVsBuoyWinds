from pathlib import Path

# =================== CONFIGURATION ===================
LAT_LIST = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
LON_LIST = [-95, -110, -125, -140, -155, -170, -180, 165]
TASKS = [(lat, lon) for lat in LAT_LIST for lon in LON_LIST]

AS_DIR = Path('/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/ASCAT_data')
BUOY_DIR = Path('/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/Buoy/from2013_2020/WINDS')

def format_coord(lat, lon):
    """Format latitude and longitude into a string like '009S_095W'."""
    lat_unit = 'S' if lat < 0 else 'N'
    lon_unit = 'W' if lon < 0 else 'E'
    coord_str1 = f"{abs(lat):02d}{lat_unit}_{abs(lon):03d}{lon_unit}"
    coord_str2 = f"{abs(lat):03d}{lat_unit}_{abs(lon):03d}{lon_unit}"
    return coord_str1, coord_str2

def process_task(lat, lon):
    """Rename for the formatting of lat-lon in the Buoy filenames."""
    coord_str1, coord_str2 = format_coord(lat, lon)
    pos_folder = f'TAOpos_{coord_str2}'

    sat_filename = f'{pos_folder}_AS.nc'
    old_filename = f'T_{coord_str1}_xrr_COARE3p5_2013_2020_2hrMeanVar.nc'
    new_filename = f'T_{coord_str2}_xrr_COARE3p5_2013_2020_2hrMeanVar.nc'
    old_path = BUOY_DIR / old_filename
    new_path = BUOY_DIR / new_filename

    if old_path.is_file():
        try:
            old_path.rename(new_path)
            print(f"Renamed: {old_filename} → {new_filename}")
        except Exception as e:
            print(f"[ERROR] Failed to rename {old_filename}: {e}")
    else:
        print(f"[INFO] File not found: {old_filename}")

def main():
    for lat, lon in TASKS:
        process_task(lat, lon)

if __name__ == "__main__":
    main()
