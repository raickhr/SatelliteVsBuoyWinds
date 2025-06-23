import os
import re
import glob
import shutil

# This code organizes TAO buoy netCDF files by creating folders per buoy location for each variable.
# Ensure all *.nc files are pre-extracted and placed under their corresponding variable folders.

variables = ['WINDS', 'SST', 'RH', 'AIRT', 'BARO', 'RAD', 'LWR', 'RAIN']
base_path = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/Buoy/TAO_data'

for var in variables:
    var_folder = os.path.join(base_path, var)
    nc_files = glob.glob(os.path.join(var_folder, '*.nc'))

    buoy_ids = set()
    for file_path in nc_files:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        if len(parts) > 1:
            ### All files should have format "TAO_T<lat>N/S<lon>E/W_string*.nc"
            buoy_ids.add(parts[1])

    # Save unique buoy IDs to a text file
    with open(f'locs_{var}.txt', 'w') as f:
        for buoy_id in sorted(buoy_ids):
            f.write(f"{buoy_id}\n")

    # Process each buoy
    for buoy_id in buoy_ids:
        orig_name = f"{buoy_id}_buoy"

        # Extract numbers and cardinal directions
        match_coords = re.findall(r'\d+', orig_name)
        # skip first char in case it's part of number because file name has 'T' before pos
        match_dirs = re.findall(r'[A-Z]+', orig_name[1:])  

        if len(match_coords) < 2 or len(match_dirs) < 2:
            print(f"Skipping {buoy_id}: could not extract lat/lon info")
            continue

        lat, lon = map(int, match_coords[:2])
        lat_dir, lon_dir = match_dirs[:2]

        # Create target folder
        target_folder = os.path.join(var_folder, f"T_{lat:03d}{lat_dir}_{lon:03d}{lon_dir}")
        os.makedirs(target_folder, exist_ok=True)

        # Move matching files to the new folder
        matching_files = glob.glob(os.path.join(var_folder, f"*_{buoy_id}_*.nc"))
        for file_path in matching_files:
            shutil.move(file_path, os.path.join(target_folder, os.path.basename(file_path)))
