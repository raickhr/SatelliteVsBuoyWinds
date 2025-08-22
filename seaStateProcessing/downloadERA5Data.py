# ======================================
# Parallel ERA5 download with mpi4py
# ======================================

from mpi4py import MPI
import cdsapi

# ----------------------------
# 1️⃣ Define your lat/lon lists
# ----------------------------
LAT_LIST = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
LON_LIST = [-95, -110, -125, -140, -155, -170, -180, 165]

# ----------------------------
# 2️⃣ Create all (lat, lon, year, month) combinations
# ----------------------------
work_list = []
for lat in LAT_LIST:
    for lon in LON_LIST:
        for year in range(1999, 2021):
            for month in range(1, 13):
                work_list.append((lat, lon, year, month))

print(f"Total download tasks: {len(work_list)}")

# ----------------------------
# 3️⃣ Initialize MPI
# ----------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ----------------------------
# 4️⃣ Scatter work: Each rank gets its slice of work_list
# ----------------------------
my_work = work_list[rank::size]

print(f"Rank {rank}: Processing {len(my_work)} tasks.")

# ----------------------------
# 5️⃣ Each rank initializes its own CDS API client
# ----------------------------
client = cdsapi.Client()

# ----------------------------
# 6️⃣ Loop through tasks for this rank only
# ----------------------------
for lat, lon, year, month in my_work:
    lat_unit = 'S' if lat < 0 else 'N'
    lon_unit = 'W' if lon < 0 else 'E'
    
    folder = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/ERA5_reanalysis_data/'
    
    # Add year and month to filename to avoid overwrites
    outFileName = (
        f'T_{abs(lat):03.0f}{lat_unit}_'
        f'{abs(lon):03.0f}{lon_unit}_'
        f'Radiation_{year:04d}_{month:02d}.nc'
    )

    # Define your CDS API request for this chunk
    request = {
        "product_type": ["reanalysis"],
        "year": [f"{year:04d}"],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "variable": [
            "mean_surface_net_long_wave_radiation_flux",
            "mean_surface_net_short_wave_radiation_flux",
            "surface_solar_radiation_downwards",
            "surface_thermal_radiation_downwards"
        ],
        "area": [lat + 1, lon - 1, lat - 1, lon + 1]
    }

    print(f"Rank {rank}: Downloading {outFileName} ...")
    try:
        client.retrieve("reanalysis-era5-single-levels", request).download(folder + outFileName)
        print(f"Rank {rank}: Finished {outFileName}")
    except Exception as e:
        print(f"Rank {rank}: Failed {outFileName} → {e}")

# ----------------------------
# 7️⃣ All done
# ----------------------------
print(f"Rank {rank}: All tasks complete.")

