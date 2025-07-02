import xarray as xr
import numpy as np

LAT_LIST = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
LON_LIST = [-95, -110, -125, -140, -155, -170, -180, 165]

for lat in LAT_LIST:
    lat_unit = 'S' if lat < 0 else 'N'
    for lon in LON_LIST:
        LON = lon
        lon_unit = 'W' if lon < 0 else 'E'
        lon = (lon + 360) % 360
        folder = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/oceanReanalysis/'
        fileName = f'T_{abs(lat):03.0f}{lat_unit}_{abs(LON):03.0f}{lon_unit}_oceanReanalysis.nc'
        print(fileName)
        
        ds = xr.open_dataset(folder + fileName)

        lat_arr = ds['latitude'].to_numpy()
        lon_arr = (ds['longitude'].to_numpy() + 360)%360

        lon_indx = np.argmin(abs(lon_arr - lon))
        lat_indx = np.argmin(abs(lat_arr - lat))

        nds = ds.isel(latitude = lat_indx, longitude = lon_indx, depth = 0).drop_vars(['latitude', 'longitude', 'depth'])

        nds.to_netcdf(folder + f'T_{abs(lat):03.0f}{lat_unit}_{abs(LON):03.0f}{lon_unit}_oceanReanalysis_xarr.nc', 
                      unlimited_dims = 'time')

