import copernicusmarine


LAT_LIST = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
LON_LIST = [-95, -110, -125, -140, -155, -170, -180, 165]

for lat in LAT_LIST:
    lat_unit = 'S' if lat < 0 else 'N'
    for lon in LON_LIST:
        lon_unit = 'W' if lon < 0 else 'E'
        lon = (lon + 360) % 360
        folder = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/oceanReanalysis/'
        outFileName = f'T_{abs(lat):03.0f}{lat_unit}_{abs(lon):03.0f}{lon_unit}_oceanReanalysis.nc'
        fullFilename = folder + outFileName
        copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        dataset_version="202311",
        variables=["vo", "zos", "uo", "thetao", "so"],
        minimum_longitude=lon-0.5,
        maximum_longitude=lon+0.5,
        minimum_latitude=lat-0.5,
        maximum_latitude=lat+0.5,
        start_datetime="1999-01-01T00:00:00",
        end_datetime="2021-06-30T00:00:00",
        minimum_depth=0.49402499198913574,
        maximum_depth=0.49402499198913574,
        coordinates_selection_method="strict-inside",
        netcdf_compression_level=1,
        disable_progress_bar=True,
        output_filename= fullFilename,
        )
