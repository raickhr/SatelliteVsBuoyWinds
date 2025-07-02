import copernicusmarine


LAT_LIST = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
LON_LIST = [-95, -110, -125, -140, -155, -170, -180, 165]

for lat in LAT_LIST:
    lat_unit = 'S' if lat < 0 else 'N'
    for lon in LON_LIST:
        lon_unit = 'W' if lon < 0 else 'E'
        lon = (lon + 360) % 360
        folder = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/WaveReanalysis/'
        outFileName = f'T_{abs(lat):03.0f}{lat_unit}_{abs(lon):03.0f}{lon_unit}_waveReanalysis.nc'
        fullFilename = folder + outFileName
        copernicusmarine.subset(
			dataset_id="cmems_mod_glo_wav_my_0.2deg_PT3H-i",
			variables=["VHM0", "VHM0_SW1", "VHM0_SW2", "VHM0_WW", "VMDR", "VMDR_SW1", "VMDR_SW2", "VMDR_WW", "VPED", "VSDX", "VSDY", "VTM01_SW1", "VTM01_SW2", "VTM01_WW", "VTM02", "VTM10", "VTPK"],
			minimum_longitude=lon-1,
			maximum_longitude=lon+1,
			minimum_latitude=lat-1,
			maximum_latitude=lat+1,
			start_datetime="1999-01-01T00:00:00",
			end_datetime="2023-04-30T21:00:00",
                        output_filename= fullFilename,
		)

