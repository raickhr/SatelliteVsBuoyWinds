import os


LAT_LIST = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
LON_LIST = [-95, -110, -125, -140, -155, -170, -180, 165]

for lat in LAT_LIST:
    lat_unit = 'S' if lat < 0 else 'N'
    for lon in LON_LIST:
        LON = lon
        lon_unit = 'W' if lon < 0 else 'E'
        lon = (lon + 360) % 360
        folder = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/WaveReanalysis/'
        outFileName = f'T_{abs(lat):03.0f}{lat_unit}_{abs(lon):03.0f}{lon_unit}_waveReanalysis.nc'
        fullFilename = folder + outFileName
        newFileName = folder + f'T_{abs(lat):03.0f}{lat_unit}_{abs(LON):03.0f}{lon_unit}_waveReanalysis.nc'
        cmd = f'mv {fullFilename} {newFileName}'
        print(cmd)
        os.system(cmd)

