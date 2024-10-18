from netCDF4 import Dataset, date2num, num2date
import numpy as np
import sys
from glob import glob
import argparse


def getDataFromFileAtLonLatPoint(fileName, lonVal, latVal, epsilon):
    ds = Dataset(fileName)

    lat = np.array(ds.variables['Latitude'])
    lon = np.array(ds.variables['Longitude'])
    lon[lon<0] += 360 
    #print(lat, lat.shape)
    #print(lon, lon.shape)
    alngTrk, crsTrk = lat.shape[0], lat.shape[1]

    # epsilon = 0.1  ## tolerance in degrees
    if lonVal < 0:
        lonVal += 360

    latMask = abs(lat - latVal) < epsilon
    lonMask = abs(lon - lonVal) < epsilon
    mask = np.logical_and(latMask, lonMask)

    selData = None
    hasData = False
    timeUnits = 'microseconds since 1900-01-01'

    if np.sum(mask) > 0:
        hasData = True
        nearSurfRain = np.array(ds.variables['nearSurfRain'])
        timeArr = np.tile(np.array(ds.variables['Time']), (crsTrk, 1)).T
        timeUnits = ds.variables['Time'].units

        selLon = lon[mask].flatten()
        selLat = lat[mask].flatten()
        selNearSurfRain = nearSurfRain[mask].flatten()
        selTimeArr = timeArr[mask].flatten()

        selData = np.stack(
            (selLon, selLat, selNearSurfRain, selTimeArr), axis=1)
        selData = selData[selData[:, 3].argsort()]
        print('shape selData', selData.shape)
        sys.stdout.flush()
    else:
        print(f'no data in lat {latVal} and lon {lonVal} in file {fileName}')
    ds.close()
    return selData, hasData, timeUnits


parser = argparse.ArgumentParser()

parser.add_argument('--lat', type=float, default=0, help='latitude')
parser.add_argument('--lon', type=float, default=-140, help='longitude')
parser.add_argument('--epsilon', type=int, default=0.1, help='tolerance in degrees')
parser.add_argument('--folder', type=str, default='./', help='read folder location')
parser.add_argument('--wfolder', type=str, default='./', help='write folder location')

args = parser.parse_args()

dirLoc = args.folder
wdirLoc = args.wfolder
lat = args.lat
lon = args.lon
epsilon = args.epsilon

if lat < 0:
    latUnit = 'S'
else:
    latUnit = 'N'

if lon < 0:
    lonUnit = 'W'
else:
    lonUnit = 'E'

latLonVal = f'T_{abs(lat):.0f}{latUnit:s}_{abs(lon):.0f}{lonUnit:s}_'

fileList = glob(dirLoc + "/dateFixed_2A25.*.nc")
AllData = np.empty((0, 4), dtype=float)

# print(fileList[0])

for file in fileList:
    selData, hasData, timeUnits = getDataFromFileAtLonLatPoint(file, lon, lat, epsilon)
    if hasData:
        AllData = np.vstack((AllData, selData))

AllData = AllData[AllData[:, 3].argsort()]

startDate = num2date(AllData[0, 3], timeUnits)
endDate = num2date(AllData[-1, 3], timeUnits)

wfile = wdirLoc + latLonVal + 'TRMM_{0:04d}_{1:02d}_{2:02d}_T{3:02d}.{4:02d}.{5:02d}'.format(
    startDate.year,
    startDate.month,
    startDate.day,
    startDate.hour,
    startDate.minute,
    startDate.second
)

wfile += '_to_{0:04d}_{1:02d}_{2:02d}_T{3:02d}.{4:02d}.{5:02d}.nc'.format(
    endDate.year,
    endDate.month,
    endDate.day,
    endDate.hour,
    endDate.minute,
    endDate.second
)


wds = Dataset(wfile, 'w', format='NETCDF4')
wds.createDimension('Time', None)

wds_time = wds.createVariable('Time', float, ('Time'))
wds_lon = wds.createVariable('lon', float, ('Time'))
wds_lat = wds.createVariable('lat', float, ('Time'))

wds_nearSurfRain = wds.createVariable('nearSurfRain', float, ('Time'))
wds_time.units = timeUnits
wds_nearSurfRain.long_name = 'nearSurfRain'


wds_lon[:] = AllData[:, 0]
wds_lat[:] = AllData[:, 1]
wds_nearSurfRain[:] = AllData[:, 2]
wds_time[:] = AllData[:, 3]

wds.close()
