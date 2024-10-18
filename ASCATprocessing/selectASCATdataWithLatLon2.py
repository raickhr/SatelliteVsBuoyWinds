from netCDF4 import Dataset, date2num, num2date
import numpy as np
import sys
from glob import glob
import argparse

writeTimeUnit = 'seconds since 1990-01-01 00:00:00'

def getDataFromFileAtLonLatPoint(fileName, lonVal, latVal, epsilon):
    #print('opening file', fileName)
    try:
        ds = Dataset(fileName)
    except:
        print('ERROR OPENING:', fileName)

    lat = np.array(ds.variables['lat'][:], dtype=float)
    lon = np.array(ds.variables['lon'][:], dtype=float)
    
    # epsilon = 0.1  ## tolerance in degrees
    if lonVal < 0:
        lonVal += 360

    latMask = abs(lat - latVal) < epsilon
    lonMask = abs(lon - lonVal) < epsilon
    mask = np.logical_and(latMask, lonMask)

    selData = None
    hasData = False

    if np.sum(mask) > 0:
        hasData = True
        wspd = np.array(ds.variables['wind_speed'][:,:], dtype=float)
        wdir = np.array(ds.variables['wind_dir'][:,:], dtype=float)
        timeArr = np.array(ds.variables['time'][:,:], dtype=float)
        readTimeUnit = ds.variables['time'].units

        selLon = lon[mask].flatten()
        selLat = lat[mask].flatten()
        selWsp = wspd[mask].flatten()
        selWdir = wdir[mask].flatten()
        selTimeArr = timeArr[mask].flatten()

        dummy = num2date(selTimeArr, readTimeUnit)
        selTimeArr = date2num(dummy, writeTimeUnit)

        selData = np.stack(
            (selLon, selLat, selWsp, selWdir, selTimeArr), axis=1)
        selData = selData[selData[:, 4].argsort()]
        print('shape selData', selData.shape)
        sys.stdout.flush()
    ds.close()
    return selData, hasData


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

fileList = glob(dirLoc + "/ascat_*.l2.nc")
AllData = np.empty((0, 5), dtype=float)

# print(fileList[0])

for fileName in fileList:
    selData, hasData = getDataFromFileAtLonLatPoint(fileName, lon, lat, epsilon)
    if hasData:
        AllData = np.vstack((AllData, selData))

AllData = AllData[AllData[:, 4].argsort()]

startDate = num2date(AllData[0, 4], writeTimeUnit)
endDate = num2date(AllData[-1, 4], writeTimeUnit)

wfile = wdirLoc + latLonVal + 'ASCAT_{0:04d}_{1:02d}_{2:02d}_T{3:02d}.{4:02d}.{5:02d}'.format(
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
wds.createDimension('time', None)

wds_time = wds.createVariable('time', float, ('time'))
wds_lon = wds.createVariable('lon', float, ('time'))
wds_lat = wds.createVariable('lat', float, ('time'))

wds_wspd = wds.createVariable('wspd', float, ('time'))
wds_wdir = wds.createVariable('wdir', float, ('time'))

wds_time.units = writeTimeUnit
wds_wspd.long_name = 'equivalent neutral winds wind speed at 10m'
wds_wdir.long_name = 'equivalent neutral winds wind direction at 10m'


wds_lon[:] = AllData[:, 0]
wds_lat[:] = AllData[:, 1]
wds_wspd[:] = AllData[:, 2]
wds_wdir[:] = AllData[:, 3]
wds_time[:] = AllData[:, 4]

wds.close()
