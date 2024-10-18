from netCDF4 import Dataset, date2num, num2date
import numpy as np
from glob import glob
import argparse


def getDataFromFileAtLonLatPoint(fileName, lonVal, latVal, epsilon):
    ds = Dataset(fileName)

    lat = np.array(ds.variables['lat'])
    lon = np.array(ds.variables['lon'])
    alngTrk, crsTrk = lat.shape[0], lat.shape[1]

    #epsilon = 0.1  ## tolerance in degrees
    if lonVal < 0:
        lonVal += 360

    latMask = abs(lat - latVal) < epsilon
    lonMask = abs(lon - lonVal) < epsilon
    mask = np.logical_and(latMask, lonMask)

    selData = None
    hasData = False

    if np.sum(mask) > 0:
        hasData = True
        wspd = np.array(ds.variables['retrieved_wind_speed'])
        wdir = np.array(ds.variables['retrieved_wind_direction'])
        timeArr = np.tile(np.array(ds.variables['time']), (crsTrk,1)).T


        selLon = lon[mask].flatten()
        selLat = lat[mask].flatten()
        selWsp = wspd[mask].flatten()
        selWdir = wdir[mask].flatten()
        selTimeArr = timeArr[mask].flatten()

        selData = np.stack((selLon, selLat, selWsp, selWdir, selTimeArr), axis=1)
        selData = selData[selData[:,4].argsort()]
        print('shape selData', selData.shape)
    ds.close()
    return selData, hasData


parser = argparse.ArgumentParser()

parser.add_argument('--lat', type=np.float, default=0, help='latitude')
parser.add_argument('--lon', type=np.float, default=140, help='longitude')
parser.add_argument('--epsilon', type=int, default=0.1, help='FOO!')
parser.add_argument('--folder', type=str, default='./', help='FOO!')

args = parser.parse_args()

dirLoc = args.folder
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

fileList = glob(dirLoc + "/*.nc")
AllData = np.empty((0,5), dtype=float)

for file in fileList:
    selData, hasData = getDataFromFileAtLonLatPoint(file, lon, lat, epsilon)
    if hasData:
        AllData = np.vstack((AllData,selData))

AllData = AllData[AllData[:, 4].argsort()]

startDate = num2date(AllData[0, 4], "seconds since 1999-1-1 0:0:0")
endDate = num2date(AllData[-1, 4], "seconds since 1999-1-1 0:0:0")

wfile = latLonVal + '/QuikSCAT_{0:04d}_{1:02d}_{2:02d}_T{3:02d}.{4:02d}.{5:02d}'.format(
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
wds.createDimension('time',None)

wds_time = wds.createVariable('time', float, ('time'))
wds_lon = wds.createVariable('lon', float, ('time'))
wds_lat = wds.createVariable('lat', float, ('time'))

wds_wspd = wds.createVariable('wspd', float, ('time'))
wds_wdir = wds.createVariable('wdir', float, ('time'))

wds_time.units = 'seconds since 1999-01-01 00:00:00'
wds_wspd.long_name = 'equivalent neutral winds wind speed at 10m'
wds_wdir.long_name = 'equivalent neutral winds wind direction at 10m'


wds_lon[:] = AllData[:,0]
wds_lat[:] = AllData[:,1]
wds_wspd[:] = AllData[:,2]
wds_wdir[:] = AllData[:,3]
wds_time[:] = AllData[:,4]

wds.close()



