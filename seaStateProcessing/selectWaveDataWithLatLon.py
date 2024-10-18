from netCDF4 import Dataset, date2num, num2date
import numpy as np
import sys
from glob import glob
import argparse

wtimeUnits = "hours since 1950-01-01 00:00:0.0"

def getDataFromFileAtLonLatPoint(fileName, lonVal, latVal, epsilon):
    ds = Dataset(fileName)

    lat1d = np.array(ds.variables['latitude'])
    lon1d = np.array(ds.variables['longitude'])%360

    #print('range of longitute in file', min(lon1d), max(lon1d))
    
    if lonVal < 0:
        lonVal += 360

    lon, lat = np.meshgrid(lon1d, lat1d)
    #epsilon = 0.1  ## tolerance in degrees
    
    #latIdex = np.argmin(abs())
    latMask = abs(lat - latVal) < epsilon
    lonMask = abs(lon - lonVal) < epsilon
    mask = np.logical_and(latMask, lonMask)

    selData = None
    hasData = False

    nDataPoints = np.sum(mask)
    #print('points inside the range', nDataPoints)
    if nDataPoints > 0:
        hasData = True
        VTPK = np.array(ds.variables['VTPK'])

        VPED = np.array(ds.variables['VPED'])

        VHM0 = np.array(ds.variables['VHM0'])

        timeVal = np.array(ds.variables['time'])
        timeUnits = ds.variables['time'].units

        dates = num2date(timeVal, timeUnits)
        timeVal = date2num(dates, wtimeUnits)

        tlen = len(timeVal)
        #print('time instances in the file is ', tlen)
        selTimeArr = np.zeros((tlen * np.sum(mask)), dtype=float)
        selLon = np.zeros((tlen * np.sum(mask)), dtype=float)
        selLat = np.zeros((tlen * np.sum(mask)), dtype=float)
        selVTPK = np.zeros((tlen * np.sum(mask)), dtype=float)
        selVPED = np.zeros((tlen * np.sum(mask)), dtype=float)
        selVHM0 = np.zeros((tlen * np.sum(mask)), dtype=float) 

        for t in range(tlen):
            start = t*nDataPoints
            end = start + nDataPoints
            #print('start and end',start, end)
            selTimeArr[start:end] = timeVal[t]

            selLon[start:end] = lon[mask].flatten()
            selLat[start:end] = lat[mask].flatten()
            #print(VTPK.shape, mask.shape)
            selVTPK[start:end] = VTPK[t,mask].flatten()
            selVPED[start:end] = VPED[t,mask].flatten()
            selVHM0[start:end] = VHM0[t,mask].flatten() 

        selData = np.stack((selLon, selLat, selVTPK, selVPED, selVHM0, selTimeArr), axis=1)
        selData = selData[selData[:,5].argsort()]
        #print('shape selData', selData.shape)
        sys.stdout.flush()
    ds.close()
    return selData, hasData


parser = argparse.ArgumentParser()

parser.add_argument('--lat', type=float, default=0, help='latitude')
parser.add_argument('--lon', type=float, default=-140, help='longitude')
parser.add_argument('--epsilon', type=float, default=0.1, help='FOO!')
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
#print('The list of file reading is \n', fileList)
AllData = np.empty((0,6), dtype=float)

# print(fileList[0])

for file in fileList:
    selData, hasData = getDataFromFileAtLonLatPoint(file, lon, lat, epsilon)
    if hasData:
        AllData = np.vstack((AllData,selData))

startDate = num2date(AllData[0, 5], wtimeUnits )
endDate = num2date(AllData[-1, 5], wtimeUnits )

wfile = dirLoc +'/'+ latLonVal + 'wave_{0:04d}_{1:02d}_{2:02d}_T{3:02d}.{4:02d}.{5:02d}'.format(
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
wds_time.units = wtimeUnits

wds_lon = wds.createVariable('lon', float, ('time'))
wds_lat = wds.createVariable('lat', float, ('time'))

wds_VTPK = wds.createVariable('VTPK', float, ('time'))
wds_VTPK.units = "s"
wds_VTPK.long_name = "Wave period at spectral peak / peak period (Tp)"

wds_VPED = wds.createVariable('VPED', float, ('time'))
wds_VPED.units = "degree"
wds_VPED.long_name = "Wave principal direction at spectral peak"

wds_VHM0 = wds.createVariable('VHM0', float, ('time'))
wds_VHM0.units = "m"
wds_VHM0.long_name = "Spectral significant wave height (Hm0)"


wds_lon[:] = AllData[:,0]
wds_lat[:] = AllData[:,1]
wds_time[:] = AllData[:,5]
wds_VTPK[:] = AllData[:, 2]
wds_VPED[:] = AllData[:, 3]
wds_VHM0[:] = AllData[:, 4]

wds.close()



