from netCDF4 import Dataset, date2num, num2date
import numpy as np
import sys
from glob import glob
import argparse


def getDataFromFileAtLonLatPoint(fileName, lonVal, latVal, epsilon):
    ds = Dataset(fileName)

    lat = np.array(ds.variables['lat'])
    lon = np.array(ds.variables['lon'])
    alngTrk, crsTrk = lat.shape[0], lat.shape[1]

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
        wspd = np.array(ds.variables['retrieved_wind_speed'])
        wdir = np.array(ds.variables['retrieved_wind_direction'])
        eflags = np.array(ds.variables['eflags'])
        flags = np.array(ds.variables['flags'])
        rain_impact = np.array(ds.variables['rain_impact'])
        nAmbiguity = np.array(ds.variables['num_ambiguities'])
        #ambiguity_dir = np.array(ds.variables['ambiguity_direction'])
        #ambiguity_speed = np.array(ds.variables['ambiguity_speed'])

        timeArr = np.tile(np.array(ds.variables['time']), (crsTrk, 1)).T

        selLon = lon[mask].flatten()
        selLat = lat[mask].flatten()
        selWsp = wspd[mask].flatten()
        selWdir = wdir[mask].flatten()
        selTimeArr = timeArr[mask].flatten()
        seleflags = eflags[mask].flatten()
        selflags = flags[mask].flatten()
        selRainImpact = rain_impact[mask].flatten()
        selnAmbiguity = nAmbiguity[mask].flatten()
        #selambiguity_dir = ambiguity_dir[mask].flatten()
        #selambiguity_speed = ambiguity_speed[mask].flatten()

        # print(f'shape {selLon.shape}')
        # print(f'shape {selLat.shape}')
        # print(f'shape {selWsp.shape}')
        # print(f'shape {selWdir.shape}')
        # print(f'shape {selTimeArr.shape}')
        # print(f'shape {seleflags.shape}')
        # print(f'shape {selnAmbiguity.shape}')
        #print(f'shape {selambiguity_dir.shape}')
        #print(f'shape {selambiguity_speed.shape}')

        selData = np.stack(
            (selLon, 
             selLat, #1
             selWsp, #2
             selWdir, #3
             seleflags, #4 
             selnAmbiguity, #5 
             selflags, #6
             selRainImpact, #7
             selTimeArr), axis=1) #8
        selData = selData[selData[:, -1].argsort()]
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

test = wdirLoc + latLonVal + '*'

calculate = True
if len(glob(test)) < 1:
    calculate = True
else:
    calculate = False

if calculate:

    fileList = glob(dirLoc + "/qs_l2b_?????_v4.1_????????????.nc")
    AllData = np.empty((0, 9), dtype=float)

    # print(fileList[0])

    for file in fileList:
        selData, hasData = getDataFromFileAtLonLatPoint(file, lon, lat, epsilon)
        if hasData:
            AllData = np.vstack((AllData, selData))

    AllData = AllData[AllData[:, -1].argsort()]

    startDate = num2date(AllData[0, -1], "seconds since 1999-1-1 0:0:0")
    endDate = num2date(AllData[-1, -1], "seconds since 1999-1-1 0:0:0")

    wfile = wdirLoc + latLonVal + 'QS_{0:04d}_{1:02d}_{2:02d}_T{3:02d}.{4:02d}.{5:02d}'.format(
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
    wds_eflags = wds.createVariable('eflags', int, ('time'))
    wds_nAmbiguity = wds.createVariable('num_ambiguities', float, ('time'))
    wds_flags = wds.createVariable('flags', float, ('time'))
    wds_RainImpact = wds.createVariable('rain_impact', float, ('time'))

    wds_time.units = 'seconds since 1999-01-01 00:00:00'
    wds_wspd.long_name = 'equivalent neutral winds wind speed at 10m'
    wds_wdir.long_name = 'equivalent neutral winds wind direction at 10m'


    wds_lon[:] = AllData[:, 0]
    wds_lat[:] = AllData[:, 1]
    wds_wspd[:] = AllData[:, 2]
    wds_wdir[:] = AllData[:, 3]
    wds_eflags[:] = AllData[:,4]
    wds_nAmbiguity[:] = AllData[:,5]
    wds_flags[:] = AllData[:,6]
    wds_RainImpact[:] = AllData[:,7]
    wds_time[:] = AllData[:, -1]

    wds.close()

else:
    print(f'skipping {test}')
