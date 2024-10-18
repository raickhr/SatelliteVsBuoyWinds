from netCDF4 import Dataset, num2date, date2num
from datetime import datetime, timedelta
import os
import sys
import pandas as pd
from glob import glob
import numpy as np
from numpy import sin, cos
from mpi4py import MPI

def getGreatCircDist(lat1, lon1, lat2, lon2):
    earthRad = 6371.0 #km
    dLambda = np.deg2rad(abs(lon2 - lon1))
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    ## from Wikipedia
    numerator = ( cos(phi2)*sin(dLambda) )**2 +(cos(phi1)*sin(phi2) - sin(phi1)*cos(phi2)*cos(dLambda))**2
    numerator = np.sqrt(numerator)

    denominator = sin(phi1)*sin(phi2) + cos(phi1)*cos(phi2)*cos(dLambda)
    
    dsigma = np.arctan2(numerator, denominator)

    dist = earthRad * dsigma
    return dist

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

print('nprocs = ', nprocs)
latList = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
lonList = [-95, -110, -125, -140, -155, -170, -180, 165]


ylen = len(latList)
xlen = len(lonList)

taskList = []

for latId  in range(ylen):
    for lonId in range(xlen):
        taskList.append([latList[latId], lonList[lonId]])

ntasks = len(taskList)

remainder = ntasks%nprocs
ntasksForMe = int(ntasks//nprocs)

# startIndex in each processor fileIndex start from 1
taskListInMe = [rank]  # the list goes on as [0,5,10] for 5 processors

if rank < remainder:
    ntasksForMe += 1

for i in range(1, ntasksForMe):
    taskListInMe.append(taskListInMe[-1]+nprocs)

folder = "../../downloads/WaveReanalysis/all/"

for task in taskListInMe:
    lat = taskList[task][0]
    lon = taskList[task][1]

    LAT = lat
    LON = lon

    if lat < 0:
        latUnits = 'S'
    else:
        latUnits = 'N'

    if lon < 0:
        lonUnits = 'W'
        LON += 360
    else:
        lonUnits = 'E'
    
    lat=abs(lat)
    lon=abs(lon)

    bFname = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_COARE3p5_2000.nc'
    waveFname= glob(f'../../downloads/WaveReanalysis/all/T_{lat}{latUnits}_{lon}{lonUnits}_wave_*.nc')[0]
    writeFname = f'../../downloads/WaveReanalysis/all/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_wave_matchupNearestFour.nc'
    if os.path.isfile(bFname):
        buoyDS = Dataset(bFname)
        waveDS = Dataset(waveFname)

        ### reading data
        waveTime = np.array(waveDS.variables['time'])
        waveTimeUnits = waveDS.variables['time'].units
        cftimes = num2date(waveTime, waveTimeUnits)
        waveDateTime = np.array([datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) for dtm in cftimes])
        waveWdir = np.array(waveDS.variables['wdir'])
        waveLat = np.array(waveDS.variables['lat'])
        waveLon = np.array(waveDS.variables['lon'])

        buoyWspd = np.array(buoyDS.variables['U10N'])
        buoyTime = np.array(buoyDS.variables['time'])
        buoyTimeUnits = buoyDS.variables['time'].units
        cftimes = num2date(buoyTime, buoyTimeUnits)
        buoyDateTime = np.array([datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) for dtm in cftimes])
        buoyWdir = np.array(buoyDS.variables['U10_direction'])

        buoyWspd2 = np.array(buoyDS.variables['U10N2'])
        buoyWdir2 = np.array(buoyDS.variables['U10_direction2'])

        buoyUwnd2 = np.array(buoyDS.variables['U10N2_x'])
        buoyVwnd2 = np.array(buoyDS.variables['U10N2_y'])

        buoySST = np.array(buoyDS.variables['SST'])
        buoyRH = np.array(buoyDS.variables['RH'])
        buoyAIRT = np.array(buoyDS.variables['AIRT'])

        nData = len(buoyDateTime)
        selSatWspd = np.zeros((4, nData), dtype=float)
        
        selSatWdir = np.zeros((4, nData), dtype=float)
        
        selSatWdist = np.zeros((4, nData), dtype=float)
        
        selSatWtDiff = np.zeros((4, nData), dtype=float)

        selSatWLon = np.zeros((4, nData), dtype=float)
        
        selSatWLat = np.zeros((4, nData), dtype=float)
        
        for j in range(nData):
            curTime = buoyDateTime[j]
            mask = abs(waveDateTime - curTime) < timedelta(minutes = 10)
            npoints = sum(mask)

            if npoints > 0:
                selLat = waveLat[mask]
                selLon = waveLon[mask]
                selWspd = VTPK[mask]
                selWdir = waveWdir[mask]
                selTime = waveDateTime[mask]

                dist = getGreatCircDist(selLat, selLon, LAT, LON)
                timeDiff = selTime-curTime

                timeDiffInSecs = np.zeros(npoints, dtype=float)
                for i in range(npoints):
                    timeDiffInSecs[i] = timeDiff[i].days * 24*3600 + timeDiff[i].seconds

                arr = np.stack((dist, selWspd, selWdir, timeDiffInSecs, selLon, selLat), axis=0)

                arr = arr[:,arr[0,:].argsort()]  ## sorting All Rows According to row 0

            for i in range(4):
                if i < npoints:
                    selSatWspd[i,j] = arr[1,i]
                    selSatWdir[i,j] = arr[2,i]
                    selSatWdist[i,j] = arr[0,i]
                    selSatWtDiff[i,j] = arr[3,i]
                    selSatWLon[i,j] = arr[4,i]
                    selSatWLat[i,j] = arr[5,i]
                else:
                    selSatWspd[i,j] = float('nan')
                    selSatWdir[i,j] = float('nan')
                    selSatWdist[i,j] = float('nan')
                    selSatWtDiff[i,j] = float('nan')
                    selSatWLon[i,j] = float('nan')
                    selSatWLat[i,j] = float('nan')

        ########################## REMOVING THE NAN VALUES IN SATELLITE DATA AND BUOY DATA #############################

        mask = np.isnan(buoyWspd) + np.isnan(selSatWspd[0,:])
        mask += abs(buoyWspd) > 100
        mask += abs(selSatWspd[0,:]) > 100

        selSatWspd = selSatWspd[:,~mask]
        selSatWdir = selSatWdir[:,~mask]
        selSatWdist = selSatWdist[:,~mask]
        selSatWtDiff = selSatWtDiff[:,~mask]
        selSatWLon = selSatWLon[:,~mask]
        selSatWLat = selSatWLat[:,~mask]

        buoyDateTime = buoyDateTime[~mask]

        buoyWspd = buoyWspd[~mask] 

        buoyWspd2 = buoyWspd2[~mask] 

        buoyWdir = buoyWdir[~mask] 

        buoyWdir2 = buoyWdir2[~mask] 

        buoyUwnd2 = buoyUwnd2[~mask] 
        buoyVwnd2 = buoyVwnd2[~mask] 

        buoySST = buoySST[~mask]
        buoyRH = buoyRH[~mask]
        buoyAIRT = buoyAIRT[~mask]