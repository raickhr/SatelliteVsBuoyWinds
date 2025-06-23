from netCDF4 import Dataset, num2date, date2num
from datetime import datetime, timedelta
import os
import sys
import pandas as pd
from glob import glob
import numpy as np
import xarray as xr
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


def get2HrVar(wspdArr, dateArr, thisDate):
    mask = np.logical_and(dateArr < (thisDate + timedelta(hours=1)), (dateArr > (thisDate - timedelta(hours=1))))
    selWspdArr = wspdArr[mask]
    #print(np.sum(mask))
    var = np.nanvar(selWspdArr)
    return var

def make1D(speed):
    shape = speed.shape
    if len(shape) > 1:
        largerDim = np.argmax(shape)
        nanArr = np.logical_or(np.isnan(speed), abs(speed)>360) 
        numOfNanData = np.sum(nanArr, axis=largerDim)
        index = np.argmin(numOfNanData)

        speed = speed[index,:] if shape[1] > shape[0] else speed[:,index]      
    return speed

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

folder = "../../downloads/ASCAT_data/BuoyLocs/"

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

    bFname = f'../../downloads/Buoy/extractedGZ2/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_COARE3p5_2007.nc'
    satFname = glob(f'../../downloads/ASCAT_data/BuoyLocs/T_{lat}{latUnits}_{lon}{lonUnits}_ASCAT_2007_*.nc')[0]
    writeFname = f'../../downloads/Buoy/extractedGZ2/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_matchupNearestFour_2007.nc'

    if os.path.isfile(bFname):
        #print(f'T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}')
        buoyDS = Dataset(bFname)
        satDS = Dataset(satFname)

        ### reading data
        satWspd = np.array(satDS.variables['wspd'])
        satTime = np.array(satDS.variables['time'])
        satTimeUnits = satDS.variables['time'].units
        cftimes = num2date(satTime, satTimeUnits)  
        satDateTime = np.array([datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) for dtm in cftimes])
        #satDateTime -= (datetime(1999,1,1) - datetime(1990,1,1))  ### This is correction I DID error in selectASCATWithLatLon2.py file
        satWdir = np.array(satDS.variables['wdir'])
        satLat = np.array(satDS.variables['lat'])
        satLon = np.array(satDS.variables['lon'])


        buoyWspd = np.array(buoyDS.variables['U10N'])
        buoyWspd = make1D(buoyWspd)
        buoyTime = np.array(buoyDS.variables['time'])
        buoyTimeUnits = buoyDS.variables['time'].units
        cftimes = num2date(buoyTime, buoyTimeUnits)
        buoyDateTime = np.array([datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) for dtm in cftimes])
        buoyWdir = np.array(buoyDS.variables['U10_direction'])#
        buoyWdir = make1D(buoyWdir)

        buoySST = np.array(buoyDS.variables['SST'][:,0])
        buoyRH = np.array(buoyDS.variables['RH'][:,0])
        buoyAIRT = np.array(buoyDS.variables['AIRT'][:,0])

        nData = len(buoyDateTime)

        twoHrVarWspd = np.zeros((nData), dtype = float) * float('nan')
        twoHrVarWdir = np.zeros((nData), dtype = float) * float('nan')


        selSatWspd = np.zeros((4, nData), dtype=float) * float('nan')
        
        selSatWdir = np.zeros((4, nData), dtype=float) * float('nan')
        
        selSatWdist = np.zeros((4, nData), dtype=float) * float('nan')
        
        selSatWtDiff = np.zeros((4, nData), dtype=float) * float('nan')

        selSatWLon = np.zeros((4, nData), dtype=float) * float('nan')
        
        selSatWLat = np.zeros((4, nData), dtype=float) * float('nan')
        
        nmatchedData = 0
        for j in range(nData):
            curTime = buoyDateTime[j]
            
            if (j > 6) and j < (nData-6):
                twoHrVarWspd[j] = get2HrVar(buoyWspd[j-6:j+6], buoyDateTime[j-6:j+6], curTime)
                twoHrVarWdir[j] = get2HrVar(buoyWdir[j-6:j+6], buoyDateTime[j-6:j+6], curTime)
            else:
                twoHrVarWspd[j] = float('nan')
                twoHrVarWdir[j] = float('nan')

            mask = abs(satDateTime - curTime) < timedelta(minutes = 10)
            npoints = sum(mask)
            nmatchedData += npoints

            if npoints > 0:
                selLat = satLat[mask]
                selLon = satLon[mask]
                selWspd = satWspd[mask]
                selWdir = satWdir[mask]
                selTime = satDateTime[mask]

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
        
        print(f'total matched data in T_{lat:02d}{latUnits}_{lon:03d}{lonUnits} is {nmatchedData}')
        mask = np.isnan(buoyWspd) + np.isnan(selSatWspd[0,:])
        mask += abs(buoyWspd) > 100
        mask += abs(selSatWspd[0,:]) > 100
        print(f'{np.sum(mask)} nan data out of {len(buoyWspd)} in T_{lat:02d}{latUnits}_{lon:03d}{lonUnits} is {nmatchedData}')

        selSatWspd = selSatWspd[:,~mask]
        selSatWdir = selSatWdir[:,~mask]
        selSatWdist = selSatWdist[:,~mask]
        selSatWtDiff = selSatWtDiff[:,~mask]
        selSatWLon = selSatWLon[:,~mask]
        selSatWLat = selSatWLat[:,~mask]

        buoyDateTime = buoyDateTime[~mask]

        buoyWspd = buoyWspd[~mask] 

        buoyWdir = buoyWdir[~mask] 

        buoySST = buoySST[~mask]
        buoyRH = buoyRH[~mask]
        buoyAIRT = buoyAIRT[~mask]
        twoHrVarWspd = twoHrVarWspd[~mask]
        twoHrVarWdir = twoHrVarWdir[~mask]


        if nmatchedData > 0 :
            ### writing File

            wds = Dataset(writeFname, 'w', format='NETCDF4')

            wds.createDimension('time', None)
            timeUnits = 'seconds since 1990-01-01 00:00:00'

            wds.createDimension('N', 4)

            ### time ###########

            wds_time = wds.createVariable('time', float, ('time'))
            wds_time.units = timeUnits
            wds_time.long_name = 'Buoy date time'
            wds_time[:] = date2num(buoyDateTime, timeUnits)

            #### AS speed and Direction ####
            wds_U10N_AS = wds.createVariable('U10N_AS', float, ('N','time',))
            wds_U10N_AS.units = 'm/sec'
            wds_U10N_AS[:,:] = selSatWspd[:,:]

            wds_U10_dir_AS= wds.createVariable('U10N_dir_AS', float, ('N','time',))
            wds_U10_dir_AS.units = 'degrees'
            wds_U10_dir_AS[:,:] = selSatWdir[:,:]


            wds_timeDiff_AS= wds.createVariable('satTimeDiff', float, ('N','time',))
            wds_timeDiff_AS.units = 'seconds'
            wds_timeDiff_AS.long_name = 'satellite time - buoy time'
            wds_timeDiff_AS[:,:] = selSatWtDiff[:,:]

            wds_distFromBuoy_AS= wds.createVariable('dist', float, ('N','time',))
            wds_distFromBuoy_AS.units = 'kilometers'
            wds_distFromBuoy_AS.long_name = 'distance from buoy'
            wds_distFromBuoy_AS[:,:] = selSatWdist[:,:]


            wds_Lon_AS= wds.createVariable('satLon', float, ('N','time',))
            wds_Lon_AS.units = 'degrees East'
            wds_Lon_AS.long_name = 'longitude of satellite points'
            wds_Lon_AS[:,:] = selSatWLon[:,:]


            wds_Lat_AS= wds.createVariable('satLat', float, ('N','time',))
            wds_Lat_AS.units = 'degrees North'
            wds_Lat_AS.long_name = 'latitude of satellite points'
            wds_Lat_AS[:,:] = selSatWLat[:,:]

            #### Buoy speed and direction ####
            wds_U10N = wds.createVariable('U10N_TAO', float, ('time',))
            wds_U10N.units = 'm/sec'
            wds_U10N.long_name = 'obtained from wspd'
            wds_U10N[:] = buoyWspd[:]

            wds_U10_dir= wds.createVariable('U10N_dir_TAO', float, ('time',))
            wds_U10_dir.units = 'degrees'
            wds_U10_dir.long_name = 'obtained from wdir'
            wds_U10_dir[:] = buoyWdir[:]


            wds_SST = wds.createVariable('SST_TAO', float, ('time',))
            wds_SST.units = 'degree Celsius'
            wds_SST.long_name = 'Sea Surface Temperature'
            wds_SST[:] = buoySST[:]


            wds_RH = wds.createVariable('RH_TAO', float, ('time',))
            wds_RH.units = 'percent'
            wds_RH.long_name = 'Relative Humidity'
            wds_RH[:] = buoyRH[:]


            wds_AIRT = wds.createVariable('AIRT_TAO', float, ('time',))
            wds_AIRT.units = 'degree Celsius'
            wds_AIRT.long_name = 'Bulk Air Temperature'
            wds_AIRT[:] = buoyAIRT[:]

            wds_wspdVar = wds.createVariable('wspdVar2hr_TAO', float, ('time',))
            wds_wspdVar.units = 'm^2/s^2'
            wds_wspdVar.long_name = '2hr Variance of wind speed centered at this date'
            wds_wspdVar[:] = twoHrVarWspd[:]

            wds_wdirVar = wds.createVariable('wdirVar2hr_TAO', float, ('time',))
            wds_wdirVar.units = 'degree^2'
            wds_wdirVar.long_name = '2hr Variance of wind direction centered at this date'
            wds_wdirVar[:] = twoHrVarWdir[:]


            wds.close()

            print(f'{lat}{latUnits}_{lon}{lonUnits} completed: {nmatchedData} points matched')

        else:
            print(f'no data matched in {lat}{latUnits}_{lon}{lonUnits}')





        
        
        

