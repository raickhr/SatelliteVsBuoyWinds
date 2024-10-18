from netCDF4 import Dataset, num2date, date2num
from datetime import datetime, timedelta
import os
import sys
import pandas as pd
from glob import glob
import numpy as np
from numpy import sin, cos
from mpi4py import MPI

def get2HrVar(wspdArr, dateArr, thisDate):
    mask = np.logical_and(dateArr < (thisDate + timedelta(hours=1)), (dateArr > (thisDate - timedelta(hours=1))))
    #print(np.sum(mask))
    selWspdArr = wspdArr[mask]
    var = np.nanvar(selWspdArr)
    #print(var)
    return var


def get2HrStd(wspdArr, dateArr, thisDate):
    mask = np.logical_and(dateArr < (thisDate + timedelta(hours=1)), (dateArr > (thisDate - timedelta(hours=1))))
    if np.sum(mask) > 1:
        selWspdArr = wspdArr[mask]
        std = np.nanstd(selWspdArr)
    #print(var)
    return std

def get30minMean(wspdArr, dateArr, thisDate):
    mask = np.logical_and(dateArr < (thisDate + timedelta(minutes=15)), (dateArr > (thisDate - timedelta(minutes=15))))
    if np.sum(mask) > 1:
        selWspdArr = wspdArr[mask]
        mean = np.nanmean(selWspdArr)
    #print(var)
    return mean



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

folder = "../../downloads/QS_data/"

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
    satFname = glob(f'../../downloads/QS_data/T_{lat}{latUnits}_{lon}{lonUnits}_QS_2000_*.nc')[0]
    writeFname = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_matchupNearestFour_2000.nc'
    if os.path.isfile(bFname):
        buoyDS = Dataset(bFname)
        satDS = Dataset(satFname)

        ### reading data
        satWspd = np.array(satDS.variables['wspd'])
        satTime = np.array(satDS.variables['time'])
        satTimeUnits = satDS.variables['time'].units
        cftimes = num2date(satTime, satTimeUnits)
        satDateTime = np.array([datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) for dtm in cftimes])
        satWdir = np.array(satDS.variables['wdir'])
        satLat = np.array(satDS.variables['lat'])
        satLon = np.array(satDS.variables['lon'])
        satEflags = np.array(satDS.variables['eflags'])
        satNambiguity = np.array(satDS.variables['num_ambiguities'])

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

        twoHrStdWspd = np.zeros((nData), dtype = float) * float('nan')
        twoHrStdWdir = np.zeros((nData), dtype = float) * float('nan')

        mean30minWspd = np.zeros((nData), dtype = float) * float('nan')
        mean30minWdir = np.zeros((nData), dtype = float) * float('nan')

        selSatWspd = np.zeros((4, nData), dtype=float) * float('nan')
        selSatWdir = np.zeros((4, nData), dtype=float) * float('nan')
        selSatWdist = np.zeros((4, nData), dtype=float) * float('nan')
        selSatWtDiff = np.zeros((4, nData), dtype=float) * float('nan')
        selSatWLon = np.zeros((4, nData), dtype=float) * float('nan')
        selSatWLat = np.zeros((4, nData), dtype=float) * float('nan')
        selSatEflags = np.zeros((4, nData), dtype=float) * float('nan')
        selSatNambiguity = np.zeros((4, nData), dtype=float) * float('nan')
        
        nmatchedData = 0
        for j in range(nData):
            curTime = buoyDateTime[j]

            if (j > 12) and j < (nData-12):
                twoHrStdWspd[j] = get2HrStd(buoyWspd[j-12:j+12], buoyDateTime[j-12:j+12], curTime)
                twoHrStdWdir[j] = get2HrStd((buoyWdir[j-12:j+12]+360)%360, buoyDateTime[j-12:j+12], curTime)
                mean30minWspd[j] = get30minMean(buoyWspd[j-12:j+12], buoyDateTime[j-12:j+12], curTime)
                mean30minWdir[j] = get30minMean((buoyWdir[j-12:j+12]+360)%360, buoyDateTime[j-12:j+12], curTime)
            elif (j <= 12):
                twoHrStdWspd[j] = get2HrStd(buoyWspd[0:j+12], buoyDateTime[0:j+12], curTime)
                twoHrStdWdir[j] = get2HrStd((buoyWdir[0:j+12]+360)%360, buoyDateTime[0:j+12], curTime)
                mean30minWspd[j] = get30minMean(buoyWspd[0:j+12], buoyDateTime[0:j+12], curTime)
                mean30minWdir[j] = get30minMean((buoyWdir[0:j+12]+360)%360, buoyDateTime[0:j+12], curTime)
            elif (j < (nData-12)):
                twoHrStdWspd[j] = get2HrStd(buoyWspd[j-12:nData], buoyDateTime[j-12:nData], curTime)
                twoHrStdWdir[j] = get2HrStd((buoyWdir[j-12:nData]+360)%360, buoyDateTime[j-12:nData], curTime)
                mean30minWspd[j] = get30minMean(buoyWspd[j-12:nData], buoyDateTime[j-12:nData], curTime)
                mean30minWdir[j] = get30minMean((buoyWdir[j-12:nData]+360)%360, buoyDateTime[j-12:nData], curTime)

            mask = abs(satDateTime - curTime) < timedelta(minutes = 10)
            npoints = sum(mask)
            nmatchedData += npoints

            if npoints > 0:
                selLat = satLat[mask]
                selLon = satLon[mask]
                selWspd = satWspd[mask]
                selWdir = satWdir[mask]
                selEflags = satEflags[mask]
                selNambiguity = satNambiguity[mask]
                selTime = satDateTime[mask]
                

                dist = getGreatCircDist(selLat, selLon, LAT, LON)
                timeDiff = selTime-curTime

                timeDiffInSecs = np.zeros(npoints, dtype=float)
                for i in range(npoints):
                    timeDiffInSecs[i] = timeDiff[i].days * 24*3600 + timeDiff[i].seconds

                arr = np.stack((dist, selWspd, selWdir, selEflags, selNambiguity, timeDiffInSecs, selLon, selLat), axis=0)

                arr = arr[:,arr[0,:].argsort()]  ## sorting All Rows According to row 0

            for i in range(4):
                if i < npoints:
                    selSatWdist[i,j] = arr[0,i]
                    selSatWspd[i,j] = arr[1,i]
                    selSatWdir[i,j] = arr[2,i]
                    selSatEflags[i,j] = arr[3,i]
                    selSatNambiguity[i,j] = arr[4,i]
                    selSatWtDiff[i,j] = arr[5,i]
                    selSatWLon[i,j] = arr[6,i]
                    selSatWLat[i,j] = arr[7,i]
                else:
                    selSatWdist[i,j] = np.nan
                    selSatWspd[i,j] = np.nan
                    selSatWdir[i,j] = np.nan
                    selSatEflags[i,j] = np.nan
                    selSatNambiguity[i,j] = np.nan
                    selSatWtDiff[i,j] = np.nan
                    selSatWLon[i,j] = np.nan
                    selSatWLat[i,j] = np.nan

        ########################## REMOVING THE NAN VALUES IN SATELLITE DATA AND BUOY DATA #############################

        mask = np.isnan(buoyWspd) + np.isnan(selSatWspd[0,:])
        mask += abs(buoyWspd) > 100
        mask += abs(selSatWspd[0,:]) > 100

        selSatWdist = selSatWdist[:,~mask]
        selSatWspd = selSatWspd[:,~mask]
        selSatWdir = selSatWdir[:,~mask]
        selSatEflags = selSatEflags[:,~mask]
        selSatNambiguity = selSatNambiguity[:,~mask]
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
        twoHrStdWspd = twoHrStdWspd[~mask]
        twoHrStdWdir = twoHrStdWdir[~mask]
        mean30minWspd = mean30minWspd[~mask]
        mean30minWdir = mean30minWdir[~mask]

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

            #### QS speed and Direction ####
            wds_U10N_QS = wds.createVariable('U10N_QS', float, ('N','time',))
            wds_U10N_QS.units = 'm/sec'
            wds_U10N_QS[:,:] = selSatWspd[:,:]

            wds_U10_dir_QS= wds.createVariable('U10N_dir_QS', float, ('N','time',))
            wds_U10_dir_QS.units = 'degrees'
            wds_U10_dir_QS[:,:] = selSatWdir[:,:]

            wds_flags_QS = wds.createVariable('eflags_QS', float, ('N','time',))
            wds_flags_QS[:,:] = selSatEflags[:,:]

            wds_flags_QS = wds.createVariable('num_ambiguity_QS', float, ('N','time',))
            wds_flags_QS[:,:] = selSatNambiguity[:,:]

            wds_timeDiff_QS= wds.createVariable('satTimeDiff', float, ('N','time',))
            wds_timeDiff_QS.units = 'seconds'
            wds_timeDiff_QS.long_name = 'satellite time - buoy time'
            wds_timeDiff_QS[:,:] = selSatWtDiff[:,:]

            wds_distFromBuoy_QS= wds.createVariable('dist', float, ('N','time',))
            wds_distFromBuoy_QS.units = 'kilometers'
            wds_distFromBuoy_QS.long_name = 'distance from buoy'
            wds_distFromBuoy_QS[:,:] = selSatWdist[:,:]


            wds_Lon_QS= wds.createVariable('satLon', float, ('N','time',))
            wds_Lon_QS.units = 'degrees East'
            wds_Lon_QS.long_name = 'longitude of satellite points'
            wds_Lon_QS[:,:] = selSatWLon[:,:]


            wds_Lat_QS= wds.createVariable('satLat', float, ('N','time',))
            wds_Lat_QS.units = 'degrees North'
            wds_Lat_QS.long_name = 'latitude of satellite points'
            wds_Lat_QS[:,:] = selSatWLat[:,:]

            #### Buoy speed and direction ####
            wds_U10N = wds.createVariable('U10N_TAO', float, ('time',))
            wds_U10N.units = 'm/sec'
            wds_U10N.long_name = 'obtained from wspd'
            wds_U10N[:] = buoyWspd[:]

            wds_U10_dir= wds.createVariable('U10N_dir_TAO', float, ('time',))
            wds_U10_dir.units = 'degrees'
            wds_U10_dir.long_name = 'obtained from wdir'
            wds_U10_dir[:] = buoyWdir[:]

            wds_U10N2 = wds.createVariable('U10N_TAO2', float, ('time',))
            wds_U10N2.units = 'm/sec'
            wds_U10N2.long_name = 'obtained from vwnd, uwnd'
            wds_U10N2[:] = buoyWspd2[:]

            wds_U10_dir2= wds.createVariable('U10N_dir_TAO2', float, ('time',))
            wds_U10_dir2.units = 'degrees'
            wds_U10_dir2.long_name = 'obtained from vwnd, uwnd'
            wds_U10_dir2[:] = buoyWdir2[:]


            wds_U10N2_x = wds.createVariable('U10N_x_TAO2', float, ('time',))
            wds_U10N2_x.units = 'm/sec'
            wds_U10N2_x.long_name = 'zonal neutral wind obtained from vwnd, uwnd'
            wds_U10N2_x[:] = buoyUwnd2[:]


            wds_U10N2_y = wds.createVariable('U10N_y_TAO2', float, ('time',))
            wds_U10N2_y.units = 'm/sec'
            wds_U10N2_y.long_name = 'meridional neutral wind obtained from vwnd, uwnd'
            wds_U10N2_y[:] = buoyVwnd2[:]

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

            wds_wspdVar = wds.createVariable('wspdStd2hr_TAO', float, ('time',))
            wds_wspdVar.units = 'm^2/s^2'
            wds_wspdVar.long_name = '2hr Std Dev of wind speed centered at this date'
            wds_wspdVar[:] = twoHrStdWspd[:]

            wds_wdirVar = wds.createVariable('wdirStd2hr_TAO', float, ('time',))
            wds_wdirVar.units = 'degree^2'
            wds_wdirVar.long_name = '2hr Std Dev of wind direction centered at this date'
            wds_wdirVar[:] = twoHrStdWdir[:]


            wds_wspdMean = wds.createVariable('wspdMean30min_TAO', float, ('time',))
            wds_wspdMean.units = 'm^2/s^2'
            wds_wspdMean.long_name = '30 min mean of wind speed centered at this date'
            wds_wspdMean[:] = mean30minWspd[:]

            wds_wdirMean = wds.createVariable('wdirMean30min_TAO', float, ('time',))
            wds_wdirMean.units = 'degree^2'
            wds_wdirMean.long_name = '30 min mean of wind direction centered at this date'
            wds_wdirMean[:] = mean30minWdir[:]


            wds.close()

            print(f'{lat}{latUnits}_{lon}{lonUnits} completed: {nmatchedData} points matched')

        else:
            print(f'no data matched in {lat}{latUnits}_{lon}{lonUnits}')




        
        
        

