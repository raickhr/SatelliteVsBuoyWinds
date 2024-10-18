import sys
sys.path.append("../COARE3p5/COAREalgorithm/Python/COARE3p5")
from coare35vn import *
import warnings
warnings.filterwarnings("ignore")

from netCDF4 import Dataset, num2date, date2num
from datetime import datetime, timedelta
import os

import numpy as np
from mpi4py import MPI


comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

def make1D(speed):
    shape = speed.shape
    if len(shape) > 1:
        speed = speed[0,:] if shape[1] > shape[0] else speed[:,0]
    return speed

def removeNan(dateTimeArr, arr):
    mask = abs(arr) > 1e10
    mask += np.isnan(arr)
    arr = arr[~mask]
    dateTimeArr = np.array(dateTimeArr)[~mask]
    return dateTimeArr.tolist(), arr

def fillData(date1, date2, arr1, arr2):
    ### length of arr 2/date2 should be always be more than arr1/date1
    
    len1 = len(date1)
    len2 = len(date2)
    #print(f'length date1 {len1}, length date2 {len2}')
    
    loop = True
    
    i = 0
    j = 0
    
    while loop:
        #print(i,j)
        #print('date1 ', date1[i], ' date2', date2[j], '\n i,j',i,j)
        if i < len(date1):
            if date1[i] < date2[j]:
                print('condition 1')
                print('date1 ', date1[i], ' date2', date2[j], '\n i,j',i,j)
                
                dateval = date1[i]
                arrval = float('nan')
                
                index = j
                date2.insert(index, dateval)
                arr2 = np.insert(arr2, index, arrval)
                
                #print('after adding in date2')
                #print('date1 ', date1[i], '\n date2', date2[j], '\n i,j',i,j, '\n')
                
            elif date1[i] > date2[j]:
                #print('condition 2')
                #print('date1', date1[i], '\n date2', date2[j], '\n i,j',i,j)
                #sys.exit()
                
                dateval = date2[j]
                arrval = float('nan')
                
                index = i
                date1.insert(index, dateval)
                arr1 = np.insert(arr1, index, arrval)
                
                #print('after adding in date1')
                #print('date1 ', date1[i], '\n date2', date2[j], '\n i,j',i,j, '\n')
                
            else:
                i+=1
                j+=1
        else:
            date1.append(date2[j])
            arr1 = np.insert(arr1, i, float('nan'))
            i+=1
            j+=1
                
                
        if j >= len(date2):
            #print(f'ending i = {i}, j = {j}, dataDate = {date1[i-1]}, nanDate = {date2[j-1]}')
            loop = False
            
        
        
    return date1, date2, arr1, arr2

def fillMask(buoy, speed, airt, rh, sst):
    if (len(speed) + len(airt) + len(rh) + len(sst))/4 != len(airt):
        print(buoy, len(speed), len(airt), len(rh), len(sst))
        MPI.Finalize()
        sys.exit()
    mask = np.isnan(speed)
    mask += np.isnan(airt)
    mask += np.isnan(rh)
    mask += np.isnan(sst)
    speed[mask] = float('nan')
    airt[mask] = float('nan')
    rh[mask] = float('nan')
    sst[mask] = float('nan')
    return speed, airt, rh, sst


def removeDuplicateAndSort(date, arr):
    #print('before removing duplicate')
    #print(f'length date {len(date)}, length array {len(arr)}')
    date = np.array(date)
    date = date[np.argsort(date)]
    udate, indx = np.unique(date, return_index=True)
    
    # for i in range(len(indx)-1):
    #     if indx[i]+1 != indx[i+1]:
    #         #print(date[i], date[i+1])
    #         #print(arr[i+1], arr[i+1]) 

    
    arr = arr[indx]
    date = udate.tolist()
    #print('after removing duplicate')
    #print(f'length date {len(date)}, length array {len(arr)}\n')
    return date, arr

print('nprocs = ', nprocs)
latList = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
lonList = [-95, -110, -125, -140, -155, -170, -180, 165]

# latList = [5]
# lonList = [-125]

# latList = [5]
# lonList = [-125]

# latList = [5]
# lonList = [-125]

# latList = [5]
# lonList = [-125]

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


# timeList = []

# deltaT = timedelta(minutes=10)
# starttime = datetime(2000, 1, 1, 0, 0)
# endtime = datetime(2001, 1, 1, 0, 0)
# curtime = starttime

# while curtime <= endtime:
#     timeList.append(curtime)
#     curtime+=deltaT
    
# nanArr = np.ones(len(timeList), dtype=float)

for task in taskListInMe:
    lat = taskList[task][0]
    lon = taskList[task][1]

    if lat < 0:
        latUnits = 'S'
    else:
        latUnits = 'N'

    if lon < 0:
        lonUnits = 'W'
    else:
        lonUnits = 'E'
    
    lat=abs(lat)
    lon=abs(lon)

    bWinds = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_WINDS_2000.nc'
    bAirT = f'../../downloads/Buoy/extractedGZ/AIRT/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_AIRT_2000.nc'
    bSST = f'../../downloads/Buoy/extractedGZ/SST/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_SST_2000.nc'
    bRH = f'../../downloads/Buoy/extractedGZ/RH/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_RH_2000.nc'
    bRAIN = f'../../downloads/Buoy/extractedGZ/RAIN/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_RAIN_2000.nc'

    if os.path.isfile(bWinds) and os.path.isfile(bRAIN):
        #print(f'{bWinds} present')
        writeFname = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_withRAIN_COARE3p5_2000.nc'

        ds_WIND = Dataset(bWinds)
        ds_SST = Dataset(bSST)
        ds_AIRT = Dataset(bAirT)
        ds_RH = Dataset(bRH)
        ds_RAIN = Dataset(bRAIN)

        speed = np.array(ds_WIND.variables['WSPD'])
        QLTY = np.array(ds_WIND.variables['WSPD_QC'])
        mask = QLTY == 0 
        mask += QLTY >= 4
        speed[mask] = float('nan')
        speed = make1D(speed)
        

        wdir = np.array(ds_WIND.variables['WDIR'])
        QLTY = np.array(ds_WIND.variables['WDIR_QC'])
        mask = QLTY == 0 
        mask += QLTY >= 4
        wdir[mask] = float('nan')
        wdir = make1D(wdir)
        wdir[wdir<0] += 360

        uwnd = np.array(ds_WIND.variables['UWND'])
        uwnd = make1D(uwnd)

        vwnd = np.array(ds_WIND.variables['VWND'])
        vwnd = make1D(vwnd)

        rh = np.array(ds_RH.variables['RELH'])
        QLTY = np.array(ds_RH.variables['RELH_QC'])
        mask = QLTY == 0 
        mask += QLTY >= 4
        rh[mask] =float('nan')
        rh = make1D(rh)

        rain = np.array(ds_RAIN.variables['RAIN'])
        QLTY = np.array(ds_RAIN.variables['RAIN_QC'])
        mask = QLTY == 0 
        mask += QLTY >= 4
        rain[mask] =float('nan')
        rain = make1D(rain)
        

        sst = np.array(ds_SST.variables['SST'])
        QLTY = np.array(ds_SST.variables['SST_QC'])
        mask = QLTY == 0 
        mask += QLTY >= 4
        sst[mask] = float('nan')
        sst = make1D(sst)
        

        airt = np.array(ds_AIRT.variables['AIRT'])
        QLTY = np.array(ds_AIRT.variables['AIRT_QC'])
        mask = QLTY == 0 
        mask += QLTY >= 4
        airt[mask] = float('nan')
        airt = make1D(airt)
        
        

        cdfTime = ds_WIND.variables['TIME']
        timeUnit = cdfTime.units
        timeArr = np.array(cdfTime)
        cftimes=num2date(timeArr, timeUnit)
        wind_dateTimeArr = [datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) for dtm in cftimes]
        wspd_dateTimeArr = wind_dateTimeArr.copy()
        wdir_dateTimeArr = wind_dateTimeArr.copy()
        uwnd_dateTimeArr = wind_dateTimeArr.copy()
        vwnd_dateTimeArr = wind_dateTimeArr.copy()
        
        wspd_dateTimeArr, speed = removeNan(wspd_dateTimeArr, speed)
        wdir_dateTimeArr, wdir = removeNan(wdir_dateTimeArr, wdir)
        #print('removing duplicate wspd')
        wspd_dateTimeArr, speed = removeDuplicateAndSort(wspd_dateTimeArr, speed)

        #print('removing duplicate wdir')
        wdir_dateTimeArr, wdir = removeDuplicateAndSort(wdir_dateTimeArr, wdir)

        uwnd_dateTimeArr, uwnd = removeNan(uwnd_dateTimeArr, uwnd)
        vwnd_dateTimeArr, vwnd = removeNan(vwnd_dateTimeArr, vwnd)

        #print('removing duplicate uwind')
        uwnd_dateTimeArr, uwnd = removeDuplicateAndSort(uwnd_dateTimeArr, uwnd)

        #print('removing duplicate vwind')
        vwnd_dateTimeArr, vwnd = removeDuplicateAndSort(vwnd_dateTimeArr, vwnd)


        cdfTime = ds_RH.variables['TIME']
        timeUnit = cdfTime.units
        timeArr = np.array(cdfTime)
        cftimes=num2date(timeArr, timeUnit)
        rh_dateTimeArr = [datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) for dtm in cftimes]

        rh_dateTimeArr, rh = removeNan(rh_dateTimeArr, rh)

        #print('removing duplicate rh')
        rh_dateTimeArr, rh = removeDuplicateAndSort(rh_dateTimeArr, rh)

        cdfTime = ds_SST.variables['TIME']
        timeUnit = cdfTime.units
        timeArr = np.array(cdfTime)
        cftimes=num2date(timeArr, timeUnit)
        sst_dateTimeArr = [datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) for dtm in cftimes]
        
        sst_dateTimeArr, sst = removeNan(sst_dateTimeArr, sst)

        #print('removing duplicate sst')
        sst_dateTimeArr, sst = removeDuplicateAndSort(sst_dateTimeArr, sst)


        #############################3

        cdfTime = ds_RAIN.variables['TIME']
        timeUnit = cdfTime.units
        timeArr = np.array(cdfTime)
        cftimes=num2date(timeArr, timeUnit)
        rain_dateTimeArr = [datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) for dtm in cftimes]
        
        rain_dateTimeArr, rain = removeNan(rain_dateTimeArr, rain)

        #print('removing duplicate sst')
        rain_dateTimeArr, rain = removeDuplicateAndSort(rain_dateTimeArr, rain)

        ################################


        cdfTime = ds_AIRT.variables['TIME']
        timeUnit = cdfTime.units
        timeArr = np.array(cdfTime)
        cftimes=num2date(timeArr, timeUnit)
        airt_dateTimeArr = [datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) for dtm in cftimes]

        airt_dateTimeArr, airt = removeNan(airt_dateTimeArr, airt)

        #print('removing duplicate airt')
        airt_dateTimeArr, airt = removeDuplicateAndSort(airt_dateTimeArr, airt)


        timeList = wind_dateTimeArr +rh_dateTimeArr +sst_dateTimeArr +airt_dateTimeArr + rain_dateTimeArr
        timeList.sort()
        timeList = list(set(timeList))
        timeList.sort()
        nanArr = np.ones(len(timeList), dtype=float) * float('nan')
        

        #print(f'length nanTimelist {len(timeList)}, length windtimelist {len(wspd_dateTimeArr)}')
        
        wdir_dateTimeArr, timeList, wdir, nanArr = fillData(wdir_dateTimeArr, timeList, wdir, nanArr)
        wspd_dateTimeArr, timeList, speed, nanArr = fillData(wspd_dateTimeArr, timeList, speed, nanArr)

        uwnd_dateTimeArr, timeList, uwnd, nanArr = fillData(uwnd_dateTimeArr, timeList, uwnd, nanArr)
        vwnd_dateTimeArr, timeList, vwnd, nanArr = fillData(vwnd_dateTimeArr, timeList, vwnd, nanArr)

        rh_dateTimeArr, timeList, rh, nanArr = fillData(rh_dateTimeArr, timeList, rh, nanArr)
        sst_dateTimeArr, timeList, sst, nanArr = fillData(sst_dateTimeArr, timeList, sst, nanArr)
        airt_dateTimeArr, timeList, airt, nanArr = fillData(airt_dateTimeArr, timeList, airt, nanArr)

        rain_dateTimeArr, timeList, rain, nanArr = fillData(rain_dateTimeArr, timeList, rain, nanArr)

        buoy =f'{lat:02d}{latUnits}_{lon:03d}{lonUnits}'
        
        speed, airt, rh, sst = fillMask(buoy, speed, airt, rh, sst)
        try:
            coareOutPutArr = coare35vn(speed, airt, rh, sst, zu=4.0, zt = 3, zq = 3)

            speed2 = np.sqrt(uwnd**2 + vwnd**2)
            
            coareOutPutArr2 = coare35vn(speed2, airt, rh, sst, zu=4.0, zt = 3, zq = 3)
        except:
            print('ERROR ',bWinds)
        
        U10N = coareOutPutArr[0,:]
        u10 = coareOutPutArr[1,:]

        U10N2 = coareOutPutArr2[0,:]
        u102 = coareOutPutArr2[1,:]

        

        wdir[wdir < 0] += 360

        wdir2 = np.arctan2(vwnd, uwnd)
        wdir2[wdir2 < 0] += 2 * np.pi 

        U10N2_x = U10N2 * np.cos(wdir2)
        U10N2_y = U10N2 * np.sin(wdir2)

        wds = Dataset(writeFname, 'w', format='NETCDF4')

        wds.createDimension('time', None)
        wds_time = wds.createVariable('time', float, ('time'))
        wds_time.units = timeUnit
        wds_time[:] = date2num(airt_dateTimeArr, timeUnit)

        wds_U10N = wds.createVariable('U10N', float, ('time',))
        wds_U10N.units = 'm/sec'
        wds_U10N.long_name = 'neutral winds at 10 m from COARE3.5'
        wds_U10N[:] = U10N[:]

        wds_U10 = wds.createVariable('U10', float, ('time',))
        wds_U10.units = 'm/sec'
        wds_U10.long_name = 'winds at 10 m from COARE3.5'
        wds_U10[:] = u10[:]

        wds_U10_dir= wds.createVariable('U10_direction', float, ('time',))
        wds_U10_dir.units = 'degrees_true'
        wds_U10_dir.long_name = 'winds direction'
        wds_U10_dir[:] = wdir[:]


        wds_U10N2 = wds.createVariable('U10N2', float, ('time',))
        wds_U10N2.units = 'm/sec'
        wds_U10N2.long_name = 'neutral winds at 10 m from COARE3.5 using vwnd and uwnd'
        wds_U10N2[:] = U10N2[:]

        wds_U102 = wds.createVariable('U102', float, ('time',))
        wds_U102.units = 'm/sec'
        wds_U102.long_name = 'winds at 10 m from COARE3.5 using vwnd and uwnd'
        wds_U102[:] = u102[:]

        wds_U102_dir= wds.createVariable('U10_direction2', float, ('time',))
        wds_U102_dir.units = 'degrees with respect to zonal direction'
        wds_U102_dir.long_name = 'winds direction using vwnd and uwnd'
        wds_U102_dir[:] = np.rad2deg(wdir2[:])

        wds_U10N2_x= wds.createVariable('U10N2_x', float, ('time',))
        wds_U10N2_x.units = 'zonal neutral wind'
        wds_U10N2_x.long_name = 'winds direction using vwnd and uwnd'
        wds_U10N2_x[:] = U10N2_x[:]

        wds_U10N2_y= wds.createVariable('U10N2_y', float, ('time',))
        wds_U10N2_y.units = 'meridional neutral wind'
        wds_U10N2_y.long_name = 'winds direction using vwnd and uwnd'
        wds_U10N2_y[:] = U10N2_y[:]


        wds_SST = wds.createVariable('SST', float, ('time',))
        wds_SST.units = 'degree_Celsius'
        wds_SST.long_name = 'Sea Surface Temperature'
        wds_SST[:] = sst[:]


        wds_RH = wds.createVariable('RH', float, ('time',))
        wds_RH.units = 'percent'
        wds_RH.long_name = 'Relative Humidity'
        wds_RH[:] = rh[:]


        wds_RAIN = wds.createVariable('RAIN', float, ('time',))
        wds_RAIN.units = 'mm/hr'
        wds_RAIN.long_name = 'rain'
        wds_RAIN[:] = rain[:]


        wds_AIRT = wds.createVariable('AIRT', float, ('time',))
        wds_AIRT.units = 'degree_Celsius'
        wds_AIRT.long_name = 'Bulk Air Temperature'
        wds_AIRT[:] = airt[:]

        wds.close()
