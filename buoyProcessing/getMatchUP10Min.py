from netCDF4 import Dataset, num2date, date2num
from datetime import datetime, timedelta
import os
import sys
import pandas as pd
from glob import glob
import numpy as np
from mpi4py import MPI

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

    bFname = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_COARE3p5_2000.nc'
    satFname = glob(f'../../downloads/QS_data/T_{lat}{latUnits}_{lon}{lonUnits}_QS_2000_*.nc')[0]
    writeFname = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_matchup_2000.nc'
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

        buoyWspd = np.array(buoyDS.variables['U10N'])
        buoyTime = np.array(buoyDS.variables['time'])
        buoyTimeUnits = buoyDS.variables['time'].units
        cftimes = num2date(buoyTime, buoyTimeUnits)
        buoyDateTime = np.array([datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) for dtm in cftimes])
        buoyWdir = np.array(buoyDS.variables['U10_direction'])

        ### binning satellite data
        selDateTime = []
        selWspd = []
        selWdir = []

        atEnd = False

        while not atEnd:
            curDate = satDateTime[0]
            mask = abs(satDateTime - curDate) < timedelta(minutes = 10)
            datArr = satDateTime[mask]
            
            selDateTime.append(pd.to_datetime(pd.Series(datArr)).mean().to_pydatetime())
            selWspd.append(np.nanmean(satWspd[mask]))
            selWdir.append(np.nanmean(satWdir[mask]))
            satDateTime = satDateTime[~mask]
            satWspd = satWspd[~mask]
            satWdir = satWdir[~mask]

            if sum(~mask) == 0:
                atEnd = True


        satDateTime = np.array(selDateTime)
        satWspd = np.array(selWspd)
        satWdir = np.array(selWdir)


        ### selecting data when satellite and buoy time matches
        satLen = len(satDateTime)

        selBuoyDate = []
        selBuoyWspd = []
        selBuoyWdir = []

        selQSDate = []
        selQSWspd = []
        selQSWdir = []

        for i in range(satLen):
            mask = abs(buoyDateTime - satDateTime[i]) < timedelta(minutes=10)
            if sum(mask) > 0:
                selBuoyWspd.append(np.nanmean(buoyWspd[mask]))
                selBuoyWdir.append(np.nanmean(buoyWdir[mask]))
                selBuoyDate.append(pd.to_datetime(pd.Series(buoyDateTime[mask])).mean().to_pydatetime())
                
                selQSWspd.append(satWspd[i])
                selQSWdir.append(satWdir[i])
                selQSDate.append(satDateTime[i])

        selBuoyWdir= np.array(selBuoyWdir)%360
        selQSWdir= (-1* np.array(selQSWdir)+360 + 90)%360

        ### writing File

        wds = Dataset(writeFname, 'w', format='NETCDF4')

        wds.createDimension('time', None)
        timeUnits = 'seconds since 1999-01-01 00:00:00'

        ### time ###########

        wds_time = wds.createVariable('time_QS', float, ('time'))
        wds_time.units = timeUnits
        wds_time[:] = date2num(selQSDate, timeUnits)

        wds_time_TAO = wds.createVariable('time_TAO', float, ('time'))
        wds_time_TAO.units = timeUnits
        wds_time_TAO[:] = date2num(selBuoyDate, timeUnits)

        #### QS speed and Direction ####
        wds_U10N_QS = wds.createVariable('U10N_QS', float, ('time',))
        wds_U10N_QS.units = 'm/sec'
        wds_U10N_QS[:] = selQSWspd[:]

        wds_U10_dir_QS= wds.createVariable('U10N_dir_QS', float, ('time',))
        wds_U10_dir_QS.units = 'degrees'
        wds_U10_dir_QS[:] = selQSWdir[:]

        #### Buoy speed and direction ####
        wds_U10N = wds.createVariable('U10N_TAO', float, ('time',))
        wds_U10N.units = 'm/sec'
        wds_U10N[:] = selBuoyWspd[:]

        wds_U10_dir= wds.createVariable('U10N_dir_TAO', float, ('time',))
        wds_U10_dir.units = 'degrees'
        wds_U10_dir[:] = selBuoyWdir[:]

        wds.close()




        
        
        

