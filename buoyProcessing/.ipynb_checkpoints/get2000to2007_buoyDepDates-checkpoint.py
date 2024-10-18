import os
from netCDF4 import Dataset, date2num, num2date
import datetime
import numpy as np
from glob import glob
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

    timeUnit = 'seconds since 1990-01-01 00:00:00'

    folder = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}'

    if os.path.exists(folder):

        fileList = glob(folder + '/*_WIND_10min*.nc')

        wds= Dataset(folder + f'/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_DeploymentDates.nc', 'w', 'NETCDF4')
        wds.createDimension('filenumber', len(fileList))
        
        fname = wds.createVariable('fileName', str, ('filenumber'))
        startDate = wds.createVariable('startDate', float, ('filenumber'))
        startDate.units = timeUnit

        endDate = wds.createVariable('endDate', float, ('filenumber'))
        endDate.units = timeUnit

        sdateList = []
        edateList = []
        fnameList = []
        
        for fileName in fileList:
            ds = Dataset(fileName)
            print(fileName)
            time = ds.variables['TIME']
            tUnits = time.units
            tarr = np.array(time)

            fnameList.append(fileName.rsplit("/")[-1])
            sdate = num2date(tarr[0], tUnits)
            sdateList.append(date2num(sdate, timeUnit))

            edate = num2date(tarr[-1], tUnits)
            edateList.append(date2num(edate, timeUnit))

            ds.close()


        fname[:] = np.array(fnameList, dtype=str)[:]
        startDate[:] = np.array(sdateList)[:]
        endDate[:] = np.array(edateList)[:]

        wds.close()


