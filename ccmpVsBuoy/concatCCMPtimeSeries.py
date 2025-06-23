import xarray as xr
import numpy as np
from numpy import sin, cos
import gc
import sys
from mpi4py import MPI
import glob
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

latList = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
lonList = [-95, -110, -125, -140, -155, -170, -180, 165]

ylen = len(latList)
xlen = len(lonList)

taskList = []

for latId  in range(ylen):
    for lonId in range(xlen):
        taskList.append([latList[latId], lonList[lonId]])

ntasks = len(taskList)

def writeTimeSeriesAtPosForAllTime(lon, lat):
    years = np.arange(2000, 2008)
    months = np.arange(1,13)
    listDS = []
    for year in years:
        for month in months:

            writeDir = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/CCMP/atBuoyLocs/'
            if lat < 0:
                rfname = f'CCMP_tseries_{lon:03d}_{abs(lat):02d}S_{year:04d}_{month:02d}.nc'
                wfname = f'CCMP_tseries_{lon:03d}_{abs(lat):02d}S.nc'
            else:
                rfname = f'CCMP_tseries_{lon:03d}_{abs(lat):02d}N_{year:04d}_{month:02d}.nc'
                wfname = f'CCMP_tseries_{lon:03d}_{abs(lat):02d}N.nc'

            if os.path.isfile(writeDir + wfname):
                continue
            
            else:
                if os.path.isfile(writeDir + rfname):
                    ds = xr.open_dataset(writeDir + rfname)
                    listDS.append(ds)
                
    wds = xr.merge(listDS)
    wds = wds.sortby('time')
    wds.to_netcdf(writeDir + wfname)
    wds.close()
    ds.close()
    del wds, ds
    gc.collect()




## dividing work 
ntasksInAllProcs = np.zeros((size), dtype=int)

avgTasks = int(ntasks//size)
remainder = ntasks%size

ntasksInAllProcs[:] = avgTasks

ntasksInAllProcs[0:remainder] += 1

endIndices = np.cumsum(ntasksInAllProcs)
startIndices = np.roll(endIndices,1)
startIndices[0] = 0

# if rank == 0:
#     for i in range(size):
#         print(f' {startIndices[i]}   {endIndices[i]}  {ntasksInAllProcs[i]}')


for i in range(startIndices[rank], endIndices[rank]):
    currentTask = taskList[i]
    lon = (currentTask[1] + 360)%360
    lat = currentTask[0]
    print(f'rank (lon, lat) = {lon} {lat}')
    writeTimeSeriesAtPosForAllTime(lon, lat)
    

    
MPI.Finalize()