import xarray as xr
import numpy as np
#import matplotlib.pyplot as plt
import sys
from mpi4py import MPI
import glob
import os
import warnings
warnings.filterwarnings("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


latList = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
lonList = [-95, -110, -125, -140, -155, -170, -180, 165]


readDir = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/ASCAT_data/downloaded/tracks/'
fileList = glob.glob(readDir + '/*.l2.nc')
writeDir = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/ASCAT_data/'
tol = 0.1


## dividing work 
startIndices = np.zeros((size), dtype=int)
endIndices = np.zeros((size), dtype=int)

if rank == 0:
    ## remove previously written files
        
    for fileName in glob.glob(writeDir + 'T_????_????_ASCAT_*.nc'):
        os.remove(fileName)

    
    nfiles = np.zeros((size),dtype=int)
    nfiles[:] = int(len(fileList)//size)
    remainder = len(fileList)%size
    
    for i in range(size):
        if i < remainder:
            nfiles[i] += 1
            
    endIndices = np.cumsum(nfiles)
    startIndices = np.roll(endIndices, 1)
    startIndices[0] = 0

    
startIndex = np.zeros((1), dtype=int)
endIndex = np.zeros((1), dtype=int)

comm.Scatter(startIndices, startIndex, root=0)
comm.Scatter(endIndices, endIndex, root=0)

print(f'at rank {rank} fileList from {startIndex[0]} to {endIndex[0]}')
sys.stdout.flush()


count = 0 
for fileIndex in range(startIndex[0], endIndex[0]):
    fileName = fileList[fileIndex]
    print(f'at rank {rank:3d}: {(fileIndex - startIndex[0])/(endIndex[0] - startIndex[0]) * 100: 6.2f} %')
    sys.stdout.flush()
    ds = xr.open_dataset(fileName)
    timeArr = ds['time'].to_numpy()[:,0]
    ds = ds.drop('time')
    ds['NUMROWS'] = xr.DataArray(timeArr, dims=['NUMROWS'])
    ds = ds.rename({'NUMROWS':'time'})
    
    for thisLat in latList:
        for thisLon in lonList:
            if thisLat < 0:
                latUnit = 'S'
            else:
                latUnit = 'N'
            
            if thisLon < 0:
                lonUnit = 'W'
            else:
                lonUnit = 'E'
    
            wFile = f'T_{abs(thisLat):03.0f}{latUnit:s}_{abs(thisLon):03.0f}{lonUnit:s}_ASCAT_fileNumber{count:06d}_rank{rank:02d}.nc'
            thisLon = (thisLon + 360)%360

            mask = np.logical_and((abs(ds['lat'] - thisLat) < tol), (abs(ds['lon'] - thisLon) < tol))
            if np.sum(mask) > 0:
                sub_ds = ds.where(mask, drop=True)
                tlen = len(sub_ds['time'])
                wds = xr.Dataset()
                
                #### ALL THIS LENGTHY PROCESS JUST TO MAKE THE ARRAY SIZE IS SAME EXCEPT FOR THE APPENDING DIMENSION
                for var in list(sub_ds.keys()):
                    newArr = np.zeros((tlen, 3)) * np.nan
                    for t in range(tlen):
                        for i in range(min(len(sub_ds[var][t,:]), 3)):
                            newArr[t,i] = sub_ds[var][t,i]
                    wds[var] = xr.DataArray(newArr, 
                                            dims = ['time', 'NUMCELLS'],
                                            coords = {'time':sub_ds['time'],
                                                        'NUMCELLS':np.arange(3)},
                                            attrs = sub_ds[var].attrs)
                    
                
                ## NOW WRITE
                wds.to_netcdf(writeDir + wFile, unlimited_dims='time')
                
    ds.close()
    count += 1
