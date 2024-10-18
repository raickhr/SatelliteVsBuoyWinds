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


readDir = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/TRMM_data/downloaded/all_ncfiles/'
fileList = glob.glob(readDir + 'dateFixed*.nc')
writeDir = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/TRMM_data/TRMM_byTAOpos/'
tol = 0.1


## dividing work 
startIndices = np.zeros((size), dtype=int)
endIndices = np.zeros((size), dtype=int)

if rank == 0:
    ## remove previously written files
        
    for fileName in glob.glob(writeDir + 'T_????_????_TRMM_*.nc'):
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
    ds = ds.rename({'Time':'time'})

    ds['Longitude'] = (ds['Longitude'] + 360)%360
    
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
    
            wFile = f'T_{abs(thisLat):03.0f}{latUnit:s}_{abs(thisLon):03.0f}{lonUnit:s}_TRMM_fileNumber{count:06d}_rank{rank:02d}.nc'
            thisLon = (thisLon + 360)%360

            mask = np.logical_and((abs(ds['Latitude'] - thisLat) < tol), (abs(ds['Longitude'] - thisLon) < tol))
            
            
            if np.sum(mask) > 0:
                #print('tlen', len(ds['time']))
                #print('total point', np.sum(mask.to_numpy()))
                sub_ds = ds.where(mask, drop=True)
                #print(sub_ds)
                tlen = len(sub_ds['time'])
                wds = xr.Dataset()
                
                #### ALL THIS LENGTHY PROCESS JUST TO MAKE THE ARRAY SIZE IS SAME EXCEPT FOR THE APPENDING DIMENSION
                for var in list(sub_ds.keys()):
                    if var not in ['rainAve', 'rain']:
                        newArr = np.zeros((tlen, 6)) * np.nan
                        for t in range(tlen):
                            for i in range(min(len(sub_ds[var][t,:]), 6)):
                                newArr[t,i] = sub_ds[var][t,i]
                        wds[var] = xr.DataArray(newArr, 
                                                dims = ['time', 'nray'],
                                                coords = {'time':sub_ds['time'],
                                                          'nray':np.arange(6)},
                                                attrs = sub_ds[var].attrs)
                    elif var == 'rainAve':
                        newArr = np.zeros((tlen, 6, 2)) * np.nan
                        for t in range(tlen):
                            for i in range(min(len(sub_ds[var][t,:]), 6)):
                                newArr[t,i,:] = sub_ds[var][t,i,:]
                        wds[var] = xr.DataArray(newArr, 
                                                dims = ['time', 'nray', 'fakeDim8'],
                                                coords = {'time':sub_ds['time'],
                                                          'nray':np.arange(6),
                                                          'fakeDim8':np.arange(2)},
                                                attrs = sub_ds[var].attrs)
                        
                    elif var == 'rain':
                        newArr = np.zeros((tlen, 6, 80)) * np.nan
                        for t in range(tlen):
                            for i in range(min(len(sub_ds[var][t,:]), 6)):
                                newArr[t,i,:] = sub_ds[var][t,i,:]
                        wds[var] = xr.DataArray(newArr, 
                                                dims = ['time', 'nray', 'ncell1'],
                                                coords = {'time':sub_ds['time'],
                                                          'nray':np.arange(6),
                                                          'ncell1':np.arange(80)},
                                                attrs = sub_ds[var].attrs)
                    
                
                ## NOW WRITE
                wds = wds.sortby('time')
                wds.to_netcdf(writeDir + wFile, unlimited_dims='time')
                
    ds.close()
    count += 1
