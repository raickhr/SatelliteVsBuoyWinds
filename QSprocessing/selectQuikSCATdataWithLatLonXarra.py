import xarray as xr
import numpy as np
from numpy import sin, cos
#import matplotlib.pyplot as plt
import sys
from mpi4py import MPI
import glob
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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

latList = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
lonList = [-95, -110, -125, -140, -155, -170, -180, 165]


readDir = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/QS_data/qs_l2b_v4p1/'
fileList = glob.glob(readDir + '/qs_l2b_?????_v4.1_*.nc')
writeDir = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/QS_data/'
tol = 0.1


## dividing work 
startIndices = np.zeros((size), dtype=int)
endIndices = np.zeros((size), dtype=int)

if rank == 0:
    ## remove previously written files
        
    for fileName in glob.glob(writeDir + 'T_????_????_QS_*.nc'):
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
    ds = ds.set_coords('time')
    ds = ds.swap_dims({'along_track':'time'})
    
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
    
            wFile = f'T_{abs(thisLat):03.0f}{latUnit:s}_{abs(thisLon):03.0f}{lonUnit:s}_QS_fileNumber{count:06d}_rank{rank:02d}.nc'
            thisLon = (thisLon + 360)%360

            mask = np.logical_and((abs(ds['lat'] - thisLat) < tol), (abs(ds['lon'] - thisLon) < tol))
            if np.sum(mask) > 0:
                sub_ds = ds.where(mask, drop=True)
                sub_ds = sub_ds.reset_coords(['lat', 'lon'])
                sub_ds = sub_ds.rename_vars({'lat' : 'QS_LAT',
                                             'lon' : 'QS_LON'})
                
                QS_lat = sub_ds['QS_LAT'].to_numpy()
                QS_lon = sub_ds['QS_LON'].to_numpy()

                dist_inKM = getGreatCircDist(thisLat, thisLon, QS_lat, QS_lon)
                sub_ds['dist_from_TAO_pos'] = xr.DataArray(dist_inKM, dims = ['time', 'cross_track'],
                                                       attrs = {'units' : 'kilometers',
                                                                'long_name' : 'great circle distance from TAO position'})

                tlen = len(sub_ds['time'])
                wds = xr.Dataset()
                
                #### ALL THIS LENGTHY PROCESS JUST TO MAKE THE ARRAY SIZE IS SAME EXCEPT FOR THE APPENDING DIMENSION
                for var in list(sub_ds.keys()):
                    if var not in ['ambiguity_speed', 'ambiguity_direction', 'ambiguity_obj']:
                        newArr = np.zeros((tlen, 3)) * np.nan
                        for t in range(tlen):
                            for i in range(min(len(sub_ds[var][t,:]), 3)):
                                newArr[t,i] = sub_ds[var][t,i]
                        wds[var] = xr.DataArray(newArr, 
                                                dims = ['time', 'cross_track'],
                                                coords = {'time':sub_ds['time'],
                                                          'cross_track':np.arange(3)},
                                                attrs = sub_ds[var].attrs)
                    else:
                        newArr = np.zeros((tlen, 3, 4)) * np.nan
                        for t in range(tlen):
                            for i in range(min(len(sub_ds[var][t,:]), 3)):
                                newArr[t,i,:] = sub_ds[var][t,i,:]
                        wds[var] = xr.DataArray(newArr, 
                                                dims = ['time', 'cross_track', 'ambiguities'],
                                                coords = {'time':sub_ds['time'],
                                                          'cross_track':np.arange(3),
                                                          'ambiguities':np.arange(4)},
                                                attrs = sub_ds[var].attrs)
                
                ## NOW WRITE
                wds.to_netcdf(writeDir + wFile, unlimited_dims='time')
                # if os.path.exists(writeDir + wFile):
                #     rds = xr.open_dataset(writeDir + wFile)
                #     wds = xr.concat((rds, wds), dim='time')
                #     os.remove(writeDir + wFile)
                #     wds.to_netcdf(writeDir + wFile, unlimited_dims='time')
                # else:
                #     wds.to_netcdf(writeDir + wFile, unlimited_dims='time')
                # sub_ds.close()
    
    ds.close()
    count += 1
