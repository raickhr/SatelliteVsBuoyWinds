import sys
sys.path.append("../COARE3p5/COAREalgorithm/Python/COARE3p5")
from coare35vn import *
import pandas as pd
from mpi4py import MPI


from netCDF4 import Dataset, num2date, date2num
from datetime import datetime, timedelta, date
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

def getCorr(x,y):
    mask = np.isnan(x)
    mask += np.isnan(y)
    r = np.corrcoef(x[~mask],y[~mask])[0][1]
    return r

def printFig(lat, lon):
    
    if lat < 0:
        latUnits = 'S'
    else:
        latUnits = 'N'

    if lon < 0:
        lonUnits = 'W'
    else:
        lonUnits = 'E'

    LAT = lat
    LON = lon
    lat=abs(lat)
    lon=abs(lon)

    #returnText = f'2000 to 2007 | {lat:02d}{latUnits} | {lon:03d}{lonUnits} | '
    

    returnText = f'2000 to 2007 | {LAT:3d} | {LON:4d} | '
    matchFileName = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_matchupNearestFour_withWave.nc'
    matchFileName2 = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_matchupNearestFour_2000.nc'
    if not os.path.isfile(matchFileName):
        return
        #print(matchFileName, 'not found')
    else:
        ds2 = Dataset(matchFileName2)
        ds = Dataset(matchFileName)
        #print(ds.variables.keys())
        wspdB = np.array(ds2.variables['U10N_TAO'])
        
        wspdBwithWave = np.array(ds.variables['U10N_TAO2'])
        
        nData1 = len(wspdB)
        nData2 = len(wspdBwithWave)

        returnText = f'{LAT:3d} | {LON:4d} | {nData1:d} | {nData2:d} | {nData2/nData1 * 100:6.2f} '
        
        print(returnText)
        return returnText



def main():

    #print('nprocs = ', nprocs)
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

    if rank == 0:
        returnText = 'LAT | LON | num. matchup | num. matchup with wave | percent of data with wave'
        print(returnText)

    comm.Barrier()

    dataInfo = []
    for task in taskListInMe:
        lat = taskList[task][0]
        lon = taskList[task][1]
        
        dataInfo.append(printFig(lat, lon))

    # dataInfo= np.array(dataInfo)

    # allDataInfo = comm.gather(dataInfo, root = 0)
    # print(allDataInfo[0])


if __name__ == '__main__':
    main()


