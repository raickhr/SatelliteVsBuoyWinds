import sys
sys.path.append("../COARE3p5/COAREalgorithm/Python/COARE3p5")
from coare35vn import *
import pandas as pd
from scipy import stats
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

def plotRegSlopeInterceptAndRsq(x,y,ax, xlabel, ylabel, cmapName='jet', nbins=200, fsize=20):
    res = stats.linregress(x, y)
    # cmap=plt.get_cmap(cmapName).copy()
    # h = ax.hist2d(x, y, cmap=cmap, bins=nbins)
    # plt.colorbar(h[3],ax=ax)
    ax.scatter(x,y,s=5)
    xmin = max(np.min(x), np.min(y))
    xmax = max(np.max(x), np.max(y))
    dx = (xmax - xmin)/1000
    x = np.arange(xmin, xmax, dx)
    ax.plot(x, res.intercept + res.slope*x, 'r', label='fitted line')

    ax.plot(x, x, 'green', label='ideal line')
    txt = ax.text(0.5, 0.85, f'slope = {res.slope:3.1f}, rsq = {res.rvalue**2:3.2f}', 
                  transform=ax.transAxes, color='white', verticalalignment='top', 
                  horizontalalignment='center', fontsize =fsize,
                  bbox=dict(facecolor='black', alpha=1))
    
    ax.legend(fontsize=fsize)
    ax.set_xlabel(xlabel, fontsize = fsize)
    ax.set_ylabel(ylabel, fontsize = fsize)


def plotScatter(x,y,ax, xlabel, ylabel, fsize=20):
    ax.scatter(x,y,s=5)
    ax.set_xlabel(xlabel, fontsize = fsize)
    ax.set_ylabel(ylabel, fontsize = fsize)




def printFig(lat, lon):
    
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

    returnText = f'{lat:02d}{latUnits} | {lon:03d}{lonUnits} | 2000 '
    matchFileName = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_matchupNearestFour_2000.nc'
    if not os.path.isfile(matchFileName):
        return
        #print(matchFileName, 'not found')
    else:
        ds = Dataset(matchFileName)
        #print(ds.variables.keys())
        wspdB = np.array(ds.variables['U10N_TAO'])
        wdirB = np.array(ds.variables['U10N_dir_TAO'])
        wdirB = 360+90 - wdirB
        wdirB = wdirB%360
        
        #wdirB[wdirB>(180+90)]

        wspdS = np.array(ds.variables['U10N_QS'][0,:])
        wdirS = np.array(ds.variables['U10N_dir_QS'][0,:])
        wdirS = 360+90 - wdirS
        wdirS = wdirS%360

        bTime = np.array(ds.variables['time'])
        bTimeUnits = ds.variables['time'].units
        cftimes = num2date(bTime, bTimeUnits)
        bTime = np.array([datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) for dtm in cftimes])


        mask = np.isnan(wspdB)
        mask += np.isnan(wdirB)
        mask += np.isnan(wspdS)
        mask += np.isnan(wdirB)


        mask += np.logical_or(mask, abs(wspdS) > 1000)
        mask += np.logical_or(mask, abs(wdirS) > 1000)
        mask += np.logical_or(mask, abs(wspdB) > 1000)
        mask += np.logical_or(mask, abs(wdirB) > 1000)

        wspdS= wspdS[~mask] 
        wdirS= wdirS[~mask] 

        wspdB= wspdB[~mask] 
        wdirB= wdirB[~mask] 

        angleB = (450-wdirB)%360
        uwndB = wspdB * np.cos(np.deg2rad(angleB))
        vwndB = wspdB * np.sin(np.deg2rad(angleB))
        
        angleS = (450-wdirS)%360
        uwndS = wspdS * np.cos(np.deg2rad(angleS))
        vwndS = wspdS * np.sin(np.deg2rad(angleS))
        
        
        ds.close()

        fig, axes = plt.subplots(nrows=2, ncols =4, figsize=(40,16))
        

        ###################################################################
        x = wspdS
        xlabel = 'QS wind speed'

        y = wspdB
        ylabel = 'buoy wind speed'
        plotRegSlopeInterceptAndRsq(x,y,axes[0, 0], xlabel, ylabel)

        ###################################################################
        
        x = wdirS
        xlabel = 'QS wind direction'

        y = wdirB
        ylabel = 'buoy wind direction'
        plotRegSlopeInterceptAndRsq(x,y,axes[0, 1], xlabel, ylabel)

        ###################################################################

        x = uwndS
        xlabel = 'QS zonal wind speed'

        y = uwndB
        ylabel = 'buoy zonal wind speed'
        plotRegSlopeInterceptAndRsq(x,y,axes[0, 2], xlabel, ylabel)

        ###################################################################
        
        x = vwndS
        xlabel = 'QS meridional wind speed'

        y = vwndB
        ylabel = 'buoy meriodional wind speed'
        plotRegSlopeInterceptAndRsq(x,y,axes[0, 3], xlabel, ylabel)

        ###################################################################

        ###################################################################
        x = uwndS
        xlabel = 'QS zonal wind speed'

        y = wspdS - wspdB
        ylabel = 'QS wind speed  - buoy wind speed'
        plotScatter(x,y,axes[1,0], xlabel, ylabel)

        ###################################################################
        
        x = vwndS
        xlabel = 'QS meridional wind direction'

        y = wspdS - wspdB
        ylabel = 'QS wind speed  - buoy wind speed'
        plotScatter(x,y,axes[1,1], xlabel, ylabel)

        ###################################################################

        x = uwndS
        xlabel = 'QS zonal wind speed'

        y = wdirB - wdirS
        ylabel = 'buoy wind direction - QS wind direction'
        plotScatter(x,y,axes[1,2], xlabel, ylabel)

        ###################################################################
        
        x = vwndS
        xlabel = 'QS meridional wind speed'

        y = wdirB - wdirS
        ylabel = 'buoy wind direction - QS wind direction'
        plotScatter(x,y,axes[1,3], xlabel, ylabel)

        ###################################################################

        text = f'{lat:d}{latUnits} {lon:d}{lonUnits} 10min-matchup buoy Neutral Winds with QS data 2000'
        fig.suptitle(text, y = 0.9)
        plt.subplots_adjust(hspace=0.05, wspace=0.05)

        fname = f'images/Regression/{lat:02d}{latUnits}_{lon:03d}{lonUnits}_Regression_2000to2007.png'
        plt.savefig(fname,dpi=70)
        plt.close()



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

    for task in taskListInMe:
        lat = taskList[task][0]
        lon = taskList[task][1]
        #print(lat, lon)
        printFig(lat, lon)

    # dataInfo= np.array(dataInfo)

    # allDataInfo = comm.gather(dataInfo, root = 0)
    # print(allDataInfo[0])


if __name__ == '__main__':
    main()


