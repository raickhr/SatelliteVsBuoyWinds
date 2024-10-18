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
    matchFileName = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_matchup_2000.nc'
    if not os.path.isfile(matchFileName):
        return
        #print(matchFileName, 'not found')
    else:
        ds = Dataset(matchFileName)
        #print(ds.variables.keys())
        wspdB = np.array(ds.variables['U10N_TAO'])
        wdirB = np.array(ds.variables['U10N_dir_TAO'])
        #wdirB[wdirB>(180+90)]

        wspdS = np.array(ds.variables['U10N_QS'])
        wdirS = np.array(ds.variables['U10N_dir_QS'])

        bTime = np.array(ds.variables['time_TAO'])
        bTimeUnits = ds.variables['time_TAO'].units
        cftimes = num2date(bTime, bTimeUnits)
        bTime = np.array([datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) for dtm in cftimes])


        sTime = np.array(ds.variables['time_QS'])
        sTimeUnits = ds.variables['time_QS'].units
        cftimes = num2date(sTime, sTimeUnits)
        sTime = np.array([datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) for dtm in cftimes])

        mask = np.isnan(wspdB)
        mask += np.isnan(wdirB)
        mask += np.isnan(wspdS)
        mask += np.isnan(wdirB)


        mask = np.logical_or(mask, abs(wspdS) > 1000)
        mask = np.logical_or(mask, abs(wdirS) > 1000)
        mask = np.logical_or(mask, abs(wspdB) > 1000)
        mask = np.logical_or(mask, abs(wdirB) > 1000)

        wspdS[mask] = float('nan')
        wdirS[mask] = float('nan')

        wspdB[mask] = float('nan')
        wdirB[mask] = float('nan')

        ds.close()

        fig, axes = plt.subplots(nrows=6, ncols =1, figsize=(20,16))
        x = sTime
        #plt.plot(wind_dateTimeArr1[st1:ed1], angleOrig[st1:ed1], label = 'buoy')


        ###################################################################
        text = 'Actual Data'

        ax = axes[0]
        ax.plot(x, wspdS, label = 'QS')
        ax.plot(x, wspdB, label = 'buoy')
        ax.legend()
        ax.text(0.5, 0.9, text, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
        ax.grid(visible=True, which='both', axis='both',alpha = 0.5)

        ###################################################################
        text = 'Diff. in Data'
        ax = axes[1]

        diff = wspdS-wspdB

        mean = np.nanmean(diff)
        std = np.nanstd(diff)
        tstat = f'mean = {mean:.2f} \n std. dev = {std:.2f}'
        ax.text(0.2, 0.7, tstat, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)

        returnText += f'| {mean:.2f} | {std:.2f}'

        ax.plot(x, diff, label = 'QS-buoy')
        ax.legend()
        ax.text(0.5, 0.9, text, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
        ax.grid(visible=True, which='both', axis='both',alpha = 0.5)

        ###################################################################
        text = 'Ratio of Data'

        deno = wspdB.copy()
        deno[abs(deno)<1e-5] = float('nan')

        r = wspdS/deno
        ax = axes[2]
        ax.plot(x, r, label = 'QS/buoy')
        ax.legend()
        ax.text(0.5, 0.9, text, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
        ax.grid(visible=True, which='both', axis='both',alpha = 0.5)

        ###################################################################

        text = 'Actual Data : Direction'
        ax = axes[3]
        ax.plot(x, wdirS, label = 'QS')
        ax.plot(x, wdirB, label = 'buoy')
        ax.legend()
        ax.text(0.5, 0.9, text, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
        ax.grid(visible=True, which='both', axis='both',alpha = 0.5)

        ###################################################################
        ax = axes[4]
        text = 'Diff. in Data : Direction'
        diff = (wdirS-wdirB)%360
        diff[diff > 180] -= 360 
        diff[diff < -180] += 360 

        mean = np.nanmean(diff)
        std = np.nanstd(diff)
        tstat = f'mean = {mean:.2f} \n std. dev = {std:.2f}'
        ax.text(0.2, 0.7, tstat, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
        
        returnText += f'| {mean:.2f} |{std:.2f} '


        ax.plot(x, diff, label = 'QS-buoy')
        ax.legend()
        ax.text(0.5, 0.9, text, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
        ax.grid(visible=True, which='both', axis='both',alpha = 0.5)

        ###################################################################

        text = 'Ratio of Data : Direction'

        deno = wdirB.copy()
        deno[abs(deno)<1e-5] = float('nan')

        r = wdirS/deno
        ax = axes[5]
        ax.plot(x, r, label = 'QS/buoy')
        ax.legend()
        ax.text(0.5, 0.9, text, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
        ax.grid(visible=True, which='both', axis='both',alpha = 0.5)

        ###################################################################

        text = f'{lat:d}{latUnits} {lon:d}{lonUnits} 5min-matchup buoy Neutral Winds with QS data 2000'
        fig.suptitle(text, y = 0.9)
        plt.subplots_adjust(hspace=0.5)

        fname = f'images/{lat:02d}{latUnits}_{lon:03d}{lonUnits}_Comp_2000.png'
        plt.savefig(fname,dpi=70)
        plt.close()
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

    dataInfo = []
    for task in taskListInMe:
        lat = taskList[task][0]
        lon = taskList[task][1]
        #print(lat, lon)
        dataInfo.append(printFig(lat, lon))

    # dataInfo= np.array(dataInfo)

    # allDataInfo = comm.gather(dataInfo, root = 0)
    # print(allDataInfo[0])


if __name__ == '__main__':
    main()


