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
    if not os.path.isfile(matchFileName):
        return
        #print(matchFileName, 'not found')
    else:
        ds = Dataset(matchFileName)
        #print(ds.variables.keys())
        wspdB = np.array(ds.variables['U10N_TAO'])
        wdirB = np.array(ds.variables['U10N_dir_TAO'])

        wspdBwithWave = np.array(ds.variables['U10N_TAO2'])
        wdirBwithWave = np.array(ds.variables['U10N_dir_TAO2'])

        #wdirB[wdirB>(180+90)]
        wdirB = (450-wdirB)%360
        wdirBwithWave = wdirBwithWave%360

        wspdS = np.array(ds.variables['U10N_QS'][0,:])
        wdirS = np.array(ds.variables['U10N_dir_QS'][0,:])
        wdirS = (450-wdirS)%360

        timeDiff = np.array(ds.variables['satTimeDiff'][0,:])/60
        dist = np.array(ds.variables['dist'][0,:])

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

        wspdS[mask] = float('nan')
        wdirS[mask] = float('nan')

        wspdB[mask] = float('nan')
        wdirB[mask] = float('nan')

        timeDiff[mask] = float('nan')
        dist[mask] = float('nan')

        ds.close()

        fig, axes = plt.subplots(nrows=4, ncols =1, figsize=(20,14))
        x = bTime
        #plt.plot(wind_dateTimeArr1[st1:ed1], angleOrig[st1:ed1], label = 'buoy')


        ###################################################################
        text = 'Actual Data: Wind speed ,'

        ax = axes[0]
        ax.plot(x, wspdS, alpha = 0.5, label = 'QS')
        ax.plot(x, wspdB, alpha = 0.5, label = 'buoy')
        ax.plot(x, wspdBwithWave, alpha = 0.5, label = 'buoyWithWave')
        ax.legend()

        r = getCorr(wspdS, wspdB)
        r2 = getCorr(wspdS, wspdBwithWave)

        text += f'corr = {r:5.3f}, corr with wave =  {r2:5.3f}'

        ax.text(0.5, 0.9, text, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
        ax.grid(visible=True, which='both', axis='both',alpha = 0.5)

        ###################################################################
        text = 'Diff. in Data'
        ax = axes[1]

        diff = wspdS-wspdB
        diff2 = wspdS-wspdBwithWave

        mean = np.nanmean(diff)
        std = np.nanstd(diff)

        mean2 = np.nanmean(diff2)
        std2 = np.nanstd(diff2)

        tstat = f'mean = {mean:.2f},  mean with wave = {mean2:.2f} \n std. dev = {std:.2f},  std. dev with wave = {std2:.2f}'

        ax.text(0.2, 0.7, tstat, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
        

        returnText += f' {mean:.2f} | {std:.2f} |{r:.2f} | {mean2:.2f} | {std2:.2f} |{r2:.2f} |'

        ax.plot(x, diff, label = 'QS-buoy')
        ax.plot(x, diff2, label = 'QS-buoyWithWave')
        ax.legend()
        ax.text(0.5, 0.9, text, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
        ax.grid(visible=True, which='both', axis='both',alpha = 0.5)


        ###################################################################

        text = 'Actual Data : Direction ,'
        ax = axes[2]
        ax.plot(x, wdirS, alpha= 0.5, label = 'QS')
        ax.plot(x, wdirB, alpha= 0.5, label = 'buoy')
        ax.plot(x, wdirBwithWave, alpha= 0.5, label = 'buoyWithWave')
        ax.legend()

        r = getCorr(wdirS, wdirB)
        r2 = getCorr(wdirS, wdirBwithWave)

        text += f'corr = {r:5.3f}'

        ax.text(0.5, 0.9, text, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
        ax.grid(visible=True, which='both', axis='both',alpha = 0.5)

        ###################################################################
        ax = axes[3]
        text = 'Diff. in Data : Direction'
        diff = (wdirS-wdirB)%360
        diff[diff > 180] -= 360 
        diff[diff < -180] += 360 

        diff2 = (wdirS-wdirBwithWave)
        diff2[diff2 > 180] -= 360 
        diff2[diff2 < -180] += 360 

        mean = np.nanmean(diff)
        std = np.nanstd(diff)

        mean2 = np.nanmean(diff2)
        std2 = np.nanstd(diff2)

        tstat = f'mean = {mean:.2f},  mean with wave = {mean2:.2f} \n std. dev = {std:.2f},  std. dev with wave = {std2:.2f}'

    
        ax.text(0.2, 0.7, tstat, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
        
        returnText += f' {mean:.2f} |{std:.2f} |{r:.2f} | {mean2:.2f} |{std2:.2f} |{r2:.2f} '


        ax.plot(x, diff, label = 'QS-buoy')
        ax.plot(x, diff2, label = 'QS-buoyWithWave')
        ax.legend()
        ax.text(0.5, 0.9, text, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
        ax.grid(visible=True, which='both', axis='both',alpha = 0.5)

        ###################################################################

        # ###################################################################
        # text = 'Spatial Distance between QS data and buoy Pos'

        # ax = axes[4]
        # ax.plot(x, dist)
        # ax.set_ylabel('KM')
        # ax.set_ylim(-2,15)
        # ax.text(0.5, 0.9, text, horizontalalignment='center',
        #     verticalalignment='center', transform=ax.transAxes)
        # ax.grid(visible=True, which='both', axis='both',alpha = 0.5)

        # ###################################################################
        # text = 'Difference in time between QS measuremnt and buoy measurement'

        # ax = axes[5]
        # ax.plot(x, timeDiff)
        # ax.set_ylabel('minutes')
        # ax.set_ylim(-15,15)
        # ax.text(0.5, 0.9, text, horizontalalignment='center',
        #     verticalalignment='center', transform=ax.transAxes)
        # ax.grid(visible=True, which='both', axis='both',alpha = 0.5)

        # ###################################################################


        text = f'{lat:d}{latUnits} {lon:d}{lonUnits} 10min-matchup buoy Neutral Winds with QS data 2000'
        fig.suptitle(text, y = 0.9)
        plt.subplots_adjust(hspace=0.5)

        fname = f'images/withWave/{lat:02d}{latUnits}_{lon:03d}{lonUnits}_Comp_2000to2007withWave.png'
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

    if rank == 0:
        returnText = 'data Range | LAT | LON | mean wspd diff | std. wspd diff | corr. wspd | mean wspd diff(wave)| std. wspd diff(wave)| corr. wspd(wave)| '
        returnText  += 'mean wdir diff | std. wdir diff | corr. wdir | mean wdir diff(wave)| std. wdir diff(wave)| corr. wdir(wave)'
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


