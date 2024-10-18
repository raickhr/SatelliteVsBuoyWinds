import sys
from mpi4py import MPI


from netCDF4 import Dataset, num2date, date2num
from datetime import datetime, timedelta, date
import os
from glob import glob
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()


def plotRegSlopeInterceptAndRsq(xDates,y,ax, regLabel = '', colors = ['blue', 'purple']):
    sDate=xDates[0]
    eDate=xDates[-1]
    xUnit = f'seconds since {sDate.year}-{sDate.month:02d}-{sDate.day:02d}-{sDate.hour:02d}:{sDate.minute:02d}:{sDate.second:02d}'
    seconds = date2num(xDates,xUnit)
    
    ax.scatter(xDates, y, s=1, color=colors[0], alpha=0.25)

    #res = stats.linregress(seconds, y)
    
    res = stats.siegelslopes(y, seconds)

    xs = float(seconds[0])
    xe = float(seconds[-1])

    dx = (xe - xs)/100

    try:
        x=np.arange(xs, xe, dx)
        #print('ok', xs, xe, dx)
    except:
        print(xs, xe, dx)
        print('len xDates', len(xDates))
        sys.exit()

    xDates = num2date(x,xUnit)
    
    ax.plot(xDates, res.intercept + res.slope*x, color=colors[1], linewidth = 3, label='fitted line ' + regLabel)
    return res.slope/3600, res.intercept + res.slope*xs


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
    deployFileName = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_DeploymentDates.nc'

    if not os.path.isfile(matchFileName):
        return
        #print(matchFileName, 'not found')
    else:
        ds = Dataset(matchFileName)

        bTime = np.array(ds.variables['time'])
        bTimeUnits = ds.variables['time'].units
        cftimes = num2date(bTime, bTimeUnits)
        bTime = np.array([datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) for dtm in cftimes])

        ds2 = Dataset(deployFileName)
        startDates = np.array(ds2.variables['startDate'])
        units = ds2.variables['startDate'].units
        cftimes = num2date(startDates, units)
        startDates = np.array([datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) for dtm in cftimes])

        endDates = np.array(ds2.variables['endDate'])
        units = ds2.variables['endDate'].units
        cftimes = num2date(endDates, units)
        endDates = np.array([datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) for dtm in cftimes])


        #print(ds.variables.keys())
        wspdB = np.array(ds.variables['U10N_TAO'])
        wdirB = np.array(ds.variables['U10N_dir_TAO'])
        

        wspdS = np.array(ds.variables['U10N_QS'][0,:])
        wdirS = np.array(ds.variables['U10N_dir_QS'][0,:])


        mask = np.isnan(wspdB)
        mask += np.isnan(wdirB)
        mask += np.isnan(wspdS)
        mask += np.isnan(wdirB)


        mask += np.logical_or(mask, abs(wspdS) > 1000)
        mask += np.logical_or(mask, abs(wdirS) > 1000)
        mask += np.logical_or(mask, abs(wspdB) > 1000)
        mask += np.logical_or(mask, abs(wdirB) > 1000)

        wspdS = wspdS[~mask] 
        wdirS = wdirS[~mask] 

        wspdB = wspdB[~mask] 
        wdirB = wdirB[~mask] 

        bTime = bTime[~mask]

        angleB = (450-wdirB)%360
        #angleB = np.deg2rad(angleB)
        uwndB = wspdB * np.cos(angleB)
        vwndB = wspdB * np.sin(angleB)
        #Bwind = uwndB + 1j*vwndB

        angleS = (450-wdirS)%360
        #angleS = np.deg2rad(angleS)
        uwndS = wspdS * np.cos(angleS)
        vwndS = wspdS * np.sin(angleS)
        #Swind = uwndS + 1j*vwndS

        r = wspdB-wspdS

        #r= abs(R)

        angleDiff = angleB - angleS
        angleDiff[angleDiff>180] = 360 - angleDiff[angleDiff>180]
        angleDiff[angleDiff<-180] += 360 
        #angleDiff = np.rad2deg(np.arctan2(R.imag, R.real))

        
        ds.close()

        fig, axes = plt.subplots(nrows=2, ncols =1, figsize=(20,5))

        deployInfoFile = open('{lat:02d}{latUnits} | {lon:03d}{lonUnits}.txt')
        
        for i in range(len(startDates)):
            startDate=startDates[i]
            endDate=endDates[i]
            s = np.argmin(abs(bTime - startDate))
            e = np.argmin(abs(bTime - endDate))
            if (e-s)>2:
                x = bTime[s:e].copy()
                y1 = r[s:e].copy()
                y2 = angleDiff[s:e].copy()

                
                wspdDevRate, wspdStartDev = plotRegSlopeInterceptAndRsq(x,y1, axes[0], regLabel = '')
                axes[0].set_ylim(-5,5)
                axes[0].yaxis.grid(visible=True, which='major')
                axes[0].set_ylabel('Buoy - QS wind Speed')

                wdirDevRate, wdirStartDev = plotRegSlopeInterceptAndRsq(x,y2, axes[1], regLabel = '')
                axes[1].set_ylim(-25,25)
                axes[1].yaxis.grid(visible=True, which='major')
                axes[1].set_ylabel('Buoy - QS wind direction')


        ###################################################################

        text = f'{lat:d}{latUnits} {lon:d}{lonUnits} 10min-matchup buoy Neutral Winds with QS data 2000'
        fig.suptitle(text, y = 0.9)
        plt.subplots_adjust(hspace=0.5)

        fname = f'images/SusanLike_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_Comp_2000to2007.png'
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


