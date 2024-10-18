import sys
sys.path.append("../COARE3p5/COAREalgorithm/Python/COARE3p5")
from coare35vn import *
import random
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

def plotRegSlopeInterceptAndRsq(x,y,ax, xlabel, ylabel, regLabel = '', cmapName='jet', color = 'blue', nbins=200, fsize=20):
    res = stats.linregress(x, y)
    # cmap=plt.get_cmap(cmapName).copy()
    # h = ax.hist2d(x, y, cmap=cmap, bins=nbins)
    # plt.colorbar(h[3],ax=ax)
    ax.scatter(x,y,s=5, color=color)
    xmin = min(np.nanmin(x), np.nanmin(y))
    xmax = max(np.nanmax(x), np.nanmax(y))
    dx = (xmax - xmin)/1000
    x = np.arange(xmin, xmax, dx)
    ax.plot(x, res.intercept + res.slope*x, color=color, label='fitted line ' + regLabel)

    ax.plot(x, x, 'green', label='ideal line')

    txt = ax.text(0.5, 0.85, f'slope = {res.slope:3.1f}, rsq = {res.rvalue**2:3.2f}', 
                  transform=ax.transAxes, color='white', verticalalignment='top', 
                  horizontalalignment='center', fontsize =fsize,
                  bbox=dict(facecolor='black', alpha=1))
    
    ax.legend(fontsize=fsize)
    ax.set_xlabel(xlabel, fontsize = fsize)
    ax.set_ylabel(ylabel, fontsize = fsize)


def plotRegSlopeInterceptAndRsqMultiple(xList, yList, ax, xlabel, ylabel, 
                                        regLabels = [''], 
                                        cmapName='jet', 
                                        colorList = [''], 
                                        fsize=20):
    
    if len(colorList) != len(xList):
        hexadecimal_alphabets = '0123456789ABCDEF'
        colorList = ["#" + ''.join([random.choice(hexadecimal_alphabets) for j in
                        range(6)]) for i in range(len(xList))]
    colorList.append('red')
        
    if len(regLabels) != len(xList):
        regLabels = [str(l) for l in range(len(xList))]

    regLabels.append('All points')

    ypos = 0.95
    xAll = np.empty((0), dtype=float)
    yAll = np.empty((0), dtype=float)
    for i in range(len(xList)+1):
        if i == len(xList):
            x = xAll
            y = yAll
        else:
            x = np.array(xList[i])
            y = np.array(yList[i])
            #print(x)
            #print(y)
            xAll = np.concatenate((xAll,x), axis=0)
            yAll = np.concatenate((yAll,y), axis=0)

        color = colorList[i]

        res = stats.linregress(x, y)
        if i < len(xList):
            ax.scatter(x,y,s=5, color=color)
        xmin = max(np.min(x), np.min(y))
        xmax = max(np.max(x), np.max(y))
        dx = (xmax - xmin)/1000
        x = np.arange(xmin, xmax, dx)
        ax.plot(x, res.intercept + res.slope*x, color=color, linewidth = 3, label='fitted line ' + regLabels[i])

        txt = ax.text(0.15, ypos, f'slope = {res.slope:3.1f}, rsq = {res.rvalue**2:3.2f}', 
                    transform=ax.transAxes, color='white', verticalalignment='top', 
                    horizontalalignment='left', fontsize =fsize,
                    bbox=dict(facecolor=color, alpha=1))
        
        ypos -= 0.1

    ax.plot(x, x, 'k', linewidth = 5, linestyle = "--", label='ideal line') 
    ax.legend(fontsize=fsize, loc = 'lower right')
    ax.set_xlabel(xlabel, fontsize = fsize)
    ax.set_ylabel(ylabel, fontsize = fsize)



def plotScatter(x,y,ax, xlabel, ylabel, fsize=20, color = 'blue', alpha=0.5, giveYlim=False, ylim=(-180,180)):
    ax.scatter(x,y,s=5, color=color, alpha=alpha)
    ax.set_xlabel(xlabel, fontsize = fsize)
    ax.set_ylabel(ylabel, fontsize = fsize)
    if giveYlim:
        ax.set_ylim(ylim[0], ylim[1])


def getDataAt(lat, lon):
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

    empty = np.empty((0), dtype=float)
    matchFileName = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_matchupNearestFour_2000.nc'
    if not os.path.isfile(matchFileName):
        return empty, empty, empty, empty
        #print(matchFileName, 'not found')
    else:
        ds = Dataset(matchFileName)
        #print(ds.variables.keys())
        wspdB = np.array(ds.variables['U10N_TAO'])
        wdirB = np.array(ds.variables['U10N_dir_TAO'])
        # wdirB = 360+90 - wdirB
        # wdirB = wdirB%360
        
        #wdirB[wdirB>(180+90)]

        wspdS = np.array(ds.variables['U10N_QS'][0,:])
        wdirS = np.array(ds.variables['U10N_dir_QS'][0,:])
        # wdirS = 360+90 - wdirS
        # wdirS = wdirS%360

        bTime = np.array(ds.variables['time'])
        bTimeUnits = ds.variables['time'].units
        cftimes = num2date(bTime, bTimeUnits)
        bTime = np.array([datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) for dtm in cftimes])

        ds.close()

        mask = np.isnan(wspdB)
        mask += np.isnan(wdirB)
        mask += np.isnan(wspdS)
        mask += np.isnan(wdirB)


        mask += np.logical_or(mask, abs(wspdS) > 1000)
        mask += np.logical_or(mask, abs(wdirS) > 1000)
        mask += np.logical_or(mask, abs(wspdB) > 1000)
        mask += np.logical_or(mask, abs(wdirB) > 1000)

        mask += abs(wspdS-wspdB) > 2

        wspdS= wspdS[~mask] 
        wdirS= wdirS[~mask] 

        wspdB= wspdB[~mask] 
        wdirB= wdirB[~mask]

        return wspdS, wdirS, wspdB, wdirB

def printFig(wspdS, wdirS, wspdB, wdirB):
        angleB = (450-wdirB)%360
        uwndB = wspdB * np.cos(np.deg2rad(angleB))
        vwndB = wspdB * np.sin(np.deg2rad(angleB))
        
        angleS = (450-wdirS)%360
        uwndS = wspdS * np.cos(np.deg2rad(angleS))
        vwndS = wspdS * np.sin(np.deg2rad(angleS))

        Emask =  uwndB > 0
        Wmask =  uwndB < 0 
        

        fig, axes = plt.subplots(nrows=3, ncols =4, figsize=(40,24))
        

        ###################################################################
        x = wspdS.copy()
        xlabel = 'QS wind speed'

        y = wspdB.copy()
        ylabel = 'buoy wind speed'
        #plotRegSlopeInterceptAndRsq(x,y,axes[0, 0], xlabel, ylabel)

        plotRegSlopeInterceptAndRsqMultiple([x[Emask], x[Wmask]], 
                                            [y[Emask], y[Wmask]], 
                                            axes[0,0], xlabel, ylabel, 
                                            colorList = ['blue', 'green'],
                                            regLabels = ['westerlies', 'easterlies'])
        

        ###################################################################
        
        x = wdirS.copy()
        xlabel = 'QS wind direction'

        y = wdirB.copy()
        ylabel = 'buoy wind direction'
        #plotRegSlopeInterceptAndRsq(x,y,axes[0, 1], xlabel, ylabel)
        plotRegSlopeInterceptAndRsqMultiple([x[Emask], x[Wmask]], 
                                            [y[Emask], y[Wmask]], 
                                            axes[0,1], xlabel, ylabel, 
                                            colorList = ['blue', 'green'],
                                            regLabels = ['westerlies', 'easterlies'])

        ###################################################################

        x = uwndS.copy()
        xlabel = 'QS zonal wind speed'

        y = uwndB.copy()
        ylabel = 'buoy zonal wind speed'
        #plotRegSlopeInterceptAndRsq(x,y,axes[0, 2], xlabel, ylabel)
        plotRegSlopeInterceptAndRsqMultiple([x[Emask], x[Wmask]], 
                                            [y[Emask], y[Wmask]], 
                                            axes[0,2], xlabel, ylabel, 
                                            colorList = ['blue', 'green'],
                                            regLabels = ['westerlies', 'easterlies'])

        ###################################################################
        
        x = vwndS.copy()
        xlabel = 'QS meridional wind speed'

        y = vwndB.copy()
        ylabel = 'buoy meriodional wind speed'
        #plotRegSlopeInterceptAndRsq(x,y,axes[0, 3], xlabel, ylabel)
        plotRegSlopeInterceptAndRsqMultiple([x[Emask], x[Wmask]], 
                                            [y[Emask], y[Wmask]], 
                                            axes[0,3], xlabel, ylabel, 
                                            colorList = ['blue', 'green'],
                                            regLabels = ['westerlies', 'easterlies'])

        ###################################################################

        ###################################################################
        x = uwndB.copy()
        xlabel = 'Buoy zonal wind speed'

        y = wspdB.copy() - wspdS.copy()
        ylabel = 'Buoy - QS wind speed'
        #plotScatter(x,y,axes[1,0], xlabel, ylabel)
        plotScatter(x[Emask],y[Emask],axes[1,0], xlabel, ylabel, color='blue')
        plotScatter(x[Wmask],y[Wmask],axes[1,0], xlabel, ylabel, color='green')
        axes[1,1].set_ylim(-10,10)

        ###################################################################
        
        x = vwndB.copy()
        xlabel = 'Buoy meriodional wind speed'

        y = wspdB.copy() - wspdS.copy()
        ylabel = 'Buoy - QS wind speed'
        #plotScatter(x,y,axes[1,1], xlabel, ylabel)
        plotScatter(x[Emask],y[Emask],axes[1,1], xlabel, ylabel, color='blue')
        plotScatter(x[Wmask],y[Wmask],axes[1,1], xlabel, ylabel, color='green')
        axes[1,1].set_ylim(-10,10)

        ###################################################################

        x = uwndB.copy()
        xlabel = 'Buoy zonal wind speed'

        y = wdirB.copy() - wdirS.copy()
        y[y>180] = 360-y[y>180]
        y[y<-180] +=  360 
        ylabel = 'buoy wind direction - QS wind direction'
        #plotScatter(x,y,axes[1,2], xlabel, ylabel)
        plotScatter(x[Emask],y[Emask],axes[1,2], xlabel, ylabel, color='blue')
        plotScatter(x[Wmask],y[Wmask],axes[1,2], xlabel, ylabel, color='green')
        axes[1,1].set_ylim(-90,90)

        ###################################################################
        
        x = vwndB.copy()
        xlabel = 'Buoy meridional wind speed'

        y = wdirB.copy() - wdirS.copy()
        y[y>180] = 360-y[y>180]
        y[y<-180] +=  360
        ylabel = 'buoy wind direction - QS wind direction'
        #plotScatter(x,y,axes[1,3], xlabel, ylabel)
        plotScatter(x[Emask],y[Emask],axes[1,3], xlabel, ylabel, color='blue')
        plotScatter(x[Wmask],y[Wmask],axes[1,3], xlabel, ylabel, color='green')
        axes[1,1].set_ylim(-90,90)

        ###################################################################

        ###################################################################
        x = uwndB.copy()
        xlabel = 'Buoy zonal wind speed'

        y = wspdB.copy() / wspdS.copy()
        ylabel = 'Buoy / QS wind speed'
        #plotScatter(x,y,axes[1,0], xlabel, ylabel)
        plotScatter(x[Emask],y[Emask],axes[2,0], xlabel, ylabel, color='blue', giveYlim=True, ylim=[-2,3])
        plotScatter(x[Wmask],y[Wmask],axes[2,0], xlabel, ylabel, color='green', giveYlim=True, ylim=[-2,3])

        ###################################################################
        
        x = vwndB.copy()
        xlabel = 'Buoy meriodional wind speed'

        y = wspdB.copy() / wspdS.copy()
        ylabel = 'Buoy / QS wind speed'
        #plotScatter(x,y,axes[1,1], xlabel, ylabel)
        plotScatter(x[Emask],y[Emask],axes[2,1], xlabel, ylabel, color='blue', giveYlim=True, ylim=[-2,3])
        plotScatter(x[Wmask],y[Wmask],axes[2,1], xlabel, ylabel, color='green', giveYlim=True, ylim=[-2,3])

        ###################################################################

        x = uwndB.copy()
        xlabel = 'Buoy zonal wind speed'

        y = uwndB.copy() / uwndS.copy()
        ylabel = 'buoy zonal wind / QS zonal wind'
        #plotScatter(x,y,axes[1,2], xlabel, ylabel)
        plotScatter(x[Emask],y[Emask],axes[2,2], xlabel, ylabel, color='blue', giveYlim=True, ylim=[-2,3])
        plotScatter(x[Wmask],y[Wmask],axes[2,2], xlabel, ylabel, color='green', giveYlim=True, ylim=[-2,3])

        ###################################################################
        
        x = vwndB.copy()
        xlabel = 'Buoy meridional wind speed'

        y = vwndB.copy() / vwndS.copy()
        ylabel = 'buoy meridional wind / QS meridioanl wind'
        #plotScatter(x,y,axes[1,3], xlabel, ylabel)
        plotScatter(x[Emask],y[Emask],axes[2,3], xlabel, ylabel, color='blue', giveYlim=True, ylim=[-2,3])
        plotScatter(x[Wmask],y[Wmask],axes[2,3], xlabel, ylabel, color='green', giveYlim=True, ylim=[-2,3])

        ###################################################################

        text = '10min-matchup buoy Neutral Winds with QS data 2000 to 2007 for All Buoy Pos'
        fig.suptitle(text, y = 0.9, fontsize = 25)
        plt.subplots_adjust(hspace=0.20, wspace=0.5)

        fname = f'images/Regression_speedDiffOutlierRemoved/AllBuoyPos_Regression_2000to2007.png'
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

    wspdS = np.empty((0), dtype=float)
    wdirS = np.empty((0), dtype=float)
    wspdB = np.empty((0), dtype=float)
    wdirB = np.empty((0), dtype=float)

    for task in taskList:
        lat = task[0]
        lon = task[1]
        #print(lat, lon)
        wspdS_f, wdirS_f, wspdB_f, wdirB_f = getDataAt(lat, lon)
        wspdS = np.concatenate((wspdS, wspdS_f), axis=0) 
        wdirS = np.concatenate((wdirS, wdirS_f), axis=0) 
        wspdB = np.concatenate((wspdB, wspdB_f), axis=0) 
        wdirB = np.concatenate((wdirB, wdirB_f), axis=0)

    printFig(wspdS, wdirS, wspdB, wdirB)


if __name__ == '__main__':
    main()


