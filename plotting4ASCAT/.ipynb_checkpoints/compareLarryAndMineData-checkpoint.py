from netCDF4 import Dataset, date2num, num2date
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from glob import glob
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()


def printFig(lat, lon):
    if lat < 0:
        latUnit = 'S'
    else:
        latUnit = 'N'

    if lon < 0:
        lonUnit = 'W'
    else:
        lonUnit = 'E'

    lat = abs(lat)
    lon = abs(lon)

    myID = f'{lat:02d}{latUnit}_{lon:03d}{lonUnit}'
    larrID = f'{lat}{latUnit.lower()}{lon}{lonUnit.lower()}'

    myQSfile = f'../../downloads/Buoy/extractedGZ/WINDS/T_{myID}_matchup_2000.nc'
    larryQSfile = f'../../downloads/larryNielData/larry2020/EXP11/fromLarry_{larrID}_data.nc'
    if not (os.path.isfile(myQSfile) and os.path.isfile(larryQSfile)):
        return

    mDS = Dataset(myQSfile)
    lDS = Dataset(larryQSfile)

    cdfTime = mDS.variables['time_QS']
    timeUnit = cdfTime.units
    timeArr = np.array(cdfTime)
    cftimes=num2date(timeArr, timeUnit)
    myTime = np.array([datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) for dtm in cftimes])

    cdfTime = mDS.variables['time_TAO']
    timeUnit = cdfTime.units
    timeArr = np.array(cdfTime)
    cftimes=num2date(timeArr, timeUnit)
    myTime_TAO = np.array([datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) for dtm in cftimes])


    exact = abs(myTime_TAO - myTime) < timedelta(seconds=1)

    mWspd = np.array(mDS.variables['U10N_QS'])
    mWdir = np.array(mDS.variables['U10N_dir_QS'])
    #print(np.max(mWdir), np.max(mWdir))
    mask = abs(mWspd)>1000  # np.logical_or(abs(mWspd) > 1000,abs(mWdir) == 0)
    mWspd[mask] = float('nan')
    mWdir[mask] = float('nan')

    mWspd_TAO = np.array(mDS.variables['U10N_TAO'])
    mWdir_TAO = np.array(mDS.variables['U10N_dir_TAO'])
    mask_TAO = abs(mWspd_TAO) > 1000 # np.logical_or(abs(mWspd_TAO) > 1000,abs(mWdir_TAO) == 0)

    ########################## CHANGED THE DIRECTION HERE #############################
    mWdir_TAO = (-mWdir_TAO+450)%360
    ########################## CHANGED THE DIRECTION HERE #############################

    mWspd_TAO[mask_TAO] = float('nan')
    mWdir_TAO[mask_TAO] = float('nan')



    cdfTime = lDS.variables['time']
    timeUnit = 'days since 0001-01-01 00:00:0.0'
    timeArr = np.array(cdfTime)
    cftimes=num2date(timeArr, timeUnit, has_year_zero=True)
    larryTime = np.array([datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) - timedelta(365) for dtm in cftimes])

    lWspd = np.array(lDS.variables['sat_wspd10n'])
    u = np.array(lDS.variables['sat_u10n'])
    v = np.array(lDS.variables['sat_v10n'])
    lWdir = np.rad2deg(np.arctan2(v,u))%360

    lWspd_TAO = np.array(lDS.variables['buoy_wspd10n'])
    u = np.array(lDS.variables['buoy_u10n'])
    v = np.array(lDS.variables['buoy_v10n'])

    lWdir_TAO = np.rad2deg(np.arctan2(v,u))%360

    fig, axes = plt.subplots(nrows=2, ncols =1, figsize = (20,8))

    ax = axes[0]
    ax.plot(myTime, mWspd, label='me')
    ax.plot(larryTime, lWspd, label='larry', alpha = 0.8)
    ax.set_title(f'QuikSCAT wind speed {myID}')
    ax.legend()

    ax = axes[1]
    ax.plot(myTime, mWdir, label='me')
    ax.plot(larryTime, (lWdir), label='larry', alpha = 0.8)
    ax.legend()
    ax.set_title(f'QuikSCAT wind dir {myID}')

    plt.savefig(f'images/QS_CompWithLarryMatchUp_{myID}.png', dpi=70)

    plt.close()

    fig, axes = plt.subplots(nrows=2, ncols =1, figsize = (20,8))

    ax = axes[0]
    ax.plot(myTime, mWspd_TAO, label='me')
    ax.plot(larryTime, lWspd_TAO, label='larry', alpha = 0.8)
    ax.set_title(f'TAO wind speed {myID}')
    ax.legend()

    ax = axes[1]
    ax.plot(myTime, mWdir_TAO, label='me')
    ax.plot(larryTime, (lWdir_TAO), label='larry', alpha = 0.8)
    ax.legend()
    ax.set_title(f'TAO wind dir {myID}')

    plt.savefig(f'images/Buoy_CompWithLarryMatchUp_{myID}.png', dpi=70)

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
