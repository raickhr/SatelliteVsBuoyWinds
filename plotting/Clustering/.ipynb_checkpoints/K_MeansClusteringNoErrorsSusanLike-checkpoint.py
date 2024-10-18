from sklearn.cluster import KMeans
import numpy as np
from scipy import stats
import sys
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime, timedelta
import os

def plotRegSlopeInterceptAndRsq(xDates,y,ax, regLabel = '', colors = ['blue', 'purple']):
    sDate=xDates[0]
    eDate=xDates[-1]
    xUnit = f'seconds since {sDate.year}-{sDate.month:02d}-{sDate.day:02d}-{sDate.hour:02d}:{sDate.minute:02d}:{sDate.second:02d}'
    seconds = np.array(date2num(xDates,xUnit), dtype=float)
    y = np.array(y, dtype=float)
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


def getClusteredDF():
    latList = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
    lonList = [-95, -110, -125, -140, -155, -170, -180, 165]

    ylen = len(latList)
    xlen = len(lonList)

    taskList = []

    for latId  in range(ylen):
        for lonId in range(xlen):
            taskList.append([latList[latId], lonList[lonId]])

    ntasks = len(taskList)

    i = 0
    tlen = 0

    metaData = np.empty((0,20), dtype=float)
    metaDataIndex = ['dateTime',
                    'time',
                    'lat', 
                    'lon',
                    'U10N_QS',
                    'U10N_dir_QS',
                    'satTimeDiff',
                    'dist',
                    'satLon',
                    'satLat',
                    'U10N_TAO',
                    'U10N_dir_TAO',
                    'U10N_TAO2',
                    'U10N_dir_TAO2',
                    'U10N_x_TAO2',
                    'U10N_y_TAO2',
                    'SST_TAO',
                    'RH_TAO',
                    'AIRT_TAO',
                    'Deployment Classifier']

    for task in taskList:
        lat = task[0]
        lon = task[1]

        LAT = lat
        LON = lon

        if lat < 0:
            latUnits = 'S'
        else:
            latUnits = 'N'

        if lon < 0:
            lonUnits = 'W'
            LON += 360
        else:
            lonUnits = 'E'
        
        lat=abs(lat)
        lon=abs(lon)

        matchFname = f'../../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_matchupNearestFour_2000.nc'
        deployFileName = f'../../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_DeploymentDates.nc'
        
        if os.path.isfile(matchFname):

            ds = Dataset(matchFname)
            arr = np.empty((0,1), dtype=float)
            for i in range(4,19):
                if i <= 9:
                    #print(i, metaDataIndex[i])
                    readArr = np.array([ds.variables[metaDataIndex[i]][0,:]]).T
                    #print(readArr.shape)
                    #print(arr.shape)
                    if len(arr) == 0:
                        arr = np.concatenate((arr, readArr), axis = 0)
                    else:
                        arr = np.concatenate((arr, readArr), axis = 1)
                else:
                    #print(i, metaDataIndex[i])
                    readArr = np.array([ds.variables[metaDataIndex[i]][:]]).T
                    #print(readArr.shape)
                    #print(arr.shape)
                    arr = np.concatenate((arr, readArr), axis = 1)

            latArr = np.array([len(arr)*[LAT]]).T
            lonArr = np.array([len(arr)*[LON]]).T
            timeArr = np.array(ds.variables['time'])
            timeUnit = ds.variables['time'].units
            cftimes = num2date(timeArr, timeUnit)
            dateTimeArr = np.array([datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute) for dtm in cftimes])

            # latArr = latArr[:,np.newaxis]
            # lonArr = lonArr[:,np.newaxis]
            
            
            #print('lonArr', lonArr.shape)
            arr = np.concatenate((lonArr, arr), axis=1)

            #print('latArr', latArr.shape)
            arr = np.concatenate((latArr, arr), axis=1)

            #print('timeArr', timeArr.shape)
            timeArr = timeArr[:,np.newaxis]
            arr = np.concatenate((timeArr, arr), axis=1)

            dateTimeArr = dateTimeArr[:,np.newaxis]
            arr = np.concatenate((dateTimeArr, arr), axis=1)

            ds2 = Dataset(deployFileName)
            startDates = np.array(ds2.variables['startDate'])
            endDates = np.array(ds2.variables['endDate'])

            depNum = np.zeros((len(arr),1), dtype=int)
            c = 1
            for i in range(1,len(startDates)):
                mask = timeArr >= startDates[i]
                mask *= timeArr <= endDates[i]
                depNum[mask] = c
                c = c+1

            arr = np.concatenate((arr, depNum), axis=1)


            metaData = np.concatenate((metaData, arr), axis=0)
            #ds = ds.expand_dims('lat', axis= 0)
            

    nanVals = np.array(np.sum(metaData[:,1:], axis=1), dtype=float)
    mask = np.isnan(nanVals)
    print('MetaData.shape', metaData.shape, 'masked rows', np.sum(mask))
    metaData = metaData[~mask, :]
    print('new MetaData.shape', metaData.shape)


    df = pd.DataFrame(metaData, columns = metaDataIndex )
    df['speedDiff'] = df['U10N_TAO'] - df['U10N_QS'] 
    df['absSpeedDiff'] = abs(df['U10N_TAO']-df['U10N_QS'] )

    dDiff = (df['U10N_dir_QS'] - df['U10N_dir_TAO'])%360
    dDiff[dDiff > 180] -= 360 #- dDiff[dDiff > 180] 
    dDiff[dDiff < -180] += 360
    df['directionDiff'] = dDiff
    df['absDirectionDiff'] = abs(dDiff)

    yearmonth = np.zeros((len(df)), dtype=int)
    month = np.zeros((len(df)), dtype=int)
    for year in range(2000,2008):
        for mth in range(1,13):
            curYearMonth = year*100 + mth
            mask =np.array([dtm.year == year and dtm.month == mth for dtm in df['dateTime']])
            yearmonth[mask] = curYearMonth

    for mth in range(1,13):
        curMonth = mth
        mask =np.array([dtm.month == mth for dtm in df['dateTime']])
        month[mask] = curMonth


    df['yearmonth'] = yearmonth
    df['month'] = month

    selectX = [#'time',
           #'U10N_QS',
           #'U10N_dir_QS',
           #'satTimeDiff',
           #'dist',
           #'U10N_TAO',
           #'U10N_dir_TAO',
           #'U10N_x_TAO2',
           #'U10N_y_TAO2',
           #'SST_TAO',
           #'RH_TAO',
           #'AIRT_TAO',
           'speedDiff',
           'directionDiff',
           #'absSpeedDiff',
           #'absDirectionDiff'
            ]

    X = df[selectX]
    normX = (X - X.mean(axis=0))/ X.std(axis=0)
    kmeans = KMeans(n_clusters=9, random_state=0, n_init="auto", max_iter = 10000).fit(normX)
    df['label'] = kmeans.labels_

    del X, normX

    return df

def printFig(df, lat, lon):
    LAT = lat
    LON = lon%360
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
    
    subDF = df.loc[df['lat'] == LAT]
    subDF = subDF.loc[subDF['lon'] == LON]

    ndata = len(subDF)

    if ndata > 1:
        depLb = subDF['Deployment Classifier'].to_list()
        depLb = list(set(depLb))

        fig, axes = plt.subplots(nrows=2, ncols =1, figsize=(20,5))

        for i in range(len(depLb)):
            dep = depLb[i]
            ssubDF = subDF.loc[subDF['Deployment Classifier'] == dep]
            if len(ssubDF) < 5:
                continue
            dates = pd.to_datetime(ssubDF['dateTime'])
            y = dates.dt.year.to_list()
            m = dates.dt.month.to_list()
            d = dates.dt.day.to_list()
            h = dates.dt.hour.to_list()
            mm = dates.dt.minute.to_list()
            ss = dates.dt.second.to_list()
            dateList = np.array([datetime(y[i],m[i],d[i],h[i],mm[i],ss[i]) for i in range(len(dates))])
            
            
            if len(dates)>2:
                x = dateList
                y1 = ssubDF['speedDiff'].to_numpy()
                angleDiff = ssubDF['directionDiff'].to_numpy()
                angleDiff[angleDiff>180] = 360 - angleDiff[angleDiff>180]
                angleDiff[angleDiff<-180] += 360
                y2 = angleDiff
                

                
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

        #fname = f'images/SusanLike/SusanLike_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_Comp_2000to2007_minErrorClusters.png'
        fname = f'images/SusanLike/SusanLike_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_Comp_2000to2007_maxErrorClusters.png'
        #fname = f'images/SusanLike/SusanLike_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_Comp_2000to2007_AllClusters.png'
        plt.savefig(fname,dpi=70)
        plt.close()
        del subDF, ssubDF
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

    dataInfo = []

    df = getClusteredDF()
    #df = df.loc[df['label'].isin([1,4,7])]
    df = df.loc[df['label'].isin([0,2,3,5,6,8])]

    for task in taskList:
        lat = task[0]
        lon = task[1]
        #print(lat, lon)
        dataInfo.append(printFig(df, lat, lon))

    # dataInfo= np.array(dataInfo)

    # allDataInfo = comm.gather(dataInfo, root = 0)
    # print(allDataInfo[0])


if __name__ == '__main__':
    main()
    



            


        

