from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime, timedelta
import os


def plotTimeSeriesByClusterLabel(df, lat, lon, ax1, ax2, cluster_label=0, color='blue'):
    try:
        subDF = df.loc[df['lat'] == lat]
        subDF = subDF.loc[subDF['lon'] == lon%360]
        subDF = subDF.loc[subDF['label'] == cluster_label]
    except:
        return

    if len(subDF) < 1:
        return

    LAT = lat
    LON = lon
    if lat < 0:
        lat = abs(lat)
        latUnits = 'S'
    else:
        latUnits = 'N'

    if lon < 0:
        lonUnits = 'W'
        lon = abs(lon) # 360
    else:
        lonUnits = 'E'

    deployFileName = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_DeploymentDates.nc'
        
    if os.path.isfile(deployFileName):
        print(f'cluster :{cluster_label}', LAT, LON, deployFileName)
        ds2 = Dataset(deployFileName)
        startDates = np.array(ds2.variables['startDate'])
        units = ds2.variables['startDate'].units
        cftimes = num2date(startDates, units)
        startDates = np.array([datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) for dtm in cftimes])

        endDates = np.array(ds2.variables['endDate'])
        units = ds2.variables['endDate'].units
        cftimes = num2date(endDates, units)
        endDates = np.array([datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute, dtm.second) for dtm in cftimes])

        ds2.close()
        
        cftimes = num2date(subDF['time'], 'seconds since 1990-01-01 00:00:00')
        dates = np.array([datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute) for dtm in cftimes])
        y1 = subDF['speedDiff']
        y2 = subDF['directionDiff']

        absSpeedDiffMean = np.mean(subDF['absSpeedDiff'])
        absSpeedDiffStd = np.std(subDF['absSpeedDiff'])
        absDirectionDiffMean = np.mean(subDF['absDirectionDiff'])
        absDirectionDiffStd = np.std(subDF['absDirectionDiff'])

        text1 = f'label {cluster_label}, meanAbsSpeedDiff = {absSpeedDiffMean:5.2f}, stdAbsSpeedDiff =  {absSpeedDiffStd:5.2f}\n'
        text2 = f'label {cluster_label}, meanAbsDirDiff = {absDirectionDiffMean:5.2f}, stdAbsDirDiff =  {absDirectionDiffStd:5.2f}\n' 

        for i in range(len(startDates)):
            colorR =list(np.random.choice(range(256), size=3)/256)
            
            ax1.axvspan(startDates[i], endDates[i], 
                    alpha=0.05, color=colorR)
            ax2.axvspan(startDates[i], endDates[i], 
                    alpha=0.05, color=colorR)
        
        #ax1.scatter(subDF['time'], y1, s=2, label = cluster_label)#, c = subDF['label'])
        #ax2.scatter(subDF['time'], y2, s=2, label = cluster_label)#, c = subDF['label'])

        ax1.scatter(dates, y1, s=3, label = cluster_label, c = color)
        ax2.scatter(dates, y2, s=3, label = cluster_label, c = color)
        ax1.set_title(f'{lat:02d}{latUnits} {lon:03d}{lonUnits} speed Diff')
        ax2.set_title(f'{lat:02d}{latUnits} {lon:03d}{lonUnits} direction Diff')
        return text1, text2
    

def plotTimeSeriesByClusterLabelForAllData(df, ax1, ax2, cluster_label=0, color='blue'):
    subDF = df.loc[df['label'] == cluster_label]
    cftimes = num2date(subDF['time'], 'seconds since 1990-01-01 00:00:00')
    dates = np.array([datetime(dtm.year, dtm.month, dtm.day, dtm.hour, dtm.minute) for dtm in cftimes])
    y1 = subDF['speedDiff']
    y2 = subDF['directionDiff']

    absSpeedDiffMean = np.mean(subDF['absSpeedDiff'])
    absSpeedDiffStd = np.std(subDF['absSpeedDiff'])
    absDirectionDiffMean = np.mean(subDF['absDirectionDiff'])
    absDirectionDiffStd = np.std(subDF['absDirectionDiff'])

    text1 = f'label {cluster_label}, meanAbsSpeedDiff = {absSpeedDiffMean:5.2f}, stdAbsSpeedDiff =  {absSpeedDiffStd:5.2f}\n'
    text2 = f'label {cluster_label}, meanAbsDirDiff = {absDirectionDiffMean:5.2f}, stdAbsDirDiff =  {absDirectionDiffStd:5.2f}\n' 


    ax1.scatter(dates, y1, s=3, label = cluster_label, c = color)
    ax2.scatter(dates, y2, s=3, label = cluster_label, c = color)
    ax1.set_title(f'All Pos Speed Diff')
    ax2.set_title(f'All Pos Direction Diff')
    return text1, text2

def plotTimeSeriesByClusterLabelForRegion(lat=None, lon=None, zonal='East', meridional='North'):

    val = None
    letter = None
    text = None
    dfSel = None
    if lat != None and lon == None: 
        if meridional == 'North':
            dfSel = df.loc[df['lat']>lat]
            text = 'Norther'
        elif meridional == 'South':
            dfSel = df.loc[df['lat']<lat]
            text = 'Souther'

        if lat < 0:
            letter = 'S'
        else:
            letter = 'N'
        
        val = abs(lat)

    if lon != None and lat == None: 
        if zonal == 'East':
            dfSel = df.loc[df['lon']>(lon%360)]
            text = 'East'
        elif zonal == 'West':
            dfSel = df.loc[df['lon']<(lon%360)]
            text = 'West'

        if lon < 0:
            letter = 'W'
        else:
            letter = 'E'

        val = abs(lon)

    titleText = f'{text:s} Than {val:d}{letter:s}'
    print(titleText)
    fname = f'images/clustering/{text}Than{val:d}{letter:s}_Clustering.png'
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20,8))
    textA1, textB1 = plotTimeSeriesByClusterLabelForAllData(dfSel, axes[0], axes[1], cluster_label=0, color = 'blue')
    textA2, textB2 = plotTimeSeriesByClusterLabelForAllData(dfSel, axes[0], axes[1], cluster_label=1, color = 'green')
    textA3, textB3 = plotTimeSeriesByClusterLabelForAllData(dfSel, axes[0], axes[1], cluster_label=2, color = 'red')
    textA = textA1 + textA2 + textA3
    axes[0].text(0.1,0.85, textA,  color='white', horizontalalignment='left',
                    verticalalignment='center', transform=axes[0].transAxes,
                    bbox=dict(facecolor='black', alpha=0.5))
    axes[0].set_ylabel('speed diff.')
    axes[0].legend()

    textB = textB1 + textB2 + textB3
    axes[1].text(0.1,0.85, textB,  color='white', horizontalalignment='left',
                    verticalalignment='center', transform=axes[1].transAxes,
                    bbox=dict(facecolor='black', alpha=0.5))
    axes[1].set_ylabel('direction diff.')
    axes[1].legend()
    plt.suptitle(titleText)
    plt.savefig(fname, dpi=100)
    plt.close()


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

metaData = np.empty((0,19), dtype=float)
metaDataIndex = ['time',
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

    matchFname = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_matchupNearestFour_2000.nc'
    deployFileName = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_DeploymentDates.nc'
    
    if os.path.isfile(matchFname):

        ds = Dataset(matchFname)
        arr = np.empty((0,1), dtype=float)
        for i in range(3,18):
            if i <= 8:
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

        # latArr = latArr[:,np.newaxis]
        # lonArr = lonArr[:,np.newaxis]
        timeArr = timeArr[:,np.newaxis]
        
        #print('lonArr', lonArr.shape)
        arr = np.concatenate((lonArr, arr), axis=1)

        #print('latArr', latArr.shape)
        arr = np.concatenate((latArr, arr), axis=1)

        #print('timeArr', timeArr.shape)
        arr = np.concatenate((timeArr, arr), axis=1)

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
        

nanVals = np.sum(metaData, axis=1)
mask = np.isnan(nanVals)
print('MetaData.shape', metaData.shape, 'masked rows', np.sum(mask))
metaData = metaData[~mask, :]
print('new MetaData.shape', metaData.shape)


df = pd.DataFrame(metaData, columns = metaDataIndex )
df['speedDiff'] = df['U10N_QS'] - df['U10N_TAO']
df['absSpeedDiff'] = abs(df['U10N_QS'] - df['U10N_TAO'])

dDiff = (df['U10N_dir_QS'] - df['U10N_dir_TAO'])%360
dDiff[dDiff > 180] -= 360 #- dDiff[dDiff > 180] 
dDiff[dDiff < -180] += 360
df['directionDiff'] = dDiff
df['absDirectionDiff'] = abs(dDiff)


selectX = [#'time',
           'U10N_QS',
           'U10N_dir_QS',
           'satTimeDiff',
           'dist',
           'U10N_TAO',
           'U10N_dir_TAO',
           'U10N_x_TAO2',
           'U10N_y_TAO2',
           'SST_TAO',
           'RH_TAO',
           'AIRT_TAO',
           #'speedDiff',
           #'directionDiff',
           'absSpeedDiff',
           'absDirectionDiff']

X = df[selectX]

normX = (X - X.mean(axis=0))/ X.std(axis=0)

kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto", max_iter = 10000).fit(normX)
df['label'] = kmeans.labels_

del X, normX

latList = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
lonList = [-95, -110, -125, -140, -155, -170, -180, 165]

#latList = [-5]
#lonList = [-110]

ylen = len(latList)
xlen = len(lonList)

taskList = []

for latId  in range(ylen):
    for lonId in range(xlen):
        taskList.append([latList[latId], lonList[lonId]])

ntasks = len(taskList)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20,8))
textA1, textB1 = plotTimeSeriesByClusterLabelForAllData(df, axes[0], axes[1], cluster_label=0, color = 'blue')
textA2, textB2 = plotTimeSeriesByClusterLabelForAllData(df, axes[0], axes[1], cluster_label=1, color = 'green')
textA3, textB3 = plotTimeSeriesByClusterLabelForAllData(df, axes[0], axes[1], cluster_label=2, color = 'red')
textA = textA1 + textA2 + textA3
axes[0].text(0.1,0.85, textA,  color='white', horizontalalignment='left',
                verticalalignment='center', transform=axes[0].transAxes,
                bbox=dict(facecolor='black', alpha=0.5))
axes[0].set_ylabel('speed diff.')
axes[0].legend()

textB = textB1 + textB2 + textB3
axes[1].text(0.1,0.85, textB,  color='white', horizontalalignment='left',
                verticalalignment='center', transform=axes[1].transAxes,
                bbox=dict(facecolor='black', alpha=0.5))
axes[1].set_ylabel('direction diff.')
axes[1].legend()

fname = f'images/clustering/AllPos_Clustering.png'
plt.savefig(fname, dpi=100)
plt.close()


plotTimeSeriesByClusterLabelForRegion(lat=2, lon=None, zonal=None, meridional='North')
plotTimeSeriesByClusterLabelForRegion(lat=3, lon=None, zonal=None, meridional='North')

plotTimeSeriesByClusterLabelForRegion(lat=-2, lon=None, zonal=None, meridional='South')
plotTimeSeriesByClusterLabelForRegion(lat=-3, lon=None, zonal=None, meridional='South')


plotTimeSeriesByClusterLabelForRegion(lat=None, lon=140, zonal='East', meridional=None)
plotTimeSeriesByClusterLabelForRegion(lat=None, lon=-150, zonal='West', meridional=None)
plotTimeSeriesByClusterLabelForRegion(lat=None, lon=-160, zonal='West', meridional=None)
plotTimeSeriesByClusterLabelForRegion(lat=None, lon=-170, zonal='West', meridional=None)



# dfSel = df.loc[df['lon']>140]
# titleText = 'East Than 140E'
# fname = f'images/clustering/EastThan140E_Clustering.png'
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20,8))
# textA1, textB1 = plotTimeSeriesByClusterLabelForAllData(dfSel, axes[0], axes[1], cluster_label=0, color = 'blue')
# textA2, textB2 = plotTimeSeriesByClusterLabelForAllData(dfSel, axes[0], axes[1], cluster_label=1, color = 'green')
# textA3, textB3 = plotTimeSeriesByClusterLabelForAllData(dfSel, axes[0], axes[1], cluster_label=2, color = 'red')
# textA = textA1 + textA2 + textA3
# axes[0].text(0.1,0.85, textA,  horizontalalignment='left',
#                 verticalalignment='center', transform=axes[0].transAxes)
# axes[0].set_ylabel('speed diff.')
# axes[0].legend()

# textB = textB1 + textB2 + textB3
# axes[1].text(0.1,0.85, textB,  horizontalalignment='left',
#                 verticalalignment='center', transform=axes[1].transAxes)
# axes[1].set_ylabel('direction diff.')
# axes[1].legend()

# plt.savefig(fname, dpi=100)
# plt.close()


# for task in taskList:
#     LAT = task[0]
#     LON = task[1]
#     lat = LAT
#     lon = LON

#     if lat < 0:
#         lat = abs(lat)
#         latUnits = 'S'
#     else:
#         latUnits = 'N'

#     if lon < 0:
#         lonUnits = 'W'
#         lon = abs(lon) # 360
#     else:
#         lonUnits = 'E'

#     deployFileName = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_DeploymentDates.nc'
#     matchFname = f'../../downloads/Buoy/extractedGZ/WINDS/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_matchupNearestFour_2000.nc'
        
#     if os.path.isfile(matchFname):
#         try:
#             fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20,8))
#             textA1, textB1 = plotTimeSeriesByClusterLabel(df, LAT, LON, axes[0], axes[1], cluster_label=0, color = 'blue')
#             textA2, textB2 = plotTimeSeriesByClusterLabel(df, LAT, LON, axes[0], axes[1], cluster_label=1, color = 'green')
#             textA3, textB3 = plotTimeSeriesByClusterLabel(df, LAT, LON, axes[0], axes[1], cluster_label=2, color = 'red')
#             textA = textA1 + textA2 + textA3
#             axes[0].text(0.1,0.85, textA,  horizontalalignment='left',
#                          verticalalignment='center', transform=axes[0].transAxes)
#             axes[0].set_ylabel('speed diff.')
#             axes[0].legend()

#             textB = textB1 + textB2 + textB3
#             axes[1].text(0.1,0.85, textB,  horizontalalignment='left',
#                          verticalalignment='center', transform=axes[1].transAxes)
#             axes[1].set_ylabel('direction diff.')
#             axes[1].legend()
            
#             fname = f'images/clustering/T_{lat:02d}{latUnits}_{lon:03d}{lonUnits}_Clustering.png'
#             plt.savefig(fname, dpi=100)
#             #print(fname, deployFileName)
#             plt.close()
#         except:
#             print('skipping ',deployFileName)
            


        

