import os
import re
from netCDF4 import Dataset, date2num, num2date
import datetime
import numpy as np
import glob


## this code creates a folder for each buoy data
## first extract the *.nc file for each variable and then run this code
## then create the name of variable and and make a parent folder for this variable

folder = f'./'
fileList = glob.glob(folder + '*.nc')

prefixList = []
for file in fileList:
    fname = file.rsplit("/")[-1]
    print(fname)
    prefix = fname.rsplit("_")[1]
    prefixList.append(prefix)

buoyNames = list(set(prefixList))


for bname in buoyNames:
    origBname = bname+'_buoy'
    match = re.findall('\d+', origBname)
    lat = int(match[0])
    lon = int(match[1])

    strMatch = re.findall(r"([A-Z]+)", origBname[1:len(origBname)])
    latS = strMatch[0]
    lonS = strMatch[1]

    print('lat = ',lat, latS)
    print('lon = ',lon, lonS)
    print('\n')

    folderName = f'T_{lat:02d}{latS:s}_{lon:03d}{lonS:s}'
    cmd = 'mkdir ' + folderName
    print(cmd)
    os.system(cmd)

    cmd = 'mv *_'+bname+'_* '+folderName
    print(cmd)
    os.system(cmd)
    
    

