import os
import re
#from netCDF4 import Dataset, date2num, num2date
import datetime
import numpy as np
import glob


## this code creates a folder for each buoy data
## first extract the *.nc file for each variable and then run this code
## then create the name of variable and and make a parent folder for this variable


variables = ['WINDS', 'SST', 'RH', 'AIRT', 'BARO', 'RAD', 'LWR', 'RAIN']

for var in variables:
    folder = f'/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/Buoy/TAO_data/{var}/'
    fileList = glob.glob(folder + '*.nc')

    prefixList = []
    for file in fileList:
        fname = file.rsplit("/")[-1]
        print(fname)
        prefix = fname.rsplit("_")[1]
        prefixList.append(prefix)

    buoyNames = list(set(prefixList))

    txtFileName = f'locs_{var}.txt'
    with open(txtFileName, "w") as file:
        for item in buoyNames:
            file.write(f"{item}\n")


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

        folderName = f'{folder:s}T_{lat:03d}{latS:s}_{lon:03d}{lonS:s}'
        cmd = 'mkdir -p ' + folderName
        print(cmd)
        os.system(cmd)

        cmd = f'mv {folder:s}*_'+bname+'_*.nc '+folderName
        print(cmd)
        os.system(cmd)
    
    

