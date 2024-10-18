from netCDF4 import Dataset
import os
import sys
from datetime import datetime, timedelta


startDate = datetime(2000,1,1)
endDate = datetime(2012,12,31)

curDate = startDate

while curDate <= endDate:

    year = curDate.year
    month = curDate.month
    day = curDate.day

    folder = f'../../downloads/WaveReanalysis/{year}/'
    fileName = folder + f'WAVERYS_3hourly_{year:04d}-{month:02d}-{day:02d}.nc'

    #print('Date:', curDate)
    if not os.path.isfile(fileName):
        dd = f'{year:04d}-{month:02d}-{day:02d}'
        print(dd)

    

    curDate += timedelta(days=1)