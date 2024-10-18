import os
from functools import partial
import multiprocessing
from datetime import datetime, timedelta
#import yaml



#!/usr/bin/env python
outDir= '../../downloads/WaveReanalysis/'
usr='srai'
pwd='CMEMSDamrewA4'

def downloadFile(currentDateTime,
                 outDir,
                 usr,
                 pwd):

    start_year = currentDateTime.year
    start_month = currentDateTime.month
    start_day = currentDateTime.day

    end_year = currentDateTime.year
    end_month = currentDateTime.month
    end_day = currentDateTime.day

    outFileName = f'GLORYS12v1_dailyAvg_{start_year:04d}-{start_month:02d}-{start_day:02d}.nc'
    cmdStrList = ['python',
                  '-m motuclient',
                  '--motu https://my.cmems-du.eu/motu-web/Motu',
                  '--service-id GLOBAL_MULTIYEAR_WAV_001_032-TDS',
                  '--product-id cmems_mod_glo_wav_my_0.2_PT3H-i',
                  '--longitude-min -180',
                  '--longitude-max 180',
                  '--latitude-min -90',
                  '--latitude-max 90',
                  f'--date-min "{start_year:04d}-{start_month:02d}-{start_day:02d} 00:00:00"',
                  f'--date-max "{end_year:04d}-{end_month:02d}-{end_day:02d} 23:59:59"',
                  '--variable VHM0 --variable VTPK --variable VPED',
                  f'--out-dir {outDir:s}',
                  f'--out-name {outFileName:s}',
                  f'--user {usr:s}',
                  f'--pwd {pwd:s}']

    cmd = ' '.join(map(str, cmdStrList))
    print(cmd)
    os.system(cmd)


def main():
    # Get number of processros
    nprocs = multiprocessing.cpu_count()
    print(f'The number of processors is {nprocs}')

    startDate = '2001-01-01'
    endDate = '2001-12-31'

    start_year, start_month, start_day = startDate.rsplit('-')
    end_year, end_month, end_day = endDate.rsplit('-')

    startDateTime = datetime(int(start_year), int(
        start_month), int(start_day), 0, 0, 0)
    endDateTime = datetime(int(end_year), int(
        end_month), int(end_day), 0, 0, 0)
    currentDateTime = startDateTime

    print(f'Downloading files from {startDateTime} to {endDateTime}')
    # List of all dates
    dateList = []
    while currentDateTime <= endDateTime:
        dateList.append(currentDateTime)
        currentDateTime += timedelta(days=1)

    ntasksInHand = len(dateList)

    partial_do_file = partial(downloadFile,
                              outDir=outDir,
                              usr=usr,
                              pwd=pwd)

    if nprocs > 36:
        print('To avoid making too many concurrent download request limiting concurrent request to 36')
        nprocs = 36

    if ntasksInHand > nprocs:
        print(f'Using {nprocs} processors to download')
        p = multiprocessing.Pool(nprocs)
    else:
        print(f'Using {ntasksInHand} processors to download')
        p = multiprocessing.Pool(ntasksInHand)

    # divide work in all processors
    p.map(partial_do_file, dateList)


if __name__ == '__main__':
    main()
