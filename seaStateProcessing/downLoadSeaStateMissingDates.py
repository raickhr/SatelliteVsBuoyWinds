import os
from functools import partial
import multiprocessing
from datetime import datetime, timedelta
#import yaml



#!/usr/bin/env python
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

    outFileName = f'{start_year:d}/WAVERYS_3hourly_{start_year:04d}-{start_month:02d}-{start_day:02d}.nc'
    
    #if file is not present or the filesize is less than 50MB download
    if (not os.path.isfile(outDir+outFileName)) or (os.path.getsize(outDir+outFileName)/1024/1024 < 70): 
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

    else:
        print(f'{outDir+outFileName} already present')


def main():
    # Get number of processros
    nprocs = multiprocessing.cpu_count()
    print(f'The number of processors is {nprocs}')
    
    
    f = open('checkFile.log')
    dateList = []
    for l in f:
        year = int(l.rstrip().split('-')[0])
        month = int(l.rstrip().split('-')[1])
        day = int(l.rstrip().split('-')[2])
        readDate = datetime(year, month, day)
        dateList.append(readDate)
        
        
    ntasksInHand = len(dateList)

    outDir= f'../../downloads/WaveReanalysis/'
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
    p.close()
    p.join()


if __name__ == '__main__':
    main()
