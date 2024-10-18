import os
import sys
import numpy as np
from mpi4py import MPI

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--year', type=int, default=2000, help='year data to convert to nc')
args = parser.parse_args()

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

print('nprocs = ', nprocs)
year = args.year
latList = [-9, -8, -5, -2, 0, 2, 5, 8, 9]
lonList = [-95, -110, -125, -140, -155, -170, -180, 165]

ylen = len(latList)
xlen = len(lonList)

taskList = []

for latId in range(ylen):
    for lonId in range(xlen):
        taskList.append([latList[latId], lonList[lonId]])

ntasks = len(taskList)

remainder = ntasks % nprocs
ntasksForMe = int(ntasks//nprocs)

# startIndex in each processor fileIndex start from 1
taskListInMe = [rank]  # the list goes on as [0,5,10] for 5 processors

if rank < remainder:
    ntasksForMe += 1

for i in range(1, ntasksForMe):
    taskListInMe.append(taskListInMe[-1]+nprocs)

folder = f'../../downloads/TRMM_data/downloaded/TRMM_nc_{year}/'
wfolder = f'../../downloads/TRMM_data/downloaded/TRMM_nc_{year}/'

for task in taskListInMe:
    # for lat in latList:
    #     for lon in lonList:
    lat = taskList[task][0]
    lon = taskList[task][1]
    cmd = f'python selectTRMMdataWithLatLon.py --folder={folder} --wfolder={wfolder} --lat={lat} --lon={lon}'
    print(cmd)
    os.system(cmd)
