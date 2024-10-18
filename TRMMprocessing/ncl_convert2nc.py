import os
import numpy as np
import sys
from glob import glob
from mpi4py import MPI

# comm = MPI.COMM_WORLD
# nprocs = comm.Get_size()
# rank = comm.Get_rank()

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--year', type=int, default=2000, help='year data to convert to nc')
args = parser.parse_args()

year = args.year

mainFolder = '/srv/seolab/srai/observation/SatelliteVsBuoy/downloads/TRMM_data/downloaded'


def convert2nc(file, outDir):
    cmd = f'ncl_convert2nc {file} -o {outDir}'
    print(cmd)
    sys.stdout.flush()
    os.system(cmd)


def parallelConvert2nc(mainFolder, year, nprocs):
    dirLoc = f'{mainFolder}/TRMM_{year}/'
    outDir = f'{mainFolder}/TRMM_nc_{year}/'
    fileList = glob(dirLoc + f"/2A25.{year}*.HDF")
    nfiles = len(fileList)
    nfilesPerProc = int(nfiles/nprocs) * np.ones((nprocs), dtype=int)
    remainder = nfiles%nprocs
    nfilesPerProc[0:remainder]+= 1
    endIndices = np.cumsum(nfilesPerProc)
    startIndices = np.roll(endIndices, 1)
    startIndices[0] = 0
    
    filesInMe = fileList[startIndices[rank]:endIndices[rank]]
    for file in filesInMe:
        convert2nc(file, outDir)


def serialConvert2nc(mainFolder, year):
    dirLoc = f'{mainFolder}/TRMM_{year}/'
    outDir = f'{mainFolder}/TRMM_nc_{year}/'
    fileList = glob(dirLoc + f"/2A25.{year}*.HDF")
    
    for file in fileList:
        convert2nc(file, outDir)

#parallelConvert2nc(mainFolder, year, nprocs)
serialConvert2nc(mainFolder, year)