from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime, timedelta
import os
import sys
from glob import glob
import argparse

from mpi4py import MPI

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()


parser = argparse.ArgumentParser()

parser.add_argument('--year', type=int, default=2000, help='year data to convert to nc')
args = parser.parse_args()

year = args.year

mainFolder = '/srv/seolab/srai/observation/SatelliteVsBuoy/downloads/TRMM_data/downloaded'


def rewriteTimeDimension(dir, file):
    file = file.replace(dir, '')
    wfile = 'dateFixed_' +file
    if os.path.isfile(dir + wfile):
        print(f'skipping {wfile}')
        return
    
    print(f'rewriting time dimenson for file {file}')
    sys.stdout.flush()

    ds = Dataset(dir + file)

    Year = np.array(ds.variables['Year'])

    Month = np.array(ds.variables['Month'])

    DayOfMonth = np.array(ds.variables['DayOfMonth'])

    Hour = np.array(ds.variables['Hour'])

    Minute = np.array(ds.variables['Minute'])

    Second = np.array(ds.variables['Second'])

    MilliSecond = np.array(ds.variables['MilliSecond'])

    # DayOfYear = np.array(ds.variables['DayOfYear'])

    # scanTime_sec = np.array(ds.variables['scanTime_sec'])

    ds.close()

    dateTimeArr = []
    for i in range(len(Year)):
        dateTimeArr.append(datetime(Year[i],Month[i],DayOfMonth[i],Hour[i],Minute[i],Second[i],MilliSecond[i]))
    dateTimeArr = np.array(dateTimeArr)

    xds = xr.open_dataset(dir+file)

    new_coords = {'Time': dateTimeArr}
    xds = xds.rename({'nscan': 'Time'}).assign_coords(new_coords)
    xds = xds.drop_vars(["Year",
                        "Month",
                        "DayOfMonth",
                        "Hour",
                        "Minute",
                        "Second",
                        "MilliSecond",
                        "DayOfYear",
                        "scanTime_sec"])

    xds.to_netcdf(dir + wfile)

    xds.close()

def rewriteSerial():
    dirLoc = f'{mainFolder}/TRMM_nc_{year}/'
    fileList = glob(dirLoc + f"/2A25.{year}*.nc")
        
    for file in fileList:
        rewriteTimeDimension(dirLoc, file)

def rewriteParallel():
    dirLoc = f'{mainFolder}/TRMM_nc_{year}/'
    fileList = glob(dirLoc + f"/2A25.{year}*.nc")

    nfiles = len(fileList)
    nfilesPerProc = int(nfiles/nprocs) * np.ones((nprocs), dtype=int)
    remainder = nfiles%nprocs
    nfilesPerProc[0:remainder]+= 1
    endIndices = np.cumsum(nfilesPerProc)
    startIndices = np.roll(endIndices, 1)
    startIndices[0] = 0
    
    filesInMe = fileList[startIndices[rank]:endIndices[rank]]
    for file in filesInMe:
        rewriteTimeDimension(dirLoc, file)


rewriteParallel()
