import os
import sys
from glob import glob
from mpi4py import MPI

import xarray as xr
import numpy as np
from netCDF4 import Dataset
from datetime import datetime, timedelta
import pandas as pd
from pyhdf.SD import SD, SDC

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--year', type=int, default=2000, help='year data to convert to nc')
args = parser.parse_args()

year = args.year

mainFolder = '/srv/data2/srai_poseidon/srai_poseidon/observation/SatelliteVsBuoy/downloads/TRMM_data/downloaded'


def convert2nc(file, inDir, outDir):
    file = file.replace(inDir, '')
    wfile = 'dateFixed_' +file
    wfile = wfile.replace('.HDF', '.nc')
    ds = SD(inDir+file, SDC.READ)
    try:
        Year = np.array(ds.select('Year'))
        Month = np.array(ds.select('Month'))
        DayOfMonth = np.array(ds.select('DayOfMonth'))
        Hour = np.array(ds.select('Hour'))
        Minute = np.array(ds.select('Minute'))
        Second = np.array(ds.select('Second'))
        MilliSecond = np.array(ds.select('MilliSecond'))
        DayOfYear = np.array(ds.select('DayOfYear'))
        scanTime_sec = np.array(ds.select('scanTime_sec'))
        lat = np.array(ds.select('Latitude'))
        lon = np.array(ds.select('Longitude'))
        nearSurfRain = np.array(ds.select('nearSurfRain')) #(nscan, nray) mm/hr
        rainAve = np.array(ds.select('rainAve')) #(nscan, nray, fakeDim8) mm/hr
        rainFlag = np.array(ds.select('rainFlag'))#(nscan, nray)
        rain = np.array(ds.select('rain')) #(nscan, nray, ncell1)
        e_SurfRain = np.array(ds.select('e_SurfRain')) #(nscan, nray)(nscan, nray) mm/hr
        rainType = np.array(ds.select('rainType')) #rainType(nscan, nray)
        ds.end()

        dateTimeArr = []
        for i in range(len(Year)):
            dateTimeArr.append(datetime(Year[i],Month[i],DayOfMonth[i],Hour[i],Minute[i],Second[i],MilliSecond[i]))
        dateTimeArr = np.array(dateTimeArr)

        xds = xr.Dataset(
                            {
                                "nearSurfRain": xr.DataArray(
                                    nearSurfRain,
                                    dims=["Time", "nray"],
                                    attrs={
                                        "units": "mm/hr",
                                    },
                                ),
                                "e_SurfRain": xr.DataArray(
                                    e_SurfRain,
                                    dims=["Time", "nray"],
                                    attrs={
                                        "units": "mm/hr",
                                    },
                                ),
                                "rainAve": xr.DataArray(
                                    rainAve,
                                    dims=["Time", "nray", 'fakeDim8'],
                                    attrs={
                                        "units": "mm/hr",
                                    },
                                ),
                                "rainFlag": xr.DataArray(
                                    rainFlag,
                                    dims=["Time", "nray"],
                                ),
                                "rainType": xr.DataArray(
                                    rainType,
                                    dims=["Time", "nray"],
                                ),
                                "rain": xr.DataArray(
                                    rain,
                                    dims=["Time", "nray", "ncell1"],
                                    attrs={
                                        "units": "mm/hr",
                                    },
                                ),
                                "Latitude": xr.DataArray(
                                    lat,
                                    dims=["Time", "nray"],
                                    attrs={
                                        "units": "degrees",
                                    },
                                ),
                                "Longitude": xr.DataArray(
                                    lon,
                                    dims=["Time", "nray"],
                                    attrs={
                                        "units": "degrees",
                                    },
                                ),
                            },
                            coords={
                                    "Time": dateTimeArr,
                                    },
                        )
        
        reference_time = pd.to_datetime('1900-01-01 00:00:00')
        xds['Time'].encoding['units'] = f'microseconds since {reference_time}'


        xds.to_netcdf(outDir+wfile, unlimited_dims=['Time'])
        print(f'rewriting time dimenson for file {file} SUCCESS')
    except:
        print(f'rewriting time dimenson for file {file} FAILED')
    
    sys.stdout.flush()


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
        convert2nc(file, dirLoc, outDir)

parallelConvert2nc(mainFolder, year, nprocs)
#serialConvert2nc(mainFolder, year)