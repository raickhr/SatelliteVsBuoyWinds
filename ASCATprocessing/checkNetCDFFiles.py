from netCDF4 import Dataset, date2num, num2date
import numpy as np
import sys
from glob import glob
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--folder', type=str, default='./', help='read folder location')

args = parser.parse_args()

dirLoc = args.folder

fileList = glob(dirLoc + "/*.nc")

for fileName in fileList:
    try:
        ds = Dataset(fileName)
        ds.close()
    except:
        print(fileName)