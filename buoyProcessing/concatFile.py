import os
import glob
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--var', type=str, default='WINDS', help='variable')
args = parser.parse_args()

varname = args.var

folder = f'{varname:s}/'

foldList = glob.glob(folder + '*')
print(foldList)

for folder in foldList:
    cmd = f'ncrcat -h {folder:s}/*.nc  {folder:s}_{varname:s}_2000.nc'
    print(cmd)
    os.system(cmd)
