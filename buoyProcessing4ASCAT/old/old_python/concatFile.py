import os
import glob
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--var', type=str, default='WINDS', help='variable')
args = parser.parse_args()

varname = args.var

folder = f'{varname:s}/'

foldList = glob.glob(folder + '*')
#print(foldList)

def makeRecDim(folder):
    subFolders = glob.glob(folder +'/*')
    print(subFolders)
    for subFolder in subFolders:
        files = glob.glob(subFolder)
        for fname in files:
            cmd = f'ncks -O --mk_rec_dmn TIME {fname} {fname}'
            print(cmd)
            os.system(cmd)

for folder in foldList:
    makeRecDim(folder)
    print('\n')
    cmd = f'ncrcat -h {folder:s}/*.nc  {folder:s}_{varname:s}_2007.nc'
    print(cmd)
    os.system(cmd)

    # fname = f'{folder:s}_{varname:s}_2007.nc'
    # cmd = f'ncpdq -O --sort TIME {fname} {fname}'
    # print(cmd)
    # os.system(cmd)

