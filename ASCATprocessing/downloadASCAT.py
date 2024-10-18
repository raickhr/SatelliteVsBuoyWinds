import os
import numpy as np
from glob import glob

folder = '../../downloads/ASCAT_data'
for year in range(2015,2020):
    
    cmdStrList = ['podaac-data-downloader',
                 '-c ASCATA_L2_25KM_CDR',
                 f'-d {folder}',
                 f'--start-date {year}-01-01T00:00:00Z',
                 f'--end-date {year}-12-31T23:59:59Z',
                 '-e .nc']
    
    cmd = ' '.join(map(str, cmdStrList))
    print(cmd)
    os.system(cmd)


