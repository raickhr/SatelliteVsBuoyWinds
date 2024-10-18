import os
import numpy as np
from glob import glob

folder = './data'
for year in range(2000,2001):
    cmdStrList =['podaac-data-downloader',
                 '-c QSCAT_LEVEL_2B_OWV_COMP_12_KUSST_LCRES_4.1',
                 f'-d {folder}',
                 f'--start-date {year}-01-01T00:00:00Z',
                 f'--end-date {year}-12-31T23:59:59Z',
                 '-b="-180,-90,180,90"' ]
    
    cmd = ' '.join(map(str, cmdStrList))
    print(cmd)
    os.system(cmd)


