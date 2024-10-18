import bulk
import coare35vn
import meteo
import util

import os

for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    __import__(module[:-4], locals(), globals())
del module