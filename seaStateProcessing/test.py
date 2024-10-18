from datetime import datetime
f = open('checkFile.log')
dateList = []
for l in f:
    print(l)
    year = int(l.rstrip().split('-')[0])
    month = int(l.rstrip().split('-')[1])
    day = int(l.rstrip().split('-')[2])
    print(year, month, day)
    readDate = datetime(year, month, day)
    print(readDate)