import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import datetime

'''
os.chdir(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Trips_All')
all_files = glob.glob('*.csv')
alltrips = pd.DataFrame()
for jj in all_files:
    print(jj)
    tem = pd.read_csv(jj)
    # print(tem.columns)
    if jj in ['Divvy_Trips_2020_Q1.csv', '202004-divvy-tripdata.csv', '202005-divvy-tripdata.csv',
              '202006-divvy-tripdata.csv', '202007-divvy-tripdata.csv']:
        tem.columns = ['trip_id', 'rideytype', 'starttime', 'stoptime', 'from_station_name', 'from_station_id',
                       'to_station_name', 'to_station_id', 'start_lat', 'start_lng', 'end_lat', 'end_lng', 'usertype']
    else:
        tem.columns = ['trip_id', 'starttime', 'stoptime', 'bikeid', 'tripduration', 'from_station_id',
                       'from_station_name', 'to_station_id', 'to_station_name', 'usertype', 'gender', 'birthday']
    tem = tem[
        ['trip_id', 'starttime', 'stoptime', 'from_station_id', 'from_station_name', 'to_station_id', 'to_station_name',
         'usertype']]
    tem['starttime'] = pd.to_datetime(tem['starttime'])
    tem['stoptime'] = pd.to_datetime(tem['stoptime'])
    alltrips = alltrips.append(tem)
alltrips.to_pickle('alltrips.chicago_202007')
'''
os.chdir(r'D:\COVID19-Transit_Bikesharing\Divvy_Data')
alltrips = pd.read_pickle(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Trips_All\alltrips.chicago_202007')
raw_length = len(alltrips)
alltrips.info()
alltrips.isnull().sum()

# Duration
alltrips['Duration'] = (alltrips['stoptime'] - alltrips['starttime']).dt.total_seconds()
alltrips = alltrips[(alltrips['Duration'] > 60) & (alltrips['Duration'] < 60 * 60 * 6)].reset_index(drop=True)
print('Delete: ' + str(raw_length - len(alltrips)))
# sns.distplot(alltrips['Duration'])

# Daily count
alltrips['startyear'] = alltrips['starttime'].dt.year
alltrips['startmonth'] = alltrips['starttime'].dt.month
alltrips['startdate'] = alltrips['starttime'].dt.date
month_count = alltrips.groupby(['startyear', 'startmonth']).count()['trip_id'].reset_index()
Day_count = alltrips.groupby(['startdate']).count()['trip_id'].reset_index()
plt.plot(Day_count['trip_id'], '-')

# Merge with station info

# Merge with Weather

# Merge with holidays

# Output to BSTS
