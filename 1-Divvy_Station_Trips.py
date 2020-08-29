import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob

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
