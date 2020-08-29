import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import datetime
import json
import requests
from pandas.tseries.holiday import USFederalHolidayCalendar

'''
# Read all trips
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
# alltrips.info()

# Duration
alltrips['Duration'] = (alltrips['stoptime'] - alltrips['starttime']).dt.total_seconds()
alltrips = alltrips[(alltrips['Duration'] > 60) & (alltrips['Duration'] < 60 * 60 * 6)].reset_index(drop=True)
print('Delete: ' + str(raw_length - len(alltrips)))
# sns.distplot(alltrips['Duration'])
# Drop na
alltrips = alltrips.dropna().reset_index(drop=True)

# # Daily count
alltrips['startyear'] = alltrips['starttime'].dt.year
alltrips['startmonth'] = alltrips['starttime'].dt.month
alltrips['startdate'] = alltrips['starttime'].dt.date

# Merge with station info
# Only consider those with trips in 2020
print('No of Stations in 2020: ' + str(len(set(alltrips.loc[alltrips['startyear'] == 2020, 'from_station_id']))))
alltrips = alltrips[
    alltrips['from_station_id'].isin(set(alltrips.loc[alltrips['startyear'] == 2020, 'from_station_id']))].reset_index(
    drop=True)
alltrips.isnull().sum()
Day_count = alltrips.groupby(['from_station_id', 'startdate']).count()['trip_id'].reset_index()
Day_count = Day_count[Day_count.startdate.notnull()].reset_index(drop=True)
'''
# Read Station in 2016-2017, to get online data
Stations_online = pd.DataFrame()
os.chdir(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Station_All')
for files in ['Divvy_Stations_2016_Q1Q2.csv', 'Divvy_Stations_2016_Q3.csv', 'Divvy_Stations_2016_Q4.csv',
              'Divvy_Stations_2017_Q1Q2.csv', 'Divvy_Stations_2017_Q3Q4.csv']:
    tem = pd.read_csv(files)
    tem['online_date'] = pd.to_datetime(tem['online_date'])
    Stations_online = Stations_online.append(tem)
Stations_online = Stations_online[['id', 'online_date']]
Stations_online = Stations_online.drop_duplicates(subset='id')

# Read station from gbts, to get all stations
return_data = requests.get('https://gbfs.divvybikes.com/gbfs/en/station_information.json')
All_Station = pd.DataFrame(json.loads(return_data.text)['data']['stations'])
All_Station = All_Station[['station_id', 'station_type', 'name', 'lon', 'lat', 'capacity']]
All_Station = All_Station[All_Station['station_type'] == 'classic']
All_Station = All_Station.rename({'station_id': 'id'}, axis=1)
All_Station['id'] = All_Station['id'].astype(int)
All_Station = All_Station.drop_duplicates().reset_index(drop=True)
All_Station = All_Station.merge(Stations_online, on='id', how='left')
All_Station.to_csv('D:\COVID19-Transit_Bikesharing\Divvy_Data\Divvy_Station.csv')
'''

# Merge with trips
All_Station = pd.read_csv('D:\COVID19-Transit_Bikesharing\Divvy_Data\Divvy_Station.csv')
All_Station_Need = All_Station[['id', 'lon', 'lat', 'capacity']]
All_Station_Need.columns = ['from_station_id', 'from_station_lon', 'from_station_lat', 'from_station_capacity']
Day_count = Day_count.merge(All_Station_Need, on='from_station_id')
Day_count = Day_count.sort_values(by=['from_station_id', 'startdate']).reset_index(drop=True)
Day_count['startdate'] = pd.to_datetime(Day_count['startdate'])

'''
# Merge with Weather
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


# Get the weather station info
Station_raw = pd.read_csv(r'D:\Transit\Weather\ghcnd-stations1.csv', header=None)
Station_raw = Station_raw.loc[:, 0:2]
Station_raw.columns = ['Sid', 'LAT', 'LON']
# Select the weather station close to transit stop
Need_Weather = []
for jj in range(0, len(All_Station)):
    # print(jj)
    tem = All_Station.loc[jj]
    Station_raw['Ref_Lat'] = tem['lat']
    Station_raw['Ref_Lng'] = tem['lon']
    Station_raw['Distance'] = haversine_array(Station_raw['Ref_Lat'], Station_raw['Ref_Lng'], Station_raw['LAT'],
                                              Station_raw['LON'])
    tem_id = list(Station_raw[Station_raw['Distance'] < 50]['Sid'])
    Need_Weather.extend(tem_id)
Need_Weather = set(Need_Weather)

## ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/
ALL_WEATHER = pd.DataFrame()
for eachyear in range(2013, 2021):
    print(eachyear)
    Weather_raw = pd.read_csv('D:\\Transit\\Weather\\' + str(eachyear) + '.csv.gz', header=None, compression='gzip')
    Weather_raw = Weather_raw.loc[:, 0:3]
    Weather_raw.columns = ['Sid', 'date', 'Type', 'Number']
    Weather_raw = Weather_raw[Weather_raw['Sid'].isin(Need_Weather)]
    PV_Weather = pd.pivot_table(Weather_raw, values='Number', index=['Sid', 'date'], columns=['Type']).reset_index()
    tem = PV_Weather.isnull().sum()
    PV_Weather = PV_Weather[['Sid', 'date', 'PRCP', 'TAVG', 'TMAX', 'TMIN']]
    # Find the nearest stations for each CT_Info
    All_Weather = pd.DataFrame()
    for jj in range(0, len(All_Station)):
        # print(jj)
        tem = All_Station.loc[jj]
        Station_raw['Ref_Lat'] = tem['lat']
        Station_raw['Ref_Lng'] = tem['lon']
        Station_raw['Distance'] = haversine_array(Station_raw['Ref_Lat'], Station_raw['Ref_Lng'], Station_raw['LAT'],
                                                  Station_raw['LON'])
        # sns.distplot(Station_raw['Distance'])
        tem_id = Station_raw[Station_raw['Distance'] < 20]['Sid']
        tem_weather_PRCP = PV_Weather[PV_Weather['Sid'].isin(tem_id)].groupby('date').mean()['PRCP'].reset_index()
        tem_id = Station_raw[Station_raw['Distance'] < 30]['Sid']
        tem_weather_T = PV_Weather[PV_Weather['Sid'].isin(tem_id)].groupby('date').mean()[
            ['TMAX', 'TMIN']].reset_index()
        tem_weather_PRCP = tem_weather_PRCP.merge(tem_weather_T, on='date', how='outer')
        tem_weather_PRCP['station_id'] = tem['id']
        All_Weather = All_Weather.append(tem_weather_PRCP)
    ALL_WEATHER = ALL_WEATHER.append(All_Weather)

# Unit: Precipitation (tenths of mm); Maximum temperature (tenths of degrees C)
ALL_WEATHER.isnull().sum()
ALL_WEATHER['TMAX'] = ALL_WEATHER['TMAX'].fillna(method='ffill').fillna(method='bfill')
ALL_WEATHER['TMIN'] = ALL_WEATHER['TMIN'].fillna(method='ffill').fillna(method='bfill')
ALL_WEATHER['PRCP'] = ALL_WEATHER['PRCP'].fillna(0)
# Change to mm and C
ALL_WEATHER['TMAX'] = ALL_WEATHER['TMAX'] * 0.1
ALL_WEATHER['TMIN'] = ALL_WEATHER['TMIN'] * 0.1
ALL_WEATHER['PRCP'] = ALL_WEATHER['PRCP'] * 0.1
# plt.plot(ALL_WEATHER['TMIN'], 'ok', alpha=0.2)
# plt.plot(All_Weather['PRCP'], 'ok', alpha=0.2)
ALL_WEATHER.to_csv('D:\COVID19-Transit_Bikesharing\Divvy_Data\All_Weather_Chicago_Divvy.csv')
'''

# Merge with Weather
ALL_WEATHER = pd.read_csv('D:\COVID19-Transit_Bikesharing\Divvy_Data\All_Weather_Chicago_Divvy.csv',
                          index_col=0).reset_index(drop=True)
ALL_WEATHER['date'] = pd.to_datetime(ALL_WEATHER['date'], format='%Y%m%d')
ALL_WEATHER.columns = ['startdate', 'PRCP', 'TMAX', 'TMIN', 'from_station_id']
ALL_WEATHER_Whole = ALL_WEATHER.groupby('startdate').median().reset_index()
ALL_WEATHER_Whole = ALL_WEATHER_Whole[['startdate', 'PRCP', 'TMAX', 'TMIN']]
ALL_WEATHER_Whole.columns = ['startdate', 'APRCP', 'ATMAX', 'ATMIN']
Day_count = Day_count.merge(ALL_WEATHER, on=['from_station_id', 'startdate'], how='left')
Day_count = Day_count.merge(ALL_WEATHER_Whole, on='startdate', how='left')
Day_count.isnull().sum()

# Is holidays
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start='2013-01-01', end='2020-08-01').to_pydatetime()
Day_count['Holidays'] = 0
Day_count.loc[Day_count['startdate'].isin(holidays), 'Holidays'] = 1

# Output to BSTS
Day_count.to_csv('Day_count_Divvy.csv')
Day_count.columns
# Total count
All_Day_count = Day_count.groupby(['startdate']).sum()['trip_id'].reset_index()
All_Others = Day_count.groupby(['startdate']).median()[['PRCP', 'TMAX', 'TMIN', 'Holidays']].reset_index()
All_Day_count = All_Day_count.merge(All_Others, on='startdate')
All_Day_count.to_csv('All_Day_count_Divvy.csv')
