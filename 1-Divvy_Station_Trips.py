import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
alltrips.to_pickle('alltrips_chicago_202007.pkl')
alltrips.to_csv('alltrips.chicago_202007.csv')
'''
os.chdir(r'D:\COVID19-Transit_Bikesharing\Divvy_Data')
alltrips = pd.read_pickle(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Trips_All\alltrips_chicago_202007.pkl')
raw_length = len(alltrips)

# Drop Outliers: Duration
alltrips['Duration'] = (alltrips['stoptime'] - alltrips['starttime']).dt.total_seconds()
alltrips = alltrips[(alltrips['Duration'] > 60) & (alltrips['Duration'] < 60 * 60 * 6)].reset_index(drop=True)
print('Delete: ' + str(raw_length - len(alltrips)))
# sns.distplot(alltrips['Duration'])
# Drop NA
alltrips = alltrips.dropna().reset_index(drop=True)

# Daily count
alltrips['startyear'] = alltrips['starttime'].dt.year
alltrips['startmonth'] = alltrips['starttime'].dt.month
alltrips['startdate'] = alltrips['starttime'].dt.date

# Merge with station info
# Only consider those with trips in 2020
print('No of Stations in 2020: ' + str(len(set(alltrips.loc[alltrips['startyear'] == 2020, 'from_station_id']))))
alltrips = alltrips[alltrips['from_station_id'].
    isin(set(alltrips.loc[alltrips['startyear'] == 2020, 'from_station_id']))].reset_index(drop=True)
# alltrips.isnull().sum()
Day_count = alltrips.groupby(['from_station_id', 'startdate']).count()['trip_id'].reset_index()
Day_count['startdate'] = pd.to_datetime(Day_count['startdate'])
# Range the date
Day_count = Day_count.set_index('startdate').groupby(['from_station_id']).resample('d')[
    ['trip_id']].asfreq().reset_index()
Day_count = Day_count.sort_values(by=['from_station_id', 'startdate'])
# Day_count.isnull().sum()
Day_count = Day_count.fillna(0)

# Merge with stations
All_Station = pd.read_csv('D:\COVID19-Transit_Bikesharing\Divvy_Data\Divvy_Station.csv')
All_Station_Need = All_Station[['id', 'lon', 'lat', 'capacity']]
All_Station_Need.columns = ['from_station_id', 'from_station_lon', 'from_station_lat', 'from_station_capacity']
Day_count = Day_count.merge(All_Station_Need, on='from_station_id')
Day_count = Day_count.sort_values(by=['from_station_id', 'startdate']).reset_index(drop=True)

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
holidays = USFederalHolidayCalendar().holidays(start='2013-01-01', end='2020-08-01').to_pydatetime()
Day_count['Holidays'] = 0
Day_count.loc[Day_count['startdate'].isin(holidays), 'Holidays'] = 1

# Output: Daily Counts for each station
Day_count.to_csv('D:\COVID19-Transit_Bikesharing\Divvy_Data\Day_count_Divvy.csv')
# Output: Daily Counts for Each day
All_Day_count = Day_count.groupby(['startdate']).sum()['trip_id'].reset_index()
All_Others = Day_count.groupby(['startdate']).median()[['PRCP', 'TMAX', 'TMIN', 'Holidays']].reset_index()
All_Day_count = All_Day_count.merge(All_Others, on='startdate')
All_Day_count.to_csv('D:\COVID19-Transit_Bikesharing\Divvy_Data\All_Day_count_Divvy.csv')

'''
# Plot the figure for each station
Day_count = pd.read_csv('D:\COVID19-Transit_Bikesharing\Divvy_Data\Day_count_Divvy.csv', index_col=0)
Day_count['startdate'] = pd.to_datetime(Day_count['startdate'])
# All_Day_count.columns
for jj in list(set(Day_count['from_station_id'])):
    tem = Day_count[Day_count['from_station_id'] == jj]
    tem = tem.set_index('startdate')
    # Find
    fig, ax = plt.subplots(figsize=(12, 6), nrows=1, ncols=1)
    ax.plot(tem['trip_id'], color='k')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=20))
    plt.tight_layout()
    plt.savefig('D:\\COVID19-Transit_Bikesharing\\Divvy_Data\\Time-series-All\\' + str(jj) + '.png')
    plt.close()
'''

# Some Stations should be dropped: Based on their time series
Not_Need_ID = [95, 102, 270, 356, 384, 386, 388, 390, 391, 392, 393, 395, 396, 398, 399, 400, 407, 408, 409, 411, 412,
               421, 426, 427, 429, 430, 431, 433, 435, 436, 437, 438, 439, 440, 441, 443, 444, 445, 446, 524] \
              + list(range(528, 589)) \
              + [593, 594, 595, 559, 564, 567, 570, 571, 572, 574, 576, 579, 580, 583, 585, 588, 642, 646, 647, 648,
                 649, 650, 652, 653, 665, 674, 677, 678, 679, 681, 683, 666, 673, 672, 662, 661]
len(Not_Need_ID)
Day_count = Day_count[~Day_count['from_station_id'].isin(Not_Need_ID)].reset_index(drop=True)
Day_count.to_csv('D:\COVID19-Transit_Bikesharing\Divvy_Data\Day_count_Divvy_dropOutlier.csv')

# Read cases
cases = pd.read_csv(r'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv')
cases = cases[(cases['county'] == 'Cook') & (cases['state'] == 'Illinois')].reset_index(drop=True)
cases['date'] = pd.to_datetime(cases['date'])
cases.set_index('date', inplace=True)
cases['cases'] = cases['cases'].diff()
cases = cases.fillna(0)

# Plot the daily figure
All_ride = pd.read_csv('D:\COVID19-Transit_Bikesharing\Divvy_Data\All_Day_count_Divvy.csv')
All_ride['startdate'] = pd.to_datetime(All_ride['startdate'])
Rider_2019 = All_ride[(All_ride['startdate'] < datetime.datetime(2020, 1, 1)) & (
        All_ride['startdate'] >= datetime.datetime(2019, 1, 1))]
Rider_2019['startdate'] = Rider_2019['startdate'] + datetime.timedelta(days=365)
All_ride.set_index('startdate', inplace=True)
Rider_2019.set_index('startdate', inplace=True)

# Plot the daily figure
plt.rcParams.update({'font.size': 22, 'font.family': "Times New Roman"})
fig, ax = plt.subplots(figsize=(14, 8))  # create a new figure with a default 111 subplot
ax.plot(All_ride['trip_id'], color='#1b96f3', alpha=0.8, lw=1)
ax.set_ylabel('Pickups')
ax.set_xlabel('Date')
ax.set_ylim(0, 3.5 * 1e4)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)

axins = inset_axes(ax, 8, 1.8, loc=9)
axins.plot(Rider_2019['trip_id'], '--', color='#1b96f3')
axins.plot(All_ride['trip_id'], color='#1b96f3')
axins.set_xlim(datetime.datetime(2020, 2, 1), datetime.datetime(2020, 7, 30))
axins.spines['top'].set_visible(False)
axins.spines['right'].set_visible(False)
axins.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
axins.set_ylabel('Pickups')
axins.legend(['2019', '2020'], frameon=False)

axtwins = axins.twinx()
axtwins.yaxis.set_offset_position('right')
axtwins.bar(cases.index, cases['cases'], color='#869ba0', alpha=0.5)
axtwins.set_ylim(0, 2500)
axtwins.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
axtwins.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
axtwins.set_ylabel('Cases')
mark_inset(ax, axins, loc1=1, loc2=1, fc="none", ec="#869ba0", lw=2, ls='--')
plt.subplots_adjust(top=0.951, bottom=0.088, left=0.067, right=0.987, hspace=0.225, wspace=0.2)

plt.savefig(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Results\FIG1.png', dpi=600)
plt.savefig(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Results\FIG1.svg')

# For ARCGIS
Day_count = pd.read_csv('D:\COVID19-Transit_Bikesharing\Divvy_Data\Day_count_Divvy_dropOutlier.csv')
Day_count['startdate'] = pd.to_datetime(Day_count['startdate'])
Day_count_2019 = Day_count[Day_count['startdate'].dt.year == 2019]
Day_count_2019.columns
Day_count_2019 = Day_count_2019.groupby(['from_station_id']).mean()[
    ['trip_id', 'from_station_lon', 'from_station_lat', 'from_station_capacity']].reset_index()
Day_count_2019.to_csv('D:\COVID19-Transit_Bikesharing\Divvy_Data\Day_count_2019.csv')
