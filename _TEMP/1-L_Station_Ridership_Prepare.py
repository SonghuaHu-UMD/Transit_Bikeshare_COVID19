import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pylab as plt
from scipy import stats
from pandas.tseries.holiday import USFederalHolidayCalendar
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.dates as mdates

os.chdir(r'D:\Transit')


def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


# Read the L station ridership
Daily_Lstaion = pd.read_csv(r'CTA_-_Ridership_-__L__Station_Entries_-_Daily_Totals.csv')
Daily_Lstaion['date'] = pd.to_datetime(Daily_Lstaion['date'])
Daily_Lstaion = Daily_Lstaion.sort_values(by=['station_id', 'date']).reset_index(drop=True)
Daily_Lstaion['Year'] = Daily_Lstaion.date.dt.year

# Daily_Lstaion.groupby('Year').sum()['rides'].plot()
# Only need after 2015
# Daily_Lstaion = Daily_Lstaion[Daily_Lstaion['Year'] >= 2015].reset_index(drop=True)
Daily_Lstaion = Daily_Lstaion.drop_duplicates(subset=['station_id', 'date'])
# Some stations drop before 2020
Daily_Lstaion = Daily_Lstaion[
    Daily_Lstaion['station_id'].isin(set(Daily_Lstaion[Daily_Lstaion['Year'] == 2020]['station_id']))].reset_index(
    drop=True)
print(len(set(Daily_Lstaion['station_id'])))

# Range the date
Daily_Lstaion = Daily_Lstaion.set_index('date').groupby(['station_id']).resample('d')[
    ['rides', 'daytype']].asfreq().reset_index()
Daily_Lstaion = Daily_Lstaion.sort_values(by=['station_id', 'date'])


# Read cases
cases = pd.read_csv(r'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv')
cases_0430 = cases[cases['date'] == '2020-04-30']
cases_0430.to_csv('cases_0430.csv')
cases = cases[(cases['county'] == 'Cook') & (cases['state'] == 'Illinois')].reset_index(drop=True)
cases['date'] = pd.to_datetime(cases['date'])
cases.set_index('date', inplace=True)
cases['cases'] = cases['cases'].diff()
cases = cases.fillna(0)
# cases.to_csv('cases_chicago.csv')
'''
# Plot time-varying figure
All_ride = Daily_Lstaion.groupby('date').sum()['rides'].reset_index()
All_ride.set_index('date', inplace=True)
myFmt = mdates.DateFormatter('%b-%d')
plt.rcParams.update({'font.size': 24, 'font.family': "Times New Roman"})
fig, ax = plt.subplots(figsize=(14, 9))  # create a new figure with a default 111 subplot
ax.plot(All_ride['rides'], color='#8fc0a9', alpha=0.8, lw=1)
ax.set_ylabel('Ridership')
ax.set_xlabel('Date')
ax.set_ylim(10000, 1000000)
ax.set_xlim(datetime.datetime(2001, 1, 1), datetime.datetime(2020, 4, 30))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)

axins = inset_axes(ax, 10, 1.8, loc=9)
axins.plot(All_ride['rides'], color='#111b1e')
axins.set_xlim(datetime.datetime(2020, 1, 1), datetime.datetime(2020, 4, 30))
axins.set_ylim(10000, 680000)
axins.spines['top'].set_visible(False)
axins.spines['right'].set_visible(False)
axins.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
axins.set_ylabel('Ridership')

axtwins = axins.twinx()
axtwins.yaxis.set_offset_position('right')
axtwins.bar(cases.index, cases['cases'], color='#111b1e', alpha=0.5)
axtwins.set_ylim(0, 2500)
axtwins.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
axtwins.xaxis.set_major_formatter(myFmt)
axtwins.set_ylabel('Cases')

# plt.yticks(visible=False)
# plt.xticks(visible=False)
mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="#ff6d69", lw=2, ls='--')
# plt.tight_layout()
plt.subplots_adjust(top=0.951, bottom=0.088, left=0.067, right=0.987, hspace=0.225, wspace=0.2)
plt.savefig('FIG1.png', dpi=600)
plt.savefig('FIG1.svg')
'''
# Merge with weather and holidays
# W=Weekday, A=Saturday, U=Sunday/Holiday
Daily_Lstaion['Week'] = Daily_Lstaion['date'].dt.dayofweek
Daily_Lstaion['IsWeekend'] = (Daily_Lstaion['Week'].isin([5, 6])).astype(int)
Daily_Lstaion['Holidays'] = 0
Daily_Lstaion.loc[(Daily_Lstaion['IsWeekend'] == 0) & (Daily_Lstaion['daytype'] != 'W'), 'Holidays'] = 1

# Merge with Station Location
Stations = pd.read_csv(r'CTA_-_System_Information_-_List_of__L__Stops.csv')
Stations = Stations[['MAP_ID', 'Location']]
Stations.columns = ['station_id', 'Location']
Stations['Location'] = Stations['Location'].str.replace('(', ',')
Stations['Location'] = Stations['Location'].str.replace(')', ',')
Stations['LAT'] = [var.split(',')[1] for var in Stations['Location']]
Stations['LNG'] = [var.split(',')[2] for var in Stations['Location']]
Stations = Stations[['station_id', 'LAT', 'LNG']]
Stations['LAT'] = Stations['LAT'].astype(float)
Stations['LNG'] = Stations['LNG'].astype(float)
Stations = Stations.drop_duplicates(subset='station_id').reset_index(drop=True)

'''
# Merge with weather
# Get the weather station info
Station_raw = pd.read_csv(r'Weather\ghcnd-stations1.csv', header=None)
Station_raw = Station_raw.loc[:, 0:2]
Station_raw.columns = ['Sid', 'LAT', 'LON']
# Select the weather station close to transit stop
Need_Weather = []
for jj in range(0, len(Stations)):
    # print(jj)
    tem = Stations.loc[jj]
    Station_raw['Ref_Lat'] = tem['LAT']
    Station_raw['Ref_Lng'] = tem['LNG']
    Station_raw['Distance'] = haversine_array(Station_raw['Ref_Lat'], Station_raw['Ref_Lng'], Station_raw['LAT'],
                                              Station_raw['LON'])
    tem_id = list(Station_raw[Station_raw['Distance'] < 50]['Sid'])
    Need_Weather.extend(tem_id)
Need_Weather = set(Need_Weather)

## ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/
ALL_WEATHER = pd.DataFrame()
for eachyear in range(2001, 2021):
    print(eachyear)
    Weather_raw = pd.read_csv('Weather\\' + str(eachyear) + '.csv.gz', header=None, compression='gzip')
    Weather_raw = Weather_raw.loc[:, 0:3]
    Weather_raw.columns = ['Sid', 'date', 'Type', 'Number']
    Weather_raw = Weather_raw[Weather_raw['Sid'].isin(Need_Weather)]
    PV_Weather = pd.pivot_table(Weather_raw, values='Number', index=['Sid', 'date'], columns=['Type']).reset_index()
    tem = PV_Weather.isnull().sum()
    PV_Weather = PV_Weather[['Sid', 'date', 'PRCP', 'TAVG', 'TMAX', 'TMIN']]
    # Find the nearest stations for each CT_Info
    All_Weather = pd.DataFrame()
    for jj in range(0, len(Stations)):
        # print(jj)
        tem = Stations.loc[jj]
        Station_raw['Ref_Lat'] = tem['LAT']
        Station_raw['Ref_Lng'] = tem['LNG']
        Station_raw['Distance'] = haversine_array(Station_raw['Ref_Lat'], Station_raw['Ref_Lng'], Station_raw['LAT'],
                                                  Station_raw['LON'])
        # sns.distplot(Station_raw['Distance'])
        tem_id = Station_raw[Station_raw['Distance'] < 20]['Sid']
        tem_weather_PRCP = PV_Weather[PV_Weather['Sid'].isin(tem_id)].groupby('date').mean()['PRCP'].reset_index()
        tem_id = Station_raw[Station_raw['Distance'] < 30]['Sid']
        tem_weather_T = PV_Weather[PV_Weather['Sid'].isin(tem_id)].groupby('date').mean()[
            ['TMAX', 'TMIN']].reset_index()
        tem_weather_PRCP = tem_weather_PRCP.merge(tem_weather_T, on='date', how='outer')
        tem_weather_PRCP['station_id'] = tem['station_id']
        All_Weather = All_Weather.append(tem_weather_PRCP)
    ALL_WEATHER = ALL_WEATHER.append(All_Weather)

# Unit: Precipitation (tenths of mm); Maximum temperature (tenths of degrees C)
ALL_WEATHER.isnull().sum()
ALL_WEATHER['TMAX'] = ALL_WEATHER['TMAX'].fillna(method='ffill').fillna(method='bfill')
ALL_WEATHER['TMIN'] = ALL_WEATHER['TMIN'].fillna(method='ffill').fillna(method='bfill')
ALL_WEATHER['PRCP'] = ALL_WEATHER['PRCP'].fillna(0)
# ALL_WEATHER = ALL_WEATHER.groupby('station_id')[['TMAX', 'TMIN', 'PRCP']].fillna(method='ffill').fillna(method='bfill')
# Change to mm and C
ALL_WEATHER['TMAX'] = ALL_WEATHER['TMAX'] * 0.1
ALL_WEATHER['TMIN'] = ALL_WEATHER['TMIN'] * 0.1
ALL_WEATHER['PRCP'] = ALL_WEATHER['PRCP'] * 0.1
# plt.plot(ALL_WEATHER['TMIN'], 'ok', alpha=0.2)
# plt.plot(All_Weather['PRCP'], 'ok', alpha=0.2)
ALL_WEATHER.to_csv('All_Weather_2001_2020.csv')
'''
ALL_WEATHER = pd.read_csv(r'All_Weather_2001_2020.csv', index_col=0)
ALL_WEATHER['date'] = pd.to_datetime(ALL_WEATHER['date'], format='%Y%m%d')

# Merge with weather
Daily_Lstaion_Final = Daily_Lstaion.merge(ALL_WEATHER, on=['station_id', 'date'], how='left')
Daily_Lstaion_Final = Daily_Lstaion_Final.fillna(0)
Daily_Lstaion_Final['Year'] = Daily_Lstaion_Final['date'].dt.year
Daily_Lstaion_Final.isnull().sum()

Daily_Lstaion_Final[
    ['station_id', 'date', 'daytype', 'rides', 'Week', 'IsWeekend', 'Holidays', 'PRCP', 'TMAX', 'TMIN']].to_csv(
    'Daily_Lstaion_Final_0806.csv', index=False)

# Output for arcgis
Count_sta = \
    Daily_Lstaion_Final[(Daily_Lstaion_Final['Year'] == 2019) & (Daily_Lstaion_Final['daytype'] == 'W')].groupby(
        ['station_id']).mean()[['rides']].reset_index()
Stations = Stations.merge(Count_sta, on='station_id')
Stations.to_csv('LStations_Chicago_W.csv')

# Daily_Lstaion_Final = pd.read_csv(r'Daily_Lstaion_Final_0806.csv')
# Daily_Lstaion_Final.describe().T.to_csv('Desc_BSTS.csv')
