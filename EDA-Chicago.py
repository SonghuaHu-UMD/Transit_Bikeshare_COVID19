import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 各年提供的网点数据
os.chdir(r'C:\Users\hsonghua\Desktop\Chiago\data\Stations')
# 2017年的网点
shop_2017 = pd.read_csv('Divvy_Stations_2017_Q3Q4.csv')
shop_2017['online_date'] = pd.to_datetime(shop_2017['online_date'])
shop_2017['online_year'] = shop_2017.online_date.dt.year
# shop_2017.to_csv('shop_2017_withyear.csv')
shop_2017.groupby('online_year').count()['id']
sns.lmplot('longitude', 'latitude', data=shop_2017, hue='online_year', fit_reg=False)

# 读取订单数据
os.chdir(r'C:\Users\hsonghua\Desktop\Chiago\data')
alltrips = pd.read_csv(r'C:\Users\hsonghua\Desktop\Chiago\data\alltrips_Divvy.csv', index_col=0)
# 时间格式存在问题 Two format exist
alltrips['starttime1'] = alltrips['starttime'].str.replace(' ', '/')
alltrips['starttime1'] = alltrips['starttime1'].str.replace('-', '/')
alltrips['starttime1'] = alltrips['starttime1'].str.replace(':', '/')
alltrips = alltrips.reset_index(drop=True)
tem = pd.DataFrame(alltrips.starttime1.str.split('/').tolist(),
                   columns=['year0', 'month0', 'date0', 'hour0', 'minute0', 'second0'])

# get year
tem['year1'] = '-1'
tem.loc[tem['year0'].isin(['2013', '2014', '2015', '2016', '2017', '2018', '2019']), 'year1'] = tem.loc[
    tem['year0'].isin(['2013', '2014', '2015', '2016', '2017', '2018', '2019']), 'year0']
tem.loc[tem['date0'].isin(['2013', '2014', '2015', '2016', '2017', '2018', '2019']), 'year1'] = tem.loc[
    tem['date0'].isin(['2013', '2014', '2015', '2016', '2017', '2018', '2019']), 'date0']
alltrips['startyear'] = tem['year1']
# get month
tem['month1'] = '-1'
tem.loc[tem['year0'].isin(['2013', '2014', '2015', '2016', '2017', '2018', '2019']), 'month1'] = tem.loc[
    tem['year0'].isin(['2013', '2014', '2015', '2016', '2017', '2018', '2019']), 'month0']
tem.loc[tem['date0'].isin(['2013', '2014', '2015', '2016', '2017', '2018', '2019']), 'month1'] = tem.loc[
    tem['date0'].isin(['2013', '2014', '2015', '2016', '2017', '2018', '2019']), 'year0']
# tem.groupby('month1').count()['year0']
alltrips['startmonth'] = tem['month1']
# get date
tem['date1'] = '-1'
tem.loc[tem['year0'].isin(['2013', '2014', '2015', '2016', '2017', '2018', '2019']), 'date1'] = tem.loc[
    tem['year0'].isin(['2013', '2014', '2015', '2016', '2017', '2018', '2019']), 'date0']
tem.loc[tem['date0'].isin(['2013', '2014', '2015', '2016', '2017', '2018', '2019']), 'date1'] = tem.loc[
    tem['date0'].isin(['2013', '2014', '2015', '2016', '2017', '2018', '2019']), 'month0']
# tem.groupby('date1').count()['year0']
alltrips['startdate'] = tem['date1']
# get time
alltrips['hour'] = tem['hour0']
alltrips['minute'] = tem['minute0']

alltrips.groupby('startyear').count()['hour']

# 统一转化为int
alltrips['startyear'] = alltrips['startyear'].astype(int)
alltrips['startmonth'] = alltrips['startmonth'].astype(int)
alltrips['startdate'] = alltrips['startdate'].astype(int)
alltrips['hour'] = alltrips['hour'].astype(int)
alltrips['minute'] = alltrips['minute'].astype(int)
alltrips['tripduration'] = alltrips['tripduration'].str.replace(',', '')
alltrips['tripduration'] = alltrips['tripduration'].astype(float)
alltrips.to_csv('alltrips_havetime.csv')

# 统计
os.chdir(r'C:\Users\Songhua Hu\Desktop\chicago land use\data\data')
alltrips = pd.read_csv('alltrips_havetime.csv', index_col=0)
shop_2017 = pd.read_csv('Divvy_Stations_2017_Q3Q4.csv')
shop_2017['online_date'] = pd.to_datetime(shop_2017['online_date'])
shop_2017['online_year'] = shop_2017.online_date.dt.year

# 各站点的日均取车量
alltrips['start_date'] = alltrips['startyear'].astype('str') + '-' + alltrips['startmonth'].astype('str') + '-' + \
                         alltrips['startdate'].astype('str')
alltrips_pick_day = alltrips.groupby('from_station_id').count()['trip_id'] / alltrips.groupby(
    'from_station_id').start_date.nunique()
alltrips_pick_day = alltrips_pick_day.reset_index()
alltrips_pick_day.columns = ['id', 'pickups']
shop_need = shop_2017[['id', 'latitude', 'longitude', 'dpcapacity', 'online_date', 'online_year']]
alltrips_pick_day = alltrips_pick_day.merge(shop_need, on='id')
# 各站点的日均还车量
alltrips_return_day = alltrips.groupby('to_station_id').count()['trip_id'] / alltrips.groupby(
    'to_station_id').start_date.nunique()
alltrips_return_day = alltrips_return_day.reset_index()
alltrips_return_day.columns = ['id', 'returns']
alltrips_pick_day = alltrips_pick_day.merge(alltrips_return_day, on='id')
alltrips_pick_day.to_csv('alltrips_pick_day.csv')
# imbalance
alltrips_pick_day['imbalance'] = alltrips_pick_day['pickups'] / alltrips_pick_day['returns']
sns.distplot(alltrips_pick_day['imbalance'])
# 骑行时间
sns.distplot(alltrips['tripduration'][(~alltrips['tripduration'].isna()) & (alltrips['tripduration'] < 10000)],
             color='grey')
alltrips['tripduration'][(~alltrips['tripduration'].isna()) & (alltrips['tripduration'] < 10000)].describe()

# 骑行频率时变图
# 各年的变化
alltrips.groupby('startyear').count()['trip_id'].plot()
# 各月的变化
alltrips.groupby(['startmonth']).count()['trip_id'].plot(marker='o')
# 各小时的变化
alltrips.groupby('hour').count()['trip_id'].plot(marker='o')

# 骑行频率空间分布图
Trips_Sta_Fre = alltrips.groupby(['from_station_id', 'to_station_id']).count()['trip_id'].reset_index()
shop_need = shop_2017[['id', 'latitude', 'longitude', 'dpcapacity', 'online_date', 'online_year']]
shop_need.columns = ['from_station_id', 'latitude', 'longitude', 'dpcapacity', 'online_date', 'online_year']
Trips_Sta_Fre = Trips_Sta_Fre.merge(shop_need, on='from_station_id')
shop_need.columns = ['to_station_id', 'latitude', 'longitude', 'dpcapacity', 'online_date', 'online_year']
Trips_Sta_Fre = Trips_Sta_Fre.merge(shop_need, on='to_station_id')
Trips_Sta_Fre1 = Trips_Sta_Fre[Trips_Sta_Fre['trip_id'] > 10].reset_index(drop=True)
# plt.rcParams.update({'font.size': 16})
alltrips.groupby(['from_station_id']).count()['trip_id'].plot()

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 9))
tem = ax.scatter(Trips_Sta_Fre1['longitude_x'], Trips_Sta_Fre1['latitude_x'], s=Trips_Sta_Fre1['trip_id'] / 80,
                 c=Trips_Sta_Fre1['trip_id'], cmap=cm.Set3)
Trips_Sta_Fre2 = Trips_Sta_Fre1[Trips_Sta_Fre1['trip_id'] > 1000].reset_index(drop=True)
for kk in range(0, len(Trips_Sta_Fre2)):
    ax.annotate('',
                xy=(Trips_Sta_Fre2.loc[kk, 'longitude_x'], Trips_Sta_Fre2.loc[kk, 'latitude_x']),
                xytext=(Trips_Sta_Fre2.loc[kk, 'longitude_y'], Trips_Sta_Fre2.loc[kk, 'latitude_y']),
                arrowprops={'arrowstyle': '->', 'lw': Trips_Sta_Fre2.loc[kk, 'trip_id'] / 10844,
                            'alpha': 0.5}, va='center')
fig.colorbar(tem, ax=ax)

plt.plot(Trips_Sta_Fre['trip_id'])

Trips_Sta_Fre = Trips_Sta_Fre.sort_values(by='trip_id').reset_index(drop=True)

# BUS ROUTE
Bus_volume = pd.read_csv(
    r'C:\Users\Songhua Hu\Desktop\chicago land use\CTA_-_Ridership_-_Bus_Routes_-_Daily_Totals_by_Route.csv')
Bus_volumeD = Bus_volume.groupby('route').mean()['rides'].reset_index()
Bus_volumeD.to_csv('Bus_volumeD.csv')

# L station
Metro_volume = pd.read_csv(
    r'C:\Users\Songhua Hu\Desktop\chicago land use\CTA_-_Ridership_-__L__Station_Entries_-_Daily_Totals.csv')
Metro_volumeD = Metro_volume.groupby('station_id').mean()['rides'].reset_index()
Metro_volumeD.to_csv('Metro_volumeD.csv')

# CRIME
crime_data = pd.read_csv(r'C:\Users\Songhua Hu\Desktop\chicago land use\Crimes_-_2001_to_present.csv')
crime_data['Year'] = [var.split('/')[2].split(' ')[0] for var in crime_data['Date']]
crime_data1 = crime_data[crime_data['Year'] > '2012']
crime_data1.to_csv('crime_data_2013.csv')
