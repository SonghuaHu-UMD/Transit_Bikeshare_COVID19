import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pylab as plt
import matplotlib.dates as mdates
import seaborn as sns
import geopandas as gpd
from scipy.stats import pearsonr
import scipy.stats
from scipy import stats

os.chdir(r'D:\COVID19-Transit_Bikesharing\Divvy_Data')
plt.rcParams.update({'font.size': 24, 'font.family': "Times New Roman"})

# Calculate the impact from last year
# _dropOutlier
ridership_old = pd.read_csv(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Day_count_Divvy_dropOutlier.csv', index_col=0)
ridership_old.columns
ridership_old['startdate'] = pd.to_datetime(ridership_old['startdate'])

# Calculate the direct decrease compared with 2019
Rider_2019 = ridership_old[(ridership_old['startdate'] <= datetime.datetime(2019, 12, 31)) & (
        ridership_old['startdate'] >= datetime.datetime(2019, 1, 1))]
Rider_2019['Month'] = Rider_2019.startdate.dt.month
Rider_2019['Day'] = Rider_2019.startdate.dt.day
Rider_2019.columns
# Rider_2019['Week'] = Rider_2019.startdate.dt.dayofweek
Rider_2019 = Rider_2019[['from_station_id', 'trip_id', 'Month', 'Day', 'PRCP', 'TMAX']]
# Rider_2019_mean = Rider_2019[['from_station_id', 'trip_id', 'Month']].groupby(
#     ['from_station_id', 'Month', 'Week']).agg(lambda x: stats.trim_mean(x, 0.1)).reset_index()
# Rider_2019_mean.columns = ['stationid', 'Month', 'Reference']
Rider_2019.columns = ['stationid', 'Reference', 'Month', 'Day', 'PRCP_2019', 'TMAX_2019']
Rider_2019_avg = Rider_2019.groupby('stationid').agg(lambda x: stats.trim_mean(x, 0.05))['Reference'].reset_index()
Rider_2019_avg.columns = ['stationid', '2019_Avg']
Rider_2019 = Rider_2019.merge(Rider_2019_avg, on='stationid')

# In 2020
Rider_2020 = ridership_old[(ridership_old['startdate'] < datetime.datetime(2020, 8, 1)) & (
        ridership_old['startdate'] >= datetime.datetime(2020, 1, 1))]
Rider_2020['Month'] = Rider_2020.startdate.dt.month
Rider_2020['Day'] = Rider_2020.startdate.dt.day
Rider_2020['Week'] = Rider_2020.startdate.dt.dayofweek
Rider_2020 = Rider_2020[['from_station_id', 'trip_id', 'Month', 'Week', 'Day', 'startdate', 'PRCP', 'TMAX']]
Rider_2020.columns = ['stationid', 'Response', 'Month', 'Week', 'Day', 'Date', 'PRCP_2020', 'TMAX_2020']

# Rider_2020 = Rider_2020.merge(Rider_2019_mean, on=['stationid', 'Month', 'Week'])
Rider_2020 = Rider_2020.merge(Rider_2019, on=['stationid', 'Month', 'Day'])
Rider_2020.columns
Rider_2020['point.effect'] = Rider_2020['Response'] - Rider_2020['Reference']
Rider_2020['Relative_Impact'] = (Rider_2020['Response'] - Rider_2020['Reference']) / Rider_2020['Reference']
Rider_2020['Cum_effect'] = Rider_2020.groupby(['stationid'])['point.effect'].cumsum()
Rider_2020['Cum_Response'] = Rider_2020.groupby(['stationid'])['Response'].cumsum()
Rider_2020['Cum_Reference'] = Rider_2020.groupby(['stationid'])['Reference'].cumsum()
Rider_2020['Cum_Relative_Impact'] = (Rider_2020['Cum_Response'] - Rider_2020['Cum_Reference']) / Rider_2020[
    'Cum_Reference']
Rider_2020['PRCP'] = Rider_2020['PRCP_2020'] - Rider_2020['PRCP_2019']
Rider_2020['TMAX'] = Rider_2020['TMAX_2020'] - Rider_2020['TMAX_2019']

# Plot relative impact
Impact_0101 = Rider_2020[(~Rider_2020['stationid'].isin(
    set(Rider_2020[(Rider_2020['Cum_Relative_Impact'] > 3) & (Rider_2020['Month'] > 1)]['stationid']))) & (
                                 Rider_2020['Month'] > 1)]
print(len(set(Impact_0101['stationid'])))
print(len(set(Rider_2020['stationid'])))
sns.set_palette(sns.color_palette("GnBu_d"))
plt.rcParams.update({'font.size': 18, 'font.family': "Times New Roman"})
fig, ax = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
sns.lineplot(data=Impact_0101, x='Date', hue='stationid', y='Cum_Relative_Impact', legend=False, ax=ax,
             palette=sns.color_palette("YlGnBu", Impact_0101.stationid.unique().shape[0]), alpha=0.1)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
ax.plot(Impact_0101.groupby('Date').mean()['Cum_Relative_Impact'], color='#eab354', lw=2, label='Mean')
ax.plot(Impact_0101.groupby('Date').median()['Cum_Relative_Impact'], color='k', lw=2, label='Median')
ax.plot([datetime.datetime(2020, 3, 11), datetime.datetime(2020, 3, 11)], [-0.8, 3], '--',
        color='#2f4c58', lw=2)
plt.text(0.25, 0.06, 'WHO Pandemic Claim', horizontalalignment='left', verticalalignment='center',
         transform=ax.transAxes)
ax.legend(frameon=False, ncol=2)
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative relative change')
plt.tight_layout()
plt.savefig('D:\COVID19-Transit_Bikesharing\Divvy_Data\Results\FIG3.png', dpi=600)
plt.savefig('D:\COVID19-Transit_Bikesharing\Divvy_Data\Results\FIG3.svg')

'''
for jj in list(set(Rider_2020['stationid'])):
    # jj = 3
    tem = Rider_2020[Rider_2020['stationid'] == jj]
    tem = tem.set_index('Date')
    # Find
    fig, ax = plt.subplots(figsize=(12, 6), nrows=2, ncols=1)
    ax[0].plot(tem['Response'], '--', color='k')
    ax[0].plot(tem['Predict'], color='g')
    ax[1].plot(tem['Cum_Relative_Impact'])
    plt.tight_layout()
    plt.savefig('D:\\COVID19-Transit_Bikesharing\\Divvy_Data\\Cum_2019\\' + str(jj) + '.png')
    plt.close()
'''

# To GAM in R
Rider_2020_New = Impact_0101
[['stationid', 'Response', 'Month', 'Day', 'Date', 'Reference', 'point.effect', 'Relative_Impact', 'Cum_effect',
  'Cum_Response', 'Cum_Reference', 'Cum_Relative_Impact']]
Rider_2020_New = Rider_2020_New.rename({'stationid': 'from_stati'}, axis=1)
Rider_2020_New = Rider_2020_New.replace([np.inf, -np.inf], np.nan)

# Merge with features
All_final = pd.read_csv('D:\COVID19-Transit_Bikesharing\Divvy_Data\Features_Divvy_0906.csv', index_col=0)
All_final = All_final.merge(Rider_2020_New, on='from_stati')
# Lat Lon Capacity
All_Station = pd.read_csv('D:\COVID19-Transit_Bikesharing\Divvy_Data\Divvy_Station.csv')
All_Station_Need = All_Station[['id', 'lon', 'lat', 'capacity']]
All_Station_Need.columns = ['from_stati', 'lon', 'lat', 'capacity']
All_final = All_final.merge(All_Station_Need, on='from_stati')
All_final['Time_Index'] = (All_final['Date'] - datetime.datetime(2020, 3, 12)).dt.days
All_final.isnull().sum()[All_final.isnull().sum() > 0]
All_final = All_final.fillna(0)
All_final = All_final.rename({'Pct.Male': 'Prop.Male', 'Pct.Age_0_24': 'Prop.Age_0_24'})
All_final.to_csv(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\All_final_Divvy_R2019_1005.csv')

# annot=corr_p.values,
All_final1 = All_final.groupby(['from_stati']).tail(1)
All_final1.columns
corr_matr = All_final1[
    ['Pct.Male', 'Pct.Age_0_24', 'Pct.Age_25_40', 'Pct.Age_40_65', 'Pct.White', 'Pct.Black', 'Pct.Indian', 'Pct.Asian',
     'Pct.Unemploy', 'Total_Population', 'Income', 'College', 'Pct.Car', 'Pct.Transit', 'Pct.Bicycle', 'Pct.Walk',
     'Pct.WorkHome', 'Cumu_Cases', 'Cumu_Death', 'Cumu_Cases_Rate', 'Cumu_Death_Rate', 'COMMERCIAL',
     'INDUSTRIAL', 'INSTITUTIONAL', 'OPENSPACE', 'OTHERS', 'RESIDENTIAL',
     'Primary', 'Secondary', 'Minor', 'All_Road_Length', 'Bike_Route', 'Pct.WJob_Goods_Product', 'Pct.WJob_Utilities',
     'Pct.WJob_OtherServices', 'WTotal_Job_Density', 'Bus_stop_count', 'boardings', 'alightings', 'Distance_Busstop',
     'Rail_stop_count', 'rides', 'Distance_Rail', 'Near_Bike_station_Count', 'Near_Bike_Capacity',
     'Distance_Bikestation', 'Near_bike_pickups', 'Distance_City', 'PopDensity', 'EmployDensity',
     'Response', 'Cum_Relative_Impact', 'Relative_Impact', 'capacity', ]]
fig, ax = plt.subplots(figsize=(11, 9))
plt.rcParams.update({'font.size': 10, 'font.family': "Times New Roman"})
sns.heatmap(All_final.corr(), fmt='',
            cmap=sns.diverging_palette(240, 130, as_cmap=True),
            square=True, xticklabels=True, yticklabels=True, linewidths=.5)
plt.tight_layout()
plt.savefig('CORR_Divvy.svg')
