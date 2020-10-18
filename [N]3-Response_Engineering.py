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

# Calculate 2019 average during 03/11 to 07/31
Rider_2019_AVG = ridership_old[(ridership_old['startdate'] <= datetime.datetime(2019, 7, 31)) & (
        ridership_old['startdate'] >= datetime.datetime(2019, 3, 11))]
Rider_2019_avg = Rider_2019_AVG.groupby('from_station_id').agg(lambda x: stats.trim_mean(x, 0.05))[
    'trip_id'].reset_index()
Rider_2019_avg.columns = ['stationid', '2019_Avg']

# Calculate 2020 average during 03/11 to 07/31
Rider_2020_AVG = ridership_old[(ridership_old['startdate'] <= datetime.datetime(2020, 7, 31)) & (
        ridership_old['startdate'] >= datetime.datetime(2020, 3, 11))]
Rider_2020_avg = Rider_2020_AVG.groupby('from_station_id').agg(lambda x: stats.trim_mean(x, 0.05))[
    'trip_id'].reset_index()
Rider_2020_avg.columns = ['stationid', '2020_Avg']

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

Rider_2019 = Rider_2019.merge(Rider_2019_avg, on='stationid')
Rider_2019 = Rider_2019.merge(Rider_2020_avg, on='stationid')

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
# Select need stations
Impact_0101 = Rider_2020[(~Rider_2020['stationid'].isin(
    set(Rider_2020[(Rider_2020['Cum_Relative_Impact'] > 3) & (Rider_2020['Month'] > 1)]['stationid']))) & (
                                 Rider_2020['Month'] > 1)]
print(len(set(Impact_0101['stationid'])))
print(len(set(Rider_2020['stationid'])))
Impact_0101[(Impact_0101['Date'] <= datetime.datetime(2020, 7, 31)) & (
        Impact_0101['Date'] >= datetime.datetime(2020, 3, 11))]['Cum_Relative_Impact'].describe()

# Plot relative impact
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
ax.plot([datetime.datetime(2020, 2, 1), datetime.datetime(2020, 7, 31)], [0, 0], '--',
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
Impact_0101.columns
Rider_2020_New = Impact_0101.copy()
# [['stationid', 'Response', 'Month', 'Day', 'Date', 'Reference', 'point.effect', 'Relative_Impact', 'Cum_effect',
#   'Cum_Response', 'Cum_Reference', 'Cum_Relative_Impact']]
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
All_final = All_final.rename(
    {'Pct.Male': 'Prop.of Male', 'Pct.Age_0_24': 'Prop.of Age_0_24', 'Pct.Age_25_40': 'Prop.of Age_25_40',
     'Pct.Age_40_65': 'Prop.of Age_40_65', 'Pct.White': 'Prop.of White',
     'Pct.Black': 'Prop.of Black', 'Pct.Indian': 'Prop.of Indian', 'Pct.Asian': 'Prop.of Asian',
     'Income': 'Median Income', 'College': 'Prop.of College Degree', 'Cumu_Cases': 'No.of Cases',
     'Cumu_Death': 'No.of Death', 'Cumu_Cases_Rate': 'Infection Rate',
     'Cumu_Death_Rate': 'Death Rate', 'COMMERCIAL': 'Prop.of Commercial',
     'INDUSTRIAL': 'Prop.of Industrial', 'INSTITUTIONAL': 'Prop.of Institutional', 'OPENSPACE': 'Prop.of Openspace',
     'RESIDENTIAL': 'Prop.of Residential', 'Primary': 'Primary Road Density', 'Secondary': 'Secondary Road Density',
     'Minor': 'Minor Road Density', 'All_Road_Length': 'Road Density', 'Bike_Route': 'Bike Route Density',
     'Pct.WJob_Goods_Product': 'Prop.of Goods_Product Jobs', 'Pct.WJob_Utilities': 'Prop.of Utilities Jobs',
     'Pct.WJob_OtherServices': 'Prop.of Other Jobs', 'WTotal_Job_Density': 'Job Density',
     'Bus_stop_count': 'No.of Nearby Busstops', 'Distance_Busstop': 'Distance to Nearest Busstop',
     'Rail_stop_count': 'No.of Nearby Rail Stations', 'Distance_Rail': 'Distance to Nearest Rail Station',
     'Near_Bike_station_Count': 'No.of Nearby Bike Stations', 'Near_Bike_Capacity': 'Capacity of Nearby Bike Stations',
     'Distance_Bikestation': 'Distance to Nearest Bike Station', 'Near_bike_pickups': 'Nearby Bike Pickups',
     'Distance_City': 'Distance to City Center', 'PopDensity': 'Population Density', 'capacity': 'Capacity',
     'Pct.Car': 'Prop.of.Car', 'Pct.Transit': 'Prop.of.Transit'}, axis=1)
All_final.columns
All_final['Transit.Ridership'] = All_final.alightings + All_final.boardings + All_final.rides
All_final["Prop.of.Walk_Bike"] = All_final['Pct.Bicycle'] + All_final['Pct.Walk']

# All_final.to_csv(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\All_final_Divvy_R2019_1005.csv')
All_final.to_csv(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\All_final_Divvy_R2019_1015.csv')

# For ARCGIS
ArcGIS_Divvy_Plot = All_final.groupby('from_stati').tail(1).reset_index(drop=True)
ArcGIS_Divvy_Plot.columns
ArcGIS_Divvy_Plot = ArcGIS_Divvy_Plot[
    ['from_stati', 'lon', 'lat', 'Capacity', 'Cum_Relative_Impact', '2019_Avg', '2020_Avg']]
ArcGIS_Divvy_Plot.to_csv('D:\COVID19-Transit_Bikesharing\Divvy_Data\ArcGIS_Divvy_Plot.csv')

# For Describe
All_final.groupby('from_stati').tail(1).describe().T.to_csv(
    r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Describe_LandUse.csv')
All_final.describe().T.to_csv(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Describe_All.csv')

# Correlation
All_final.groupby('from_stati').tail(1).corr().to_csv(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Divvy_Corr.csv')
