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


def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues


os.chdir(r'D:\COVID19-Transit_Bikesharing\Divvy_Data')
plt.rcParams.update({'font.size': 24, 'font.family': "Times New Roman"})

# Calculate the impact from last year
ridership_old = pd.read_csv(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Day_count_Divvy_dropOutlier.csv', index_col=0)
ridership_old.columns
ridership_old['startdate'] = pd.to_datetime(ridership_old['startdate'])
# Calculate the direct decrease
# 2020-3-14 to 2020-4-30 : 2019-3-14 to 2019-4-30
Rider_2020 = ridership_old[(ridership_old['startdate'] < datetime.datetime(2020, 8, 1)) & (
        ridership_old['startdate'] > datetime.datetime(2020, 3, 12))]
Rider_2020['Month'] = Rider_2020.startdate.dt.month
Rider_2020['Day'] = Rider_2020.startdate.dt.day
Rider_2020['Week'] = Rider_2020.startdate.dt.dayofweek
Rider_2020 = Rider_2020[['from_station_id', 'trip_id', 'Month', 'Day', 'Week', 'startdate']]
Rider_2020.columns = ['stationid', 'Response', 'Month', 'Day', 'Week', 'Date']

Rider_2019 = ridership_old[(ridership_old['startdate'] < datetime.datetime(2019, 8, 1)) & (
        ridership_old['startdate'] > datetime.datetime(2019, 3, 12))]
Rider_2019['Month'] = Rider_2019.startdate.dt.month
Rider_2019['Day'] = Rider_2019.startdate.dt.day
Rider_2019 = Rider_2019[['from_station_id', 'trip_id', 'Month', 'Day']]
Rider_2019.columns = ['stationid', 'Predict', 'Month', 'Day']

Rider_2020 = Rider_2020.merge(Rider_2019, on=['stationid', 'Month', 'Day'])
Rider_2020['point.effect'] = Rider_2020['Response'] - Rider_2020['Predict']
Rider_2020['Relative_Impact'] = (Rider_2020['Response'] - Rider_2020['Predict']) / Rider_2020['Predict']
Rider_2020['Cum_effect'] = Rider_2020.groupby(['stationid'])['point.effect'].cumsum()
Rider_2020['Cum_Response'] = Rider_2020.groupby(['stationid'])['Response'].cumsum()
Rider_2020['Cum_Predict'] = Rider_2020.groupby(['stationid'])['Predict'].cumsum()
Rider_2020['Cum_Relative_Impact'] = (Rider_2020['Cum_Response'] - Rider_2020['Cum_Predict']) / Rider_2020['Cum_Predict']

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

Rider_2020 = Rider_2020.replace([np.inf, -np.inf], np.nan)
sns.distplot(Rider_2020['Relative_Impact'])
Impact_Sta_plot = Rider_2020.groupby(['Date']).mean().reset_index()

sns.set_palette(sns.color_palette("GnBu_d"))
plt.rcParams.update({'font.size': 18, 'font.family': "Times New Roman"})
fig, ax = plt.subplots(figsize=(12, 8), nrows=4, ncols=1, sharex=True)
ax[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
ax[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
sns.lineplot(data=Rider_2020, x='Date', hue='stationid', y='Predict', ax=ax[0], legend=False,
             palette=sns.color_palette("GnBu_d", Rider_2020.stationid.unique().shape[0]), alpha=0.4)
ax[0].plot(Impact_Sta_plot['Date'], Impact_Sta_plot['Predict'], color='#2f4c58', lw=2)
# ax[0].plot(Impact_Sta_plot['time'], Impact_Sta_plot['point.pred.lower'], '--', color='#2f4c58')
# ax[0].plot(Impact_Sta_plot['time'], Impact_Sta_plot['point.pred.upper'], '--', color='#2f4c58')
ax[0].set_ylabel('Prediction')

sns.lineplot(data=Rider_2020, x='Date', hue='stationid', y='Response', ax=ax[1], legend=False,
             palette=sns.color_palette("GnBu_d", Rider_2020.stationid.unique().shape[0]), alpha=0.4)
ax[1].plot(Impact_Sta_plot['Date'], Impact_Sta_plot['Response'], color='#2f4c58', lw=2)
ax[1].set_ylabel('Response')

sns.lineplot(data=Rider_2020, x='Date', hue='stationid', y='point.effect', ax=ax[2], legend=False,
             palette=sns.color_palette("GnBu_d", Rider_2020.stationid.unique().shape[0]), alpha=0.4)
ax[2].plot([datetime.datetime(2020, 2, 1), datetime.datetime(2020, 7, 30)], [0, 0], '--', color='r')
ax[2].plot(Impact_Sta_plot['Date'], Impact_Sta_plot['point.effect'], color='#2f4c58', lw=2)
# ax[2].plot([datetime.datetime(2020, 3, 11), datetime.datetime(2020, 3, 11)], [-2 * 10e3, 0.2 * 10e3], '--',
#            color='#2f4c58', lw=2)
plt.text(0.35, 0.1, 'Pre-Intervention', horizontalalignment='center', verticalalignment='center',
         transform=ax[2].transAxes)
plt.text(0.53, 0.1, 'Intervention', horizontalalignment='center', verticalalignment='center',
         transform=ax[2].transAxes)
# ax[2].plot(Impact_Sta_plot['time'], Impact_Sta_plot['point.effect.lower'], '--', color='#2f4c58')
# ax[2].plot(Impact_Sta_plot['time'], Impact_Sta_plot['point.effect.upper'], '--', color='#2f4c58')
ax[2].set_ylabel('Piecewise impact')
sns.lineplot(data=Rider_2020, x='Date', hue='stationid', y='Relative_Impact', ax=ax[3], legend=False,
             palette=sns.color_palette("GnBu_d", Rider_2020.stationid.unique().shape[0]), alpha=0.4)
ax[3].plot(Impact_Sta_plot['Date'], Impact_Sta_plot['Relative_Impact'], color='#2f4c58', lw=2)
ax[3].plot([datetime.datetime(2020, 2, 1), datetime.datetime(2020, 7, 30)], [0, 0], '--', color='r')
# ax[3].plot([datetime.datetime(2020, 3, 11), datetime.datetime(2020, 3, 11)], [-1, 1], '--', color='#2f4c58', lw=2)
# ax[3].plot(Impact_Sta_plot['time'], Impact_Sta_plot['Relative_Impact_lower'], '--', color='#2f4c58')
# ax[3].plot(Impact_Sta_plot['time'], Impact_Sta_plot['Relative_Impact_upper'], '--', color='#2f4c58')
ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
ax[3].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax[3].set_xlabel('Date')
ax[3].set_ylabel('Relative impact')
plt.text(0.35, 0.1, 'Pre-Intervention', horizontalalignment='center', verticalalignment='center',
         transform=ax[3].transAxes)
plt.text(0.53, 0.1, 'Intervention', horizontalalignment='center', verticalalignment='center',
         transform=ax[3].transAxes)
# plt.tight_layout()
plt.subplots_adjust(top=0.954, bottom=0.078, left=0.068, right=0.985, hspace=0.233, wspace=0.2)

# To GAM in R
Rider_2020.head().T
Rider_2020_New = Rider_2020.copy()
Rider_2020_New = Rider_2020_New.rename({'stationid': 'from_stati'}, axis=1)

# Merge with features
All_final = pd.read_csv('D:\COVID19-Transit_Bikesharing\Divvy_Data\Features_Divvy_0906.csv', index_col=0)
All_final = All_final.merge(Rider_2020_New, on='from_stati')
# Lat Lon Capacity
All_Station = pd.read_csv('D:\COVID19-Transit_Bikesharing\Divvy_Data\Divvy_Station.csv')
All_Station_Need = All_Station[['id', 'lon', 'lat', 'capacity']]
All_Station_Need.columns = ['from_stati', 'lon', 'lat', 'capacity']
All_final = All_final.merge(All_Station_Need, on='from_stati')
All_final['Time_Index'] = (All_final['Date'] - datetime.datetime(2020, 3, 12)).dt.days
All_final.to_csv(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\All_final_Divvy_R_0907.csv')

# Directly cumsum and minus
Sum_Impact = Rider_2020.groupby(['stationid']).sum()[['Response', 'Predict']].reset_index()
Sum_Impact['Cum_Rela_Imp'] = (Sum_Impact['Response'] - Sum_Impact['Predict']) / Sum_Impact['Predict']
sns.distplot(Sum_Impact['Cum_Rela_Imp'])
Sum_Impact = Sum_Impact[['stationid', 'Cum_Rela_Imp', 'Predict']]
Sum_Impact.columns = ['from_stati', 'Cum_Effect', 'Pickups']
Sum_Impact.describe()
# Merge with features
All_final = pd.read_csv('D:\COVID19-Transit_Bikesharing\Divvy_Data\All_final_Divvy_R_0907.csv', index_col=0)
All_final = All_final.drop_duplicates(subset=['from_stati'])
All_final = All_final.merge(Sum_Impact, on='from_stati')
All_final.to_csv(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Avg_final_Divvy_R_0907.csv')
All_final = All_final.drop(['GEOID', 'AREA', 'from_stati', 'ZIP_CODE', 'OTHERS', ], axis=1)
All_final.columns

corr_p = calculate_pvalues(All_final)
corr_p[(corr_p > 0.1)] = 0.1
corr_p[(corr_p < 0.1) & (corr_p >= 0.05)] = 0.05
corr_p[(corr_p < 0.05) & (corr_p >= 0.01)] = 0.01
corr_p[(corr_p < 0.01) & (corr_p >= 0.001)] = 0.001
corr_p[(corr_p < 0.001) & (corr_p >= 0)] = 0
corr_p = corr_p.replace({0.1: '', 0.05: '.', 0.01: '*', 0.001: '**', 0: "***"})

# annot=corr_p.values,
fig, ax = plt.subplots(figsize=(11, 9))
plt.rcParams.update({'font.size': 10, 'font.family': "Times New Roman"})
sns.heatmap(All_final.corr(), fmt='',
            cmap=sns.diverging_palette(240, 130, as_cmap=True),
            square=True, xticklabels=True, yticklabels=True, linewidths=.5)
plt.tight_layout()
plt.savefig('CORR_Divvy.svg')
