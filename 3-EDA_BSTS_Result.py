import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pylab as plt
import matplotlib.dates as mdates
import seaborn as sns
import geopandas as gpd

os.chdir(r'D:\COVID19-Transit_Bikesharing\Divvy_Data')

Not_Need_ID = [95, 102, 270, 356, 384, 386, 388, 390, 391, 392, 393, 395, 396, 398, 399, 400, 407, 408, 409, 411, 412,
               421, 426, 427, 429, 430, 431, 433, 435, 436, 437, 438, 439, 440, 441, 443, 444, 445, 446, 524] + list(
    range(528, 589)) + [593, 594, 595, 559, 564, 567, 570, 571, 572, 574, 576, 579, 580, 583, 585, 588, 642, 646, 647,
                        648, 649, 650, 652, 653, 665, 674, 677, 678, 679, 681, 683, 666, 673, 672, 662, 661]

# Plot the time series
Results_All = pd.read_csv(r'finalMatrix_Divvy_0906.csv', index_col=0)
Results_All['Date'] = pd.to_datetime(Results_All['Date'])
Results_All.columns
Results_All = Results_All[~Results_All['stationid'].isin(Not_Need_ID)].reset_index(drop=True)
'''
plt.rcParams.update({'font.size': 20, 'font.family': "Times New Roman"})
for jj in set(Results_All['stationid']):
    print(jj)
    # jj = 2
    Temp_time = Results_All[Results_All['stationid'] == jj]
    # Temp_time = Temp_time[Temp_time['Date'] >= '2019-01-01']
    Temp_time.set_index('Date', inplace=True)

    fig, ax = plt.subplots(nrows=7, ncols=1, figsize=(16, 9.5), sharex=True)  # 12,9.5
    # ax[0].set_title('Station_ID: ' + str(jj))
    ax[0].plot(Temp_time.loc[Temp_time['Component'] == 'Trend', 'Value'], color='#2f4c58', alpha=0.7, lw=1)
    ax[0].set_ylabel('Trend')

    ax[1].plot(Temp_time.loc[Temp_time['Component'] == 'Seasonality', 'Value'], color='#2f4c58', alpha=0.7, lw=1)
    ax[1].set_ylabel('Week')

    ax[2].plot(Temp_time.loc[Temp_time['Component'] == 'Monthly', 'Value'], color='#2f4c58', alpha=0.7, lw=1)
    ax[2].set_ylabel('Month')

    ax[3].plot(Temp_time.loc[Temp_time['Component'] == 'Regression', 'Value'], color='#2f4c58', alpha=0.7, lw=1)
    ax[3].set_ylabel('Regress')

    ax[4].plot(Temp_time.loc[Temp_time['Component'] == 'Response', 'Value'], color='#2f4c58', alpha=0.7, lw=1)
    ax[4].plot(Temp_time.loc[Temp_time['Component'] == 'Predict', 'Value'], '--', color='#62760c', alpha=0.7, lw=0.5)
    ax[4].fill_between(Temp_time.loc[Temp_time['Component'] == 'Predict_Lower', 'Value'].index,
                       Temp_time.loc[Temp_time['Component'] == 'Predict_Lower', 'Value'],
                       Temp_time.loc[Temp_time['Component'] == 'Predict_Upper', 'Value'], facecolor='#96bb7c',
                       alpha=0.5)
    ax[4].set_ylabel('Predict')

    ax[5].plot(Temp_time.loc[Temp_time['Component'] == 'Response', 'Value'] - Temp_time.loc[
        Temp_time['Component'] == 'Predict', 'Value'], color='#2f4c58', alpha=0.7, lw=1)
    ax[5].fill_between(Temp_time.loc[Temp_time['Component'] == 'Response', 'Value'].index,
                       Temp_time.loc[Temp_time['Component'] == 'Response', 'Value'] - Temp_time.loc[
                           Temp_time['Component'] == 'Predict_Upper', 'Value'],
                       Temp_time.loc[Temp_time['Component'] == 'Response', 'Value'] - Temp_time.loc[
                           Temp_time['Component'] == 'Predict_Lower', 'Value'], facecolor='#96bb7c', alpha=0.5)

    ax[6].plot(Temp_time.loc[Temp_time['Component'] == 'Cum_effect', 'Value'], color='#2f4c58', alpha=0.7, lw=1)
    ax[6].fill_between(Temp_time.loc[Temp_time['Component'] == 'Cum_effect', 'Value'].index,
                       Temp_time.loc[Temp_time['Component'] == 'Cum_effect_Lower', 'Value'],
                       Temp_time.loc[Temp_time['Component'] == 'Cum_effect_Upper', 'Value'], facecolor='#96bb7c',
                       alpha=0.5)
    ax[6].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    ax[6].xaxis.set_major_locator(mdates.WeekdayLocator(interval=30))
    ax[6].set_ylabel('Effect')
    ax[6].set_xlabel('Date')
    plt.xlim(xmin=min(Temp_time.index), xmax=max(Temp_time.index))
    # for axx in ax:
    #     axx.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=False)
    # fig.autofmt_xdate()
    plt.subplots_adjust(top=0.987, bottom=0.087, left=0.087, right=0.982, hspace=0.078, wspace=0.09)
    # plt.savefig('FIG2-1.png', dpi=600)
    # plt.savefig('FIG2-1.svg')
    plt.savefig(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\BSTS_Predict\Full_Time_Range_' + str(jj) + 'png', dpi=500)
    plt.close()

# Plot the zoom in figure
jj = 2
myFmt = mdates.DateFormatter('%b-%d')
Temp_time = Results_All[Results_All['stationid'] == jj]
Temp_time = Temp_time[Temp_time['Date'] >= '2020-01-01']
Temp_time.set_index('Date', inplace=True)
fig, ax = plt.subplots(nrows=6, ncols=1, figsize=(6, 9.5), sharex=True)  # 12,9.5
ax[0].plot(Temp_time.loc[Temp_time['Component'] == 'Trend', 'Value'], color='#2f4c58')
ax[1].plot(Temp_time.loc[Temp_time['Component'] == 'Seasonality', 'Value'], color='#2f4c58')
ax[2].plot(Temp_time.loc[Temp_time['Component'] == 'Monthly', 'Value'], color='#2f4c58')
ax[3].plot(Temp_time.loc[Temp_time['Component'] == 'Regression', 'Value'], color='#2f4c58')
ax[4].plot(Temp_time.loc[Temp_time['Component'] == 'Response', 'Value'], color='#2f4c58')
ax[4].plot(Temp_time.loc[Temp_time['Component'] == 'Predict', 'Value'], '--', color='#62760c', alpha=0.8)
ax[4].fill_between(Temp_time.loc[Temp_time['Component'] == 'Predict_Lower', 'Value'].index,
                   Temp_time.loc[Temp_time['Component'] == 'Predict_Lower', 'Value'],
                   Temp_time.loc[Temp_time['Component'] == 'Predict_Upper', 'Value'], facecolor='#96bb7c',
                   alpha=0.5)
ax[5].plot(Temp_time.loc[Temp_time['Component'] == 'Response', 'Value'] - Temp_time.loc[
    Temp_time['Component'] == 'Predict', 'Value'], color='#2f4c58')
ax[5].fill_between(Temp_time.loc[Temp_time['Component'] == 'Response', 'Value'].index,
                   Temp_time.loc[Temp_time['Component'] == 'Response', 'Value'] - Temp_time.loc[
                       Temp_time['Component'] == 'Predict_Upper', 'Value'],
                   Temp_time.loc[Temp_time['Component'] == 'Response', 'Value'] - Temp_time.loc[
                       Temp_time['Component'] == 'Predict_Lower', 'Value'], facecolor='#96bb7c', alpha=0.5)
ax[5].xaxis.set_major_formatter(myFmt)
ax[5].xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
ax[5].set_xlabel('Date')
plt.xlim(xmin=min(Temp_time.index), xmax=max(Temp_time.index))
for ax0 in ax:
    ax0.axes.yaxis.set_visible(False)
plt.tight_layout()
plt.subplots_adjust(top=0.987, bottom=0.087, left=0.022, right=0.987, hspace=0.078, wspace=0.09)
plt.savefig('FIG2-2.png', dpi=600)
plt.savefig('FIG2-2.svg')
'''
# Calculate the casual impact
# For build the PLS model
Impact = pd.pivot_table(Results_All, values='Value', index=['Date', 'stationid'], columns=['Component']).reset_index()
del Results_All
Impact_0312 = Impact[Impact['Date'] >= datetime.datetime(2020, 3, 13)]
Impact_0312 = Impact_0312.sort_values(by=['stationid', 'Date']).reset_index(drop=True)
Impact_0312.columns
Impact_0312['Cum_Predict'] = Impact_0312.groupby(['stationid'])['Predict'].cumsum()
Impact_0312['Cum_Relt_Effect'] = Impact_0312['Cum_effect'] / Impact_0312['Cum_Predict']
Impact_0312.rename({'stationid': 'from_stati'}, axis=1, inplace=True)
# Merge with station
# Merge with features
All_final = pd.read_csv('D:\COVID19-Transit_Bikesharing\Divvy_Data\Features_Divvy_0906.csv', index_col=0)
All_final = All_final.merge(Impact_0312, on='from_stati')
# Lat Lon Capacity
All_Station = pd.read_csv('D:\COVID19-Transit_Bikesharing\Divvy_Data\Divvy_Station.csv')
All_Station_Need = All_Station[['id', 'lon', 'lat', 'capacity']]
All_Station_Need.columns = ['from_stati', 'lon', 'lat', 'capacity']
All_final = All_final.merge(All_Station_Need, on='from_stati')
All_final['Time_Index'] = (All_final['Date'] - datetime.datetime(2020, 3, 12)).dt.days
All_final['Week'] = All_final['Date'].dt.dayofweek
All_final['Month'] = All_final['Date'].dt.month
All_final.to_csv(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\All_final_Divvy_R_BSTS_0907.csv')

# Impact_0312.to_csv('Impact_0312_tem.csv')

'''
for jj in list(set(Impact_0312['stationid'])):
    # jj = 3
    tem = Impact_0312[Impact_0312['stationid'] == jj]
    tem = tem.set_index('Date')
    # Find
    fig, ax = plt.subplots(figsize=(12, 6), nrows=1, ncols=1)
    ax.plot(tem['Cum_Relt_Effect'], '-', color='k')
    plt.tight_layout()
    plt.savefig('D:\\COVID19-Transit_Bikesharing\\Divvy_Data\\Cum_BSTS\\' + str(jj) + '.png')
    plt.close()
'''
# Calculate the relative impact
Impact_0312['Relative_Impact'] = ((Impact_0312['Response'] - Impact_0312['Predict']) / Impact_0312['Predict'])
Impact_0312['Relative_Impact_lower'] = (
        (Impact_0312['Response'] - Impact_0312['Predict_Lower']) / Impact_0312['Predict_Lower'])
Impact_0312['Relative_Impact_upper'] = (
        (Impact_0312['Response'] - Impact_0312['Predict_Upper']) / Impact_0312['Predict_Upper'])
Impact_Sta = Impact_0312.groupby(['stationid']).mean()['Relative_Impact'].reset_index()
# plt.plot(Impact_Sta['Relative_Impact'])
# sns.distplot(Impact_Sta['Relative_Impact'])
Impact_Sta.to_csv('Divvy_Impact_Sta.csv')

# Plot the impact for each station
Impact_0101 = Impact[Impact['Date'] >= datetime.datetime(2020, 2, 1)]
Impact_0101['point.effect'] = Impact_0101['Response'] - Impact_0101['Predict']
Impact_0101['Relative_Impact'] = ((Impact_0101['Response'] - Impact_0101['Predict']) / Impact_0101['Predict'])
Impact_0101['Relative_Impact_lower'] = (
        (Impact_0101['Response'] - Impact_0101['Predict_Lower']) / Impact_0101['Predict_Lower'])
Impact_0101['Relative_Impact_upper'] = (
        (Impact_0101['Response'] - Impact_0101['Predict_Upper']) / Impact_0101['Predict_Upper'])
Impact_Sta_plot = Impact_0101.groupby(['Date']).mean().reset_index()

sns.set_palette(sns.color_palette("GnBu_d"))
plt.rcParams.update({'font.size': 18, 'font.family': "Times New Roman"})
fig, ax = plt.subplots(figsize=(12, 8), nrows=4, ncols=1, sharex=True)
ax[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
ax[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
sns.lineplot(data=Impact_0101, x='Date', hue='stationid', y='Predict', ax=ax[0], legend=False,
             palette=sns.color_palette("GnBu_d", Impact_0101.stationid.unique().shape[0]), alpha=0.4)
ax[0].plot(Impact_Sta_plot['Date'], Impact_Sta_plot['Predict'], color='#2f4c58', lw=2)
# ax[0].plot(Impact_Sta_plot['time'], Impact_Sta_plot['point.pred.lower'], '--', color='#2f4c58')
# ax[0].plot(Impact_Sta_plot['time'], Impact_Sta_plot['point.pred.upper'], '--', color='#2f4c58')
ax[0].set_ylabel('Prediction')

sns.lineplot(data=Impact_0101, x='Date', hue='stationid', y='Response', ax=ax[1], legend=False,
             palette=sns.color_palette("GnBu_d", Impact_0101.stationid.unique().shape[0]), alpha=0.4)
ax[1].plot(Impact_Sta_plot['Date'], Impact_Sta_plot['Response'], color='#2f4c58', lw=2)
ax[1].set_ylabel('Response')

sns.lineplot(data=Impact_0101, x='Date', hue='stationid', y='point.effect', ax=ax[2], legend=False,
             palette=sns.color_palette("GnBu_d", Impact_0101.stationid.unique().shape[0]), alpha=0.4)
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
sns.lineplot(data=Impact_0101, x='Date', hue='stationid', y='Relative_Impact', ax=ax[3], legend=False,
             palette=sns.color_palette("GnBu_d", Impact_0101.stationid.unique().shape[0]), alpha=0.4)
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
plt.savefig(r'Fig-3.png', dpi=600)
plt.savefig(r'Fig-3.svg')

# Residual
Results_All_Res = Results_All[Results_All['Date'] < datetime.datetime(2020, 3, 2)]
Results_All_Res = pd.pivot_table(Results_All_Res, values='Value', index=['Date', 'stationid'],
                                 columns=['Component']).reset_index()
Results_All_Res.columns
Results_All_Res['Residual'] = Results_All_Res['Response'] - Results_All_Res['Predict']
Results_All_Res['MAE'] = abs(Results_All_Res['Response'] - Results_All_Res['Predict'])
Results_All_Res['MAPE'] = abs(
    (Results_All_Res['Response'] - Results_All_Res['Predict']) / (Results_All_Res['Response']))
Results_All_Res = Results_All_Res.replace([np.inf, -np.inf], np.nan)
Results_All_Res.isnull().sum()
Results_All_Res = Results_All_Res.fillna(0)
Results_All_Res.describe()
Results_All_Res.groupby(['stationid']).median()['MAPE'].min()
Results_All_Res.groupby(['stationid']).median()['MAPE'].max()
Results_All_Res.groupby(['stationid']).median()['MAPE'].mean()
# Plot MAPE
fig, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(x="stationid", y="MAPE", data=Results_All_Res, palette=sns.color_palette("GnBu_d"), showfliers=False, ax=ax,
            linewidth=1)
ax.tick_params(labelbottom=False)
ax.set_xlabel('Transit Station')
# ax.set_ylim([0, 0.15])
plt.tight_layout()
plt.savefig('Fig-MAPE.png', dpi=600)
plt.savefig('Fig-MAPE.svg')
# Tem_DA = pd.DataFrame({'Response': Results_All_Res.loc[Results_All_Res['Component'] == 'Response', 'Value'].values,
#                        'Predict': Results_All_Res.loc[Results_All_Res['Component'] == 'Predict', 'Value'].values})

# Plot Coeeff
Coeffic = pd.read_csv(r'finalCoeff_Transit_0810.csv', index_col=0)
Coeffic.columns
Coeffic.describe().T
plt.rcParams.update({'font.size': 20, 'font.family': "Times New Roman"})
fig, ax = plt.subplots(figsize=(14, 4), nrows=1, ncols=3)
ax[0].set_ylabel('Frequency')
sns.distplot(Coeffic['PRCP'], ax=ax[0], rug_kws={"color": "g"}, axlabel='Coeff. of Precipitation',
             hist_kws={"histtype": "step", "linewidth": 2, "alpha": 0.8, "color": "g"}, kde=False)
sns.distplot(Coeffic['TMAX'], ax=ax[1], rug_kws={"color": "g"}, axlabel='Coeff. of Temperature',
             hist_kws={"histtype": "step", "linewidth": 2, "alpha": 0.8, "color": "g"}, kde=False)
sns.distplot(Coeffic['Holidays'], ax=ax[2], rug_kws={"color": "g"}, axlabel='Coeff. of Is Holiday',
             hist_kws={"histtype": "step", "linewidth": 2, "alpha": 0.8, "color": "g"}, kde=False)
plt.text(0.3, 0.9, 'Mean = -0.6207', horizontalalignment='center', verticalalignment='center',
         transform=ax[0].transAxes)
plt.text(0.3, 0.9, 'Mean = 1.2913', horizontalalignment='center', verticalalignment='center',
         transform=ax[1].transAxes)
plt.text(0.3, 0.9, 'Mean = -1564.2068', horizontalalignment='center', verticalalignment='center',
         transform=ax[2].transAxes)
for axx in ax:
    axx.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
plt.subplots_adjust(top=0.885, bottom=0.213, left=0.049, right=0.984, hspace=0.2, wspace=0.198)
plt.savefig('Hist.png', dpi=600)
plt.savefig('Hist.svg')

# Calculate Impact for plot and describe
Impact_OLD = pd.read_csv(r'finalImpact_Transit_0810_old.csv')
Impact_OLD_AVG = Impact_OLD[::2]
Impact_OLD_AVG.describe().T.to_csv('Impact_OLD_AVG.csv')
Impact_OLD_AVG = Impact_OLD_AVG.rename({'stationid': 'station_id'}, axis=1)
Stations = pd.read_csv('LStations_Chicago.csv', index_col=0)
Impact_OLD_AVG = Impact_OLD_AVG.merge(Stations, on='station_id')
Impact_OLD_AVG.to_csv('Impact_Sta_ARCGIS_NEW.csv')

# Calculate the impact from last year
ridership_old = pd.read_csv(r'Daily_Lstaion_Final.csv')
ridership_old.columns
ridership_old['date'] = pd.to_datetime(ridership_old['date'])
# Calculate the direct decrease
# 2020-3-14 to 2020-4-30 : 2019-3-14 to 2019-4-30
Rider_2020 = ridership_old[
    (ridership_old['date'] < datetime.datetime(2020, 5, 1)) & (ridership_old['date'] > datetime.datetime(2020, 3, 12))]
Rider_2020['Month'] = Rider_2020.date.dt.month
Rider_2020['Day'] = Rider_2020.date.dt.day

Rider_2019 = ridership_old[
    (ridership_old['date'] < datetime.datetime(2019, 5, 1)) & (ridership_old['date'] > datetime.datetime(2019, 3, 12))]
Rider_2019['Month'] = Rider_2019.date.dt.month
Rider_2019['Day'] = Rider_2019.date.dt.day

Rider_2020 = Rider_2020.merge(Rider_2019, on=['station_id', 'Month', 'Day'])
Rider_2020['RELIMP'] = (Rider_2020['rides_x'] - Rider_2020['rides_y']) / Rider_2020['rides_y']
Rider_2020 = Rider_2020.replace([np.inf, -np.inf], np.nan)
Rider_2020_Impact = Rider_2020.groupby(['station_id']).mean()[['RELIMP', 'rides_x', 'rides_y']].reset_index()
# plt.plot(Rider_2020_Impact['RELIMP'])
Rider_2020_Impact = Rider_2020_Impact[Rider_2020_Impact['RELIMP'] < 0]
Rider_2020_Impact['RELIMP'] = Rider_2020_Impact['RELIMP'] - 0.08
Rider_2020_Impact.describe()
Rider_2020.describe().T

# Compare with inferred impact
Impact_OLD_AVG.rename({'stationid': 'station_id'}, axis=1, inplace=True)
Rider_2020_Impact = Rider_2020_Impact.merge(Impact_OLD_AVG, on='station_id')
Rider_2020_Impact['ABS_diff'] = abs(Rider_2020_Impact['RelEffect'] - Rider_2020_Impact['RELIMP'])
# plt.plot(Rider_2020_Impact['RelEffect'], Rider_2020_Impact['RELIMP'], 'o')
# plt.plot(Rider_2020_Impact.loc[Rider_2020_Impact['p'] > 0.1, 'RelEffect'],
#          Rider_2020_Impact.loc[Rider_2020_Impact['p'] > 0.1, 'RELIMP'], 'o')
np.corrcoef(Rider_2020_Impact['RelEffect'], Rider_2020_Impact['RELIMP'])
fig, ax = plt.subplots(figsize=(8, 6))
sns.regplot(x=Rider_2020_Impact['RelEffect'], y=(Rider_2020_Impact['RELIMP']), color='#2f4c58',
            scatter_kws={'s': (Rider_2020_Impact['rides_y'] / 50), 'alpha': 0.5}, ax=ax)
tem = Rider_2020_Impact[
    Rider_2020_Impact['station_id'].isin([41420, 40890, 41000, 41030, 40160, 40030, 40730])].reset_index()
plt.scatter(tem['RelEffect'], tem['RELIMP'], alpha=0.5, s=tem['rides_y'] / 50, color='green')
ax.set_xlim([-1.05, -0.4])
ax.set_ylim([-1.05, -0.4])
ax.set_ylabel('Relative Decrease (Baseline:2019)')
ax.set_xlabel('Relative Effect')
plt.tight_layout()
plt.savefig('Compare.png', dpi=600)
plt.savefig('Compare.svg')
Rider_2020_Impact.to_csv('Rider_2020_Impact_compare.csv')

# See the station without significant causal impact
# 40890
jj = 40630
myFmt = mdates.DateFormatter('%b-%d')
Temp_time = Results_All[Results_All['stationid'] == jj]
Temp_time = Temp_time[Temp_time['Date'] >= '2019-01-01']
Temp_time.set_index('Date', inplace=True)
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6.5), sharex=True)  # 12,9.5
ax[0].plot(Temp_time.loc[Temp_time['Component'] == 'Response', 'Value'], color='#2f4c58')
ax[0].plot(Temp_time.loc[Temp_time['Component'] == 'Predict', 'Value'], '--', color='#62760c', alpha=0.8)
ax[0].fill_between(Temp_time.loc[Temp_time['Component'] == 'Predict_Lower', 'Value'].index,
                   Temp_time.loc[Temp_time['Component'] == 'Predict_Lower', 'Value'],
                   Temp_time.loc[Temp_time['Component'] == 'Predict_Upper', 'Value'], facecolor='#96bb7c',
                   alpha=0.5)
ax[1].plot(Temp_time.loc[Temp_time['Component'] == 'Response', 'Value'] - Temp_time.loc[
    Temp_time['Component'] == 'Predict', 'Value'], color='#2f4c58')
ax[1].fill_between(Temp_time.loc[Temp_time['Component'] == 'Response', 'Value'].index,
                   Temp_time.loc[Temp_time['Component'] == 'Response', 'Value'] - Temp_time.loc[
                       Temp_time['Component'] == 'Predict_Upper', 'Value'],
                   Temp_time.loc[Temp_time['Component'] == 'Response', 'Value'] - Temp_time.loc[
                       Temp_time['Component'] == 'Predict_Lower', 'Value'], facecolor='#96bb7c', alpha=0.5)
ax[1].xaxis.set_major_formatter(myFmt)
ax[1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
ax[1].set_xlabel('Date')
plt.xlim(xmin=min(Temp_time.index), xmax=max(Temp_time.index))
for ax0 in ax:
    ax0.axes.yaxis.set_visible(False)
plt.tight_layout()
plt.subplots_adjust(top=0.987, bottom=0.087, left=0.022, right=0.987, hspace=0.078, wspace=0.09)
