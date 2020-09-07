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

os.chdir(r'D:\COVID19-Transit_Bikesharing\Divvy_Data')

# Plot the time series
Results_All = pd.read_csv(r'finalMatrix_Divvy_0906.csv', index_col=0)
Results_All['Date'] = pd.to_datetime(Results_All['Date'])
Results_All.columns
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 24, 'font.family': "Times New Roman"})

# Calculate the impact from last year
ridership_old = pd.read_csv(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Day_count_Divvy_dropOutlier.csv', index_col=0)
ridership_old.columns
ridership_old['startdate'] = pd.to_datetime(ridership_old['startdate'])
# Calculate the direct decrease
# 2020-3-14 to 2020-4-30 : 2019-3-14 to 2019-4-30
Rider_2020 = ridership_old[
    (ridership_old['startdate'] < datetime.datetime(2020, 8, 1)) & (
            ridership_old['startdate'] > datetime.datetime(2020, 3, 12))]
Rider_2020['Month'] = Rider_2020.startdate.dt.month
Rider_2020['Day'] = Rider_2020.startdate.dt.day

Rider_2019 = ridership_old[
    (ridership_old['startdate'] < datetime.datetime(2019, 8, 1)) & (
            ridership_old['startdate'] > datetime.datetime(2019, 3, 12))]
Rider_2019['Month'] = Rider_2019.startdate.dt.month
Rider_2019['Day'] = Rider_2019.startdate.dt.day

Rider_2020 = Rider_2020.merge(Rider_2019, on=['from_station_id', 'Month', 'Day'])
Rider_2020['RELIMP'] = (Rider_2020['trip_id_x'] - Rider_2020['trip_id_y']) / Rider_2020['trip_id_y']
Rider_2020 = Rider_2020.replace([np.inf, -np.inf], np.nan)
sns.distplot(Rider_2020['RELIMP'])
Rider_2020_Impact = Rider_2020.groupby(['from_station_id']).mean()[['RELIMP', 'trip_id_x', 'trip_id_y']].reset_index()
# plt.plot(Rider_2020_Impact['RELIMP'])
Rider_2020_Impact = Rider_2020_Impact[Rider_2020_Impact['RELIMP'] < 0]
Rider_2020_Impact.describe()
Rider_2020.describe().T

# Directly cumsum and minus
Sum_Impact = Rider_2020.groupby(['from_station_id']).sum()[['trip_id_x', 'trip_id_y']].reset_index()
Sum_Impact['Cum_Rela_Imp'] = (Sum_Impact['trip_id_x'] - Sum_Impact['trip_id_y']) / Sum_Impact['trip_id_y']
sns.distplot(Sum_Impact['Cum_Rela_Imp'])
Sum_Impact = Sum_Impact[['from_station_id', 'Cum_Rela_Imp', 'trip_id_y']]
Sum_Impact.columns = ['from_stati', 'Cum_Effect', 'Pickups']
# Merge with features
All_final = pd.read_csv('D:\COVID19-Transit_Bikesharing\Divvy_Data\Features_Divvy_0906.csv', index_col=0)
All_final = All_final.merge(Sum_Impact, on='from_stati')
All_final = All_final.drop(['GEOID', 'AREA', 'from_stati', 'ZIP_CODE', 'OTHERS', ], axis=1)
All_final.columns


def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues


corr_p = calculate_pvalues(All_final)
corr_p[(corr_p > 0.1)] = 0.1
corr_p[(corr_p < 0.1) & (corr_p >= 0.05)] = 0.05
corr_p[(corr_p < 0.05) & (corr_p >= 0.01)] = 0.01
corr_p[(corr_p < 0.01) & (corr_p >= 0.001)] = 0.001
corr_p[(corr_p < 0.001) & (corr_p >= 0)] = 0

corr_p = corr_p.replace({0.1: '', 0.05: '.', 0.01: '*', 0.001: '**', 0: "***"})

# annot=corr_p.values,
fig, ax = plt.subplots(figsize=(11, 9))
plt.rcParams.update({'font.size': 14, 'font.family': "Times New Roman"})
sns.heatmap(All_final.corr(), fmt='',
            cmap=sns.diverging_palette(240, 130, as_cmap=True),
            square=True, xticklabels=True, yticklabels=True, linewidths=.5)
plt.tight_layout()
plt.savefig('CORR_Divvy.svg')
