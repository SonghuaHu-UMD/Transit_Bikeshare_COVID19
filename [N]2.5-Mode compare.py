import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import datetime
import matplotlib as mpl
import matplotlib.dates as mdates

plt.rcParams.update(
    {'font.size': 13, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': True, "grid.alpha": 0, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})
mpl.rcParams['axes.prop_cycle'] = \
    mpl.cycler('color', ['#4477AA', '#EE6677', '#228833', '#BBBBBB', '#66CCEE',
                         '#AA3377', '#CCBB44', '#000000'])
l_styles = ['-', '--', '-.']  # , ':' , '-.'
m_styles = ['o', '^', '*', '.', ]

# Here we try to compare three different modes: Driving; Transit; Bikesharing, and overall mobility
# Driving from Apple; Overall from Safegraph
# Apple: compared to a baseline volume on January 13th, 2020.
# Cook County

data_path = r'D:\\COVID19-Mobility-DataSet\\'
result_path = r'D:\\COVID19-Socio\\'

##### ##### ##### ##### ##### ##### ##### ##### #####
##### Read Apple #####
Human_mobility = pd.read_csv(data_path + 'Apple\\applemobilitytrends.csv')
Human_mobility = Human_mobility[
    (Human_mobility['country'] == 'United States') & (Human_mobility['geo_type'] == 'county')]
Human_mobility['index'] = Human_mobility['region'] + '$' + Human_mobility['sub-region'] + '$' + Human_mobility[
    'transportation_type']
Human_mobility = Human_mobility.drop(['region', 'sub-region', 'geo_type', 'transportation_type', 'alternative_name',
                                      'country'], axis=1).set_index('index')
Human_mobility = Human_mobility.unstack().reset_index()
Human_mobility = Human_mobility.join(
    Human_mobility["index"].str.split('$', 2, expand=True).rename(columns={0: 'County', 1: 'State', 2: 'Type'}))
Human_mobility.columns = ['Date', 'CTID', 'Mobility', 'County', 'State', 'Type']
Human_mobility['Date'] = pd.to_datetime(Human_mobility['Date'])

# Fill na by the average of pervious week and next week
Human_mobility.isnull().sum()
Human_mobility = Human_mobility.sort_values(by=['State', 'County', 'Date']).reset_index(drop=True)
Human_mobility['Week1'] = Human_mobility.groupby(['CTID']).shift(-7)['Mobility']
Human_mobility['Week2'] = Human_mobility.groupby(['CTID']).shift(7)['Mobility']
Human_mobility['Mobility_Inp'] = (Human_mobility['Week1'] + Human_mobility['Week2']) / 2
Human_mobility.loc[Human_mobility['Mobility'].isnull(), 'Mobility'] = Human_mobility.loc[
    Human_mobility['Mobility'].isnull(), 'Mobility_Inp']
Human_mobility.isnull().sum()
Human_mobility.drop(['Week1', 'Week2', 'Mobility_Inp'], axis=1, inplace=True)
Human_mobility_Apple = Human_mobility[
    (Human_mobility['County'] == 'Cook County') & (Human_mobility['State'] == 'Illinois')].reset_index(drop=True)
Human_mobility_Apple = Human_mobility_Apple[['Date', 'Type', 'Mobility']]
Human_mobility_Apple['Type'] = Human_mobility_Apple['Type'] + ' (Apple)'
Human_mobility_Apple['Type'] = Human_mobility_Apple['Type'].str.title()
##### ##### ##### ##### ##### ##### ##### ##### #####

##### ##### ##### ##### ##### ##### ##### ##### #####
# Read Safegraph # All_County_Visit_POI.pkl
Human_mobility = pd.read_pickle(data_path + 'SafeGraph\\All_County_Metrics.pkl')
Human_mobility = Human_mobility[Human_mobility['CTFIPS'] == '17031'].reset_index(drop=True)
Human_mobility['Daily_Visits_no_parent'] = Human_mobility['OutFlow'] + Human_mobility['InFlow'] + Human_mobility[
    'IntraFlow']
Human_mobility['Date'] = pd.to_datetime(Human_mobility['Date'])
Human_mobility = Human_mobility.groupby(['CTFIPS', 'Date']).sum()[
    'Daily_Visits_no_parent'].reset_index()  # Daily_Visits_no_parent
# Baseline: 2020-01-13
Human_mobility['Baseline_Mobility'] = Human_mobility.loc[
    Human_mobility['Date'] == datetime.datetime(2020, 1, 13), 'Daily_Visits_no_parent'].values[0]
Human_mobility['Mobility'] = (Human_mobility['Daily_Visits_no_parent'] / Human_mobility['Baseline_Mobility']) * 100
Human_mobility['Type'] = 'All travel (SafeGraph)'
Human_mobility_SG = Human_mobility[['Date', 'Type', 'Mobility']]
Human_mobility_SG = Human_mobility_SG[Human_mobility_SG['Date'] >= datetime.datetime(2020, 1, 1)].reset_index(
    drop=True)
##### ##### ##### ##### ##### ##### ##### ##### #####

##### ##### ##### ##### ##### ##### ##### ##### #####
# Read bikeshare
ridership_old = pd.read_csv(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Day_count_Divvy_dropOutlier.csv', index_col=0)
# ridership_old = pd.read_pickle(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Trips_All\alltrips_chicago_202007.pkl')
ridership_old['Date'] = pd.to_datetime(ridership_old['startdate'])
ridership_old_2020 = ridership_old[(ridership_old['Date'] >= datetime.datetime(2020, 1, 1)) & (
        ridership_old['Date'] <= datetime.datetime(2020, 7, 31))]
ridership_old_2020 = ridership_old_2020.groupby(['Date']).sum()['trip_id'].reset_index()
ridership_old_2020['Baseline_Mobility'] = \
    ridership_old_2020.loc[ridership_old_2020['Date'] == datetime.datetime(2020, 1, 13), 'trip_id'].values[0]
ridership_old_2020['Mobility'] = (ridership_old_2020['trip_id'] / ridership_old_2020['Baseline_Mobility']) * 100
ridership_old_2020['Type'] = 'Bike-sharing (Divvy)'
ridership_old_2020 = ridership_old_2020[['Date', 'Type', 'Mobility']]
# plt.plot(ridership_old['Date'], ridership_old['Mobility'], '-o')

ridership_old_2019 = ridership_old[(ridership_old['Date'] >= datetime.datetime(2019, 1, 1)) & (
        ridership_old['Date'] <= datetime.datetime(2019, 7, 31))]
ridership_old_2019 = ridership_old_2019.groupby(['Date']).sum()['trip_id'].reset_index()
ridership_old_2019['Baseline_Mobility'] = \
    ridership_old_2019.loc[ridership_old_2019['Date'] == datetime.datetime(2019, 1, 14), 'trip_id'].values[0]
ridership_old_2019['Mobility'] = (ridership_old_2019['trip_id'] / ridership_old_2019['Baseline_Mobility']) * 100
ridership_old_2019['Type'] = 'Bikesharing (2019)'
ridership_old_2019 = ridership_old_2019[['Date', 'Type', 'Mobility']]
ridership_old_2019['Date'] = ridership_old_2019['Date'] + datetime.timedelta(days=365)

# Merge
Human_mobility_All = pd.concat([Human_mobility_SG, Human_mobility_Apple, ridership_old_2020], axis=0)
Human_mobility_All['Mobility'] = Human_mobility_All['Mobility'] / 100
Human_mobility_All = Human_mobility_All[(Human_mobility_All['Date'] >= datetime.datetime(2020, 1, 1)) & (
        Human_mobility_All['Date'] <= datetime.datetime(2020, 7, 31))]
# Human_mobility_All['weekofyear'] = Human_mobility_All['Date'].dt.weekofyear
Human_mobility_All.set_index('Date', inplace=True)

# Human_mobility_All = Human_mobility_All.groupby(['Type', 'weekofyear']).mean().reset_index()
# Human_mobility_All.set_index('weekofyear', inplace=True)
fig, ax = plt.subplots(figsize=(10, 6))
ccount = 0
# set(Human_mobility_All['Type'])
for type in ['All travel (SafeGraph)', 'Bike-sharing (Divvy)', 'Driving (Apple)', 'Transit (Apple)', 'Walking (Apple)']:
    temp = Human_mobility_All[Human_mobility_All['Type'] == type]
    ax.plot(temp['Mobility'], linestyle=l_styles[ccount % len(l_styles)], marker=m_styles[ccount // len(l_styles)],
            markersize=4, label=type, markevery=2)
    ccount += 1
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
plt.plot([datetime.datetime(2020, 1, 1), datetime.datetime(2020, 7, 31)], [1, 1], '-.', color='gray')
plt.plot([datetime.datetime(2020, 1, 13), datetime.datetime(2020, 1, 13)], [0, 3.8], '-.', color='gray')
plt.ylabel('Relative Mobility Ratio')
plt.tight_layout()
plt.legend(loc=2, ncol=3)
plt.text(0.105, 0.05, 'Baseline: Jan-13, 2020', horizontalalignment='left', verticalalignment='center',
         transform=ax.transAxes)
plt.savefig(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Results\FIG-NEW.png', dpi=600)
plt.savefig(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Results\FIG-NEW.svg')

# Describe
Human_mobility_All = Human_mobility_All.reset_index()
Human_mobility_All['Month'] = Human_mobility_All['Date'].dt.month
Human_mobility_All_Month = Human_mobility_All.groupby(['Type', 'Month']).mean().reset_index()
Human_mobility_All_Month.pivot(index='Month', columns='Type', values='Mobility').to_csv(
    r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Results\Modes.csv')
