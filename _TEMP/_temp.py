# Nearest Distance to busstop
from shapely.ops import nearest_points

# unary union of the gpd2 geomtries
pts3 = Busstop.geometry.unary_union


def near(point, pts=pts3):
    # point=Station.iloc[0].geometry
    # find the nearest point and return the corresponding Place value
    nearest = Busstop.geometry == nearest_points(point, pts)[1]
    return Busstop[nearest].geometry


Station['Nearest_Busstop'] = Station.apply(lambda row: near(row.geometry), axis=1)

# Plot year trend
Results_All['Year'] = Results_All['Date'].dt.year
Temp_time = Results_All.groupby(['Year', 'Component', 'stationid']).mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(data=Temp_time[Temp_time['Component'] == 'Trend'], x='Year', hue='stationid', y='Value', ax=ax,
             legend=False,
             palette=sns.color_palette("GnBu_d", Temp_time.stationid.unique().shape[0]), alpha=0.4)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
plt.ylabel('Trend')
plt.tight_layout()
plt.savefig('FIGN-2.png', dpi=600)
plt.savefig('FIGN-2.svg')

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6), sharex=True)  # 12,9.5
for jj in set(Results_All['stationid']):
    print(jj)
    # jj = 40930
    myFmt = mdates.DateFormatter('%Y')
    Temp_time = Results_All[Results_All['stationid'] == jj]
    Temp_time['Year'] = Temp_time['Date'].dt.year
    Temp_time = Temp_time.groupby(['Year', 'Component']).mean().reset_index()
    # Temp_time = Temp_time[Temp_time['Date'] >= '2019-01-01']
    # ax[0].set_title('Station_ID: ' + str(jj))
    ax.plot(Temp_time.loc[Temp_time['Component'] == 'Trend', 'Value'] / max(
        Temp_time.loc[Temp_time['Component'] == 'Trend', 'Value']), color='#2f4c58', alpha=0.7, lw=2)
    ax.set_ylabel('Trend')

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

# Plot Corr
All_final = pd.read_csv('D:\COVID19-Transit_Bikesharing\Divvy_Data\All_final_Divvy_R_BSTS_1003.csv')
All_final.columns
All_final1 = All_final.groupby(['from_stati']).tail(1)
corr_matr = All_final1[
    ['Pct.Male', 'Pct.Age_0_24', 'Pct.Age_25_40', 'Pct.Age_40_65', 'Pct.White', 'Pct.Black', 'Pct.Indian', 'Pct.Asian',
     'Pct.Unemploy', 'Total_Population', 'Income', 'College', 'Pct.Car', 'Pct.Transit', 'Pct.Bicycle', 'Pct.Walk',
     'Pct.WorkHome', 'Cumu_Cases', 'Cumu_Death', 'Cumu_Cases_Rate', 'Cumu_Death_Rate', 'COMMERCIAL',
     'INDUSTRIAL', 'INSTITUTIONAL', 'OPENSPACE', 'OTHERS', 'RESIDENTIAL',
     'Primary', 'Secondary', 'Minor', 'All_Road_Length', 'Bike_Route', 'Pct.WJob_Goods_Product', 'Pct.WJob_Utilities',
     'Pct.WJob_OtherServices', 'WTotal_Job_Density', 'Bus_stop_count', 'boardings', 'alightings', 'Distance_Busstop',
     'Rail_stop_count', 'rides', 'Distance_Rail', 'Near_Bike_station_Count', 'Near_Bike_Capacity',
     'Distance_Bikestation', 'Near_bike_pickups', 'Distance_City', 'PopDensity', 'EmployDensity',
     'Response', 'Cum_Relt_Effect', 'Relative_Impact', 'capacity', ]]

corr_matr.corr().to_csv('Divvy_Corr.csv')
fig, ax = plt.subplots(figsize=(11, 9))
plt.rcParams.update({'font.size': 14, 'font.family': "Times New Roman"})
sns.heatmap(corr_matr.corr(), fmt='',
            cmap=sns.diverging_palette(240, 130, as_cmap=True),
            square=True, xticklabels=True, yticklabels=True, linewidths=.5)
plt.tight_layout()

plt.rcParams.update({'font.size': 20, 'font.family': "Times New Roman"})
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(14, 6), sharex='col', sharey='row')
sns.regplot(x=corr_matr['Near_Bike_Capacity'], y=(corr_matr['Cum_Relt_Effect']), color='#2f4c58',
            scatter_kws={'s': (corr_matr['Response'] / 5), 'alpha': 0.5}, ax=ax[0][0])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=corr_matr['Near_Bike_Capacity'],
                                                                     y=corr_matr['Cum_Relt_Effect'])
plt.text(0.7, 0.9, '$R^2 = $' + str(round(r_value ** 2, 3)), horizontalalignment='center',
         verticalalignment='center', transform=ax[0][0].transAxes)

colnames(dat)
vif_test <-
  lm(Relative_Impact ~ Pct.Male + Pct.Age_0_24 + Pct.Age_25_40 + Pct.Age_40_65 + Pct.White + Pct.Black + Pct.Asian +
       Income + College + Pct.Car + Pct.BikeWalk + Pct.WorkHome + Cumu_Cases + Cumu_Death +
    COMMERCIAL + INDUSTRIAL + INSTITUTIONAL + OPENSPACE + RESIDENTIAL + Primary + Secondary + Minor + Bike_Route +
    Pct.WJob_Utilities + Pct.WJob_Goods_Product + WTotal_Job_Density + Bus_stop_count + boardings  +
    Distance_Busstop + Rail_stop_count + rides + Distance_Rail  + Near_Bike_Capacity +
    Distance_Bikestation + Near_bike_pickups + Distance_City + PopDensity + capacity,
     data = dat
  )
vif(vif_test)
summary(vif_test)

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

# Compare with 2019
ridership_old = pd.read_csv(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Day_count_Divvy_dropOutlier.csv', index_col=0)
# ridership_old = pd.read_pickle(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Trips_All\alltrips_chicago_202007.pkl')
ridership_old['Date'] = pd.to_datetime(ridership_old['startdate'])
ridership_old_2019 = ridership_old[(ridership_old['Date'] >= datetime.datetime(2019, 1, 1)) & (
        ridership_old['Date'] <= datetime.datetime(2019, 12, 31))]
ridership_old_2019 = ridership_old_2019.groupby(['Date']).sum()['trip_id'].reset_index()
# ridership_old_2019['weekofyear'] = ridership_old_2019['Date'].dt.weekofyear
# ridership_old_2019 = ridership_old_2019.groupby('weekofyear').sum().reset_index()
ridership_old_2019.columns = ['Date', 'trips_2019']

ridership_old_2020 = ridership_old[(ridership_old['Date'] >= datetime.datetime(2020, 1, 1)) & (
        ridership_old['Date'] <= datetime.datetime(2020, 12, 31))]
ridership_old_2020 = ridership_old_2020.groupby(['Date']).sum()['trip_id'].reset_index()
# ridership_old_2020['weekofyear'] = ridership_old_2020['Date'].dt.weekofyear
# ridership_old_2020 = ridership_old_2020.groupby('weekofyear').sum().reset_index()
ridership_old_2020.columns = ['Date', 'trips_2020']
ridership_old_2020['trips_2019'] = ridership_old_2019['trips_2019']
plt.plot(ridership_old_2020['trips_2020'] / ridership_old_2020['trips_2019'])

ridership_old['Baseline_Mobility'] = \
    ridership_old.loc[ridership_old['Date'] == datetime.datetime(2020, 1, 13), 'trip_id'].values[0]
ridership_old['Mobility'] = (ridership_old['trip_id'] / ridership_old['Baseline_Mobility']) * 100
ridership_old['Type'] = 'Bikesharing'
ridership_old = ridership_old[['Date', 'Type', 'Mobility']]
