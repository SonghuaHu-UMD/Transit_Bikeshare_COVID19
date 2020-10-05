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
