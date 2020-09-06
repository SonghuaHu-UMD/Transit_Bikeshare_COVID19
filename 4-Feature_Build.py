import geopandas as gpd
import pandas as pd
import os
import matplotlib.pyplot as plt
import pyproj
import numpy as np
import datetime
import math
from functools import reduce
import seaborn as sns

pyproj.__version__  # (2.6.0)
gpd.__version__


################## Calculate all land use/socio-demograhic/road/cases related features ##############################
# Get the UTM code
def convert_wgs_to_utm(lon, lat):
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0' + utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band  # lat>0: N;
    else:
        epsg_code = '327' + utm_band
    return epsg_code


os.chdir(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\GIS')
Buffer_V = 500

# Read station
Station = gpd.read_file('Divvy_Station.shp')
# Get the utm code
utm_code = convert_wgs_to_utm(Station.geometry.bounds['minx'][0], Station.geometry.bounds['miny'][0])
# Project DTA 84 to utm
Station = Station.to_crs('EPSG:' + str(utm_code))

# Buffer the station
Buffer_S = Station.buffer(Buffer_V, cap_style=1).reset_index(drop=True).reset_index()
Buffer_S.columns = ['B_ID', 'geometry']
Buffer_S = gpd.GeoDataFrame(Buffer_S, geometry='geometry', crs='EPSG:4326')
Buffer_S = Buffer_S.to_crs('EPSG:' + str(utm_code))
Buffer_S['station_id'] = Station['from_stati']
# Buffer_S.to_file('DivvyBuffer.shp')

## Socio calculation
# Read BlockGroup
BlockGroup = gpd.read_file(r'D:\Transit\GIS\BlockGroup.shp')
BlockGroup = BlockGroup.to_crs('EPSG:' + str(utm_code))
# BlockGroup['ALAND'] = BlockGroup['ALAND'] * 3.86102e-7  # to sq. miles
BlockGroup['AREA'] = BlockGroup.geometry.area * 3.86102e-7  # to sq. miles
# plt.plot(BlockGroup['ALAND'],BlockGroup['AREA'])
# SJOIN with BlockGroup
SInBG = gpd.sjoin(Station, BlockGroup, how='inner', op='within').reset_index(drop=True)
SInBG_index = SInBG[['GEOID', 'from_stati', 'AREA']]
# DF_SInBG = pd.DataFrame(SInBG)
# Read socio-demogra
Socid_Raw = pd.read_csv(
    r'D:\COVID-19\RNN_COVID_Data\Social Demography\nhgis0021_csv\nhgis0021_ds239_20185_2018_blck_grp.csv',
    encoding='gbk')
Socid_Raw['GEOID'] = Socid_Raw['GISJOIN'].str[1:3] + Socid_Raw['GISJOIN'].str[4:7] + Socid_Raw['GISJOIN'].str[8:15]
Socid_Raw = Socid_Raw.merge(SInBG_index, on='GEOID')

# Gender
Socid_Raw['Male'] = Socid_Raw['AJWBE002']
# Age
Socid_Raw['Total_Population'] = Socid_Raw['AJWBE001']
# Socid_Raw['PopDensity'] =
Socid_Raw['Age_0_24'] = \
    sum(Socid_Raw[col] for col in
        ['AJWBE%03d' % num for num in range(3, 11)] + ['AJWBE%03d' % num for num in range(27, 35)])
Socid_Raw['Age_25_40'] = \
    sum(Socid_Raw[col] for col in
        ['AJWBE%03d' % num for num in range(11, 14)] + ['AJWBE%03d' % num for num in range(35, 38)])
Socid_Raw['Age_40_65'] = \
    sum(Socid_Raw[col] for col in
        ['AJWBE%03d' % num for num in range(14, 20)] + ['AJWBE%03d' % num for num in range(38, 44)])
Socid_Raw['Age_65_'] = \
    sum(Socid_Raw[col] for col in
        ['AJWBE%03d' % num for num in range(20, 26)] + ['AJWBE%03d' % num for num in range(44, 50)])
# Race
Socid_Raw['White'] = Socid_Raw['AJWNE002']
Socid_Raw['Black'] = Socid_Raw['AJWNE003']
Socid_Raw['Indian'] = Socid_Raw['AJWNE004']
Socid_Raw['Asian'] = Socid_Raw['AJWNE005']
# Percentage
for each in ['Male', 'Age_0_24', 'Age_25_40', 'Age_40_65', 'Age_65_', 'White', 'Black', 'Indian', 'Asian']:
    Socid_Raw['Pct.' + each] = Socid_Raw[each] / Socid_Raw['Total_Population']
# Employment
Socid_Raw['LaborForce'] = Socid_Raw['AJ1CE002']
Socid_Raw['Unemployed'] = Socid_Raw['AJ1CE005']
Socid_Raw['Employed'] = Socid_Raw['AJ1CE004']
Socid_Raw['Pct.Unemploy'] = Socid_Raw['Unemployed'] / Socid_Raw['LaborForce']
# Income
Socid_Raw['Income'] = Socid_Raw["AJZAE001"]
# Travel infomation
Socid_Raw['Pct.Car'] = Socid_Raw['AJXCE002'] / Socid_Raw['AJXCE001']
Socid_Raw['Pct.Transit'] = Socid_Raw['AJXCE010'] / Socid_Raw['AJXCE001']
Socid_Raw['Pct.WorkHome'] = Socid_Raw['AJXCE021'] / Socid_Raw['AJXCE001']
# Education
Socid_Raw['College'] = (Socid_Raw['AJYPE022'] + Socid_Raw['AJYPE023'] + Socid_Raw['AJYPE024'] + Socid_Raw['AJYPE025']) / \
                       Socid_Raw['AJYPE001']
# Need
Socid_Raw = Socid_Raw[
    ['Pct.Male', 'Pct.Age_0_24', 'Pct.Age_25_40', 'Pct.Age_40_65', 'Pct.White', 'Pct.Black', 'Pct.Indian', 'Pct.Asian',
     'Pct.Unemploy', 'Total_Population', 'GEOID', 'Income', 'Employed', 'College', 'Pct.Car', 'Pct.Transit',
     'Pct.WorkHome', 'AREA', 'from_stati']]
# Socid_Raw.isnull().sum()
# fill na: 40890 40930
Socid_Raw_Final = Socid_Raw.fillna(Socid_Raw.mean())

# Read ZIPCODE
ZIPCODE = gpd.read_file(r'D:\Transit\GIS\ZIPCODEUS.shp')
ZIPCODE = ZIPCODE.to_crs('EPSG:' + str(utm_code))
# SJOIN with ZIPCODE
SInZIP = gpd.sjoin(Station, ZIPCODE, how='inner', op='within').reset_index(drop=True)
# Et the ZCODE
StationZIP = pd.DataFrame(SInZIP[['from_stati', 'ZIP_CODE']])
# Get the number of cases
Num_Cases = pd.read_csv(r'D:\Transit\COVID-19_Cases__Tests__and_Deaths_by_ZIP_Code (1).csv')
Num_Cases['Week Start'] = pd.to_datetime(Num_Cases['Week Start'])
Num_Cases['Week End'] = pd.to_datetime(Num_Cases['Week End'])
Num_Cases = Num_Cases.sort_values(by=['ZIP Code', 'Week Start']).reset_index(drop=True)
Num_Cases['Diff_Start'] = (Num_Cases['Week Start'] - datetime.datetime(2020, 4, 30)).dt.days
# Find the max negative
Num_Cases_Neg = Num_Cases[Num_Cases['Diff_Start'] <= 0]
Num_Cases_Cum = Num_Cases_Neg.groupby(['ZIP Code']).tail(1)
Num_Cases_Cum = Num_Cases_Cum[['ZIP Code', 'Cases - Cumulative', 'Deaths - Cumulative', 'Population']]
Num_Cases_Cum.columns = ['ZIP_CODE', 'Cumu_Cases', 'Cumu_Death', 'Population']
Num_Cases_Cum['Cumu_Cases_Rate'] = Num_Cases_Cum['Cumu_Cases'] / Num_Cases_Cum['Population']
Num_Cases_Cum['Cumu_Death_Rate'] = Num_Cases_Cum['Cumu_Death'] / Num_Cases_Cum['Population']
Num_Cases_Cum = Num_Cases_Cum.drop(['Population'], axis=1)
Num_Cases_Cum.to_csv('Num_Cases_Cum.csv')
StationZIP = StationZIP.merge(Num_Cases_Cum, on='ZIP_CODE', how='left')
StationZIP = StationZIP.fillna(0)

# Read Landuse
Landuse = gpd.read_file(r'D:\Transit\GIS\Landuse2013_CMAP.shp')
Landuse = Landuse.to_crs('EPSG:' + str(utm_code))
# INTERSECT with LANDUSE
SInLand = gpd.overlay(Landuse, Buffer_S, how='intersection')
SInLand['Area'] = SInLand['geometry'].area
SInLand.columns
SInLand_Area = SInLand.groupby(['from_stati', 'LANDUSE']).sum()['Area'].reset_index()
SInLand_Area['Land_Use_F2'] = [var[0:2] for var in SInLand_Area['LANDUSE']]
SInLand_Area['Land_Use_F1'] = [var[0:1] for var in SInLand_Area['LANDUSE']]
# SInLand.to_file('SInLand.shp')
# set(SInLand_Area['LANDUSE'])
SInLand_Area['Land_Use_Des'] = np.nan
SInLand_Area.loc[SInLand_Area['Land_Use_F2'] == '11', 'Land_Use_Des'] = 'RESIDENTIAL'
SInLand_Area.loc[SInLand_Area['Land_Use_F2'] == '12', 'Land_Use_Des'] = 'COMMERCIAL'
SInLand_Area.loc[SInLand_Area['Land_Use_F2'] == '13', 'Land_Use_Des'] = 'INSTITUTIONAL'
SInLand_Area.loc[SInLand_Area['Land_Use_F2'] == '14', 'Land_Use_Des'] = 'INDUSTRIAL'
SInLand_Area.loc[SInLand_Area['Land_Use_F2'] == '15', 'Land_Use_Des'] = 'TCUW'
SInLand_Area.loc[SInLand_Area['Land_Use_F1'] == '3', 'Land_Use_Des'] = 'OPENSPACE'
SInLand_Area.loc[SInLand_Area['Land_Use_F1'].isin(['2', '4', '5', '6', '9']), 'Land_Use_Des'] = 'OTHERS'
SInLand_Area.loc[SInLand_Area['Land_Use_Des'] == 'TCUW', 'Land_Use_Des'] = 'OTHERS'
SInLand_Area_New = SInLand_Area.groupby(['station_id', 'Land_Use_Des']).sum()['Area'].reset_index()

# Calculate the LUM
TAZ_LAND_USE_ALL = SInLand_Area.groupby(['station_id']).sum()['Area'].reset_index()
TAZ_LAND_USE_ALL.columns = ['station_id', 'ALL_AREA']
LandUse_Area1 = SInLand_Area_New.merge(TAZ_LAND_USE_ALL, on='station_id')
LandUse_Area1['Percen'] = LandUse_Area1['Area'] / LandUse_Area1['ALL_AREA']
LandUse_Area1['LogPercen'] = np.log(LandUse_Area1['Percen'])
LandUse_Area1['LogP*P'] = LandUse_Area1['Percen'] * LandUse_Area1['LogPercen']
LandUse_Area2 = LandUse_Area1.groupby(['station_id']).sum()['LogP*P'].reset_index()
LandUse_Area3 = SInLand_Area_New.groupby(['station_id']).count()['Land_Use_Des'].reset_index()
LandUse_Area3.columns = ['station_id', 'Count']
LandUse_Area2 = LandUse_Area2.merge(LandUse_Area3, on='station_id')
LandUse_Area2['LUM'] = LandUse_Area2['LogP*P'] * ((-1) / (np.log(LandUse_Area2['Count'])))
LUM = LandUse_Area2[['station_id', 'LUM']].fillna(LandUse_Area2.mean())
LandUse_Area_PCT = LandUse_Area1[['station_id', 'Land_Use_Des', 'Area', 'Percen']]
LandUse_Area_PCT_Final = LandUse_Area_PCT.pivot('station_id', 'Land_Use_Des', 'Percen').fillna(0).reset_index()

# Read roads
Roads = gpd.read_file(r'ROAD.shp')
Roads = Roads.to_crs('EPSG:' + str(utm_code))
# INTERSECT with LANDUSE
SInRoad = gpd.clip(Roads, Buffer_S)
SInRoad1 = gpd.overlay(Roads, Buffer_S, how='intersection')
SInRoad1['Length'] = SInRoad1['geometry'].length
SInRoad_Length = SInRoad1.groupby(['station_id', 'fclass']).sum()['Length'].reset_index()
set(SInRoad_Length['fclass'])
Road_Length_With_Type = SInRoad_Length.pivot('station_id', 'fclass', 'Length').fillna(0).reset_index()
SInRoad_Length.groupby(['fclass']).sum()['Length'].plot.bar()

# SInRoad1.to_file('SInRoad_overlay.shp')
Road_Length_With_Type['Primary'] = Road_Length_With_Type['motorway'] + Road_Length_With_Type['motorway_link'] + \
                                   Road_Length_With_Type['trunk'] + Road_Length_With_Type['trunk_link'] + \
                                   Road_Length_With_Type['primary'] + Road_Length_With_Type['primary_link'] + \
                                   Road_Length_With_Type['tertiary'] + Road_Length_With_Type['tertiary_link']
Road_Length_With_Type['Secondary'] = Road_Length_With_Type['secondary'] + Road_Length_With_Type['secondary_link'] + \
                                     Road_Length_With_Type['residential'] + Road_Length_With_Type['tertiary']
Road_Length_With_Type['Minor'] = Road_Length_With_Type['service'] + Road_Length_With_Type['steps'] + \
                                 Road_Length_With_Type['track'] + Road_Length_With_Type['living_street']
Road_Length_With_Type['All_Road_Length'] = Road_Length_With_Type.iloc[:, 1:-3].sum(axis=1)
Road_Length_With_Type = Road_Length_With_Type[['station_id', 'Primary', 'Secondary', 'Minor', 'All_Road_Length']]
# Calculate density
Road_Length_With_Type['Primary'] = (Road_Length_With_Type['Primary'] * 0.000621371) / (
        3.1415926 * (Buffer_V * 0.000621371) * (Buffer_V * 0.000621371))
Road_Length_With_Type['Secondary'] = (Road_Length_With_Type['Secondary'] * 0.000621371) / (
        3.1415926 * (Buffer_V * 0.000621371) * (Buffer_V * 0.000621371))
Road_Length_With_Type['Minor'] = (Road_Length_With_Type['Minor'] * 0.000621371) / (
        3.1415926 * (Buffer_V * 0.000621371) * (Buffer_V * 0.000621371))
Road_Length_With_Type['All_Road_Length'] = (Road_Length_With_Type['All_Road_Length'] * 0.000621371) / (
        3.1415926 * (Buffer_V * 0.000621371) * (Buffer_V * 0.000621371))

# Read job density
Job_density = pd.read_csv(r'D:\Transit\il_od_aux_JT00_2015.csv')
Job_density['w_geocode'] = Job_density['w_geocode'].astype(str).str[0:-3]

W_Job = Job_density.groupby(['w_geocode']).sum()[
    ['S000', 'SE01', 'SE02', 'SE03', 'SA01', 'SA02', 'SA03', 'SI01', 'SI02', 'SI03']].reset_index()
W_Job.columns = ['GEOID', 'WTotal_Job', 'WJob_1250', 'WJob_1250_3333', 'WJob_3333', 'WJob_29', 'WJob_30_54', 'WJob_55',
                 'WJob_Goods_Product', 'WJob_Utilities', 'WJob_OtherServices']
W_Job = W_Job.merge(SInBG_index, on='GEOID', how='right')
W_Job = W_Job.fillna(W_Job.mean())
for jj in ['WJob_1250', 'WJob_1250_3333', 'WJob_3333', 'WJob_29', 'WJob_30_54', 'WJob_55',
           'WJob_Goods_Product', 'WJob_Utilities', 'WJob_OtherServices']:
    W_Job['Pct.' + jj] = W_Job[jj] / W_Job['WTotal_Job']

W_Job['WTotal_Job_Density'] = W_Job['WTotal_Job'] / (W_Job['AREA'] * 1e3)

W_Job = W_Job.drop(
    ['GEOID', 'AREA', 'WTotal_Job', 'WJob_1250', 'WJob_1250_3333', 'WJob_3333', 'WJob_29', 'WJob_30_54', 'WJob_55',
     'WJob_Goods_Product', 'WJob_Utilities', 'WJob_OtherServices'], axis=1)

# # Read population (new)
T_pop = pd.read_csv(r'D:\Transit\Population_by_2010_Census_Block.csv')
T_pop['GEOID'] = T_pop['CENSUS BLOCK FULL'].astype(str).str[0:-3]
T_pop = T_pop[['GEOID', 'TOTAL POPULATION']]
T_pop = T_pop.groupby('GEOID').sum()['TOTAL POPULATION'].reset_index()
T_pop = T_pop.merge(SInBG_index, on='GEOID', how='right')
T_pop = T_pop.fillna(T_pop.mean())
T_pop['NPop_Density'] = T_pop['TOTAL POPULATION'] / (T_pop['AREA'] * 1e3)
T_pop = T_pop[['station_id', 'NPop_Density']]
# T_pop.describe().T

# Merge all data
dfs = [Road_Length_With_Type, LandUse_Area_PCT_Final, LUM, StationZIP, Socid_Raw_Final, W_Job]
All_final = reduce(lambda left, right: pd.merge(left, right, on='station_id'), dfs)
All_final.isnull().sum()
All_final.describe().T

# Change unit
All_final['PopDensity'] = (All_final['Total_Population'] / 1e3) / All_final['AREA']
All_final['Income'] = All_final['Income'] / 1e3
All_final['Cumu_Cases'] = All_final['Cumu_Cases'] / 1e3
All_final['Cumu_Death'] = All_final['Cumu_Death'] / 1e3
All_final['EmployDensity'] = (All_final['Employed'] / 1e3) / All_final['AREA']
# Output
All_final.to_csv('D:\Transit\Features_Transit_0822.csv')
################## Calculate all land use/socio-demograhic/road/cases related features ##############################

################## Calculate all land use/socio-demograhic/road/cases related features ##############################
# Merge with Transit and others
import pandas as pd
import os
from functools import reduce
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
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


os.chdir(r'D:\Transit')
# Ride_C = pd.read_csv(r'LStations_Chicago.csv', index_col=0)
Impact_C = pd.read_csv(r'Impact_Sta.csv', index_col=0)
Features = pd.read_csv(r'Features_Transit_0822.csv', index_col=0)
dfs = [Impact_C, Features]
All_final = reduce(lambda left, right: pd.merge(left, right, on='station_id'), dfs)
All_final.describe().T

# Read transit related data
Stop_ID = pd.read_csv(r'D:\Transit\google_transit\stops.txt')
Stop_ID = Stop_ID[Stop_ID['parent_station'].isin(All_final['station_id'])]
Stop_times = pd.read_csv(r'D:\Transit\google_transit\stop_times.txt')
Stop_times = Stop_times[Stop_times['stop_id'].isin(Stop_ID['stop_id'])]
Stop_times['hour'] = [var[0] for var in Stop_times['arrival_time'].str.split(':')]
Stop_times['minute'] = [var[1] for var in Stop_times['arrival_time'].str.split(':')]
Stop_times['second'] = [var[2] for var in Stop_times['arrival_time'].str.split(':')]
Stop_times['hour'] = Stop_times['hour'].str.replace('24', '00')
Stop_times['hour'] = Stop_times['hour'].str.replace('25', '01')
Stop_times['arrival_time'] = pd.to_datetime(
    '2020-06-01 ' + Stop_times['hour'] + ':' + Stop_times['minute'] + ':' + Stop_times['second'])
Stop_times = Stop_times.sort_values(by=['stop_id', 'arrival_time']).reset_index(drop=True)
Stop_times['Freq'] = Stop_times.groupby(['stop_id'])['arrival_time'].diff()
Stop_times['Freq'] = [var.total_seconds() / 60 for var in Stop_times['Freq']]

# First is the number of trips
No_Trips = Stop_times.groupby(['stop_id']).count()['trip_id'].reset_index()
Freq_Trips = Stop_times.groupby(['stop_id']).mean()['Freq'].reset_index()
No_Trips = No_Trips.merge(Freq_Trips, on='stop_id')
Stop_ID = Stop_ID[['stop_id', 'parent_station']]
No_Trips = No_Trips.merge(Stop_ID, on='stop_id')
No_Trips_1 = No_Trips.groupby(['parent_station']).sum()['trip_id'].reset_index()
No_Trips_2 = No_Trips.groupby(['parent_station']).mean()['Freq'].reset_index()
No_Fre_Trips = No_Trips_1.merge(No_Trips_2, on='parent_station')
No_Fre_Trips.columns = ['station_id', 'Num_trips', 'Freq']

# Calculate the decrease based on 2019
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
Rider_2020_Impact = Rider_2020_Impact[Rider_2020_Impact['RELIMP'] < 0]
Rider_2020_Impact.describe()
Rider_2020_Impact = Rider_2020_Impact[['station_id', 'RELIMP']]

# Merge all in one matrix
All_final = All_final.merge(No_Fre_Trips, on='station_id')
All_final = All_final.merge(Rider_2020_Impact, on='station_id', how='left')
All_final.loc[All_final['RELIMP'].isna(), 'RELIMP'] = All_final.loc[All_final['RELIMP'].isna(), 'Relative_Impact']
# plt.plot(All_final['Relative_Impact'],All_final['RELIMP'])
# Change unit
All_final['Num_trips'] = All_final['Num_trips'] / 1e3
All_final['rides'] = All_final['rides'] / 1e3
All_final.describe().T
# All_final.to_csv('All_final_Transit_R_0822.csv')


# PLOT CORR
corr_matr = All_final[
    ['Relative_Impact', 'rides', 'RELIMP', 'COMMERCIAL', 'INDUSTRIAL', 'INSTITUTIONAL', 'OPENSPACE', 'RESIDENTIAL',
     'LUM', 'Cumu_Cases', 'Cumu_Death', 'Cumu_Cases_Rate', 'Cumu_Death_Rate', 'Pct.Male', 'Pct.Age_0_24',
     'Pct.Age_25_40', 'Pct.Age_40_65', 'Pct.White', 'Pct.Black', 'Income', 'College', 'Pct.WJob_Goods_Product',
     'Pct.WJob_Utilities', 'Pct.WJob_OtherServices', 'WTotal_Job_Density', 'PopDensity', 'Num_trips', 'Freq']]
# corr_matr['Relative_Impact'] = -corr_matr['Relative_Impact']
corr_matr.rename({'Relative_Impact': 'Relative Impact', 'rides': 'Ridership', 'RELIMP': 'Relative Decline',
                  'COMMERCIAL': 'Commercial', 'INDUSTRIAL': 'Prop. Industrial', 'INSTITUTIONAL': 'Prop. Institutional',
                  'OPENSPACE': 'Prop. Open space', 'RESIDENTIAL': 'Prop. Residential', 'Cumu_Cases': 'Cases',
                  'Cumu_Death': 'Death', 'Cumu_Cases_Rate': 'Cases_Rate', 'Cumu_Death_Rate': 'Death_Rate',
                  'Pct.Male': 'Prop. Male', 'Pct.Age_0_24': 'Prop. Age-0_24', 'Pct.Age_25_40': 'Prop. Age-25_40',
                  'Pct.Age_40_65': 'Prop. Age-40_65', 'Pct.White': 'Prop. Race-White', 'Pct.Black': 'Prop. Race-Black',
                  'Income': 'Median Income', 'College': 'Prop. College Degree',
                  'Pct.WJob_Goods_Product': 'Prop. Job-Goods', 'Pct.WJob_Utilities': 'Prop. Job-Utilities',
                  'Pct.WJob_OtherServices': 'Prop. Job-Others', 'WTotal_Job_Density': 'Job Density',
                  'PopDensity': 'Population Density', 'Num_trips': 'Trips', 'Freq': 'Transit Frequency'}, axis=1,
                 inplace=True)
corr_matr.describe().T.to_csv('Describe.csv')
corr_matr.corr().to_csv('Corr.csv')
corr_p = calculate_pvalues(corr_matr)
corr_p[(corr_p > 0.1)] = 0.1
corr_p[(corr_p < 0.1) & (corr_p >= 0.05)] = 0.05
corr_p[(corr_p < 0.05) & (corr_p >= 0.01)] = 0.01
corr_p[(corr_p < 0.01) & (corr_p >= 0.001)] = 0.001
corr_p[(corr_p < 0.001) & (corr_p >= 0)] = 0

corr_p = corr_p.replace({0.1: '', 0.05: '.', 0.01: '*', 0.001: '**', 0: "***"})

fig, ax = plt.subplots(figsize=(11, 9))
plt.rcParams.update({'font.size': 14, 'font.family': "Times New Roman"})
sns.heatmap(corr_matr.corr(), annot=corr_p.values, fmt='',
            cmap=sns.diverging_palette(240, 130, as_cmap=True),
            square=True, xticklabels=True, yticklabels=True, linewidths=.5)
plt.tight_layout()
plt.savefig('CORR.png', dpi=600)
plt.savefig('CORR.svg')

# Plot linear regression
# #96bb7c
plt.rcParams.update({'font.size': 20, 'font.family': "Times New Roman"})
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(14, 6), sharex='col', sharey='row')
sns.regplot(x=corr_matr['Prop. College Degree'], y=(corr_matr['Relative Impact']), color='#2f4c58',
            scatter_kws={'s': (corr_matr['Ridership'] * 6), 'alpha': 0.5}, ax=ax[0][0])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=corr_matr['Prop. College Degree'],
                                                                     y=corr_matr['Relative Impact'])
plt.text(0.7, 0.9, '$R^2 = $' + str(round(r_value ** 2, 3)), horizontalalignment='center',
         verticalalignment='center', transform=ax[0][0].transAxes)

sns.regplot(x=corr_matr['Median Income'], y=(corr_matr['Relative Impact']), color='#2f4c58',
            scatter_kws={'s': (corr_matr['Ridership'] * 6), 'alpha': 0.5}, ax=ax[0][1])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=corr_matr['Median Income'],
                                                                     y=corr_matr['Relative Impact'])
plt.text(0.7, 0.9, '$R^2 = $' + str(round(r_value ** 2, 3)), horizontalalignment='center',
         verticalalignment='center', transform=ax[0][1].transAxes)

sns.regplot(x=corr_matr['Prop. Race-White'], y=(corr_matr['Relative Impact']), color='#2f4c58',
            scatter_kws={'s': (corr_matr['Ridership'] * 6), 'alpha': 0.5}, ax=ax[0][2])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=corr_matr['Prop. Race-White'],
                                                                     y=corr_matr['Relative Impact'])
plt.text(0.7, 0.9, '$R^2 = $' + str(round(r_value ** 2, 3)), horizontalalignment='center',
         verticalalignment='center', transform=ax[0][2].transAxes)

sns.regplot(x=corr_matr['Prop. Race-Black'], y=(corr_matr['Relative Impact']), color='#96bb7c',
            scatter_kws={'s': (corr_matr['Ridership'] * 6), 'alpha': 0.5}, ax=ax[0][3])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=corr_matr['Prop. Race-Black'],
                                                                     y=corr_matr['Relative Impact'])
plt.text(0.7, 0.9, '$R^2 = $' + str(round(r_value ** 2, 3)), horizontalalignment='center',
         verticalalignment='center', transform=ax[0][3].transAxes)

sns.regplot(x=corr_matr['Prop. College Degree'], y=(corr_matr['Ridership']), color='#96bb7c',
            scatter_kws={'s': (corr_matr['Ridership'] * 6), 'alpha': 0.5}, ax=ax[1][0])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=corr_matr['Prop. College Degree'],
                                                                     y=corr_matr['Ridership'])
plt.text(0.7, 0.9, '$R^2 = $' + str(round(r_value ** 2, 3)), horizontalalignment='center',
         verticalalignment='center', transform=ax[1][0].transAxes)

sns.regplot(x=corr_matr['Median Income'], y=(corr_matr['Ridership']), color='#96bb7c',
            scatter_kws={'s': (corr_matr['Ridership'] * 6), 'alpha': 0.5}, ax=ax[1][1])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=corr_matr['Median Income'],
                                                                     y=corr_matr['Ridership'])
plt.text(0.7, 0.9, '$R^2 = $' + str(round(r_value ** 2, 3)), horizontalalignment='center',
         verticalalignment='center', transform=ax[1][1].transAxes)

sns.regplot(x=corr_matr['Prop. Race-White'], y=(corr_matr['Ridership']), color='#96bb7c',
            scatter_kws={'s': (corr_matr['Ridership'] * 6), 'alpha': 0.5}, ax=ax[1][2])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=corr_matr['Prop. Race-White'],
                                                                     y=corr_matr['Ridership'])
plt.text(0.7, 0.9, '$R^2 = $' + str(round(r_value ** 2, 3)), horizontalalignment='center',
         verticalalignment='center', transform=ax[1][2].transAxes)

sns.regplot(x=corr_matr['Prop. Race-Black'], y=(corr_matr['Ridership']), color='#2f4c58',
            scatter_kws={'s': (corr_matr['Ridership'] * 6), 'alpha': 0.5}, ax=ax[1][3])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=corr_matr['Prop. Race-Black'],
                                                                     y=corr_matr['Ridership'])
plt.text(0.7, 0.9, '$R^2 = $' + str(round(r_value ** 2, 3)), horizontalalignment='center',
         verticalalignment='center', transform=ax[1][3].transAxes)

for axx in [ax[0][1], ax[0][2], ax[0][3], ax[1][1], ax[1][2], ax[1][3]]:
    axx.set_ylabel('')
for axx in [ax[0][0], ax[0][2], ax[0][3], ax[0][1]]:
    axx.set_xlabel('')
plt.subplots_adjust(top=0.975, bottom=0.12, left=0.076, right=0.986, hspace=0.123, wspace=0.142)
plt.savefig('CORR-Linear.png', dpi=600)
plt.savefig('CORR-Linear.svg')
