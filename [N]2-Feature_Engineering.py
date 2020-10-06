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
from scipy.stats import pearsonr
import scipy.stats

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
Buffer_S['from_stati'] = Station['from_stati']
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
Socid_Raw['Pct.Bicycle'] = Socid_Raw['AJXCE018'] / Socid_Raw['AJXCE001']
Socid_Raw['Pct.Walk'] = Socid_Raw['AJXCE019'] / Socid_Raw['AJXCE001']

# Education
Socid_Raw['College'] = (Socid_Raw['AJYPE022'] + Socid_Raw['AJYPE023'] + Socid_Raw['AJYPE024'] + Socid_Raw['AJYPE025']) / \
                       Socid_Raw['AJYPE001']
# Need
Socid_Raw = Socid_Raw[
    ['Pct.Male', 'Pct.Age_0_24', 'Pct.Age_25_40', 'Pct.Age_40_65', 'Pct.White', 'Pct.Black', 'Pct.Indian', 'Pct.Asian',
     'Pct.Unemploy', 'Total_Population', 'GEOID', 'Income', 'Employed', 'College', 'Pct.Car', 'Pct.Transit',
     'Pct.Bicycle', 'Pct.Walk', 'Pct.WorkHome', 'AREA', 'from_stati']]
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
SInLand_Area_New = SInLand_Area.groupby(['from_stati', 'Land_Use_Des']).sum()['Area'].reset_index()

# Calculate the LUM
TAZ_LAND_USE_ALL = SInLand_Area.groupby(['from_stati']).sum()['Area'].reset_index()
TAZ_LAND_USE_ALL.columns = ['from_stati', 'ALL_AREA']
LandUse_Area1 = SInLand_Area_New.merge(TAZ_LAND_USE_ALL, on='from_stati')
LandUse_Area1['Percen'] = LandUse_Area1['Area'] / LandUse_Area1['ALL_AREA']
LandUse_Area1['LogPercen'] = np.log(LandUse_Area1['Percen'])
LandUse_Area1['LogP*P'] = LandUse_Area1['Percen'] * LandUse_Area1['LogPercen']
LandUse_Area2 = LandUse_Area1.groupby(['from_stati']).sum()['LogP*P'].reset_index()
LandUse_Area3 = SInLand_Area_New.groupby(['from_stati']).count()['Land_Use_Des'].reset_index()
LandUse_Area3.columns = ['from_stati', 'Count']
LandUse_Area2 = LandUse_Area2.merge(LandUse_Area3, on='from_stati')
LandUse_Area2['LUM'] = LandUse_Area2['LogP*P'] * ((-1) / (np.log(LandUse_Area2['Count'])))
LUM = LandUse_Area2[['from_stati', 'LUM']].fillna(LandUse_Area2.mean())
LandUse_Area_PCT = LandUse_Area1[['from_stati', 'Land_Use_Des', 'Area', 'Percen']]
LandUse_Area_PCT_Final = LandUse_Area_PCT.pivot('from_stati', 'Land_Use_Des', 'Percen').fillna(0).reset_index()

# Read roads
Roads = gpd.read_file(r'D:\Transit\GIS\ROAD.shp')
Roads = Roads.to_crs('EPSG:' + str(utm_code))
# INTERSECT with roads
SInRoad = gpd.clip(Roads, Buffer_S)
SInRoad1 = gpd.overlay(Roads, Buffer_S, how='intersection')
SInRoad1['Length'] = SInRoad1['geometry'].length
SInRoad_Length = SInRoad1.groupby(['from_stati', 'fclass']).sum()['Length'].reset_index()
set(SInRoad_Length['fclass'])
Road_Length_With_Type = SInRoad_Length.pivot('from_stati', 'fclass', 'Length').fillna(0).reset_index()
# SInRoad_Length.groupby(['fclass']).sum()['Length'].plot.bar()

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
Road_Length_With_Type = Road_Length_With_Type[['from_stati', 'Primary', 'Secondary', 'Minor', 'All_Road_Length']]
# Calculate density
Road_Length_With_Type['Primary'] = (Road_Length_With_Type['Primary'] * 0.000621371) / (
        3.1415926 * (Buffer_V * 0.000621371) * (Buffer_V * 0.000621371))
Road_Length_With_Type['Secondary'] = (Road_Length_With_Type['Secondary'] * 0.000621371) / (
        3.1415926 * (Buffer_V * 0.000621371) * (Buffer_V * 0.000621371))
Road_Length_With_Type['Minor'] = (Road_Length_With_Type['Minor'] * 0.000621371) / (
        3.1415926 * (Buffer_V * 0.000621371) * (Buffer_V * 0.000621371))
Road_Length_With_Type['All_Road_Length'] = (Road_Length_With_Type['All_Road_Length'] * 0.000621371) / (
        3.1415926 * (Buffer_V * 0.000621371) * (Buffer_V * 0.000621371))

# Read bike lanes
Roads_Bike = gpd.read_file(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\GIS\Bikeroute.shp')
Roads_Bike = Roads_Bike.to_crs('EPSG:' + str(utm_code))
# INTERSECT with roads
SInRoadBike = gpd.clip(Roads_Bike, Buffer_S)
SInRoadBike = gpd.overlay(SInRoadBike, Buffer_S, how='intersection')
SInRoadBike['Length'] = SInRoadBike['geometry'].length
SInRoad_BikeLength = SInRoadBike.groupby(['from_stati']).sum()['Length'].reset_index()
SInRoad_BikeLength.columns = ['from_stati', 'Bike_Route']
SInRoad_BikeLength['Bike_Route'] = (SInRoad_BikeLength['Bike_Route'] * 0.000621371) / (
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
T_pop = T_pop[['from_stati', 'NPop_Density']]
# T_pop.describe().T

# Transit_Related variables
# Bus
# How many ridership
# Read bus stop
Busstop = gpd.read_file(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\GIS\Busstop_Ridership.shp')
Busstop = Busstop.to_crs('EPSG:' + str(utm_code))
# SJOIN with Buffer
SInBusstop = gpd.sjoin(Busstop, Buffer_S, how='inner', op='within').reset_index(drop=True)
SInBusstop.columns
# Number of busstop
Numofbusstop = SInBusstop.groupby('from_stati').count()['OBJECTID_1'].reset_index()
Numofbusstop.columns = ['from_stati', 'Bus_stop_count']
# Ridership
Bus_Rider = SInBusstop.groupby('from_stati').sum()[['boardings', 'alightings']].reset_index()
# Nearest Distance to busstop
Distance_Bus = pd.DataFrame({'Distance_Busstop': Station.geometry.apply(lambda x: Busstop.distance(x).min()),
                             'from_stati': Station.from_stati})

# Transit
Railstop = gpd.read_file(r'D:\Transit\GIS\Relative_Impact.shp')
Railstop = Railstop.to_crs('EPSG:' + str(utm_code))
# SJOIN with Buffer
SInRailstop = gpd.sjoin(Railstop, Buffer_S, how='inner', op='within').reset_index(drop=True)
# Number of rail stop
NumofRailstop = SInRailstop.groupby('from_stati').count()['Field1'].reset_index()
NumofRailstop.columns = ['from_stati', 'Rail_stop_count']
# Ridership
Rail_Rider = SInRailstop.groupby('from_stati').sum()[['rides']].reset_index()
# Nearest Distance to rail stop
Distance_Rail = pd.DataFrame({'Distance_Rail': Station.geometry.apply(lambda x: Railstop.distance(x).min()),
                              'from_stati': Station.from_stati})

# Bike station related
Capacity_Bike = pd.read_csv(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Day_count_2019.csv', index_col=0)
Capacity_Bike = Capacity_Bike[['from_station_id', 'from_station_capacity']]
SInBike = gpd.sjoin(Station, Buffer_S, how='inner', op='within').reset_index(drop=True)
# Not include itself
SInBike = SInBike[SInBike['from_stati_left'] != SInBike['from_stati_right']]
# Number of bike station
Numofbikestation = SInBike.groupby('from_stati_right').count()['Field1'].reset_index()
Numofbikestation.columns = ['from_stati', 'Near_Bike_station_Count']
# Capacity
Capacity_Bike.columns = ['from_stati_left', 'Bike_Capacity']
SInBike = SInBike.merge(Capacity_Bike, on='from_stati_left')
Numofbikecapacity = SInBike.groupby('from_stati_right').sum()['Bike_Capacity'].reset_index()
Numofbikecapacity.columns = ['from_stati', 'Near_Bike_Capacity']
# Nearest Distance to bike station
Distance_Bike = pd.DataFrame({'Distance_Bikestation': Station.geometry.apply(lambda x: sorted(Station.distance(x))[1]),
                              'from_stati': Station.from_stati})
# Pickups
Bike_Rider = SInBike.groupby('from_stati_left').sum()[['trip_id']].reset_index()
Bike_Rider.columns = ['from_stati', 'Near_bike_pickups']

# Distance to city center
# Station 35 as city center
Distance_City_Center = Station.copy()
Distance_City_Center['Distance_City'] = [
    x.distance(list(Distance_City_Center.loc[Distance_City_Center['from_stati'] == 35, 'geometry'])[0]) for x in
    Distance_City_Center['geometry']]
Distance_City_Center = Distance_City_Center[['from_stati', 'Distance_City']]

# Merge all data
dfs = [Socid_Raw_Final, StationZIP, LandUse_Area_PCT_Final, Road_Length_With_Type, SInRoad_BikeLength, W_Job,
       Numofbusstop, Bus_Rider, Distance_Bus, NumofRailstop, Rail_Rider, Distance_Rail, Numofbikestation,
       Numofbikecapacity, Distance_Bike, Bike_Rider, Distance_City_Center]
All_final = reduce(lambda left, right: pd.merge(left, right, on='from_stati', how='outer'), dfs)
All_final.isnull().sum()
All_final.describe().T

# Change unit and fill na
All_final = All_final.fillna(0)
All_final['PopDensity'] = (All_final['Total_Population'] / 1e3) / All_final['AREA']
All_final['Income'] = All_final['Income'] / 1e3
All_final['Cumu_Cases'] = All_final['Cumu_Cases'] / 1e3
All_final['Cumu_Death'] = All_final['Cumu_Death'] / 1e3
All_final['EmployDensity'] = (All_final['Employed'] / 1e3) / All_final['AREA']
All_final['Distance_City'] = All_final['Distance_City'] * 0.000621371
All_final['Distance_Bikestation'] = All_final['Distance_Bikestation'] * 0.000621371
All_final['Distance_Rail'] = All_final['Distance_Rail'] * 0.000621371
All_final['Distance_Busstop'] = All_final['Distance_Busstop'] * 0.000621371
All_final['rides'] = All_final['rides'] / 1e3
All_final['boardings'] = All_final['boardings'] / 1e3
All_final['alightings'] = All_final['alightings'] / 1e3
All_final.describe().T
# Output
All_final.to_csv('D:\COVID19-Transit_Bikesharing\Divvy_Data\Features_Divvy_0906.csv')
################## Calculate all land use/socio-demograhic/road/cases related features ##############################
