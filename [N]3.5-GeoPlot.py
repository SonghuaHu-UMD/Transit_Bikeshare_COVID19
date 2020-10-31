import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pylab as plt
import matplotlib.dates as mdates
import seaborn as sns
import geopandas
import contextily as ctx
import mapclassify
import geoplot as gplt
import geoplot.crs as gcrs
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'dejavuserif'
rcParams['text.usetex'] = False

os.chdir(r'D:\COVID19-Transit_Bikesharing\Divvy_Data')

# Spatial Plot
ArcGIS_Divvy_Plot = pd.read_csv(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\ArcGIS_Divvy_Plot.csv')
Coummunity_plot = pd.read_csv(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Community_chicago.csv')
ArcGIS_Divvy_Plot = ArcGIS_Divvy_Plot.merge(Coummunity_plot, on='from_stati', how='right')
# Plot Spatial
Station_poly = geopandas.GeoDataFrame.from_file(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\GIS\Divvy_Station.shp')
Station_poly = Station_poly.merge(ArcGIS_Divvy_Plot, on='from_stati', how='left')
Station_poly = Station_poly.fillna(0)
bikeroute = geopandas.GeoDataFrame.from_file(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\GIS\Bikeroute.shp')
boundary = geopandas.GeoDataFrame.from_file(
    r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Boundaries - City\geo_export_07e6fdf7-f530-46a9-b1bc-95b15607e039.shp')
# boundary = boundary.to_crs({'init': 'EPSG:3857'})
# Station_poly = Station_poly.to_crs({'init': 'EPSG:3857'})
# bikeroute = bikeroute.to_crs({'init': 'EPSG:3857'})

# Size for pickups
Station_poly['2019_size_label'] = mapclassify.UserDefined(Station_poly['2019_Avg'], bins=[5, 15, 30, 60, 90]).yb
Station_poly.loc[Station_poly['2019_size_label'] == 0, '2019_size'] = 5
Station_poly.loc[Station_poly['2019_size_label'] == 1, '2019_size'] = 10
Station_poly.loc[Station_poly['2019_size_label'] == 2, '2019_size'] = 30
Station_poly.loc[Station_poly['2019_size_label'] == 3, '2019_size'] = 80
Station_poly.loc[Station_poly['2019_size_label'] == 4, '2019_size'] = 120
Station_poly.loc[Station_poly['2019_size_label'] == 5, '2019_size'] = 160

Station_poly['2020_size_label'] = mapclassify.UserDefined(Station_poly['2020_Avg'], bins=[5, 15, 30, 60, 90]).yb
Station_poly.loc[Station_poly['2020_size_label'] == 0, '2020_size'] = 5
Station_poly.loc[Station_poly['2020_size_label'] == 1, '2020_size'] = 10
Station_poly.loc[Station_poly['2020_size_label'] == 2, '2020_size'] = 30
Station_poly.loc[Station_poly['2020_size_label'] == 3, '2020_size'] = 80
Station_poly.loc[Station_poly['2020_size_label'] == 4, '2020_size'] = 120
Station_poly.loc[Station_poly['2020_size_label'] == 5, '2020_size'] = 160

# Relative Change
Station_poly_Positive = Station_poly[Station_poly['Cum_Relative_Impact'] >= 0]
Station_poly_Positive['Change_size_label'] = mapclassify.UserDefined(Station_poly_Positive['Cum_Relative_Impact'],
                                                                     bins=[0.15, 0.3, 0.45, 0.7, 1]).yb
Station_poly_Positive.loc[Station_poly_Positive['Change_size_label'] == 0, 'Change_size'] = 5
Station_poly_Positive.loc[Station_poly_Positive['Change_size_label'] == 1, 'Change_size'] = 10
Station_poly_Positive.loc[Station_poly_Positive['Change_size_label'] == 2, 'Change_size'] = 30
Station_poly_Positive.loc[Station_poly_Positive['Change_size_label'] == 3, 'Change_size'] = 80
Station_poly_Positive.loc[Station_poly_Positive['Change_size_label'] == 4, 'Change_size'] = 120
Station_poly_Positive.loc[Station_poly_Positive['Change_size_label'] == 5, 'Change_size'] = 160

Station_poly_Negative = Station_poly[Station_poly['Cum_Relative_Impact'] < 0]
Station_poly_Negative['Cum_Relative_Impact'] = -Station_poly_Negative['Cum_Relative_Impact']
Station_poly_Negative['Change_size_label'] = mapclassify.UserDefined(Station_poly_Negative['Cum_Relative_Impact'],
                                                                     bins=[0.15, 0.3, 0.45, 0.7, 1]).yb
Station_poly_Negative.loc[Station_poly_Negative['Change_size_label'] == 0, 'Change_size'] = 5
Station_poly_Negative.loc[Station_poly_Negative['Change_size_label'] == 1, 'Change_size'] = 10
Station_poly_Negative.loc[Station_poly_Negative['Change_size_label'] == 2, 'Change_size'] = 30
Station_poly_Negative.loc[Station_poly_Negative['Change_size_label'] == 3, 'Change_size'] = 80
Station_poly_Negative.loc[Station_poly_Negative['Change_size_label'] == 4, 'Change_size'] = 120
Station_poly_Negative.loc[Station_poly_Negative['Change_size_label'] == 5, 'Change_size'] = 160

# with plt.style.context(['science', 'ieee']):
#     fig, ax = plt.subplots(figsize=(7, 10))  # figsize=(10, 5)
#     # bikeroute.plot(ax=ax, color='k', linewidth=0.5, alpha=0.5)
#     # boundary.boundary.plot(ax=ax, color='k', linewidth=0.5, alpha=1)
#     Station_poly.plot(column='Label_2019', categorical=True, linewidth=0.1, edgecolor='k', cmap='RdGy',
#                       markersize=Station_poly['2019_size'], legend=True, ax=ax)
#     # ax.add_artist(ScaleBar(1))
#     ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)  # ,  , crs='epsg:4326'
#     plt.axis('off')
#     plt.subplots_adjust(top=0.99, bottom=0.03, left=0.0, right=1.0, hspace=0.0, wspace=0.0)
#     fig.savefig(r'D:\\COVID19-Transit_Bikesharing\\Divvy_Data\\Results\\Community_1.jpg', dpi=600)


# fig, ax = plt.subplots()  # figsize=(10, 5)
# ctx.add_basemap(ax, crs='epsg:4326', source=ctx.providers.Stamen.TonerLite)
# fig = plt.gcf()


boundary.total_bounds
fig = plt.figure(figsize=(7, 8))
ax1 = plt.subplot(121, projection=gcrs.WebMercator())
ax2 = plt.subplot(122, projection=gcrs.WebMercator())
# ax1 = gplt.webmap(boundary, projection=gcrs.WebMercator())
# gplt.sankey(bikeroute, ax=ax, color='black')
gplt.pointplot(Station_poly, scale='2019_size', limits=(4, 15), hue='Label_2019', cmap='Blues', legend=True,
               legend_var='scale', linewidth=0.2, edgecolor='Blue',
               # extent=np.array([-87.80, 41.68454312, -87.5241371, 42.12303859]),
               legend_kwargs={'bbox_to_anchor': (0.65, 1), 'frameon': False},
               legend_values=[5, 10, 30, 80, 120, 160],
               legend_labels=['< 5', '5 - 15', '15 - 30', '30 - 60', '60 - 90', '> 90'], ax=ax1)
# gplt.polyplot(boundary, ax=ax1)
# gplt.kdeplot(Station_poly, projection=gcrs.AlbersEqualArea(), cmap='Reds', ax=ax1)
# gplt.webmap(contiguous_usa, ax=ax, extent=extent)
ctx.add_basemap(ax1, source=ctx.providers.Stamen.TonerLite)
ax1.set_title('2019 Average Pickups')

# ax2 = gplt.webmap(boundary, projection=gcrs.WebMercator())
# gplt.sankey(bikeroute, ax=ax, color='black')
gplt.pointplot(Station_poly, scale='2020_size', limits=(4, 15), hue='Label_2020', cmap='Blues', legend=True,
               legend_var='scale', linewidth=0.2, edgecolor='Blue',
               legend_kwargs={'bbox_to_anchor': (0.65, 1), 'frameon': False},
               legend_values=[5, 10, 30, 80, 120, 160],
               legend_labels=['< 5', '5 - 15', '15 - 30', '30 - 60', '60 - 90', '> 90'], ax=ax2)
# gplt.kdeplot(Station_poly, projection=gcrs.AlbersEqualArea(), cmap='Reds', ax=ax2)
# gplt.webmap(contiguous_usa, ax=ax, extent=extent)
ctx.add_basemap(ax2, source=ctx.providers.Stamen.TonerLite)
plt.subplots_adjust(top=0.985, bottom=0.005, left=0.01, right=0.99, hspace=0.0, wspace=0.01)
ax2.set_title('2020 Average Pickups')
plt.savefig(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Results\Figure4-1.png', dpi=600)
plt.savefig(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Results\Figure4-1.svg')
# plt.tight_layout()

fig = plt.figure(figsize=(7, 8))
ax1 = plt.subplot(121, projection=gcrs.WebMercator())
ax2 = plt.subplot(122, projection=gcrs.WebMercator())
# ax1 = gplt.webmap(boundary, projection=gcrs.WebMercator())
# gplt.sankey(bikeroute, ax=ax, color='black')
gplt.pointplot(Station_poly[Station_poly['Cum_Relative_Impact'] < 0], ax=ax1, **pointplot_kwargs)
gplt.pointplot(Station_poly_Positive, scale='Change_size', limits=(4, 15), color='#7eb8da', legend=True,
               legend_var='scale', linewidth=0.2, edgecolor='Blue',
               # extent=np.array([-87.80, 41.68454312, -87.5241371, 42.12303859]),
               legend_kwargs={'bbox_to_anchor': (0.65, 1), 'frameon': False},
               legend_values=[5, 20, 50, 90, 120, 150],
               legend_labels=['< 0.15', '0.15 - 0.3', '0.3 - 0.45', '0.45 - 0.7', '0.7 - 1', '> 1'], ax=ax1)
pointplot_kwargs = {"facecolor": 'none', "edgecolor": 'k', 's': 2, "linewidth": 0.4, 'color': 'none'}
# gplt.polyplot(boundary, ax=ax1)
# gplt.kdeplot(Station_poly, projection=gcrs.AlbersEqualArea(), cmap='Reds', ax=ax1)
# gplt.webmap(contiguous_usa, ax=ax, extent=extent)
ctx.add_basemap(ax1, source=ctx.providers.Stamen.TonerLite)
ax1.set_title('Cumulative Relative Change (+)')

# ax2 = gplt.webmap(boundary, projection=gcrs.WebMercator())
# gplt.sankey(bikeroute, ax=ax, color='black')
gplt.pointplot(Station_poly_Negative, scale='Change_size', limits=(4, 15), color='#7eb8da', legend=True,
               legend_var='scale', linewidth=0.2, edgecolor='Blue',
               legend_kwargs={'bbox_to_anchor': (0.65, 1), 'frameon': False},
               legend_values=[5, 20, 50, 90, 120, 150],
               legend_labels=['< 0.15', '0.15 - 0.3', '0.3 - 0.45', '0.45 - 0.7', '0.7 - 1', '> 1'], ax=ax2)
pointplot_kwargs = {"facecolor": 'none', "edgecolor": 'k', 's': 2, "linewidth": 0.4, 'color': 'none'}
gplt.pointplot(Station_poly[Station_poly['Cum_Relative_Impact'] >= 0], ax=ax2, **pointplot_kwargs)
# gplt.kdeplot(Station_poly, projection=gcrs.AlbersEqualArea(), cmap='Reds', ax=ax2)
# gplt.webmap(contiguous_usa, ax=ax, extent=extent)
ctx.add_basemap(ax2, source=ctx.providers.Stamen.TonerLite)
plt.subplots_adjust(top=0.985, bottom=0.005, left=0.01, right=0.99, hspace=0.0, wspace=0.01)
ax2.set_title('Cumulative Relative Change (|-|)')
plt.savefig(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Results\Fig5-1.png', dpi=600)
plt.savefig(r'D:\COVID19-Transit_Bikesharing\Divvy_Data\Results\Fig5-1.svg')

# plt.tight_layout()
