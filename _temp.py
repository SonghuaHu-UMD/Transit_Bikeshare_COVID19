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