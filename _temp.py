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