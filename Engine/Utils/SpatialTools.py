from __future__ import annotations

from numpy import rad2deg, ndarray, zeros
from geopy.distance import geodesic
from pandas import DataFrame
from math import sin, atan2
from numpy import cos


def get_bearing(coordinate_1: list, coordinate_2: list) -> float:
    d_lon = (coordinate_2[1] - coordinate_1[1])
    y = sin(d_lon) * cos(coordinate_2[0])
    x = cos(coordinate_1[0]) * sin(coordinate_2[0]) - sin(coordinate_1[0]) * cos(coordinate_2[0]) * cos(d_lon)
    bearing = atan2(y, x)
    bearing = rad2deg(bearing)
    return bearing


def weight_angle_matrix(df: DataFrame, geo_level: str) -> tuple[ndarray, ndarray, dict]:
    locations = list(df[geo_level].unique())
    loc_dict = dict()
    for item in locations:
        loc_dict[item] = [df.loc[df[geo_level] == item, 'latitude'].mean(),
                          df.loc[df[geo_level] == item, 'longitude'].mean()]

    n = len(loc_dict)
    w_matrix = zeros((n, n))
    angle_matrix = zeros((n, n))

    for i, value1 in enumerate(loc_dict):
        for j, value2 in enumerate(loc_dict):
            if i != j:
                angle_matrix[i, j] = get_bearing(loc_dict[value1], loc_dict[value2])
                w_matrix[i, j] = geodesic(loc_dict[value1], loc_dict[value2]).km
            else:
                w_matrix[i, i] = 0
                angle_matrix[i, i] = 0
    return w_matrix, angle_matrix, loc_dict


def spatial_tensor(
        angle: DataFrame,
        angle_matrix: ndarray
        ) -> DataFrame:

    angle = angle.to_numpy()
    t = len(angle)
    n = len(angle_matrix)
    ww_tensor = zeros((t, n, n))

    for i in range(t):
        ww_tensor[i, :, :] = cos(angle_matrix - float(angle[i]))
    return DataFrame(ww_tensor.reshape(t, n * n))
