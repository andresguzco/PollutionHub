from __future__ import annotations

from numpy import rad2deg, ndarray, zeros
from geopy.distance import geodesic
from pandas import DataFrame
from math import sin, atan2
from numpy import cos


def get_bearing(coor1: list, coor2: list) -> float:
    d_lon = (coor2[1] - coor1[1])
    y = sin(d_lon) * cos(coor2[0])
    x = cos(coor1[0]) * sin(coor2[0]) - sin(coor1[0]) * cos(coor2[0]) * cos(d_lon)
    brng = atan2(y, x)
    brng = rad2deg(brng)
    return brng


def coordinate_dict(df: DataFrame, geo_level: str):
    locations = list(df[geo_level].unique())
    c_dict = dict()
    for item in locations:
        c_dict[item] = [df.loc[df[geo_level] == item, 'latitude'].mean(),
                        df.loc[df[geo_level] == item, 'longitude'].mean()]
    return c_dict


def weight_angle_matrix(loc_dict: dict) -> tuple[ndarray, ndarray]:
    n = len(loc_dict)
    w_matrix = zeros((n, n))
    angle_matrix = zeros((n, n))

    for i, value1 in enumerate(loc_dict):
        for j, value2 in enumerate(loc_dict):
            if i != j:
                theta = get_bearing(loc_dict[value1], loc_dict[value2])
                w_matrix[i, j] = geodesic(loc_dict[value1], loc_dict[value2]).km
                angle_matrix[i, j] = theta
            else:
                w_matrix[i, i] = 0
                angle_matrix[i, i] = 0
    return w_matrix, angle_matrix
