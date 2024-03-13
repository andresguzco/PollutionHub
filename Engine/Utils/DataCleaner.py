from __future__ import annotations

from Engine.Utils import SpatialTools
from itertools import product
import pandas as pd
import numpy as np


class RunFlow:
    def __init__(self) -> None:
        self.geo_lev = "tag"
        self.time_lev = "timestamp"
        self.cutoff: str = '2019-09-19 01:00:00+00:00'

        self.data = None
        self.processed_data = None

    def get_data(self, path: str, faulty: set | list) -> None:
        self.data = pd.read_csv(path)
        self.format_data(faulty=faulty)
        self.group_data()
        distance_matrix, angle_matrix, loc_dict = SpatialTools.weight_angle_matrix(
            df=self.data,
            geo_level=self.geo_lev
        )
        names = [element[0] + "_" + element[1] for element in list(product(list(loc_dict.keys()), repeat=2))]

        windAngle = self.process_data()

        angles = SpatialTools.spatial_tensor(
            angle=windAngle, angle_matrix=angle_matrix
        )
        angles.columns = ["Angle | " + element for element in names]
        angles.index = self.data.index

        distance_matrix = pd.DataFrame(np.tile(distance_matrix.flatten(), (len(windAngle), 1)))
        distance_matrix.index = self.data.index
        distance_matrix.columns = ["Distance | " + element for element in names]

        self.data = pd.concat([self.data, distance_matrix, angles], axis=1)
        return None

    def process_data(self) -> pd.DataFrame:
        def formatLoc(df):
            location = df['tag'].unique()[0]
            cols = [element + '_' + location for element in df.columns.tolist()]
            df.columns = cols
            return df.drop(columns=['tag_' + location], axis=1)

        def mergeLocations(df):
            locations = {loc: formatLoc(df[df['tag'] == loc]) for loc in list(df['tag'].unique())}
            return pd.concat([locations[loc] for loc in locations.keys()], axis=1)

        canvasDf = self.data.copy()
        canvasDf.drop(["latitude", "longitude"], axis=1, inplace=True)

        windSpeed = canvasDf.loc[:, "Wind Speed"]
        windInfo = windSpeed.reset_index().groupby('timestamp').mean()
        windAngle = canvasDf.loc[self.cutoff:, "Wind Angle"]
        windAngle = windAngle.reset_index().groupby('timestamp').mean()
        canvasDf.drop(['Wind Angle', 'Wind Speed'], axis=1, inplace=True)

        self.data: pd.DataFrame = pd.concat(
            [mergeLocations(canvasDf).loc[self.cutoff:], windInfo.loc[self.cutoff:]],
            axis=1)

        return windAngle

    def format_data(self, faulty: set | list) -> None:
        self.data["FH"] *= 0.36
        self.data.drop([
            'id', 'no2', 'pm10', 'pm10_cal', 'pm10_fac', 'pm10_max', 'pm10_min',
            'datum', 'tijd', 'components', 'sensortype', 'weekdag', 'uur', '#STN',
            'jaar', 'maand', 'weeknummer', 'dag', 'H', 'T', 'U', 'YYYYMMDD', 'name'
        ], axis=1, inplace=True)

        self.data.rename(columns={"DD": "Wind Angle", "FH": "Wind Speed"}, inplace=True)
        self.delete_entries_raw(pop_values=faulty, key=self.geo_lev)
        self.data[self.time_lev] = pd.to_datetime(self.data[self.time_lev])
        return None

    def delete_entries_raw(self, pop_values: set | list, key: str) -> None:
        if len(pop_values) > 0:
            self.data = self.data[~self.data[key].isin(pop_values)]
        else:
            pass

    def group_data(self) -> None:
        grouped_df = self.data.groupby(by=[self.geo_lev, self.time_lev]).mean().reset_index()
        grouped_df.set_index(self.time_lev, inplace=True, drop=True)
        self.data = grouped_df
        return None

    def run(self, path_to_save: str, path_to_read: str) -> None:
        no_sensors = ["Uithoorn", "Velsen-Zuid", "Koog aan de Zaan", "Wijk aan Zee"]

        self.get_data(path=path_to_read, faulty=no_sensors)
        self.data.ffill(inplace=True)
        self.data.to_csv(path_to_save)
        return None


def angle_correction(angle: int) -> int:
    if angle > 360:
        angle -= 360
    return angle
