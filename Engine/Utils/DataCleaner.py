from __future__ import annotations

from Engine.Utils import SpatialTools
import pandas as pd


class RunFlow:
    def __init__(self) -> None:
        self.geo_lev = "tag"
        self.time_lev = "timestamp"

        self.angle_matrix = None
        self.weight_matrix = None
        self.coordinate_dict = None

        self.raw_data = None
        self.grouped_data = None
        self.processed_data = None
        self.processed_data = {
            "Train Data": None,
            "Test Data": None,
            "Validation Data": None
        }

    def get_data(self, path: str, faulty: set | list) -> None:
        self.raw_data = pd.read_csv(path)
        self.format_data(faulty=faulty)
        self.group_data()
        # self.coordinate_dict = SpatialTools.coordinate_dict(df=self.grouped_data,  geo_level=self.geo_lev)
        # self.weight_matrix, self.angle_matrix = SpatialTools.weight_angle_matrix(self.coordinate_dict)
        return None

    def format_data(self, faulty: set | list) -> None:
        self.raw_data["FH"] *= 0.36
        self.raw_data.drop([
            'id', 'no2', 'pm10', 'pm10_cal', 'pm10_fac', 'pm10_max', 'pm10_min',
            'datum', 'tijd', 'components', 'sensortype', 'weekdag', 'uur', '#STN',
            'jaar', 'maand', 'weeknummer', 'dag', 'H', 'T', 'U', 'YYYYMMDD', 'name'
        ], axis=1, inplace=True)

        self.raw_data.rename(columns={"DD": "Wind Angle", "FH": "Wind Speed"}, inplace=True)
        self.delete_entries_raw(pop_values=faulty, key=self.geo_lev)
        self.raw_data[self.time_lev] = pd.to_datetime(self.raw_data[self.time_lev])
        return None

    def delete_entries_raw(self, pop_values: set | list, key: str) -> None:
        if len(pop_values) > 0:
            self.raw_data = self.raw_data[~self.raw_data[key].isin(pop_values)]
        else:
            pass

    def group_data(self) -> None:
        grouped_df = self.raw_data.groupby(by=[self.geo_lev, self.time_lev]).mean().reset_index()
        grouped_df.set_index(self.time_lev, inplace=True, drop=True)
        self.grouped_data = grouped_df
        return None

    def run(self) -> None:
        path = r"../DTO/pm25_weer.csv"

        no_sensors = ["Uithoorn", "Velsen-Zuid", "Koog aan de Zaan", "Wijk aan Zee"]

        self.get_data(path=path, faulty=no_sensors)
        # self.split()
        return None

    def split(self, separation: float = 0.75) -> None:
        self.processed_data["Train Data"]: pd.DataFrame = ...
        self.processed_data["Test Data"]: pd.DataFrame = ...
        self.processed_data["Validation Data"]: pd.DataFrame = ...
        self.delete_entries()
        return None

    def delete_entries(self) -> None:
        train_names = set(self.processed_data["Train Data"][self.geo_lev].unique())
        test_names = set(self.processed_data["Test Data"][self.geo_lev].unique())
        val_names = set(self.processed_data["Validation Data"][self.geo_lev].unique())

        misplaced_train = train_names - (test_names & train_names & val_names)
        misplaced_test = test_names - (test_names & train_names & val_names)
        misplaced_test = test_names - (test_names & train_names & val_names)

        # TODO: Drop all the rows that contain unique locations
        return None


def angle_correction(angle: int) -> int:
    if angle > 360:
        angle -= 360
    return angle
