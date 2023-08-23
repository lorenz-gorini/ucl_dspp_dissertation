import datetime
from enum import Enum
from typing import Optional, Union

import numpy as np
import pandas as pd


class TripVehicle(Enum):
    ROAD = 1
    TRAIN = 2
    AIRPLANE = 3
    BOAT = 4


class SingleTrip:
    def __init__(
        self,
        index: int,
        destination: str,
        start_date: datetime.datetime,
        weather_index: Optional[str] = None,
        trip_vehicle: TripVehicle = None,
        weather_timeserie: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> None:
        """
        Class representing a single trip

        Parameters
        ----------
        index : int
            Index of the trip
        destination : str
            Name of the destination of the trip
        start_date : datetime.datetime
            Date when the trip started
        """
        self.index = index
        self.destination = destination
        self.start_date = start_date
        self.weather_index = weather_index
        self.trip_vehicle = trip_vehicle
        self.weather_timeserie = weather_timeserie

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __repr__(self):
        attr_str = ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
        return f"SingleTrip({attr_str})"

    def to_dict(self):
        return self.__dict__

    def to_series(self):
        return pd.Series(self.__dict__)
