import datetime
from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, Union

import numpy as np
import pandas as pd

from .single_trip import SingleTrip, TripVehicle


class SingleTripOperation(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, timeserie_df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractproperty
    def output_attribute(self) -> str:
        pass

    @abstractmethod
    def __repr__(self):
        pass


class SelectPeriodBeforeTripDate(SingleTripOperation):
    def __init__(
        self,
        timeserie_df: pd.DataFrame,
        date_column: str,
        location_column: str,
    ):
        """
        Select a specific time period from the timeserie data before specific date

        This operation will clip the timeseries differently for each row of
        ``datetime_df`` that is passed to __call__ method. Particularly, the clipped
        timeserie will go from the reference date (`ref_date`), until
        `ref_date + period_length`. The ``ref_date`` is computed as:

        `ref_date = trip.start_date - time_before_date`,

        where `trip` is a representation of the ``datetime_df`` row that is being
        processed.

        Parameters
        ----------
        datetime_df : pd.DataFrame
            Dataframe containing the datetimes to use as reference to compute the time
            period, used to filter the timeserie data
        date_column : str
            Name of the column containing the dates in the dataframe.
        location_column : str
            Name of the column containing the locations in the dataframe. These will be
            used to select the timeserie related to the given locations.
        """
        super().__init__()
        self.timeserie_df = timeserie_df
        self.date_column = date_column
        self.location_column = location_column

        self.timeserie_df[self.date_column] = pd.to_datetime(
            self.timeserie_df[self.date_column]
        )

    @property
    def output_attribute(self) -> str:
        return "weather_timeserie"

    @abstractmethod
    def __call__(self, trip: SingleTrip) -> SingleTrip:
        """
        Select the timeserie data for the time period before specific date

        Parameters
        ----------
        trip : SingleTrip
            Trip containing the date and location attributes to use as reference to
            compute the time period, used to filter the timeserie data

        Returns
        -------
        SingleTrip
            Trip containing the filtered timeserie data as "weather_timeserie" attribute
        """
        pass

    @abstractmethod
    def __repr__(self):
        pass


class SelectConstantPeriodBeforeDate(SelectPeriodBeforeTripDate):
    def __init__(
        self,
        timeserie_df: pd.DataFrame,
        date_column: str,
        location_column: str,
        period_length: datetime.timedelta,
        time_before_date: datetime.timedelta,
    ):
        """
        Select the timeserie data for the time period before specific date

        This operation will clip the timeseries differently for each row of
        ``timeserie_df`` that is passed to __call__ method. Particularly, the clipped
        timeserie will go from:
        `ref_date - period_length/2`
        until
        `ref_date + period_length/2`.

        The ``ref_date`` is computed as:
        `ref_date = trip.start_date - time_before_date`
        where `trip` is a representation of the dataset row that is being
        processed.

        Parameters
        ----------
        timeserie_df : pd.DataFrame
            Dataframe containing the datetimes to use as reference to compute the time
            period, used to filter the timeserie data.
        date_column : str
            Name of the column containing the dates in the dataframe.
        location_column : str
            Name of the column containing the locations in the dataframe. These will be
            used to select the timeserie related to the given locations.
        period_length : datetime.timedelta
            Length of the time period (in days) to select the timeserie data.
        time_before_date : datetime.timedelta
            Time period before the reference date from ``date_column``.
            That will be the central date of the clipped timeserie.
        """
        super().__init__(
            timeserie_df=timeserie_df,
            date_column=date_column,
            location_column=location_column,
        )
        self.period_length = period_length
        self.time_before_date = time_before_date

    def __call__(self, trip: SingleTrip) -> SingleTrip:
        """
        Select the timeserie data for the time period before specific date

        Parameters
        ----------
        trip : SingleTrip
            Trip containing the date and location to use as reference to compute the
            time period, used to filter the timeserie data

        Returns
        -------
        SingleTrip
            Trip containing the filtered timeserie data
        """
        try:
            # Get the timeserie data for the given time range
            period_timeserie = self.timeserie_df[
                (
                    self.timeserie_df[self.date_column]
                    >= trip.start_date
                    - self.time_before_date
                    - self.period_length / 2
                    + datetime.timedelta(days=1)
                    # +1 because the trip start date is included in the period
                )
                & (
                    self.timeserie_df[self.date_column]
                    <= trip.start_date - self.time_before_date + self.period_length / 2
                )
            ][trip.location]
        except KeyError as e:
            print(f"KeyError: {e}")
            period_timeserie = pd.Series([np.nan] * self.period_length.days)

        return SingleTrip(
            index=trip.index,
            location=trip.location,
            start_date=trip.start_date,
            weather_index=trip.weather_index,
            trip_vehicle=trip.trip_vehicle,
            weather_timeserie=period_timeserie,
        )

    def __repr__(self):
        return (
            "SelectConstantPeriodBeforeDate("
            f"{self.period_length}, {self.time_before_date})"
        )


class SelectVariablePeriodBeforeDateByVehicle(SelectPeriodBeforeTripDate):
    def __init__(
        self,
        timeserie_df: pd.DataFrame,
        date_column: str,
        location_column: str,
        period_length_by_vehicle: Dict[TripVehicle, datetime.timedelta],
        time_before_date_by_vehicle: Dict[TripVehicle, datetime.timedelta],
    ):
        """
        Select the timeserie data for the time period before specific date

        This operation will clip the timeseries differently for each row of
        ``datetime_df`` that is passed to __call__ method. Particularly, the clipped
        timeserie will go from the reference date (`ref_date`), until
        `ref_date + period_length`. The ``ref_date`` is computed as:

        `ref_date = trip.start_date - time_before_date`,

        where `trip` is a representation of the ``datetime_df`` row that is being
        processed.

        Parameters
        ----------
        datetime_df : pd.DataFrame
            Dataframe containing the datetimes to use as reference to compute the time
            period, used to filter the timeserie data.
        date_column : str
            Name of the column containing the dates in the dataframe.
        location_column : str
            Name of the column containing the locations in the dataframe. These will be
            used to select the timeserie related to the given locations.
        period_length : datetime.timedelta
            Length of the time period (in days) to select the timeserie data.
        time_before_date : datetime.timedelta
            Time period before the reference date from ``date_column``.
            The clipped timeserie will go from that day and it will last
            ``period_length``.
            Possible values are: "1d", "1w", "1m", "1y"
        """
        super().__init__(
            timeserie_df=timeserie_df,
            date_column=date_column,
            location_column=location_column,
        )
        self.period_length_by_vehicle = period_length_by_vehicle
        self.time_before_date_by_vehicle = time_before_date_by_vehicle

    def __call__(self, trip: SingleTrip) -> SingleTrip:
        """
        Select the timeserie data for the time period before specific date

        Parameters
        ----------
        trip : SingleTrip
            Trip containing the date and location to use as reference to compute the
            time period, used to filter the timeserie data

        Returns
        -------
        SingleTrip
            Trip containing the filtered timeserie data
        """
        # Check if the trip has a vehicle
        if trip.trip_vehicle is None:
            print("The trip has ``trip_vehicle`` attribute == None")
            period_timeserie = pd.Series(
                [np.nan] * self.period_length_by_vehicle[TripVehicle.CAR]
            )
        else:
            try:
                # Get the timeserie data for the given time range
                period_timeserie = self.timeserie_df[
                    (
                        self.timeserie_df[self.date_column]
                        >= trip.start_date
                        - self.period_length_by_vehicle[trip.trip_vehicle]
                        - self.time_before_date_by_vehicle[trip.trip_vehicle]
                    )
                    & (
                        self.timeserie_df[self.date_column]
                        <= trip.start_date
                        - self.time_before_date_by_vehicle[trip.trip_vehicle]
                    )
                ][trip.location]

            except KeyError as e:
                print(f"KeyError: {e}")
                period_timeserie = pd.Series(
                    [np.nan] * self.period_length_by_vehicle[TripVehicle.CAR]
                )

        return SingleTrip(
            index=trip.index,
            location=trip.location,
            start_date=trip.start_date,
            weather_index=trip.weather_index,
            trip_vehicle=trip.trip_vehicle,
            weather_timeserie=period_timeserie,
        )

    def __repr__(self):
        period_length_dict_str = {
            k.name: v for k, v in self.period_length_by_vehicle.items()
        }
        time_before_dict_str = {
            k.name: v for k, v in self.time_before_date_by_vehicle.items()
        }

        return (
            "SelectVariablePeriodBeforeDateByVehicle("
            f"{period_length_dict_str}, {time_before_dict_str})"
        )


class AggregateTimeSerie(SingleTripOperation):
    def __init__(self) -> None:
        """
        Operation that aggregates the timeserie data by the given columns

        Parameters
        ----------
        columns : List[str]
            List of columns to aggregate the timeserie data by
        """
        super().__init__()

    @abstractmethod
    def aggregate(self, timeserie: Union[pd.Series, np.ndarray]) -> float:
        pass

    def __call__(self, trip: SingleTrip) -> SingleTrip:
        if trip.weather_timeserie is None:
            raise ValueError(
                "The trip does not have a weather timeserie. Did you forget to apply"
                " the SelectPeriodBeforeTripDate operation?"
            )
        trip[self.output_attribute] = self.aggregate(trip.weather_timeserie)
        return trip

    @property
    def output_attribute(self) -> str:
        return "weather_index"


class MeanAggregator(AggregateTimeSerie):
    def aggregate(self, timeserie: Union[pd.Series, np.ndarray]) -> float:
        return np.mean(timeserie)

    def __repr__(self):
        return "MeanAggregator()"


class SumAggregator(AggregateTimeSerie):
    def aggregate(self, timeserie: Union[pd.Series, np.ndarray]) -> float:
        return np.sum(timeserie)

    def __repr__(self):
        return "SumAggregator()"


class MaxAggregator(AggregateTimeSerie):
    def aggregate(self, timeserie: Union[pd.Series, np.ndarray]) -> float:
        return np.max(timeserie)

    def __repr__(self):
        return "MaxAggregator()"


class MedianAggregator(AggregateTimeSerie):
    def aggregate(self, timeserie: Union[pd.Series, np.ndarray]) -> float:
        return np.median(timeserie)

    def __repr__(self):
        return "MedianAggregator()"


class StdAggregator(AggregateTimeSerie):
    def aggregate(self, timeserie: Union[pd.Series, np.ndarray]) -> float:
        return np.std(timeserie)

    def __repr__(self):
        return "StdAggregator()"
