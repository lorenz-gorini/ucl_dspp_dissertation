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
        timeserie will be centered around the reference date (``ref_date``) and it will
        last ``period_length``:

        ( ref_date - period_length/2 , ref_date + period_length/2 )

        The ``ref_date`` is computed as:

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
        timeserie will be centered around the reference date (``ref_date``) and it will
        last ``period_length``:

        ( ref_date - period_length/2 , ref_date + period_length/2 )

        The ``ref_date`` is computed as:

        `ref_date = trip.start_date - time_before_date`

        where `trip` is a representation of the dataset row that is being
        processed.  This is to reproduce the decision-making process of tourists,
        which should be based on deciding what weather will be like around their date
        of departure, based on the weather in the timespan around their departure.

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
            ][[self.date_column, trip.location]]
            # Set the index to the date column
            period_timeserie.set_index(self.date_column, inplace=True, drop=True)

        except KeyError as e:
            print(f"KeyError: {e}")
            period_timeserie = pd.Series([np.nan] * self.period_length.days)

        return SingleTrip(
            index=trip.index,
            location=trip.location,
            start_date=trip.start_date,
            weather_index=trip.weather_index,
            trip_vehicle=trip.trip_vehicle,
            weather_timeserie=period_timeserie[trip.location],
        )

    def __repr__(self):
        return (
            "SelectConstantPeriodBeforeDate("
            f"{self.period_length}, {self.time_before_date})"
        )


class SelectPeriodBeforeDateMultipleYears(SelectPeriodBeforeTripDate):
    def __init__(
        self,
        timeserie_df: pd.DataFrame,
        date_column: str,
        location_column: str,
        years_same_period: int,
        period_length_per_year: datetime.timedelta,
        first_year_before_date: int,
        days_before_trip_period: datetime.timedelta = datetime.timedelta(days=0),
    ):
        """
        Select the data for the same time period in multiple years before trip date

        This operation will clip the timeseries differently for each row of
        ``timeserie_df`` that is passed to __call__ method. Particularly, the clipped
        timeserie will be centered around the reference date (``ref_date``) and it will
        last ``period_length_per_year`` for each year in ``years_same_period``:

        ( ref_date - period_length/2 , ref_date + period_length/2 )

        The ``ref_date`` spans through multiple years:

        ``year`` = 0 .. ``years_same_period``

        and it is computed as:

        `ref_date = trip.start_date - (first_year_before_date - year) * 365 days
                     - days_before_trip_period`

        where `trip` is a representation of the dataset row that is being
        processed.  This is to reproduce the decision-making process of tourists,
        which should be based on deciding what weather will be like around their date
        of departure, based on the weather in the timespan around their departure.

        Example
        -------
        >>> SelectPeriodBeforeDateMultipleYears(
        ...     timeserie_df,
        ...     date_column="date",
        ...     location_column="location",
        ...     years_same_period=5,
        ...     period_length_per_year=datetime.timedelta(days=60),
        ...     first_year_before_date=5,
        ...     days_before_trip_period=datetime.timedelta(days=0)
        ... )
        Let's say that the trip start date is 2020-04-01.
        The reference date for the first year will be:
        2020-04-01 - (5 - 0) * 365 - 0 = 2015-04-01
        The reference date for the second year will be:
        2020-04-01 - (5 - 1) * 365 - 0 = 2016-04-01
        ...
        So the final selected timeserie will be composed by the time periods:
        (2015-03-01, 2015-05-01) + (2016-03-01, 2016-05-01) +
        (2017-03-01, 2017-05-01) + (2018-03-01, 2018-05-01) +
        (2019-03-01, 2019-05-01).

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
        years_same_period : int
            Number of consecutive years before ``trip.start_date`` to consider to
            select the timeserie period.
        period_length_per_year : datetime.timedelta
            Length of the time period (in days) for each year in ``years_same_period``
            to select the timeserie data.
        first_year_before_date : int
            Number of years before ``trip.start_date`` when to start counting for
            ``years_same_period``. If None, it will be set equal to ``years_same_period``.
        days_before_trip_period : datetime.timedelta, optional
            Time period before the reference date from ``date_column``.
            That will be the central date of the clipped timeserie.
            Default is ``datetime.timedelta(days=0)``
        """
        super().__init__(
            timeserie_df=timeserie_df,
            date_column=date_column,
            location_column=location_column,
        )
        self.years_same_period = years_same_period
        self.period_length_per_year = period_length_per_year
        self.first_year_before_date = first_year_before_date
        self.days_before_trip_period = days_before_trip_period

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
            period_timeseries = []
            for year in range(self.years_same_period):
                ref_date = (
                    trip.start_date
                    - datetime.timedelta(
                        days=(self.first_year_before_date - year) * 365
                    )
                    - self.days_before_trip_period
                )
                period_timeseries.append(
                    self.timeserie_df[
                        (
                            self.timeserie_df[self.date_column]
                            >= ref_date
                            - self.period_length_per_year / 2
                            + datetime.timedelta(days=1)
                            # +1 because the trip start date is included in the period
                        )
                        & (
                            self.timeserie_df[self.date_column]
                            <= ref_date + self.period_length_per_year / 2
                        )
                    ][[self.date_column, trip.location]]
                )
            period_timeserie = pd.concat(period_timeseries)
            # Set the index to the date column
            period_timeserie.set_index(self.date_column, inplace=True, drop=True)

        except KeyError as e:
            print(f"KeyError: {e}")
            period_timeserie = pd.Series([np.nan] * self.period_length_per_year.days)

        return SingleTrip(
            index=trip.index,
            location=trip.location,
            start_date=trip.start_date,
            weather_index=trip.weather_index,
            trip_vehicle=trip.trip_vehicle,
            weather_timeserie=period_timeserie[trip.location],
        )

    def __repr__(self):
        return (
            "SelectPeriodBeforeDateMultipleYears("
            f"{self.date_column}, {self.location_column}, {self.years_same_period},"
            f" {self.period_length_per_year}, {self.first_year_before_date},"
            f" {self.days_before_trip_period})"
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
                ][[self.date_column, trip.location]]
                # Set the index to the date column
                period_timeserie.set_index(self.date_column, inplace=True, drop=True)

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
            weather_timeserie=period_timeserie[trip.location],
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
    def aggregate(self, trip: SingleTrip) -> float:
        pass

    def __call__(self, trip: SingleTrip) -> SingleTrip:
        if trip.weather_timeserie is None:
            raise ValueError(
                "The trip does not have a weather timeserie. Did you forget to apply"
                " the SelectPeriodBeforeTripDate operation?"
            )
        elif trip.weather_timeserie.isna().all():
            trip[self.output_attribute] = np.nan
        else:
            trip[self.output_attribute] = self.aggregate(trip)
        return trip

    @property
    def output_attribute(self) -> str:
        return "weather_index"


class MeanAggregator(AggregateTimeSerie):
    def aggregate(self, trip: SingleTrip) -> float:
        return np.mean(trip.weather_timeserie)

    def __repr__(self):
        return "MeanAggregator()"


class SumAggregator(AggregateTimeSerie):
    def aggregate(self, trip: SingleTrip) -> float:
        return np.sum(trip.weather_timeserie)

    def __repr__(self):
        return "SumAggregator()"


class MaxAggregator(AggregateTimeSerie):
    def aggregate(self, trip: SingleTrip) -> float:
        return np.max(trip.weather_timeserie)

    def __repr__(self):
        return "MaxAggregator()"


class MedianAggregator(AggregateTimeSerie):
    def aggregate(self, trip: SingleTrip) -> float:
        return np.median(trip.weather_timeserie)

    def __repr__(self):
        return "MedianAggregator()"


class StdAggregator(AggregateTimeSerie):
    def aggregate(self, trip: SingleTrip) -> float:
        return np.std(trip.weather_timeserie)

    def __repr__(self):
        return "StdAggregator()"


class HeatwavesAggregator(AggregateTimeSerie):
    def __init__(
        self,
        historical_data: pd.DataFrame,
        date_column: str,
    ) -> None:
        """
        Count the number of days considered as heatwaves (based on WMO)

        The heatwaves are defined as days of the timeserie data when the daily
        maximum temperature of more than 5 consecutive days exceeds the
        average maximum temperature by 5 °C, with regards to the same days of
        the years 1961-1990 (see https://doi.org/10.3354/cr019193)

        Parameters
        ----------
        historical_data : pd.DataFrame
            Dataframe containing the historical data to use as reference to compute
            the heatwaves. This is expected to have the daily means of the maximum
            temperatures for each day of the year. The daily means are computed by
            considering the years 1961-1990 as reference.
            The date of the year is expected to be in the ``date_column`` and is
            expected to contain always the same year.
        date_column : str
            Name of the column containing the dates in the ``historical_data``
            dataframe.
        reference_year : int, optional
            Year to use as reference to compute the heatwaves, by default 1980
        """
        super().__init__()
        self.historical_data = historical_data.copy()
        self.date_column = date_column
        self.historical_data[self.date_column] = pd.to_datetime(
            self.historical_data[self.date_column]
        )

        self.historical_data = self.historical_data.set_index(
            self.date_column, drop=True
        )
        reference_years = self.historical_data.index.year.unique()
        if len(reference_years) != 1:
            raise ValueError(
                "The historical data must contain only one year of data, but it "
                f"contains {reference_years}"
            )
        else:
            self.reference_year = reference_years[0]

    def aggregate(self, trip: SingleTrip) -> float:
        """
        Count the number of days considered as heatwaves (based on WMO)

        The heatwaves are defined as days of the timeserie data when the daily
        maximum temperature of more than 5 consecutive days exceeds the
        average maximum temperature by 5 °C, with regards to the same days of
        the years 1961-1990 (see https://doi.org/10.3354/cr019193).

        Parameters
        ----------
        trip : SingleTrip
            Trip containing:
            - the timeserie data to analyze which we extract the aggregated index from.
            Its indices must be the timeserie dates
            - the location related to the timeserie data

        Returns
        -------
        float
            Number of days in timeserie data considered as heatwaves
        """
        if trip.weather_timeserie.isna().all():
            return np.nan

        trip_timeserie = trip.weather_timeserie.copy()
        trip_timeserie.index = trip_timeserie.index.map(
            lambda x: x.replace(year=self.reference_year)
        )
        try:
            # Calculate the rolling average of maximum temperatures
            historical_avg_df = self.historical_data.loc[
                trip_timeserie.index, trip.location
            ]
        except KeyError as e:
            print(f"KeyError: {e}")
            return np.nan

        # Calculate the difference between daily max temperatures and rolling
        # average of historical data
        temp_difference = trip_timeserie - historical_avg_df

        # Identify days with temperature differences exceeding 5 °C
        days_above_hist_thresh = temp_difference > 5
        # Compute how many days are above historical threshold out of a 5-day
        # rolling window. If they are 5, then it is a heatwave because they are also
        # consecutive
        consecutive_days_above_thresh = days_above_hist_thresh.rolling(window=5).sum()
        heatwave_days = consecutive_days_above_thresh == 5
        # Cluster the consecutive True values (>5 hot consecutive days) and assign
        # a unique id to each of them, so that we understand how many heatwave periods
        # there are
        heatwave_cluster_ids = (heatwave_days != heatwave_days.shift()).cumsum()[
            heatwave_days
        ]
        # Count the number of clusters/heatwaves in the period
        heatwave_sequences_count = len(heatwave_cluster_ids.unique())

        return np.sum(heatwave_days) + 5 * heatwave_sequences_count

    def __repr__(self):
        return f"HeatwavesAggregator({self.date_column})"
