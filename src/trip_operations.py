from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import googlemaps
import numpy as np
import pandas as pd
import requests
from geopy.geocoders.base import Geocoder
from tqdm import tqdm

from .single_trip import SingleTrip, TripVehicle
from .single_trip_operations import AggregateTimeSerie, SelectPeriodBeforeTripDate
from .trip_dataset import TripDataset


class TripDatasetOperation(ABC):
    def __init__(
        self,
        input_columns: List[str],
        output_columns: List[str],
        force_repeat: bool = False,
    ) -> None:
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.force_repeat = force_repeat

    @abstractmethod
    def __call__(self, dataset: TripDataset) -> pd.DataFrame:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class ToDatetimeConverter(TripDatasetOperation):
    def __init__(
        self,
        date_column: str,
        output_column: str,
        date_format: str = "%Y-%m-%d",
        force_repeat: bool = False,
    ) -> None:
        """
        Convert a column to datetime pandas dtype
        """
        super().__init__(
            input_columns=[date_column],
            output_columns=[output_column],
            force_repeat=force_repeat,
        )
        self.date_column = date_column
        self.date_format = date_format

    def __call__(self, dataset: TripDataset) -> pd.DataFrame:
        new_df = dataset.df.copy()
        new_df[self.output_columns[0]] = pd.to_datetime(
            new_df[self.date_column], format=self.date_format
        )
        return new_df

    def __repr__(self) -> str:
        return (
            f"ConvertToDatetime({self.date_column}, {self.date_format}, "
            f"{self.output_columns[0]})"
        )


class TripStartDateCreator(TripDatasetOperation):
    def __init__(
        self,
        trip_end_date_column: str,
        trip_duration_column: str,
        output_column: str,
        trip_end_date_format: str = "%Y-%m-%d",
        duration_column_unit: str = "days",
        force_repeat: bool = False,
    ) -> None:
        """
        Compute the start date of the trip by subtracting trip duration to its end date
        """
        super().__init__(
            input_columns=[trip_end_date_column, trip_duration_column],
            output_columns=[output_column],
            force_repeat=force_repeat,
        )
        self.trip_end_date_column = trip_end_date_column
        self.trip_duration_column = trip_duration_column
        self.end_date_format = trip_end_date_format
        self.duration_column_unit = duration_column_unit

    def __call__(self, dataset: TripDataset) -> pd.DataFrame:
        new_df = dataset.df.copy()
        new_df[self.output_columns[0]] = pd.to_datetime(
            new_df[self.trip_end_date_column], format=self.end_date_format
        ) - pd.to_timedelta(
            new_df[self.trip_duration_column], unit=self.duration_column_unit
        )
        return new_df

    def __repr__(self) -> str:
        return (
            f"TripStartDateCreator({self.trip_end_date_column},"
            f" {self.trip_duration_column}, {self.output_columns[0]},"
            f" {self.end_date_format}, {self.duration_column_unit})"
        )


class ReplaceValuesByMap(TripDatasetOperation):
    def __init__(
        self,
        input_column: List[str],
        output_column: List[str],
        map_dict: Dict[Any, Any],
        force_repeat: bool = False,
    ) -> None:
        super().__init__(
            input_columns=[input_column],
            output_columns=[output_column],
            force_repeat=force_repeat,
        )
        self.map_dict = map_dict

    def __call__(self, dataset: TripDataset) -> pd.DataFrame:
        new_df = dataset.df.copy()
        new_df[self.output_columns[0]] = new_df[self.input_columns[0]].replace(
            self.map_dict
        )
        return new_df

    def __repr__(self) -> str:
        return f"ReplaceByMap({self.input_columns[0]}, {self.output_columns[0]}, {self.map_dict})"


class FilterCountries(TripDatasetOperation):
    def __init__(
        self, country_column: str, countries: List[str], force_repeat: bool = False
    ) -> None:
        """
        Filter the dataset by the given countries

        Parameters
        ----------
        country_column : str
            The name of the column in the dataset containing the countries
        countries : List[str]
            The list of countries to keep in the dataset
        force_repeat : bool, optional
            If True, the operation will be repeated even if the output columns are
            already present in the dataset. Default is False.

        Notes
        -----
        The countries must be in the same format as in the dataset, e.g. "FRANCIA"
        instead of "France".
        """
        super().__init__(
            input_columns=[country_column],
            output_columns=[],
            force_repeat=force_repeat,
        )
        self.countries = countries

    def __call__(self, dataset: TripDataset) -> pd.DataFrame:
        new_df = dataset.df.copy()
        new_df = new_df[new_df[self.input_columns[0]].isin(self.countries)]
        new_df.reset_index(inplace=True, drop=True)
        print(
            f"The dataset now contains {new_df.shape[0]} rows out of"
            f" {dataset.df.shape[0]}"
        )
        dropped_countries = set(dataset.df[self.input_columns[0]].unique()) - set(
            self.countries
        )
        print(f"The dropped countries are {dropped_countries}")
        return new_df

    def __repr__(self) -> str:
        return f"FilterCountries({self.input_columns[0]}, {self.countries})"


class CodeToStringMapper(TripDatasetOperation):
    def __init__(
        self,
        input_column: str,
        output_column: str,
        code_map: Dict[int, str],
        force_repeat: bool = False,
    ) -> None:
        super().__init__(
            input_columns=[input_column],
            output_columns=[output_column],
            force_repeat=force_repeat,
        )
        self.code_map = code_map

    def __call__(self, dataset: TripDataset) -> pd.DataFrame:
        new_df = dataset.df.copy()
        new_df[self.output_columns[0]] = new_df[self.input_columns[0]].map(
            self.code_map
        )
        print(
            "The codes not associated to a name based on the provided ``code_map`` are "
            + str(new_df[self.output_columns[0]].isna().sum())
            + " and they are:"
            # Show which codes could not be associated to a name
            + str(
                new_df[new_df[self.output_columns[0]].isna()][
                    self.input_columns[0]
                ].unique()
            )
        )
        return new_df

    def __repr__(self) -> str:
        return (
            f"CodeToStringMapper({self.input_columns[0]}, {self.output_columns[0]}, "
            f"{self.code_map})"
        )


class CodeToLocationMapperFromCSV(TripDatasetOperation):
    def __init__(
        self,
        input_column: str,
        output_column: str,
        code_map_csv: Union[str, Path],
        code_column_csv: str,
        location_name_column: str,
        nan_codes: Optional[List[int]] = None,
        separator: Optional[str] = None,
        force_repeat: bool = False,
    ) -> None:
        """
        Create a mapper from location codes to location names from a CSV file

        Parameters
        ----------
        input_column : str
            The name of the column in the dataset containing the location codes
        output_column : str
            The name of the column in the dataset containing the location names
        code_map_csv : Union[str, Path]
            The path to the CSV file containing the location codes and names
        code_column_csv : str
            The name of the column in the CSV file containing the location codes
        location_name_column : str
            The name of the column in the CSV file containing the location names
        nan_codes : Optional[List[int]], optional
            The list of codes that will be mapped to pd.NA. If None,
            the codes considered as NaN are None and pd.NA, otherwise these two values
            will be added to the list ``nan_codes`` provided. Default is None.
        separator : Optional[str], optional
            The separator used in the CSV file. If None, the
            separator is a comma ",". Default is None.
        """
        super().__init__(
            input_columns=[input_column],
            output_columns=[output_column],
            force_repeat=force_repeat,
        )
        self.code_map_csv = code_map_csv
        self.code_column_csv = code_column_csv
        self.location_name_column = location_name_column
        self.separator = "," if separator is None else separator

        self.nan_codes = set([None, pd.NA])
        if nan_codes is not None:
            self.nan_codes = self.nan_codes | set(nan_codes)
        # Load the CSV file with the location codes and names
        self._code_map_df = pd.read_csv(self.code_map_csv, sep=self.separator)

    def __call__(self, dataset: TripDataset) -> pd.DataFrame:
        # Turn the df into a dictionary mapping codes to names
        code_to_name_dict = {}
        for _, row in self._code_map_df.iterrows():
            code_to_name_dict[row[self.code_column_csv]] = (
                pd.NA
                if row[self.code_column_csv] in self.nan_codes
                else row[self.location_name_column]
            )

        new_df = CodeToStringMapper(
            input_column=self.input_columns[0],
            output_column=self.output_columns[0],
            code_map=code_to_name_dict,
        )(dataset)
        return new_df

    def __repr__(self) -> str:
        return (
            f"CodeToLocationMapperFromCSV({self.input_columns[0]}, {self.output_columns[0]}, "
            f"{self.code_map_csv}, {self.code_column_csv}, {self.location_name_column})"
        )


class LocationToCoordinatesMapper(TripDatasetOperation):
    def __init__(
        self,
        input_column: str,
        output_latit_column: str,
        output_longit_column: str,
        geolocator: Union[googlemaps.Client, Geocoder],
        location_suffix: str = "",
        nan_codes: List[Optional[int]] = [None],
        force_repeat: bool = False,
        cache_df_path: Path = Path(
            "/mnt/c/Users/loreg/OneDrive/UCL/Dissertation/ucl_dspp_dissertation/data/"
            "location_to_coords.csv"
        ),
    ) -> None:
        """
        Create a mapper from location names to coordinates

        Parameters
        ----------
        input_column : str
            The name of the column in the dataset containing the location names
        output_latit_column : str
            The name of the column in the dataset containing the latitude of the
            locations
        output_longit_column : str
            The name of the column in the dataset containing the longitude of the
            locations
        geolocator : Union[googlemaps.Client, Geocoder]
            The geolocator to use to geocode the locations. It can be either a
            googlemaps.Client or a geopy.geocoders.base.Geocoder
        location_suffix : str, optional
            The suffix to add to the locations before geocoding them. Default is "".
        nan_codes : Optional[List[int]], optional
            The list of codes that will be mapped to pd.NA. If None,
            the codes considered as NaN are None and pd.NA, otherwise these two values
            will be added to the list ``nan_codes`` provided. Default is None.
        force_repeat : bool, optional
            If True, the locations already geocoded will be geocoded again.
            Default is False.
        cache_df_path : Union[str, Path], optional
            The path to the CSV file containing the cache dataframe. Default is
            "/mnt/c/Users/loreg/OneDrive/UCL/Dissertation/ucl_dspp_dissertation/data/location_to_coords.csv".

        Notes
        -----
        The two most popular libraries to use as geolocator are googlemaps and geopy.
        The first one is a wrapper around the Google Maps API, while the second one
        is a wrapper around multiple geocoding services. One of these geocoding
        services is Nominatim, which is free but it is less precise.
        Examples:
        >>> import googlemaps
        >>> geolocator = googlemaps.Client(key="YOUR_API_KEY")
        Nominatim:
        >>> from geopy.geocoders import Nominatim
        >>> geolocator = Nominatim(user_agent="my-application")

        """
        super().__init__(
            input_columns=[input_column],
            output_columns=[output_latit_column, output_longit_column],
            force_repeat=force_repeat,
        )
        self.output_latit_column = output_latit_column
        self.output_longit_column = output_longit_column
        # Even if nan_codes is None, we want to consider the NA values as NaN
        self.nan_codes = set([None, pd.NA, np.nan]) | set(nan_codes)
        self.geolocator = geolocator
        self.location_suffix = location_suffix
        self.cache_df_path = cache_df_path
        (
            self.location_to_coords_df,
            self.location_to_latitude_map,
            self.location_to_longitude_map,
        ) = self.load_cache_df(self.cache_df_path)

    def geocode_location(self, location: str) -> Tuple[float, float]:
        """
        Geocode a location and add it to the cache

        Parameters
        ----------
        location : str
            The location to geocode

        Returns
        -------
        float
            The latitude of the location
        float
            The longitude of the location
        """
        print(f"Geocoding location: {location}")
        geocode_location = self.geolocator.geocode(location)
        if isinstance(self.geolocator, googlemaps.Client):
            geocode_location = geocode_location[0]["geometry"]["location"]
            return geocode_location["lat"], geocode_location["lng"]
        elif isinstance(self.geolocator, Geocoder):
            return geocode_location.latitude, geocode_location.longitude
        else:
            raise ValueError(
                "The geolocator must be either a googlemaps.Client or a geopy.geocoders.base.Geocoder"
            )

    @staticmethod
    def load_cache_df(cache_df_path: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Load the cache dataframe from the given path

        Parameters
        ----------
        cache_path : Union[str, Path]
            The path to the cache dataframe

        Returns
        -------
        Dict[str,float]
            The map between the locations and their latitude
        Dict[str,float]
            The map between the locations and their longitude
        """
        if cache_df_path.exists():
            location_to_coords_df = pd.read_csv(cache_df_path)
            location_to_latitude_map = {}
            location_to_longitude_map = {}
            for _, row in location_to_coords_df.iterrows():
                location_to_latitude_map[row["location"]] = row["latitude"]
                location_to_longitude_map[row["location"]] = row["longitude"]
            return (
                location_to_coords_df,
                location_to_latitude_map,
                location_to_longitude_map,
            )
        else:
            return pd.DataFrame(columns=["location", "latitude", "longitude"]), {}, {}

    def update_cache_df(
        self,
        location_to_latitude_map: Dict[str, float],
        location_to_longitude_map: Dict[str, float],
    ) -> None:
        """
        Update the cache dataframe with the new geocoded locations
        """
        location_to_coords_dict = []
        for location, latitude in location_to_latitude_map.items():
            longitude = location_to_longitude_map[location]
            location_to_coords_dict.append(
                {
                    "location": location,
                    "latitude": latitude,
                    "longitude": longitude,
                }
            )
        self.location_to_coords_df = pd.DataFrame(location_to_coords_dict)
        self.location_to_coords_df.to_csv(self.cache_df_path, index=False)

    def __call__(self, dataset: TripDataset) -> pd.DataFrame:
        """
        Geocode the locations in the dataset and add the coordinates to the dataset

        First, it identifies the unique locations to geocode in ``dataset_column``
        so that the same locations are geocoded only once, then it geocodes the
        location list and creates a map between the locations and their
        (latitude, longitude) tuple. The map will finally be used to map the
        locations in the dataset to their coordinates.
        The geocoded locations are cached.

        Parameters
        ----------
        dataset : TripDataset
            The dataset to geocode

        Returns
        -------
        pd.DataFrame
            The dataset with the coordinates of the locations
        """
        # Identify the unique locations to geocode
        locations_to_geocode = set(dataset.df[self.input_columns[0]].unique())
        # Remove the NaNs
        locations_to_geocode = locations_to_geocode - self.nan_codes
        # Add the suffix to the locations (after removing the NaNs!)
        locations_to_geocode = [
            location + self.location_suffix for location in locations_to_geocode
        ]
        # Remove the locations already geocoded
        new_locations_to_geocode = list(
            set(locations_to_geocode) - set(self.location_to_latitude_map.keys())
        )
        print(f"Found {len(new_locations_to_geocode)} locations to geocode")
        # Geocode the locations
        for location in new_locations_to_geocode:
            latitude, longitude = self.geocode_location(location)
            self.location_to_latitude_map[location] = latitude
            self.location_to_longitude_map[location] = longitude
        # Map the locations in the dataset to their coordinates
        new_df = dataset.df.copy()

        # add suffix to the location column
        LOCATION_W_SUFFIX_TEMP_COL = "location_w_suffix"
        new_df[LOCATION_W_SUFFIX_TEMP_COL] = (
            new_df[self.input_columns[0]] + self.location_suffix
        )
        new_df[self.output_latit_column] = new_df[LOCATION_W_SUFFIX_TEMP_COL].map(
            self.location_to_latitude_map
        )
        new_df[self.output_longit_column] = new_df[LOCATION_W_SUFFIX_TEMP_COL].map(
            self.location_to_longitude_map
        )
        # Remove the temporary column
        new_df.drop(columns=[LOCATION_W_SUFFIX_TEMP_COL], inplace=True)

        self.update_cache_df(
            self.location_to_latitude_map, self.location_to_longitude_map
        )
        return new_df

    def __repr__(self) -> str:
        return (
            f"LocationToCoordinatesMapper({self.input_columns[0]},"
            f" {self.output_latit_column}, {self.output_longit_column} "
            f"{type(self.geolocator)})"
        )


class CoordinateToElevationMapper(TripDatasetOperation):
    def __init__(
        self,
        dataset_lat_column: str,
        dataset_long_column: str,
        output_column: str,
        force_repeat: bool = False,
    ) -> None:
        super().__init__(
            input_columns=[dataset_lat_column, dataset_long_column],
            output_columns=[output_column],
            force_repeat=force_repeat,
        )
        self.dataset_lat_column = dataset_lat_column
        self.dataset_long_column = dataset_long_column
        self._url = "https://api.open-elevation.com/api/v1/lookup"
        self.coords_to_elevation_map_cache = {}

    def __call__(self, dataset: TripDataset) -> pd.DataFrame:
        """
        Find the elevation of the locations in the dataset and add this to the dataset

        Parameters
        ----------
        dataset : TripDataset
            The dataset to geocode

        Returns
        -------
        pd.DataFrame
            The pd.DataFrame from ``dataset`` argument, where the
            ``destination_column`` is added, filled with the elevation for each location
        """
        TEMP_COORD_TUPLE_COL = "coords_tuple"
        new_df = dataset.df.copy()
        # 1. Create a column with the tuples of latitude and longitude
        # that we will later map to the elevation
        new_df[TEMP_COORD_TUPLE_COL] = new_df[
            [self.dataset_lat_column, self.dataset_long_column]
        ].apply(
            lambda row: (row[self.dataset_lat_column], row[self.dataset_long_column]),
            axis=1,
        )
        # 2. Create a set of the unique tuples to map (so that we only send one request
        # for each unique combination) and remove the coordinates already geocoded
        missing_coords = list(
            set(new_df[TEMP_COORD_TUPLE_COL].unique())
            - set(self.coords_to_elevation_map_cache.keys())
        )
        # 3. Remove the NaNs
        missing_coords = pd.DataFrame(missing_coords, columns=["latitude", "longitude"])
        missing_coords = missing_coords.dropna(axis=0, how="any")
        missing_coords = list(
            zip(
                missing_coords["latitude"].tolist(),
                missing_coords["longitude"].tolist(),
            )
        )
        print(f"Found {len(missing_coords)} coordinates to find the elevation of")
        # Loop through the dataset getting 100 rows at a time
        for i in range(0, len(missing_coords), 100):
            rows_to_geocode = missing_coords[i : i + 100]
            response = requests.post(
                url=self._url,
                json={
                    "locations": [
                        {"latitude": latit, "longitude": longit}
                        for latit, longit in rows_to_geocode
                    ]
                },
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            results = response.json()["results"]
            # Add elevation to cache
            self.coords_to_elevation_map_cache.update(
                {
                    coord_tuple: result["elevation"]
                    for coord_tuple, result in zip(rows_to_geocode, results)
                }
            )
        # Add the elevation to the dataset
        new_df[self.output_columns[0]] = new_df[TEMP_COORD_TUPLE_COL].map(
            self.coords_to_elevation_map_cache,
        )
        print(f"Found {new_df[self.output_columns[0]].isna().sum()} missing values")
        # Remove the temporary column
        new_df.drop(columns=[TEMP_COORD_TUPLE_COL], inplace=True)

        return new_df

    def __repr__(self) -> str:
        return (
            f"CoordinateToElevationMapper({self.dataset_lat_column},"
            f" {self.dataset_long_column}, {self.output_columns})"
        )


class WeatherIndexCreator(TripDatasetOperation):
    def __init__(
        self,
        trip_location_column: str,
        trip_date_column: str,
        output_columns: List[str],
        vehicle_column: Optional[str] = None,
        force_repeat: bool = False,
    ):
        """
        Create a weather index for each trip in the dataset

        Parameters
        ----------
        trip_location_column : str
            The name of the column in the dataset containing the trip location
        trip_date_column : str
            The name of the column in the dataset containing the trip date. NOTE: the
            dtype of this column must be datetime
        output_columns : List[str]
            The name of the column in the dataset that will be created with the
            computed weather index
        """
        super().__init__(
            input_columns=(
                [trip_location_column, trip_date_column]
                if vehicle_column is None
                else [trip_location_column, trip_date_column, vehicle_column]
            ),
            output_columns=output_columns,
            force_repeat=force_repeat,
        )
        self.trip_location_column = trip_location_column
        self.trip_date_column = trip_date_column
        self.vehicle_column = vehicle_column



class WeatherIndexPerTripCreator(WeatherIndexCreator):
    def __init__(
        self,
        weather_df: pd.DataFrame,
        select_weather_period_operation: SelectPeriodBeforeTripDate,
        aggregate_timeserie_operation: AggregateTimeSerie,
        trip_destination_column: str,
        trip_date_column: str,
        output_column: str = "weather_index",
        force_repeat: bool = False,
        vehicle_column: str = "FLAG_LOCALITA_mapped",
    ):
        super().__init__(
            weather_df=weather_df,
            trip_destination_column=trip_destination_column,
            trip_date_column=trip_date_column,
            output_column=output_column,
            force_repeat=force_repeat,
        )
        self.input_columns += [vehicle_column]
        self.select_weather_period_operation = select_weather_period_operation
        self.aggregate_timeserie_operation = aggregate_timeserie_operation
        self.vehicle_column = vehicle_column

    def _compute_weather_index(self, row: pd.Series) -> float:
        """
        Add the weather index to the row

        Parameters
        ----------
        row : pd.Series
            The row to add the weather index to
        """
        if pd.isna(row[self.trip_location_column]) or pd.isna(
            row[self.trip_date_column]
        ):
            return pd.NA
        else:
            # Create a SingleTrip object
            single_trip = SingleTrip(
                index=row.name,
                location=row[self.trip_location_column],
                start_date=row[self.trip_date_column],
                trip_vehicle=(
                    None
                    if pd.isna(row[self.vehicle_column])
                    else TripVehicle(row[self.vehicle_column])
                ),
            )

            single_trip = self.select_weather_period_operation(single_trip)
            single_trip = self.aggregate_timeserie_operation(single_trip)
            return single_trip[self.aggregate_timeserie_operation.output_attribute]

    def __call__(self, dataset: TripDataset) -> pd.DataFrame:
        """
        Add a column with the weather index.

        Parameters
        ----------
        dataset : TripDataset
            The dataset to add the weather index to
        """
        # Convert date column (with format yyyymmdd to datetime with pd.to_datetime)
        dataset.df[self.trip_date_column] = pd.to_datetime(
            dataset.df[self.trip_date_column], format="%Y%m%d"
        )

        tqdm.pandas()
        # Apply function to each row
        dataset.df[self.output_columns[0]] = dataset.df.progress_apply(
            self._compute_weather_index, axis=1
        )
        return dataset.df

    def __repr__(self) -> str:
        return (
            f"WeatherIndexPerTripCreator({self.select_weather_period_operation},"
            f" {self.aggregate_timeserie_operation}, {self.trip_location_column},"
            f" {self.trip_date_column}, {self.output_columns}, {self.vehicle_column})"
        )


@dataclass
class WeatherIndexOperationsToColumnMap:
    """
    A class to represent an operation to compute the weather index from a timeserie
    """

    select_period: SelectPeriodBeforeTripDate
    output_column_to_aggregator: Dict[str, AggregateTimeSerie]

    def __repr__(self) -> str:
        dict_str = {k: str(v) for k, v in self.output_column_to_aggregator.items()}
        return (
            f"WeatherIndexOperationsToColumnMap({self.select_period}," f" {dict_str})"
        )


class MultipleWeatherIndexCreator(WeatherIndexCreator):
    def __init__(
        self,
        operations_to_column_map: WeatherIndexOperationsToColumnMap,
        trip_location_column: str,
        trip_date_column: str,
        vehicle_column: Optional[str] = None,
        force_repeat: bool = False,
    ):
        """
        Create multiple weather indexes for each trip in the dataset

        Each index aggregates the selected time serie in different ways.
        Since we only select the timeserie once for each trip, this should improve performances, compared to apply WeatherIndexCreator operation multiple times

        Parameters
        ----------
        trip_date_column : str
            The name of the column in the dataset containing the trip dates. NOTE that
            this column must already have the ``datetime`` dtype
        """
        super().__init__(
            trip_location_column=trip_location_column,
            trip_date_column=trip_date_column,
            output_columns=operations_to_column_map.output_column_to_aggregator.keys(),
            vehicle_column=vehicle_column,
            force_repeat=force_repeat,
        )
        self.operations_to_column_map = operations_to_column_map

    def _compute_weather_index(self, row: pd.Series) -> float:
        """
        Add the weather index to the row

        Parameters
        ----------
        row : pd.Series
            The row to add the weather index to
        """
        if pd.isna(row[self.trip_location_column]) or pd.isna(
            row[self.trip_date_column]
        ):
            return {col_name: pd.NA for col_name in self.output_columns}
        else:
            # Create a SingleTrip object
            single_trip = SingleTrip(
                index=row.name,
                location=row[self.trip_location_column],
                start_date=row[self.trip_date_column],
                trip_vehicle=(
                    None
                    if self.vehicle_column is None or pd.isna(row[self.vehicle_column])
                    else TripVehicle(row[self.vehicle_column])
                ),
            )
            # 1. Select period of the timeserie related to the single_trip
            single_trip = self.operations_to_column_map.select_period(single_trip)
            # 2. Compute a multiple weather indexes by applying different types of
            # aggregators to the timeserie
            output_dict = {}
            for (
                col_name,
                aggregator,
            ) in self.operations_to_column_map.output_column_to_aggregator.items():
                single_trip = aggregator(single_trip)
                output_dict[col_name] = single_trip[aggregator.output_attribute]

            return output_dict

    def __call__(self, dataset: TripDataset, n_jobs: int = 3) -> pd.DataFrame:
        """
        Add a column with the weather index.

        Parameters
        ----------
        dataset : TripDataset
            The dataset to add the weather index to
        """
        dataset.df[self.trip_date_column] = pd.to_datetime(
            dataset.df[self.trip_date_column]
        )
        print(
            f"Found {dataset.df[self.trip_date_column].isna().sum()} missing values"
            " in ``trip_date_column``"
        )
        print(
            f"Found {dataset.df[self.trip_location_column].isna().sum()} missing values"
            " in ``trip_location_column``"
        )
        tqdm.pandas()
        # Apply function to each row. The output will be a list of dictionaries (which
        # is the output type of the function applied)

        apply_output_dict = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self._compute_weather_index)(row=row)
            for _, row in tqdm(dataset.df.iterrows())
        )
        df = pd.concat([dataset.df, pd.DataFrame(apply_output_dict)], axis=1)

        # Count NA by column
        nan_count_per_row_in_output_cols = df[self.output_columns].isna().sum(axis=1)
        print(
            f"Found {(nan_count_per_row_in_output_cols).sum()} missing values "
            f"in {len(self.output_columns)} columns"
        )

        # Find which values have input_columns when output_columns are NA
        # Rows with at least 1 NA in output_columns
        rows_1_output_na = df[nan_count_per_row_in_output_cols > 0]
        print(
            "Rows with missing outputs have the following values for",
            " ``trip_location_column``: \n",
            rows_1_output_na[self.trip_location_column].value_counts(),
        )

        return df

    def __repr__(self) -> str:
        return (
            f"MultipleWeatherIndexCreator({self.operations_to_column_map},"
            f" {self.trip_location_column}, {self.trip_date_column})"
        )
