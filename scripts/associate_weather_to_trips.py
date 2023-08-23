import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.single_trip_operations import MeanAggregator, SelectConstantPeriodBeforeDate
from src.trip_dataset import TouristOrigin, TripDataset, VariableSubset
from src.trip_operations import WeatherIndexPerTripCreator

timeseries_per_province_df = pd.read_csv(
    "/mnt/c/Users/loreg/Documents/dissertation_data/timeseries_per_province_mapped.csv"
)

POST_ANALYSIS = False

weather_index_creator = WeatherIndexPerTripCreator(
    weather_df=timeseries_per_province_df,
    output_column="weather_index",
    trip_destination_column="PROVINCIA_VISITATA_mapped",
    trip_date_column="DATA_INTERVISTA",  # DATA_INIZ_VIAGGIO_computed
    vehicle_column="FLAG_LOCALITA",
    select_weather_period_operation=SelectConstantPeriodBeforeDate(
        timeserie_df=timeseries_per_province_df,
        date_column="DATE",
        location_column="PROVINCIA_VISITATA_mapped",
        # 2 months in order to have a more stable weather index that can describe
        # the expected weather in those months during the trip
        period_length=datetime.timedelta(days=60),
        # Previous Year, during the same period, 1 month before (half of the period length)
        time_before_date=datetime.timedelta(days=365),
    ),
    aggregate_timeserie_operation=MeanAggregator(),
    force_repeat=False,
)

# read tourism data
for year in tqdm(range(1997, 2023, 1)):
    dataset = TripDataset(
        variable_subset=VariableSubset.PRIMARY,
        tourist_origin=TouristOrigin.FOREIGNERS,
        year=year,
        raw_folder=Path("/mnt/c/Users/loreg/Documents/dissertation_data/raw"),
        processed_folder=Path(
            "/mnt/c/Users/loreg/Documents/dissertation_data/processed"
        ),
        # ``CHIAVE`` column is a very long int, much longer than the maximum size of Int64
        column_to_dtype_map={"CHIAVE": str},
    )
    dataset.apply(weather_index_creator)

if POST_ANALYSIS:
    # Scatter Plot with dataset.df["weather_index"] and dataset.df["DATA_INTERVISTA"].dt.month
    dataset._df["DATA_INTERVISTA_mapped"] = pd.to_datetime(
        dataset._df["DATA_INTERVISTA"], format="%Y%m%d"
    )
    dataset._df["month"] = dataset._df["DATA_INTERVISTA"].dt.month
    # Reduce size of points
    dataset.df.plot.scatter(x="month", y="weather_index", s=0.1)
    dataset.df["month"].hist()
