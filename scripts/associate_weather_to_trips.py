import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.geotimeserie_dataset import WeatherTimeSeriesEnum
from src.single_trip_operations import (
    MaxAggregator,
    MeanAggregator,
    SelectConstantPeriodBeforeDate,
    StdAggregator,
    HeatwavesAggregator,
)
from src.trip_dataset import (
    GenericTripDataset,
    TouristOrigin,
    TripDataset,
    VariableSubset,
)
from src.trip_operations import (
    MultipleWeatherIndexCreator,
    ToDatetimeConverter,
    WeatherIndexOperationsToColumnMap,
)

POST_ANALYSIS = False
USE_SAMPLED = True
weather_timeserie = WeatherTimeSeriesEnum.MAX_TEMPERATURE  # MEAN_TEMPERATURE


timeseries_per_province_df = pd.read_csv(
    "/mnt/c/Users/loreg/Documents/dissertation_data/timeserie_"
    f"{weather_timeserie.value}_per_province.csv"
)
timeseries_per_country_df = pd.read_csv(
    "/mnt/c/Users/loreg/Documents/dissertation_data/timeserie_"
    f"{weather_timeserie.value}_per_country.csv"
)
timeserie_tx_per_province_historical = pd.read_csv(
    "/mnt/c/Users/loreg/Documents/dissertation_data/timeserie_"
    f"{weather_timeserie.value}_per_province_historical_mean.csv"
)
timeserie_tx_per_country_historical = pd.read_csv(
    "/mnt/c/Users/loreg/Documents/dissertation_data/timeserie_"
    f"{weather_timeserie.value}_per_country_historical_mean.csv"
)

datetime_converter = ToDatetimeConverter(
    date_column="DATA_INTERVISTA",
    date_format="%Y%m%d",
    output_column="DATA_INTERVISTA",
    force_repeat=False,
)

# ============== European tourists to Italy ==============

visited_province_index_creator = MultipleWeatherIndexCreator(
    trip_location_column="PROVINCIA_VISITATA_mapped",
    trip_date_column="DATA_INIZ_VIAGGIO_computed",  # DATA_INIZ_VIAGGIO_computed
    operations_to_column_map=WeatherIndexOperationsToColumnMap(
        select_period=SelectConstantPeriodBeforeDate(
            timeserie_df=timeseries_per_province_df,
            date_column="DATE",
            location_column="PROVINCIA_VISITATA_mapped",
            # 2 months in order to have a more stable weather index that can describe
            # the expected weather in those months during the trip
            period_length=datetime.timedelta(days=60),
            # Previous Year, during the same period, 1 month before (half of the period length)
            time_before_date=datetime.timedelta(days=365),
        ),
        output_column_to_aggregator={
            f"PROVINCIA_VISITATA_{weather_timeserie.value}_1y_before_2m_period_mean": MeanAggregator(),
            f"PROVINCIA_VISITATA_{weather_timeserie.value}_1y_before_2m_period_std": StdAggregator(),
            f"PROVINCIA_VISITATA_{weather_timeserie.value}_1y_before_2m_period_max": MaxAggregator(),
            f"PROVINCIA_VISITATA_{weather_timeserie.value}_1y_before_2m_period_heatwaves": HeatwavesAggregator(
                historical_data=timeserie_tx_per_province_historical, date_column="DATE"
            ),
        },
    ),
    force_repeat=False,
)
residence_country_index_creator = MultipleWeatherIndexCreator(
    trip_location_column="STATO_RESIDENZA_mapped",
    trip_date_column="DATA_INIZ_VIAGGIO_computed",  # DATA_INIZ_VIAGGIO_computed
    operations_to_column_map=WeatherIndexOperationsToColumnMap(
        select_period=SelectConstantPeriodBeforeDate(
            timeserie_df=timeseries_per_country_df,
            date_column="DATE",
            location_column="STATO_RESIDENZA_mapped",
            # 3 months before leaving, 3 months period
            period_length=datetime.timedelta(days=90),
            time_before_date=datetime.timedelta(days=90),
        ),
        output_column_to_aggregator={
            f"STATO_RESIDENZA_{weather_timeserie.value}_3m_before_3m_period_mean": MeanAggregator(),
            f"STATO_RESIDENZA_{weather_timeserie.value}_3m_before_3m_period_std": StdAggregator(),
            f"STATO_RESIDENZA_{weather_timeserie.value}_3m_before_3m_period_max": MaxAggregator(),
            f"STATO_RESIDENZA_{weather_timeserie.value}_3m_before_3m_period_heatwaves": HeatwavesAggregator(
                historical_data=timeserie_tx_per_country_historical, date_column="DATE"
            ),
        },
    ),
    force_repeat=False,
)

# ============== Italian tourists to Europe ==============

visited_country_index_creator = MultipleWeatherIndexCreator(
    trip_location_column="STATO_VISITATO_mapped",
    trip_date_column="DATA_INIZ_VIAGGIO_computed",  # DATA_INIZ_VIAGGIO_computed
    operations_to_column_map=WeatherIndexOperationsToColumnMap(
        select_period=SelectConstantPeriodBeforeDate(
            timeserie_df=timeseries_per_country_df,
            date_column="DATE",
            location_column="STATO_VISITATO_mapped",
            # 2 months in order to have a more stable weather index that can describe
            # the expected weather in those months during the trip
            period_length=datetime.timedelta(days=60),
            # Previous Year, during the same period, 1 month before (half of the period length)
            time_before_date=datetime.timedelta(days=365),
        ),
        output_column_to_aggregator={
            f"STATO_VISITATO_{weather_timeserie.value}_1y_before_2m_period_mean": MeanAggregator(),
            f"STATO_VISITATO_{weather_timeserie.value}_1y_before_2m_period_std": StdAggregator(),
            f"STATO_VISITATO_{weather_timeserie.value}_1y_before_2m_period_max": MaxAggregator(),
            f"STATO_VISITATO_{weather_timeserie.value}_1y_before_2m_period_heatwaves": HeatwavesAggregator(
                historical_data=timeserie_tx_per_country_historical, date_column="DATE"
            ),
        },
    ),
    force_repeat=False,
)

residence_province_index_creator = MultipleWeatherIndexCreator(
    trip_location_column="PROV_RESIDENZA_mapped",
    trip_date_column="DATA_INIZ_VIAGGIO_computed",  # DATA_INIZ_VIAGGIO_computed
    operations_to_column_map=WeatherIndexOperationsToColumnMap(
        select_period=SelectConstantPeriodBeforeDate(
            timeserie_df=timeseries_per_province_df,
            date_column="DATE",
            location_column="PROV_RESIDENZA_mapped",
            # 3 months before leaving, 3 months period
            period_length=datetime.timedelta(days=90),
            time_before_date=datetime.timedelta(days=90),
        ),
        output_column_to_aggregator={
            f"PROV_RESIDENZA_{weather_timeserie.value}_3m_before_3m_period_mean": MeanAggregator(),
            f"PROV_RESIDENZA_{weather_timeserie.value}_3m_before_3m_period_std": StdAggregator(),
            f"PROV_RESIDENZA_{weather_timeserie.value}_3m_before_3m_period_max": MaxAggregator(),
            f"PROV_RESIDENZA_{weather_timeserie.value}_3m_before_3m_period_heatwaves": HeatwavesAggregator(
                historical_data=timeserie_tx_per_province_historical, date_column="DATE"
            ),
        },
    ),
    force_repeat=False,
)

# read tourism data
for tourist_origin in [
    TouristOrigin.ITALIANS,
    # TouristOrigin.FOREIGNERS,
]:
    if USE_SAMPLED:
        dataset = GenericTripDataset(
            raw_file_path=Path("/mnt/c/Users/loreg/Documents/dissertation_data/raw")
            / f"sample_5K_{tourist_origin.value}.csv",
            processed_file_path=Path(
                "/mnt/c/Users/loreg/Documents/dissertation_data/processed"
                f"/sample_5K_{tourist_origin.value}.csv",
            ),
            # ``CHIAVE`` column is a very long int, much longer than the maximum size of Int64
            column_to_dtype_map={"CHIAVE": str},
            force_raw=False,
        )
        dataset = dataset.apply(datetime_converter, save=False)
        if tourist_origin == TouristOrigin.ITALIANS:
            dataset = dataset.apply(visited_country_index_creator, save=True)
            dataset = dataset.apply(residence_province_index_creator, save=True)
        else:
            dataset = dataset.apply(visited_province_index_creator, save=True)
            dataset = dataset.apply(residence_country_index_creator, save=True)

    else:
        for year in tqdm(range(1997, 2023, 1)):
            dataset = TripDataset(
                variable_subset=VariableSubset.PRIMARY,
                tourist_origin=tourist_origin,
                year=year,
                raw_folder=Path("/mnt/c/Users/loreg/Documents/dissertation_data/raw"),
                processed_folder=Path(
                    "/mnt/c/Users/loreg/Documents/dissertation_data/processed"
                ),
                force_raw=False,
                column_to_dtype_map={"CHIAVE": str},
            )
            dataset = dataset.apply(datetime_converter, save=False)
            if tourist_origin == TouristOrigin.ITALIANS:
                dataset = dataset.apply(visited_country_index_creator, save=True)
                dataset = dataset.apply(residence_province_index_creator, save=True)
            else:
                dataset = dataset.apply(visited_province_index_creator, save=True)
                dataset = dataset.apply(residence_country_index_creator, save=True)


if POST_ANALYSIS:
    # Scatter Plot with dataset.df["weather_index"] and dataset.df["DATA_INTERVISTA"].dt.month
    dataset._df["DATA_INTERVISTA_mapped"] = pd.to_datetime(
        dataset._df["DATA_INTERVISTA"], format="%Y%m%d"
    )
    dataset._df["month"] = dataset._df["DATA_INTERVISTA"].dt.month
    # Reduce size of points
    dataset.df.plot.scatter(x="month", y="weather_index", s=0.1)
    dataset.df["month"].hist()
