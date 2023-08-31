"""
Script to compute daily average of historical data, which will be used to compute
heatwave occurrence, according to World Meteorological
Organization's definition, by ``src.single_trip_operations.HeatwavesAggregator``
aggregator class. 
The heatwave is defined as the period (at least 5 consecutive day-long) where the
daily temperature is 5 degrees C above the average of the daily maximum
temperature in the reference period 1961-1990.
So, this script allows to compute the average of the daily maximum temperature in the
reference period 1961-1990, contained in the file ``FILE_PATH``
"""

import datetime
from pathlib import Path
import pandas as pd

from src.geotimeserie_dataset import WeatherTimeSeriesEnum

timeserie_name = WeatherTimeSeriesEnum.MAX_TEMPERATURE

for location_type in ["province", "country"]:
    FILE_PATH = Path(
        "/mnt/c/Users/loreg/Documents/dissertation_data/"
        f"timeserie_{timeserie_name.value}_per_{location_type}_historical.csv"
    )
    df = pd.read_csv(FILE_PATH)
    df["DATE"] = pd.to_datetime(df["DATE"])

    df_mean = df.groupby([df["DATE"].dt.month, df["DATE"].dt.day]).mean()
    df_mean["month"] = df_mean.index.get_level_values(0)
    df_mean["day"] = df_mean.index.get_level_values(1)
    df_mean = df_mean.drop(columns=["DATE"])
    df_mean.reset_index(inplace=True, drop=True)
    # Create date column by combining month and day columns
    df_mean["DATE"] = df_mean.apply(
        lambda row: datetime.datetime(
            # because 1980 is a leap year so we have all the possible dates
            year=1980,
            month=int(row["month"]),
            day=int(row["day"]),
        ),
        axis=1,
    )
    df_mean.to_csv(
        "/mnt/c/Users/loreg/Documents/dissertation_data/"
        f"timeserie_{timeserie_name.value}_per_{location_type}_historical_mean.csv",
        index=False,
    )
