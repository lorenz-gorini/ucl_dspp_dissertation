from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely
import xarray as xr
from joblib import Parallel, delayed
from rioxarray.exceptions import NoDataInBounds

from src.geotimeserie_dataset import GeoTimeSerieDataset, WeatherTimeSeriesEnum
from src.geotimeserie_operations import (
    AreaClipOperation,
    CastToTypeOperation,
    InterpolateOperation,
    MeanAggregator,
    SetCRSOperation,
    TimeRangeClipOperation,
)
from src.polygon_areas import PolygonAreasFromFile

POST_ANALYSIS = False
timeserie_name = WeatherTimeSeriesEnum.MEAN_TEMPERATURE

raw_ts_data = GeoTimeSerieDataset(
    timeserie_name=timeserie_name,
    file_crs="EPSG:4326",
    nc_file_folder=Path(
        "/mnt/c/Users/loreg/Documents/dissertation_data/"
        "dataset-insitu-gridded-observations-europe/"
    ),
)

raw_ts_data = raw_ts_data.apply(
    [TimeRangeClipOperation(start_time=f"1980-01-01", end_time="2019-12-31")]
)

# read shapefile
country_shp = gpd.read_file(
    "/mnt/c/Users/loreg/Documents/dissertation_data/world-administrative-boundaries"
    "/world-administrative-boundaries.shp"
)
country_shp = country_shp.to_crs("EPSG:4326")
europe_country_shp = country_shp[country_shp["continent"] == "Europe"]

country_polygons = PolygonAreasFromFile(
    column_name="name",
    crs="EPSG:4326",
    shapefile_path=None,
    geo_df=europe_country_shp,
).polygon_names_dict

province_polygons = PolygonAreasFromFile(
    column_name="DEN_UTS",
    crs="EPSG:4326",
    shapefile_path=(
        "/mnt/c/Users/loreg/Documents/dissertation_data/"
        "Municipal_Boundaries_of_Italy_2019/ProvCM01012023_g/"
        "ProvCM01012023_g_WGS84.shp"
    ),
).polygon_names_dict


def get_province_temperature(
    province_name: str, year: int, polygon: shapely.Polygon, raw_ts_data: xr.Dataset
) -> xr.Dataset:
    print(f"Extracting time serie for {(province_name, year)}")
    try:
        one_prov_one_year_ts = raw_ts_data.apply(
            [
                SetCRSOperation(crs="EPSG:4326"),
                TimeRangeClipOperation(
                    start_time=f"{year}-01-01", end_time=f"{year}-12-31"
                ),
                AreaClipOperation(area=polygon),
                CastToTypeOperation(
                    variable_name=WeatherTimeSeriesEnum.MEAN_TEMPERATURE.value,
                    dtype="float32",
                ),
                InterpolateOperation(target_resolution=0.03),
                MeanAggregator(columns=["latitude", "longitude"]),
            ]
        )
        return one_prov_one_year_ts.to_dataframe()[
            WeatherTimeSeriesEnum.MEAN_TEMPERATURE.value
        ].rename(province_name)
    except NoDataInBounds:
        print(f"No data for {(province_name, year)}")
        return pd.Series(name=province_name)


for location_type, polygons_per_location in [
    ("country", country_polygons),
    ("province", province_polygons),
]:
    timeseries_per_prov = []
    for name, polygon in list(country_polygons.items()):
        one_prov_multi_year_list = Parallel(n_jobs=3, backend="loky")(
            delayed(get_province_temperature)(
                province_name=name, year=year, polygon=polygon, raw_ts_data=raw_ts_data
            )
            for year in range(1997, 2020)
        )
        # Combine the timeseries of each year into a single pd.Series
        single_prov_ts_serie = pd.concat(one_prov_multi_year_list, axis=0)
        timeseries_per_prov.append(single_prov_ts_serie)

    timeseries_per_prov_df = pd.DataFrame(timeseries_per_prov).T
    timeseries_per_prov_df.index = pd.to_datetime(timeseries_per_prov_df.index)
    timeseries_per_prov_df.reset_index(inplace=True, drop=False)
    timeseries_per_prov_df.rename(columns={"index": "date"}, inplace=True)
    file_path = (
        "/mnt/c/Users/loreg/Documents/dissertation_data/"
        f"timeseries_per_{location_type}.csv"
    )
    timeseries_per_prov_df.to_csv(file_path, index=False)


if POST_ANALYSIS:
    timeserie_name = WeatherTimeSeriesEnum.MEAN_TEMPERATURE
    location_type = "province"
    timeseries_per_prov_df = pd.read_csv(
        "/mnt/c/Users/loreg/Documents/dissertation_data/"
        f"timeserie_{timeserie_name.value}_per_{location_type}.csv"
    )
    timeseries_per_prov_df.T.plot()
    # count nan values and sort by number of nan values
    timeseries_per_prov_df.isna().sum(axis=0).sort_values(ascending=False)

    # Groupby the combination of (month, year) (contained in the index datetime) and get the mean temperature for each month
    df_by_year_month = timeseries_per_prov_df.groupby(
        [timeseries_per_prov_df.time.year, timeseries_per_prov_df.time.month]
    ).mean()
    # Combine the month and year into a single datetime
    df_by_year_month.time = pd.to_datetime(
        df_by_year_month.time.get_level_values(0).astype(str)
        + "-"
        + df_by_year_month.time.get_level_values(1).astype(str)
    )
    df_by_year_month.plot()

    # Groupby year (contained in the index datetime) and get the mean temperature for each year
    df_by_year = timeseries_per_prov_df.groupby(timeseries_per_prov_df.time.year).mean()
    df_by_year.index = pd.to_datetime(df_by_year.time.astype(str))
    # Drop time column
    df_by_year.drop(columns=["time"], inplace=True)
    df_by_year.plot()
