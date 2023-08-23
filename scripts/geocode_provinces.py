"""
Script to geocode the locations in the dataset and add the coordinates to the dataset.

First I map "PROVINCIA_VISITATA" column to "PROVINCIA_VISITATA_mapped" column
and then I geocode it to get the coordinates.

I will geolocate the province for the center+south of Italy, but I will geolocate the
city for the north of Italy ("PROVINCIA_VISITATA" column), because there is more
variance in snow coverage and altitude in the north of Italy, and there are only few
extended provinces.
"""
import os
from pathlib import Path

import googlemaps
from tqdm import tqdm

from src.single_trip_operations import TripVehicle
from src.trip_dataset import TouristOrigin, TripDataset, VariableSubset
from src.trip_operations import (
    CodeToLocationMapperFromCSV,
    CodeToStringMapper,
    CoordinateToElevationMapper,
    LocationToCoordinatesMapper,
    TripStartDateCreator,
)

code_mapper = CodeToLocationMapperFromCSV(
    input_column="PROVINCIA_VISITATA",
    output_column="PROVINCIA_VISITATA_mapped",
    code_map_csv=Path(
        "/mnt/c/Users/loreg/Documents/dissertation_data/Metadati-CSV/TPROVINCE.csv"
    ),
    code_column_csv="ELEM_DOMINIO",
    location_name_column="D_EL_DOMINIO",
    separator=";",
    nan_codes=[0, 99999],
)

location_to_coord_mapper = LocationToCoordinatesMapper(
    input_column="PROVINCIA_VISITATA_mapped",
    output_latit_column="PROVINCIA_VISITATA_latitude",
    output_longit_column="PROVINCIA_VISITATA_longitude",
    geolocator=googlemaps.Client(key=os.environ.get("GMAP_API_KEY")),
    location_suffix=", Italy",
    force_repeat=False,
)
coord_to_elevation_mapper = CoordinateToElevationMapper(
    dataset_lat_column="PROVINCIA_VISITATA_latitude",
    dataset_long_column="PROVINCIA_VISITATA_longitude",
    output_column="PROVINCIA_VISITATA_elevation",
    force_repeat=False,
)

# 2. Load the data by year and compute the aggregated values by year and
# by country of origin
total_dfs_by_year = []
for year in tqdm(range(1997, 2023, 1)):
    dataset = TripDataset(
        variable_subset=VariableSubset.PRIMARY,
        tourist_origin=TouristOrigin.FOREIGNERS,
        year=year,
        raw_folder=Path("/mnt/c/Users/loreg/Documents/dissertation_data/raw"),
        processed_folder=Path(
            "/mnt/c/Users/loreg/Documents/dissertation_data/processed"
        ),
    )
    dataset.apply(code_mapper)
    dataset.apply(location_to_coord_mapper)
    dataset.apply(coord_to_elevation_mapper)
