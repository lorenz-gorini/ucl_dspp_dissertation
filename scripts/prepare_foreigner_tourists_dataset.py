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
import pandas as pd
from tqdm import tqdm

from src.single_trip_operations import TripVehicle
from src.trip_dataset import TouristOrigin, TripDataset, VariableSubset
from src.trip_operations import (
    CodeToLocationMapperFromCSV,
    CodeToStringMapper,
    CoordinateToElevationMapper,
    FilterCountries,
    LocationToCoordinatesMapper,
    ReplaceValuesByMap,
    ToDatetimeConverter,
    TripStartDateCreator,
)

prov_code_mapper = CodeToLocationMapperFromCSV(
    input_column="PROVINCIA_VISITATA",
    output_column="PROVINCIA_VISITATA_mapped",
    code_map_csv=Path(
        "/mnt/c/Users/loreg/Documents/dissertation_data/Metadati-CSV/TPROVINCE.csv"
    ),
    code_column_csv="ELEM_DOMINIO",
    location_name_column="D_EL_DOMINIO",
    separator=";",
    nan_codes=[0, 99999],
    force_repeat=False,
)
timeseries_per_country_df = pd.read_csv(
    "/mnt/c/Users/loreg/Documents/dissertation_data/timeserie_tg_per_country.csv"
)
filter_european_tourists_to_italy = FilterCountries(
    country_column="STATO_RESIDENZA_mapped",
    countries=timeseries_per_country_df.columns.to_list(),
    force_repeat=False,
)
replace_names_by_map = ReplaceValuesByMap(
    input_column="PROVINCIA_VISITATA_mapped",
    output_column="PROVINCIA_VISITATA_mapped",
    map_dict={
        "CARBONIA-IGLESIAS": "SUD SARDEGNA",
        "OGLIASTRA": "NUORO",
        "OLBIA-TEMPIO": "SASSARI",
        "MEDIO CAMPIDANO": "SUD SARDEGNA",
        "SANTA SEDE CITTA' DEL VATICANO": "ROMA",
        "CITTA' DEL VATICANO": "ROMA",
    },
    force_repeat=False,
)

residence_country_mapper = CodeToLocationMapperFromCSV(
    input_column="STATO_RESIDENZA",
    output_column="STATO_RESIDENZA_mapped",
    code_map_csv=Path(
        "/mnt/c/Users/loreg/Documents/dissertation_data/Metadati-CSV/TSTATI.csv"
    ),
    code_column_csv="ELEM_DOMINIO",
    location_name_column="D_EL_DOMINIO",
    separator=";",
    nan_codes=[0, 99999],
    force_repeat=False,
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

# '1’ = Veicolo stradale ‘2’= Treno ‘3’= Aereo ‘4’ = Nave
vehicle_type_mapper = CodeToStringMapper(
    input_column="FLAG_LOCALITA",
    output_column="FLAG_LOCALITA_mapped",
    code_map={i.value: i.name for i in TripVehicle},
    force_repeat=False,
)
datetime_converter = ToDatetimeConverter(
    date_column="DATA_INTERVISTA",
    date_format="%Y%m%d",
    output_column="DATA_INTERVISTA",
    force_repeat=False,
)

trip_start_date_creator = TripStartDateCreator(
    trip_end_date_column="DATA_INTERVISTA",
    trip_duration_column="NR_NOTTI",
    output_column="DATA_INIZ_VIAGGIO_computed",
    trip_end_date_format="%Y-%m-%d",
    duration_column_unit="days",
    force_repeat=False,
)

for year in tqdm(range(1997, 2020, 1)):
    dataset = TripDataset(
        variable_subset=VariableSubset.PRIMARY,
        tourist_origin=TouristOrigin.FOREIGNERS,
        year=year,
        raw_folder=Path("/mnt/c/Users/loreg/Documents/dissertation_data/raw"),
        processed_folder=Path(
            "/mnt/c/Users/loreg/Documents/dissertation_data/processed"
        ),
        force_raw=False,
        column_to_dtype_map={"CHIAVE": str},
    )
    initial_nrows = dataset.df.shape[0]
    print("year ", year)
    dataset = dataset.apply(prov_code_mapper, save=False)
    dataset = dataset.apply(replace_names_by_map, save=False)
    dataset = dataset.apply(location_to_coord_mapper, save=False)
    dataset = dataset.apply(coord_to_elevation_mapper, save=False)

    dataset = dataset.apply(residence_country_mapper, save=False)
    dataset = dataset.apply(vehicle_type_mapper, save=False)
    dataset = dataset.apply(datetime_converter, save=False)
    dataset = dataset.apply(trip_start_date_creator, save=False)
    assert (
        initial_nrows == dataset.df.shape[0]
    ), f"{initial_nrows - dataset.df.shape[0]} rows were dropped, check the code"

    dataset = dataset.apply(filter_european_tourists_to_italy, save=True)

    print("OK")
