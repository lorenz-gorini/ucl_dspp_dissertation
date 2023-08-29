"""
Script to map the country IDs from "STATO_VISITATO" column into actual country names and save them in "STATO_VISITATA_mapped" column.
"""
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.single_trip_operations import TripVehicle
from src.trip_dataset import TouristOrigin, TripDataset, VariableSubset
from src.trip_operations import (
    CodeToLocationMapperFromCSV,
    CodeToStringMapper,
    FilterCountries,
    ReplaceValuesByMap,
    ToDatetimeConverter,
    TripStartDateCreator,
)

# 1’ = Veicolo stradale ‘2’= Treno ‘3’= Aereo ‘4’ = Nave
vehicle_type_mapper = CodeToStringMapper(
    input_column="FLAG_LOCALITA",
    output_column="FLAG_LOCALITA_mapped",
    code_map={i.name: i.value for i in TripVehicle},
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
    trip_end_date_format="%d-%m-%Y",
    duration_column_unit="days",
    force_repeat=False,
)

visited_country_code_mapper = CodeToLocationMapperFromCSV(
    input_column="STATO_VISITATO",
    output_column="STATO_VISITATO_mapped",
    code_map_csv=Path(
        "/mnt/c/Users/loreg/Documents/dissertation_data/Metadati-CSV/TSTATI.csv"
    ),
    code_column_csv="ELEM_DOMINIO",
    location_name_column="D_EL_DOMINIO",
    separator=";",
    nan_codes=[0, 99999],
    force_repeat=False,
)

residence_province_code_mapper = CodeToLocationMapperFromCSV(
    input_column="PROV_RESIDENZA",
    output_column="PROV_RESIDENZA_mapped",
    code_map_csv=Path(
        "/mnt/c/Users/loreg/Documents/dissertation_data/Metadati-CSV/TPROVINCE.csv"
    ),
    code_column_csv="ELEM_DOMINIO",
    location_name_column="D_EL_DOMINIO",
    separator=";",
    nan_codes=[0, 99999],
    force_repeat=False,
)
replace_province_names_by_map = ReplaceValuesByMap(
    input_column="PROV_RESIDENZA_mapped",
    output_column="PROV_RESIDENZA_mapped",
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
timeseries_per_country_df = pd.read_csv(
    "/mnt/c/Users/loreg/Documents/dissertation_data/timeserie_tg_per_country.csv"
)
filter_italian_tourists_to_europe = FilterCountries(
    country_column="STATO_VISITATO_mapped",
    countries=timeseries_per_country_df.columns.to_list(),
    force_repeat=False,
)


# 2. Load the data by year and compute the aggregated values by year and
# by country of origin
for year in tqdm(range(1997, 2023, 1)):
    dataset = TripDataset(
        variable_subset=VariableSubset.PRIMARY,
        tourist_origin=TouristOrigin.ITALIANS,
        year=year,
        raw_folder=Path("/mnt/c/Users/loreg/Documents/dissertation_data/raw"),
        processed_folder=Path(
            "/mnt/c/Users/loreg/Documents/dissertation_data/processed"
        ),
        force_raw=False,
    )
    initial_nrows = dataset.df.shape[0]
    print("year ", year)

    dataset.apply(visited_country_code_mapper)
    dataset.apply(residence_province_code_mapper)
    dataset.apply(replace_province_names_by_map)
    dataset.apply(vehicle_type_mapper)
    dataset.apply(datetime_converter)
    dataset.apply(trip_start_date_creator)
    assert (
        initial_nrows == dataset.df.shape[0]
    ), f"{initial_nrows - dataset.df.shape[0]} rows were dropped, check the code"

    dataset.apply(filter_italian_tourists_to_europe)
