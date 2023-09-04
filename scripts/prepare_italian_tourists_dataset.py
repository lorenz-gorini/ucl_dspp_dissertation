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
    MergeWithDataset,
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
for year in tqdm(range(1997, 2020, 1)):
    dataset = TripDataset(
        variable_subset=VariableSubset.PRIMARY,
        tourist_origin=TouristOrigin.ITALIANS,
        year=year,
        raw_folder=Path("/mnt/c/Users/loreg/Documents/dissertation_data/raw"),
        processed_folder=Path(
            "/mnt/c/Users/loreg/Documents/dissertation_data/processed"
        ),
        force_raw=True,
    )
    initial_nrows = dataset.df.shape[0]
    print("year ", year)

    dataset = dataset.apply(visited_country_code_mapper, save=False)
    dataset = dataset.apply(residence_province_code_mapper, save=False)
    dataset = dataset.apply(replace_province_names_by_map, save=False)
    dataset = dataset.apply(vehicle_type_mapper, save=False)
    dataset = dataset.apply(datetime_converter, save=False)
    dataset = dataset.apply(trip_start_date_creator, save=False)
    assert (
        initial_nrows == dataset.df.shape[0]
    ), f"{initial_nrows - dataset.df.shape[0]} rows were dropped, check the code"

    dataset = dataset.apply(filter_italian_tourists_to_europe, save=False)

    # Add expansion factors by merging with the dataset containing them
    nrows_before_merge = dataset.df.shape[0]
    dataset_exp_factor = TripDataset(
        variable_subset=VariableSubset.EXPANSION_FACTORS,
        tourist_origin=TouristOrigin.ITALIANS,
        year=year,
        raw_folder=Path("/mnt/c/Users/loreg/Documents/dissertation_data/raw"),
        force_raw=False,
    )
    dataset_merge = dataset.apply(
        MergeWithDataset(dataset_exp_factor, on="CHIAVE", how="inner"), save=False
    )
    assert nrows_before_merge == dataset.df.shape[0], (
        f"{nrows_before_merge - dataset.df.shape[0]} rows were dropped due "
        "to Merge operation"
    )
    dataset_merge.save_df()
