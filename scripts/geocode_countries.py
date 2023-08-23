"""
Script to map the country IDs from "STATO_VISITATO" column into actual country names and save them in "STATO_VISITATA_mapped" column.
"""
from pathlib import Path

from tqdm import tqdm

from src.single_trip_operations import TripVehicle
from src.trip_dataset import TouristOrigin, TripDataset, VariableSubset
from src.trip_operations import (
    CodeToLocationMapperFromCSV,
    CodeToStringMapper,
    TripStartDateCreator,
)

code_mapper = CodeToLocationMapperFromCSV(
    input_column="STATO_VISITATO",
    output_column="STATO_VISITATO_mapped",
    code_map_csv=Path(
        "/mnt/c/Users/loreg/Documents/dissertation_data/Metadati-CSV/TSTATI.csv"
    ),
    code_column_csv="ELEM_DOMINIO",
    location_name_column="D_EL_DOMINIO",
    separator=";",
    nan_codes=[0, 99999],
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
    )
    dataset.apply(code_mapper)
