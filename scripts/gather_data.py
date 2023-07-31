"""
Script to gather microdata from Bank of Italy online database
"""
from src.dataset import MicroDataset, TouristOrigin, VariableSubset

from pathlib import Path
from joblib import Parallel, delayed


# function to call in parallel
def download_file_function(variable_subset, tourist_origin, year, raw_folder):
    file = MicroDataset(
        variable_subset=variable_subset,
        tourist_origin=tourist_origin,
        year=year,
        raw_folder=raw_folder,
        processed_folder=None,
    )
    file.download()


years = range(1997, 2022, 1)
# Loop over variable_type + italian/foreigners + years
for var_subset in VariableSubset:
    for origin in TouristOrigin:
        Parallel(n_jobs=7, backend="loky")(
            delayed(download_file_function)(
                variable_subset=var_subset,
                tourist_origin=origin,
                year=year,
                raw_folder=Path(
                    "/mnt/c/Users/loreg/Documents/dissertation_data/raw"
                ),
            )
            for year in range(1997, 2022, 1)
        )

"https://www.bancaditalia.it/statistiche/tematiche/rapporti-estero/turismo-internazionale/distribuzione-microdati/file-dati/documenti/2013/csv/Residenza-dei-viaggiatori-Stranieri-Principali-CSV.zip"
# Download the expansion factors for 2022
URL = "https://www.bancaditalia.it/statistiche/tematiche/rapporti-estero/turismo-internazionale/distribuzione-microdati/file-dati/documenti/2022/csv/ita_2022_fp_csv.zip"
