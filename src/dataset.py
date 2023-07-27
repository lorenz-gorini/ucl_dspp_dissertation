import os
import zipfile

from pathlib import Path
from typing import Optional
from io import BytesIO
import enum
from functools import cached_property

import requests
import pandas as pd


class VariableSubset(enum.Enum):
    PRIMARY = "principali"
    SECONDARY = "secondarie"
    EXPANSION_FACTORS = "fp"


class TouristOrigin(enum.Enum):
    ITALIAN = "ita"
    FOREIGNERS = "stra"


# For some reason the 2013 files are not in the same format as the others, so we create
# a separate map from Enum values to URLs
from_vars_to_2013_url_map = {
    "ita-primarie": "https://www.bancaditalia.it/statistiche/tematiche/rapporti-estero/turismo-internazionale/distribuzione-microdati/file-dati/documenti/2013/csv/ita_2013_principali.zip",
    "ita-secondarie": "https://www.bancaditalia.it/statistiche/tematiche/rapporti-estero/turismo-internazionale/distribuzione-microdati/file-dati/documenti/2013/csv/Residenza-dei-viaggiatori-Italiani-Secondarie-CSV.zip",
    "ita-fp": "https://www.bancaditalia.it/statistiche/tematiche/rapporti-estero/turismo-internazionale/distribuzione-microdati/file-dati/documenti/2013/csv/Residenza-dei-viaggiatori-Italiani-Fattori-di-espansione-CSV.zip",
    "stra-principali": "https://www.bancaditalia.it/statistiche/tematiche/rapporti-estero/turismo-internazionale/distribuzione-microdati/file-dati/documenti/2013/csv/Residenza-dei-viaggiatori-Stranieri-Principali-CSV.zip",
    "stra-secondarie": "https://www.bancaditalia.it/statistiche/tematiche/rapporti-estero/turismo-internazionale/distribuzione-microdati/file-dati/documenti/2013/csv/Residenza-dei-viaggiatori-Stranieri-Secondarie-CSV.zip",
    "stra-fp": "https://www.bancaditalia.it/statistiche/tematiche/rapporti-estero/turismo-internazionale/distribuzione-microdati/file-dati/documenti/2013/csv/Residenza-dei-viaggiatori-stranieri-Fattori-di-espansione-CSV.zip",
}


class MicroDataset:
    def __init__(
        self,
        variable_subset: VariableSubset,
        tourist_origin: TouristOrigin,
        year: int,
        destination_folder: Path,
    ) -> None:
        self.variable_subset = variable_subset
        self.tourist_origin = tourist_origin
        self.year = year
        self._file_name = f"{self.year}_{self.tourist_origin.name.lower()}_{self.variable_subset.name.lower()}"

        destination_folder.mkdir(parents=True, exist_ok=True)
        self._file_path = destination_folder / f"{self._file_name}.csv"
        self._temp_path = Path(".")

    @property
    def url(self) -> str:
        if self.year == 2013:
            # For some reason the 2013 files are not in the standard format
            return from_vars_to_2013_url_map[
                f"{self.tourist_origin.value}-{self.variable_subset.value}"
            ]
        elif self.year >= 2016 or (
            self.year == 2012
            and self.variable_subset == VariableSubset.EXPANSION_FACTORS
            and self.tourist_origin == TouristOrigin.FOREIGNERS
        ):
            # The 2016-2021 file names (and another special case) have a
            # "_csv" suffix
            return (
                "https://www.bancaditalia.it/statistiche/tematiche/"
                "rapporti-estero/turismo-internazionale/distribuzione-microdati/"
                f"file-dati/documenti/{self.year}/csv/{self.tourist_origin.value}_"
                f"{self.year}_{self.variable_subset.value}_csv.zip"
            )
        else:
            return (
                "https://www.bancaditalia.it/statistiche/tematiche/"
                "rapporti-estero/turismo-internazionale/distribuzione-microdati/"
                f"file-dati/documenti/{self.year}/csv/{self.tourist_origin.value}_"
                f"{self.year}_{self.variable_subset.value}.zip"
            )

    @cached_property
    def df(self, top_n_rows: Optional[int] = None) -> pd.DataFrame:
        if not self._file_path.exists():
            self.download()
        df = pd.read_csv(self._file_path)
        if top_n_rows is None:
            return df
        else:
            return df.head(top_n_rows)

    @staticmethod
    def _unzip_file_from_stream(file_stream: BytesIO, output_folder: Path) -> None:
        output_folder.mkdir(exist_ok=True)
        with zipfile.ZipFile(file_stream) as zip_file:
            zip_file.extractall(output_folder)

    def _move_single_file(self, input_folder: Path, destination_file: Path) -> None:
        # Check if there is more than one file in the extracted folder
        extracted_files = os.listdir(input_folder)
        if len(extracted_files) != 1:
            raise ValueError(
                f"Found unexpected number of files in the extracted folder:"
                f" {extracted_files}"
            )
        else:
            # copy file to data folder
            os.rename(
                input_folder / extracted_files[0],
                destination_file,
            )
            print(f"File moved to {destination_file}")

    def download(self) -> None:
        # Download the file
        if self._file_path.exists():
            print(f"File {self._file_path} already exists. Skipping download")
            return
        else:
            print(f"Downloading from {self.url}")

            temp_folder = self._temp_path / self._file_name
            with requests.get(
                self.url, allow_redirects=True, stream=True
            ) as url_file_stream:
                self._unzip_file_from_stream(
                    BytesIO(url_file_stream.content),
                    output_folder=temp_folder,
                )
            self._move_single_file(
                input_folder=temp_folder,
                destination_file=self._file_path,
            )
            os.rmdir(temp_folder)

    def __str__(self) -> str:
        return str(self._file_path)
