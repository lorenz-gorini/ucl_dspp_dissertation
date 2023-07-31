from __future__ import annotations

import enum
import os
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List

import pandas as pd
import requests


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
        raw_folder: Path,
        processed_folder: Path,
    ) -> None:
        """
        Dataset class that handles downloading the file and loading it from disk

        This class handles the dataset defined by the argument used to initiate it
        in a completely opaque way to the user.
        So if the dataset has not been downloaded, it will automatically download
        it when needed and save it in the ``raw_folder``. Similarly, everytime a new
        operation is performed, the resulting dataset is saved in ``processed_folder``,
        because it is assumed that every operation is improving the dataset.
        Every operation name and parameter is saved in a new column called "operation"
        and the list can also be accessed with the ``operations`` class property.

        Parameters
        ----------
        variable_subset : VariableSubset
            The subset of variables to use. Can be either ``VariableSubset.PRIMARY``,
            ``VariableSubset.SECONDARY`` or ``VariableSubset.EXPANSION_FACTORS``.
        tourist_origin : TouristOrigin
            The origin of the tourists. Can be either ``TouristOrigin.ITALIAN`` or
            ``TouristOrigin.FOREIGNERS``.
        year : int
            The year of the dataset. Can be any year between 1997 and 2023.
        raw_folder : Path
            The folder where the raw file will be saved. If it doesn't exist, it will
            be created.
        processed_folder : Path
            The folder where the processed file will be saved. If it doesn't exist, it
            will be created.
        """
        self.variable_subset = variable_subset
        self.tourist_origin = tourist_origin
        self.year = year
        self.raw_folder = raw_folder
        self.processed_folder = processed_folder

        self.file_name = (
            f"{self.year}_{self.tourist_origin.name.lower()}_"
            f"{self.variable_subset.name.lower()}"
        )

        self.raw_folder.mkdir(parents=True, exist_ok=True)
        self.processed_folder.mkdir(parents=True, exist_ok=True)
        self.raw_file_path = self.raw_folder / f"{self.file_name}.csv"
        self.processed_file_path = self.raw_folder / f"{self.file_name}.csv"

        self._temp_path = Path(".")
        self._df = None

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

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            if self.processed_file_path.exists():
                self._df = pd.read_csv(self.processed_file_path)
            else:
                if not self.raw_file_path.exists():
                    self.download()
                self._df = pd.read_csv(self.raw_file_path)
        return self._df

    @property
    def operations(self) -> List[str]:
        return self.df[~self.df["operations"].isna()]["operations"].to_list()

    @staticmethod
    def _unzip_file_from_stream(file_stream: BytesIO, output_folder: Path) -> None:
        output_folder.mkdir(exist_ok=True)
        with zipfile.ZipFile(file_stream) as zip_file:
            zip_file.extractall(output_folder)

    def _add_operation(self, operation: "DatasetOperation") -> None:
        if "operations" not in self._df.columns:
            self._df["operations"] = None
        self._df.loc[len(self.operations), "operations"] = str(operation)

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

    def set_df(self, new_df: pd.DataFrame) -> None:
        self._df = new_df
        new_df.to_csv(self.processed_file_path, index=False)

    def download(self) -> None:
        # Download the file
        if self.raw_file_path.exists():
            print(f"File {self.raw_file_path} already exists. Skipping download")
            return
        else:
            print(f"Downloading from {self.url}")

            temp_folder = self._temp_path / self.file_name
            with requests.get(
                self.url, allow_redirects=True, stream=True
            ) as url_file_stream:
                self._unzip_file_from_stream(
                    BytesIO(url_file_stream.content),
                    output_folder=temp_folder,
                )
            self._move_single_file(
                input_folder=temp_folder,
                destination_file=self.raw_file_path,
            )
            os.rmdir(temp_folder)

    def apply(self, operation: "DatasetOperation") -> "MicroDataset":
        if operation.destination_column in self.df.columns:
            print(
                f"Column {operation.destination_column} already exists in dataset,"
                f" skipping operation {str(operation)}"
            )
        else:
            new_df = operation(self)
            self.set_df(new_df)
            self._add_operation(operation)
        return self

    def __str__(self) -> str:
        return str(self.raw_file_path)
