from __future__ import annotations

import enum
import os
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import requests
from sas7bdat import SAS7BDAT


class VariableSubset(enum.Enum):
    PRIMARY = "principali"
    SECONDARY = "secondarie"
    EXPANSION_FACTORS = "fp"


class TouristOrigin(enum.Enum):
    ITALIANS = "ita"
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


class GenericTripDataset:
    def __init__(
        self,
        raw_file_path: Path,
        processed_file_path: Path,
        column_to_dtype_map: Optional[Dict[str, Any]] = None,
        df: pd.DataFrame = None,
        force_raw: bool = False,
    ) -> None:
        """
        Dataset class

        This class handles the dataset defined by the argument used to initiate it
        in a completely opaque way to the user.
        Everytime a new operation is performed, the resulting dataset is saved
        in ``processed_folder``, because it is assumed that every operation is
        improving the dataset.
        Every operation name and parameter is saved in a new column called "operations"
        and the list can also be accessed with the ``operations`` class property.

        Parameters
        ----------
        raw_file_path : Path
            The path to the raw file.
        processed_file_path : Path
            The path to the processed file.
        column_to_dtype_map : Dict[str, Any], optional
            A dictionary that maps column names to their data type. This is useful
            because some columns are not correctly parsed by pandas, so we can force
            the correct data type. When set to None, it will
            be automatically set to ``{"CHIAVE": str}``, because in the trip datasets
            the ``CHIAVE`` column is a very long int, much longer than the maximum
            size of Int64. In case where
            (year==2018 and tourist_origin==TouristOrigin.FOREIGNERS)
            the "CHIAVE" key will be automatically replaced with "chiave" due to
            inconsistency of the raw data. Default is None.
        force_raw : bool
            If True, the dataset will be loaded in "raw" version from ``raw_folder``,
            even if it is present in the ``processed_folder``. The default is False.
        """
        # TODO: Make this class the base class for TripDataset
        self.raw_file_path = raw_file_path
        self.processed_file_path = processed_file_path
        self.force_raw = force_raw

        if column_to_dtype_map is None:
            column_to_dtype_map = {"CHIAVE": str}

        self.column_to_dtype_map = column_to_dtype_map
        self._df = df

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            if not self.force_raw and self.processed_file_path.exists():
                self._df = pd.read_csv(
                    self.processed_file_path, dtype=self.column_to_dtype_map
                )
            else:
                self._df = pd.read_csv(
                    self.raw_file_path, dtype=self.column_to_dtype_map
                )
        return self._df

    @property
    def operations(self) -> List[str]:
        return self.df[~self.df["operations"].isna()]["operations"].to_list()

    def _restore_operations(
        self, operations: List[Union[str, "TripDatasetOperation"]]
    ) -> None:
        for index, operation in enumerate(operations):
            self._df.loc[index, "operations"] = str(operation)

    def _add_operation(self, operation: "TripDatasetOperation") -> None:
        self._df.loc[len(self.operations), "operations"] = str(operation)

    def _remove_operation(self, operation: "TripDatasetOperation") -> None:
        self._df["operations"] = self._df["operations"].apply(
            lambda x: None if x == str(operation) else x
        )

    def save_df(self) -> None:
        """
        Set the dataset to a new dataframe and save it to disk

        WARNING: Be careful when using this method, because it will overwrite the
        previous file without keeping track of the operations performed on the dataset.
        It is highly recommended to use the ``apply`` method with a
        ``DatasetOperation`` instead.
        """
        self._df.to_csv(self.processed_file_path, index=False)

    def is_operation_applied(self, operation: "TripDatasetOperation") -> bool:
        is_applied = str(operation) in self.operations
        if is_applied:
            # Check that all the output columns are in the dataframe
            for col in operation.output_columns:
                if col not in self.df.columns:
                    raise ValueError(
                        f"Operation {str(operation)} was already performed on dataset,"
                        f" but the output column {col} is not in the dataframe."
                    )
        return is_applied

    def apply(self, operation: "TripDatasetOperation") -> "TripDataset":
        """
        Apply an operation to the dataset

        When the ``operation`` is applied, the dataset keeps track of it by saving its
        string representation in the ``operations`` column of the dataset (that can
        be accessed with ``operations`` property), so that it can also be skipped if
        it is already applied.

        NOTE
        ----
        After applying each operation, the dataset is saved to disk in the
        ``processed_folder``, so that the operations are not repeated if the script
        is interrupted and we load the same dataset again.

        Parameters
        ----------
        operation : TripDatasetOperation
            The operation to apply to the dataset.

        Returns
        -------
        TripDataset
            The dataset with the operation applied.
        """
        if operation.force_repeat is False and self.is_operation_applied(operation):
            print(f"Operation {str(operation)} already performed on dataset. Skipping")
        else:
            if operation.force_repeat is True:
                self._remove_operation(operation)
            # Need to cache the operations, because applying the ``operation``
            # may drop some rows containing previous operations already applied
            operation_cache = self.operations
            print(f"Applying operation {str(operation)} to dataset")
            self._df = operation(self)
            # Restore the operation list by writing them on the new dataset
            self._restore_operations(operation_cache)
            self._add_operation(operation)
            self.save_df()
        return self

    def __repr__(self) -> str:
        attributes = self.__dict__.copy()
        attributes.pop("_df")
        attr_str = ", ".join([f"{k}={v}" for k, v in attributes.items()])
        return f"TripDataset({attr_str})"


class TripDataset:
    def __init__(
        self,
        variable_subset: VariableSubset,
        tourist_origin: TouristOrigin,
        year: int,
        raw_folder: Path = Path("/mnt/c/Users/loreg/Documents/dissertation_data/raw"),
        processed_folder: Path = Path(
            "/mnt/c/Users/loreg/Documents/dissertation_data/processed"
        ),
        column_to_dtype_map: Optional[Dict[str, Any]] = None,
        force_raw: bool = False,
    ) -> None:
        """
        Dataset class that handles downloading the file and loading it from disk

        This class handles the dataset defined by the argument used to initiate it
        in a completely opaque way to the user.
        So if the dataset has not been downloaded, it will automatically download
        it when needed and save it in the ``raw_folder``. Similarly, everytime a new
        operation is performed, the resulting dataset is saved in ``processed_folder``,
        because it is assumed that every operation is improving the dataset.
        Every operation name and parameter is saved in a new column called "operations"
        and the list can also be accessed with the ``operations`` class property.

        Parameters
        ----------
        variable_subset : VariableSubset
            The subset of variables to use. Can be either ``VariableSubset.PRIMARY``,
            ``VariableSubset.SECONDARY`` or ``VariableSubset.EXPANSION_FACTORS``.
        tourist_origin : TouristOrigin
            The origin of the tourists. Can be either ``TouristOrigin.ITALIANS`` or
            ``TouristOrigin.FOREIGNERS``.
        year : int
            The year of the dataset. Can be any year between 1997 and 2023.
        raw_folder : Path
            The folder where the raw file will be saved. If it doesn't exist, it will
            be created.
        processed_folder : Path
            The folder where the processed file will be saved. If it doesn't exist, it
            will be created.
        column_to_dtype_map : Dict[str, Any], optional
            A dictionary that maps column names to their data type. This is useful
            because some columns are not correctly parsed by pandas, so we can force
            the correct data type. When set to None, it will
            be automatically set to ``{"CHIAVE": str}``, because in the trip datasets
            the ``CHIAVE`` column is a very long int, much longer than the maximum
            size of Int64. In case where
            (year==2018 and tourist_origin==TouristOrigin.FOREIGNERS)
            the "CHIAVE" key will be automatically replaced with "chiave" due to
            inconsistency of the raw data. Default is None.
        force_raw : bool
            If True, the dataset will be loaded in "raw" version from ``raw_folder``,
            even if it is present in the ``processed_folder``. The default is False.
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
        if (
            self.year == 2013
            and self.tourist_origin == TouristOrigin.FOREIGNERS
            and self.variable_subset == VariableSubset.SECONDARY
        ):
            self.raw_file_path = self.raw_folder / f"{self.file_name}.sas7bdat"
        else:
            self.raw_file_path = self.raw_folder / f"{self.file_name}.csv"
        self.processed_file_path = self.processed_folder / f"{self.file_name}.csv"
        self.force_raw = force_raw

        if column_to_dtype_map is None:
            column_to_dtype_map = {"CHIAVE": str}
        # Replace the "CHIAVE" key with "chiave" if year==2018
        if self.year == 2018 and tourist_origin == TouristOrigin.FOREIGNERS:
            column_to_dtype_map["chiave"] = column_to_dtype_map.pop("CHIAVE")
        self.column_to_dtype_map = column_to_dtype_map

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
            if not self.force_raw and self.processed_file_path.exists():
                self._df = pd.read_csv(
                    self.processed_file_path, dtype=self.column_to_dtype_map
                )
            else:
                if not self.raw_file_path.exists():
                    self.download()
                self._df = pd.read_csv(
                    self.raw_file_path, dtype=self.column_to_dtype_map
                )
                if "operations" in self.df.columns:
                    raise ValueError(
                        "The raw file should not have an 'operations' column"
                    )
                else:
                    self._df["operations"] = None
            if "chiave" in self._df.columns:
                self._df.rename(columns={"chiave": "CHIAVE"}, inplace=True)
        return self._df

    @property
    def operations(self) -> List[str]:
        return self.df[~self.df["operations"].isna()]["operations"].to_list()

    @staticmethod
    def _unzip_file_from_stream(file_stream: BytesIO, output_folder: Path) -> None:
        output_folder.mkdir(exist_ok=True)
        with zipfile.ZipFile(file_stream) as zip_file:
            zip_file.extractall(output_folder)

    def _add_operation(self, operation: "TripDatasetOperation") -> None:
        self._df.loc[len(self.operations), "operations"] = str(operation)

    def _remove_operation(self, operation: "TripDatasetOperation") -> None:
        self._df["operations"] = self._df["operations"].apply(
            lambda x: None if x == str(operation) else x
        )

    def _restore_operations(
        self, operations: List[Union[str, "TripDatasetOperation"]]
    ) -> None:
        for index, operation in enumerate(operations):
            self._df.loc[index, "operations"] = str(operation)

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

    def save_df(self) -> None:
        """
        Set the dataset to a new dataframe and save it to disk

        WARNING: Be careful when using this method, because it will overwrite the
        previous file without keeping track of the operations performed on the dataset.
        It is highly recommended to use the ``apply`` method with a
        ``DatasetOperation`` instead.
        """
        self._df.to_csv(self.processed_file_path, index=False)

    def download(self, force: bool = False) -> None:
        # Download the file
        if not force and self.raw_file_path.exists():
            print(f"File {self.raw_file_path} already exists. Skipping download")
            return
        else:
            print(f"Downloading from {self.url}")

            temp_extract_folder = self._temp_path / self.file_name
            with requests.get(
                self.url, allow_redirects=True, stream=True
            ) as url_file_stream:
                self._unzip_file_from_stream(
                    BytesIO(url_file_stream.content),
                    output_folder=temp_extract_folder,
                )
            if (
                self.year == 2013
                and self.tourist_origin == TouristOrigin.FOREIGNERS
                and self.variable_subset == VariableSubset.SECONDARY
            ):
                # Special case: In 2013 at the CSV URL, a SAS file List file in the extracted folder
                extracted_files = os.listdir(temp_extract_folder)
                if len(extracted_files) != 1:
                    raise ValueError(
                        f"Found unexpected number of files in the extracted folder:"
                        f" {extracted_files}"
                    )
                with SAS7BDAT(path=str(temp_extract_folder / extracted_files[0])) as f:
                    df = f.to_data_frame()

                df.to_csv(self.raw_file_path, index=False)
                # Remove file inside folder, or ``os.rmdir`` will not work
                os.remove(temp_extract_folder / extracted_files[0])
            else:
                self._move_single_file(
                    input_folder=temp_extract_folder,
                    destination_file=self.raw_file_path,
                )
            os.rmdir(temp_extract_folder)

    def is_operation_applied(self, operation: "TripDatasetOperation") -> bool:
        is_applied = str(operation) in self.operations
        if is_applied:
            # Check that all the output columns are in the dataframe
            for col in operation.output_columns:
                if col not in self.df.columns:
                    raise ValueError(
                        f"Operation {str(operation)} was already performed on dataset,"
                        f" but the output column {col} is not in the dataframe."
                    )
        return is_applied

    def apply(
        self, operation: "TripDatasetOperation", save: bool = False
    ) -> "TripDataset":
        if operation.force_repeat is False and self.is_operation_applied(operation):
            print(f"Operation {str(operation)} already performed on dataset. Skipping")
        else:
            if operation.force_repeat is True:
                self._remove_operation(operation)

            # Need to cache the operations, because applying the ``operation``
            # may drop some rows containing previous operations already applied
            operation_cache = self.operations
            print(f"Applying operation {str(operation)} to dataset")
            self._df = operation(self)
            # Restore the operation list by writing them on the new dataset
            self._restore_operations(operation_cache)
            self._add_operation(operation)
            if save:
                self.save_df()
        return self

    def __repr__(self) -> str:
        attributes = self.__dict__.copy()
        attributes.pop("_df")
        attr_str = ", ".join([f"{k}={v}" for k, v in attributes.items()])
        return f"TripDataset({attr_str})"
