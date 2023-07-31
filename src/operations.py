from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from .dataset import MicroDataset


class DatasetOperation(ABC):
    def __init__(self, dataset_column: str, destination_column: str) -> None:
        self.dataset_column = dataset_column
        self.destination_column = destination_column

    @abstractmethod
    def __call__(self, dataset: MicroDataset) -> MicroDataset:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class CodeToLocationMapper(DatasetOperation):
    def __init__(
        self,
        dataset_column: str,
        destination_column: str,
        code_map: Dict[int, str],
    ) -> None:
        super().__init__(
            dataset_column=dataset_column, destination_column=destination_column
        )
        # Load the CSV file with the location codes and names
        self.code_map = code_map

    def __call__(self, dataset: MicroDataset) -> pd.DataFrame:
        new_df = dataset.df.copy()
        new_df[self.destination_column] = new_df[self.dataset_column].map(self.code_map)
        print(
            "The codes not associated to a name are: "
            + str(new_df[self.destination_column].isna().sum())
        )
        return new_df

    def __str__(self) -> str:
        return (
            f"CodeToLocationMapper({self.dataset_column}, {self.destination_column}, "
            f"{self.code_map})"
        )


class CodeToLocationMapperFromCSV(DatasetOperation):
    def __init__(
        self,
        dataset_column: str,
        destination_column: str,
        code_map_csv: Union[str, Path],
        code_column: str,
        location_name_column: str,
        separator: Optional[str] = None,
    ) -> None:
        super().__init__(
            dataset_column=dataset_column, destination_column=destination_column
        )
        self.code_map_csv = code_map_csv
        self.code_column = code_column
        self.location_name_column = location_name_column
        self.separator = "," if separator is None else separator
        # Load the CSV file with the location codes and names
        self._code_map_df = pd.read_csv(self.code_map_csv, sep=self.separator)

    def __call__(self, dataset: MicroDataset) -> pd.DataFrame:
        # Turn the df into a dictionary mapping codes to names
        code_to_name_dict = {}
        for _, row in self._code_map_df.iterrows():
            code_to_name_dict[row[self.code_column]] = row[self.location_name_column]

        new_df = CodeToLocationMapper(
            dataset_column=self.dataset_column,
            destination_column=self.destination_column,
            code_map=code_to_name_dict,
        )(dataset)
        return new_df

    def __str__(self) -> str:
        return (
            f"CodeToLocationMapperFromCSV({self.dataset_column}, {self.destination_column}, "
            f"{self.code_map_csv}, {self.code_column}, {self.location_name_column})"
        )
