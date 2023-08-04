from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union, List

import pandas as pd

from .dataset import MicroDataset


class DatasetOperation(ABC):
    def __init__(
        self,
        input_columns: List[str],
        output_columns: List[str],
        force_repeat: bool = False,
    ) -> None:
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.force_repeat = force_repeat

    @abstractmethod
    def __call__(self, dataset: MicroDataset) -> MicroDataset:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class CodeToLocationMapper(DatasetOperation):
    def __init__(
        self,
        input_column: str,
        output_column: str,
        code_map: Dict[int, str],
        force_repeat: bool = False,
    ) -> None:
        super().__init__(
            input_columns=[input_column],
            output_columns=[output_column],
            force_repeat=force_repeat,
        )
        self.code_map = code_map

    def __call__(self, dataset: MicroDataset) -> pd.DataFrame:
        new_df = dataset.df.copy()
        new_df[self.output_columns[0]] = new_df[self.input_columns[0]].map(
            self.code_map
        )
        print(
            "The codes not associated to a name are: "
            + str(new_df[self.output_columns[0]].isna().sum())
        )
        return new_df

    def __str__(self) -> str:
        return (
            f"CodeToLocationMapper({self.input_columns[0]}, {self.output_columns[0]}, "
            f"{self.code_map})"
        )


class CodeToLocationMapperFromCSV(DatasetOperation):
    def __init__(
        self,
        input_column: str,
        output_column: str,
        code_map_csv: Union[str, Path],
        code_column: str,
        location_name_column: str,
        nan_codes: Optional[List[int]] = None,
        separator: Optional[str] = None,
        force_repeat: bool = False,
    ) -> None:
        """
        Create a mapper from location codes to location names from a CSV file

        Parameters
        ----------
        input_column : str
            The name of the column in the dataset containing the location codes
        output_column : str
            The name of the column in the dataset containing the location names
        code_map_csv : Union[str, Path]
            The path to the CSV file containing the location codes and names
        code_column : str
            The name of the column in the CSV file containing the location codes
        location_name_column : str
            The name of the column in the CSV file containing the location names
        nan_codes : Optional[List[int]], optional
            The list of codes that will be mapped to pd.NA. If None,
            the codes considered as NaN are None and pd.NA, otherwise these two values
            will be added to the list ``nan_codes`` provided. Default is None.
        separator : Optional[str], optional
            The separator used in the CSV file. If None, the
            separator is a comma ",". Default is None.
        """
        super().__init__(
            input_columns=[input_column],
            output_columns=[output_column],
            force_repeat=force_repeat,
        )
        self.code_map_csv = code_map_csv
        self.code_column = code_column
        self.location_name_column = location_name_column
        self.separator = "," if separator is None else separator
        
        self.nan_codes = set(None, pd.NA)
        if nan_codes is not None:
            self.nan_codes = self.nan_codes | set(nan_codes)
        # Load the CSV file with the location codes and names
        self._code_map_df = pd.read_csv(self.code_map_csv, sep=self.separator)

    def __call__(self, dataset: MicroDataset) -> pd.DataFrame:
        # Turn the df into a dictionary mapping codes to names
        code_to_name_dict = {}
        for _, row in self._code_map_df.iterrows():
            code_to_name_dict[row[self.code_column]] = (
                pd.NA
                if row[self.code_column] in self.nan_codes
                else row[self.location_name_column]
            )

        new_df = CodeToLocationMapper(
            input_column=self.input_columns[0],
            output_column=self.output_columns[0],
            code_map=code_to_name_dict,
        )(dataset)
        return new_df

    def __str__(self) -> str:
        return (
            f"CodeToLocationMapperFromCSV({self.input_columns[0]}, {self.output_columns[0]}, "
        )
