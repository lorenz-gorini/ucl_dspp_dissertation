from __future__ import annotations

import enum
from abc import ABC
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import rioxarray
import xarray as xr

from .timeserie_operations import TimeSerieOperation


class WeatherTimeSeriesEnum(enum.Enum):
    MEAN_TEMPERATURE = "tg"
    RAINFALL = "rr"
    HUMIDITY = "hu"
    WIND_SPEED = "fg"
    MIN_TEMPERATURE = "tn"
    MAX_TEMPERATURE = "tx"


class TimeSerieDataset(ABC):
    def __init__(
        self,
        timeserie_name: WeatherTimeSeriesEnum,
        file_crs: str = "EPSG:4326",
        nc_file_folder: Optional[Path] = Path(
            "/mnt/c/Users/loreg/Documents/dissertation_data/"
            "dataset-insitu-gridded-observations-europe/"
        ),
        ts_data: Optional[xr.Dataset] = None,
    ) -> None:
        """
        Class representing a timeserie dataset

        Parameters
        ----------
        timeserie_name : WeatherTimeSeriesEnum
            Name of the timeserie to load
        file_crs : str, optional
            When the dataset ``ts_data`` is set to None, the dataset is loaded from
            ``nc_file_folder`` and this is used as the Coordinate Reference System (CRS)
            of the timeserie data. By default "EPSG:4326"
        nc_file_folder : Optional[Path], optional
            Folder where the netCDF files are stored, by default
            ```Path(
                "/mnt/c/Users/loreg/Documents/dissertation_data/"
                "dataset-insitu-gridded-observations-europe/"
            )```
        ts_data : Optional[xr.Dataset], optional
            Dataset containing the timeserie data, by default None
        """
        self.ts_name = timeserie_name
        self.nc_file_folder = nc_file_folder
        self.crs = file_crs

        if ts_data is None:
            if nc_file_folder is None:
                raise ValueError(
                    "Either `nc_file_folder` or `ts_data` must be not None"
                )
            else:
                nc_file = (
                    nc_file_folder
                    / f"{timeserie_name.value}_ens_mean_0.1deg_reg_v27.0e.nc"
                )
                self.ts_data = self.read_ts_data(nc_file, crs=file_crs)
        else:
            self.ts_data = ts_data

    @staticmethod
    def read_ts_data(nc_file_path: Path, crs: str) -> xr.Dataset:
        """
        Loads the timeserie data from the netCDF file

        Parameters
        ----------
        timeserie_name : WeatherTimeSeriesEnum
            Name of the timeserie to load

        Returns
        -------
        xr.Dataset
            Dataset containing the timeserie data
        """
        ts_data = xr.open_dataset(nc_file_path)
        # We set the spatial dimensions and the CRS here to be sure they are set right
        ts_data = ts_data.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
        ts_data = ts_data.rio.write_crs(crs)
        return ts_data

    def to_file(self, file_path: Path) -> None:
        self.ts_data.to_netcdf(file_path)

    def bounds(self) -> Tuple[float, float, float, float]:
        """
        Returns the bounds of the timeserie data

        Returns
        -------
        Tuple[float, float, float, float]
            Bounds of the timeserie data
        """
        return self.ts_data.rio.bounds()

    def apply(
        self,
        operations: List[TimeSerieOperation],
    ) -> "TimeSerieDataset":
        """
        Apply the given operations to the timeserie data

        Parameters
        ----------
        operations : List[TimeSerieOperation]
            List of operations to apply to the timeserie data

        Returns
        -------
        TimeSerieDataset
            Timeserie data after applying the given operations
        """
        ts_data = self.ts_data
        for operation in operations:
            ts_data = operation(ts_data)

        return TimeSerieDataset(
            timeserie_name=self.ts_name,
            file_crs=self.crs,
            nc_file_folder=self.nc_file_folder,
            ts_data=ts_data,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the timeserie data to a DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame containing the timeserie data
        """
        return self.ts_data.to_dataframe()
