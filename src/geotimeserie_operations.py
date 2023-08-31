from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from typing import List, Literal

import numpy as np
import rioxarray
import shapely
import xarray as xr


class GeoTimeSerieOperation(ABC):
    @abstractmethod
    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        pass


class SetCRSOperation(GeoTimeSerieOperation):
    def __init__(self, crs: str) -> None:
        """
        Operation that sets the Coordinate Reference System of the timeserie data

        Parameters
        ----------
        crs : str
            Coordinate Reference System (CRS) to set the timeserie data to
        """
        super().__init__()
        self.crs = crs

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Set the CRS of the timeserie data

        Parameters
        ----------
        crs : str
            CRS to set the timeserie data to

        Returns
        -------
        xr.Dataset
            Dataset containing the timeserie data with the new CRS
        """
        dataset = dataset.rio.write_crs(self.crs)
        return dataset


class TimeResampleOperation(GeoTimeSerieOperation):
    def __init__(self, freq: str) -> None:
        super().__init__()
        self.freq = freq
        raise NotImplementedError

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Resample the timeserie data to the given frequency

        Parameters
        ----------
        freq : str
            Frequency to resample the timeserie data to

        Returns
        -------
        xr.Dataset
            Dataset containing the resampled timeserie data
        """
        # TODO: check how and if this works
        # return dataset.resample(time=self.freq).mean()
        raise NotImplementedError


class InterpolateOperation(GeoTimeSerieOperation):
    def __init__(self, dataset_variable: str, target_resolution: float = 0.01) -> None:
        super().__init__()
        self.target_resol = target_resolution
        self.dataset_variable = dataset_variable

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Interpolate the timeserie data to the given area

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing the entire timeserie data

        Returns
        -------
        xr.Dataset
            Dataset containing the interpolated timeserie data
        """
        min_lon, min_lat, max_lon, max_lat = dataset.rio.bounds()

        # create dummy dataset with the required resolution and rectangular bounds
        # containing area
        lats = np.arange(min_lat, max_lat, self.target_resol).astype("float32")
        lons = np.arange(min_lon, max_lon, self.target_resol).astype("float32")

        ds_interp_empty = xr.Dataset(
            data_vars={
                self.dataset_variable: (
                    ["time", "latitude", "longitude"],
                    # Create array with nan
                    np.full(
                        (len(dataset.time.data), len(lats), len(lons)),
                        fill_value=np.nan,
                    ),
                )
            },
            coords={
                "time": (["time"], dataset.time.data),
                "latitude": (["latitude"], lats),
                "longitude": (["longitude"], lons),
            },
        )

        # NOTE: The following command does not work when the dataset contains nan values
        # ds_interp = dataset.interp_like(other=ds_interp_empty)

        # Interpolate the data by merging it with the new grid
        ds_interp_merge = xr.merge([dataset, ds_interp_empty])

        ds_interp_merge[self.dataset_variable] = ds_interp_merge[
            self.dataset_variable
        ].rio.write_nodata(np.nan)
        ds_interp = ds_interp_merge.rio.interpolate_na(
            method="linear",  # kwargs={"fill_value": "extrapolate"},
        )

        ds_interp_nonan = ds_interp.dropna(dim="latitude", how="all").dropna(
            dim="longitude", how="all"
        )
        return ds_interp_nonan


class CastToTypeOperation(GeoTimeSerieOperation):
    def __init__(self, dtype: str, variable_name: str) -> None:
        """
        Operation that cast the timeserie dataset to a new type

        Parameters
        ----------
        dtype : str
            New type to cast the timeserie data to
        variable_name : str
            Name of the dataset variable to convert to the new type
        """
        super().__init__()
        self.dtype = dtype
        self.variable_name = variable_name

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Casts the timeserie data to the given type

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing the variable to cast

        Returns
        -------
        xr.Dataset
            Dataset containing the variable converted to the given type
        """
        dataset[self.variable_name] = dataset[self.variable_name].astype(self.dtype)
        return dataset


class AreaClipOperation(GeoTimeSerieOperation):
    def __init__(self, area: shapely.Polygon) -> None:
        super().__init__()
        self.area = area

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Clips the timeserie data to the given area

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing the entire timeserie data

        Returns
        -------
        xr.Dataset
            Dataset containing the clipped timeserie data

        Raises
        ------
        rioxarray.exceptions.NoDataInBounds
            If the timeserie data does not contain any data in the given area
        """
        return dataset.rio.clip([self.area])


class TimeRangeClipOperation(GeoTimeSerieOperation):
    def __init__(
        self, start_time: datetime.datetime, end_time: datetime.datetime
    ) -> None:
        super().__init__()
        self.start_time = start_time
        self.end_time = end_time

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Clips the timeserie data to the given time range

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing the entire timeserie data

        Returns
        -------
        xr.Dataset
            Dataset containing the clipped timeserie data
        """

        # Get the timeserie data for the given time range
        return dataset.sel(time=slice(self.start_time, self.end_time))


class DropNAOperation(GeoTimeSerieOperation):
    def __init__(self, dim: str, how: Literal["any", "all"] = "any") -> None:
        super().__init__()
        self.dim = dim
        self.how = how

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Drops the NA values from the timeserie data

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing the entire timeserie data

        Returns
        -------
        xr.Dataset
            Dataset containing the timeserie data without NA values
        """
        return dataset.dropna(dim=self.dim, how=self.how)


class ValueAggregator(GeoTimeSerieOperation):
    def __init__(self, columns: List[str]) -> None:
        """
        Operation that aggregates the timeserie data by the given columns

        Parameters
        ----------
        columns : List[str]
            List of columns to aggregate the timeserie data by
        """
        super().__init__()
        self.aggregate_cols = columns

    @abstractmethod
    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        pass


class MeanAggregator(ValueAggregator):
    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        return dataset.mean(dim=self.aggregate_cols)


class SumAggregator(ValueAggregator):
    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        return dataset.sum(dim=self.aggregate_cols)


class MaxAggregator(ValueAggregator):
    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        return dataset.max(dim=self.aggregate_cols)


class MedianAggregator(ValueAggregator):
    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        return dataset.median(dim=self.aggregate_cols)


class StdAggregator(ValueAggregator):
    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        return dataset.std(dim=self.aggregate_cols)
