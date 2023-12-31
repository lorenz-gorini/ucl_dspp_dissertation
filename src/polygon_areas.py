from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Union

import geog
import geopandas as gpd
import numpy as np
import rioxarray
import shapely


@dataclass
class SpotCoordinates:
    longitude: float
    latitude: float

    def circular_area(self, radius: float) -> shapely.geometry.polygon.Polygon:
        """
        Returns a circular area around the spot, with given radius (in km)

        Parameters
        ----------
        radius : float
            Radius of the circular area, in km

        Returns
        -------
        shapely.geometry.polygon.Polygon
            Polygon representing the circular area
        """
        d = radius * 1000  # meters
        angles = np.linspace(0, 360, n_points=40)
        # Move from point p, at given angle, for given distance
        # If we use multiple angles, we get multiple points (to draw a polygon around p)
        polygon = geog.propagate([(self.latitude, self.longitude)], angle=angles, d=d)
        return shapely.geometry.Polygon(polygon)


def from_point_to_area(
    spot: shapely.Point,
    radius: float,
) -> shapely.Polygon:
    """
    Returns the polygon representing a circle around the spot

    Parameters
    ----------
    spot : shapely.geometry.Point
        Point representing the spot coordinates
    radius : float
        Radius of the circular area around the spot, in km

    Returns
    -------
    shapely.geometry.polygon.Polygon
        Polygon representing a circle around the spot
    """
    n_points = 20
    d = radius * 1000  # convert `radius` to meters
    angles = np.linspace(0, 360, n_points).astype(float)
    # Move from point p, at given angle, for given distance
    # If we use multiple angles, we get multiple points (to draw a polygon around p)
    polygon = geog.propagate([(spot.x, spot.y)], angle=angles, d=d)
    return shapely.Polygon(polygon)


def filter_multi_polygons(
    multi_polygon: shapely.MultiPolygon,
    distance_ratio: float = 0.4,
    area_ratio: float = 0.001,
) -> Union[shapely.Polygon, shapely.MultiPolygon]:
    """
    Filter the polygons in the given multi_polygon to remove the small and far ones

    Compute distance of each polygon centroid to the multi_polygon centroid
    and select only the ones that are big and close enough.
    NOTE: From `shapely` library: For multipolygons the centroid is weighted by
    the area of each polygon
    """
    distances = [
        poly.centroid.distance(multi_polygon.centroid) for poly in multi_polygon.geoms
    ]
    areas = [poly.area for poly in multi_polygon.geoms]

    polygons = [
        poly
        for poly, dist, area in zip(multi_polygon.geoms, distances, areas)
        if (dist < distance_ratio * max(distances) and area > area_ratio * sum(areas))
    ]
    if len(polygons) == 1:
        return shapely.Polygon(polygons[0])
    else:
        return shapely.MultiPolygon(polygons)


class PolygonAreasFromFile:
    def __init__(
        self,
        column_name: str,
        crs: str = "EPSG:4326",
        shapefile_path: Optional[Path] = None,
        geo_df: Optional[gpd.GeoDataFrame] = None,
    ) -> None:
        self.column_name = column_name
        self.crs = crs
        if geo_df is None:
            if shapefile_path is None:
                raise ValueError("Either `shapefile_path` or `geo_df` must be not None")
            else:
                self.geo_df = self.read_geo_df(shapefile_path, crs=crs)
        else:
            self.geo_df = geo_df

    @staticmethod
    def read_geo_df(file_path: Path, crs: str) -> gpd.GeoDataFrame:
        """
        Loads the shapefile for the given country

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing the shapefile data
        """
        geo_df = gpd.read_file(file_path)
        geo_df = geo_df.to_crs(crs)
        return geo_df

    def _area_rows_from_df(
        self, geo_df: gpd.GeoDataFrame, area_name: str
    ) -> gpd.GeoDataFrame:
        """
        Return the rows corresponding to the given name

        Parameters
        ----------
        geo_df : gpd.GeoDataFrame
            GeoDataFrame containing the "geometry" (i.e. Polygon) of multiple areas and names/codes

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing only the rows corresponding to
            the given name
        """
        return geo_df[geo_df[self.column_name] == area_name]

    def get_polygon(
        self,
        area_name: str,
        use_polygon_filter: bool = False,
        distance_ratio: float = 0.4,
        area_ratio: float = 0.001,
    ) -> shapely.Polygon:
        self.country_rows = self._area_rows_from_df(
            geo_df=self.geo_df, area_name=area_name
        )
        if self.country_rows.shape[0] == 0:
            raise ValueError(f"No area found for {area_name}")
        elif self.country_rows.shape[0] > 1:
            raise ValueError(f"Multiple areas found for {area_name}")
        polygons = self.country_rows.iloc[0].geometry
        if use_polygon_filter and isinstance(polygons, shapely.MultiPolygon):
            return filter_multi_polygons(
                multi_polygon=polygons,
                distance_ratio=distance_ratio,
                area_ratio=area_ratio,
            )
        else:
            return polygons

    @cached_property
    def polygon_names(self) -> List[str]:
        """
        Returns a list of all the names of the areas in the shapefile

        Returns
        -------
        List[str]
            List of all the names of the areas in the shapefile
        """
        return self.geo_df[self.column_name].unique().tolist()

    def polygon_names_dict(
        self,
        use_polygon_filter: bool = False,
        distance_ratio: float = 0.4,
        area_ratio: float = 0.001,
    ) -> Dict[str, shapely.Polygon]:
        """
        Returns a dictionary mapping area names to the coordinates of their polygons

        Returns
        -------
        Dict[str, shapely.Polygon]
            Dictionary mapping area names to the coordinates of their polygons
        """
        return {
            area_name: self.get_polygon(
                area_name,
                use_polygon_filter=use_polygon_filter,
                distance_ratio=distance_ratio,
                area_ratio=area_ratio,
            )
            for area_name in self.polygon_names
        }
