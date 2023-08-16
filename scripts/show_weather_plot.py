import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import shapely.geometry
import xarray as xr
from typing import Union
from rioxarray.exceptions import NoDataInBounds
import pyproj


def compute_na_share(ds: Union[xr.DataArray, xr.Dataset]):
    """
    Compute the share of missing values in the first variable of the dataset
    """
    coords_prod = np.prod(
        [
            ds.coords[coord].shape[0]
            for coord in ds.coords
            if len(ds.coords[coord].shape) == 1
        ]
    )
    if isinstance(ds, xr.Dataset):
        ds = ds[list(ds.data_vars)[0]]
    return float((ds.isnull().sum() / coords_prod).values)


weather_ds = xr.open_dataset(
    "/mnt/c/Users/loreg/Documents/dissertation_data/"
    "dataset-insitu-gridded-observations-europe/tg_ens_mean_0.1deg_reg_v27.0e.nc"
)

# =================== select the time period =======================
start_time = datetime.datetime(2019, 1, 1)
end_time = datetime.datetime(2019, 12, 31)
# Select the time period

# weather_ds_2019 = weather_ds.sel(time=slice(start_time, end_time))

# weather_ds_2019.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
# # NOTE: This is ONLY if data contains only one year
# weather_ds_2019  = weather_ds_2019.groupby(weather_ds_2019.time.dt.month).mean()


weather_ds_year = weather_ds.groupby(weather_ds.time.dt.year).mean()
weather_ds_year.to_netcdf(
    "/mnt/c/Users/loreg/Documents/dissertation_data/"
    "dataset-insitu-gridded-observations-europe/weather_ds_year_mean.nc"
)


# ================== Interpolate to a regular grid and select the area =================
# Data from "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/world-administrative-boundaries/exports/shp?lang=en&timezone=Europe%2FBerlin"

# read shapefile
shapefile = gpd.read_file(
    "/mnt/c/Users/loreg/Documents/dissertation_data/world-administrative-boundaries"
    "/world-administrative-boundaries.shp"
)
shapefile = shapefile.to_crs("EPSG:4326")
# This is the file with the municipalities (comuni) of Italy
# (from "https://www.istat.it/it/archivio/222527")
# TODO: Link this with the original dataset by:
# 1. Replace code with "COMUNE" name (from "TCOMUNI.csv" file)
# 2. Combine that name with "DEN_UTS" name (from "Com01012023_g_WGS84.shp" file)
munic_shp = gpd.read_file(
    "/mnt/c/Users/loreg/Documents/dissertation_data/Municipal_Boundaries_of_Italy_2019/"
    "Com01012023_g/Com01012023_g_WGS84.shp"
)  # "COMUNE", "geometry"
munic_shp = munic_shp.to_crs("EPSG:4326")

# File with the provinces of Italy
# TODO: Link this with the original dataset so that we use those polygons to
# get the average temperature of each province and plot it
prov_shp = gpd.read_file(
    "/mnt/c/Users/loreg/Documents/dissertation_data/Municipal_Boundaries_of_Italy_2019/"
    "ProvCM01012023_g/ProvCM01012023_g_WGS84.shp"
)  # "DEN_UTS", "geometry"
prov_shp = prov_shp.to_crs("EPSG:4326")

italy_area = shapefile[shapefile["name"] == "Italy"].iloc[0].geometry
torino_area = prov_shp[prov_shp["DEN_UTS"] == "Rimini"].iloc[0].geometry

# get the bounds of Italy
min_lon, min_lat, max_lon, max_lat = italy_area.bounds
target_resol = 0.05  # resolution

# create dummy dataset with the required resolution and bounds
lats = np.flip(np.arange(min_lat, max_lat, target_resol)).astype("float32")
lons = np.arange(min_lon, max_lon, target_resol).astype("float32")

ds_out = xr.Dataset(
    {
        "latitude": (["latitude"], lats),
        "longitude": (["longitude"], lons),
    }
)

# interpolate and assign
ds_interp = weather_ds_year.interp_like(other=ds_out)
ds_interp = ds_interp.assign_coords(
    {
        "latitude": (["latitude"], lats),
        "longitude": (["longitude"], lons),
    }
)

# ======== Plot NA share per year with bokeh ===============

# compute the share of missing values
na_share = []
year_range = range(1980, 2023).start
for year in year_range:
    print(year)
    start_time = datetime.datetime(year, 1, 1)
    end_time = datetime.datetime(year, 12, 31)
    weather_ds_year = weather_ds.sel(time=slice(start_time, end_time))
    na_share.append(compute_na_share(weather_ds_year.tg))

# create the plot
from bokeh.plotting import figure, show, output_notebook, save, output_file
from bokeh.models import NumeralTickFormatter, FixedTicker

p = figure(
    title="Share of missing values per year",
    x_axis_label="Year",
    y_axis_label="Share of missing values",
    x_range=(1980, 2023),
    y_range=(0, 1),
    width=500,
    height=400,
)
p.line(x=list(year_range), y=na_share, line_width=2)
p.circle(x=list(year_range), y=na_share, fill_color="white", size=8)

output_file("na_share.html")
save(p)

# =======================================================

# clip the data to Italy
ds_interp.rio.write_crs("epsg:4326", inplace=True)
ds_interp = ds_interp.rio.clip([shapely.geometry.mapping(italy_area)])

# Set CRS to row
ds_tg = ds_interp.tg

weather_prov = {}
nodata_provinces = []
for _, row in prov_shp.iterrows():
    name = row["DEN_UTS"]
    prov_geom = row["geometry"]

    try:
        weather_single_prov = (
            ds_tg.rio.clip([prov_geom]).mean(dim=["longitude", "latitude"]).values
        )
        weather_prov[name] = weather_single_prov
    except NoDataInBounds:
        nodata_provinces.append(name)

# date as index
start_time = datetime.datetime(int(weather_ds.year.min().values), 1, 1)
end_time = datetime.datetime(int(weather_ds.year.max().values), 12, 31)
year_ids = pd.date_range(start_time, end_time, freq="1Y")
year_ids = [i.strftime("%Y") for i in year_ids]

# df of temperatures for each country
weather_prov = pd.DataFrame(weather_prov)
weather_prov.index = year_ids

# ==============  Plot the temperature of each province ================

import plotly.express as px
import imageio

# removing small countries and NA values
geo = prov_shp.dropna()
geo = geo.reset_index(drop=True)
geo = geo[["DEN_UTS", "SIGLA", "geometry"]]

# Transpose weather_prov dataset
weather_prov_temp = weather_prov.T
weather_prov_temp = weather_prov_temp.reset_index(names=["Province"])
geo = pd.merge(weather_prov_temp, geo, left_on="Province", right_on="DEN_UTS")
geo = geo.drop(columns=["Province"])


# converting the temperature columns into rows
geo = geo.melt(
    id_vars=["SIGLA", "DEN_UTS", "geometry"],
    var_name="Date",
    value_name="Temperature (deg C)",
)

geo = gpd.GeoDataFrame(geo)
geo = geo.to_crs(pyproj.CRS.from_epsg(4326))
geo.head()

for year in year_ids:
    # Plot the temperature of each province
    geo_single_year = geo[geo["Date"] == year]
    geo_single_year = geo_single_year.set_index("DEN_UTS")
    fig = px.choropleth(
        geo_single_year,
        geojson=geo_single_year.geometry,
        locations=geo_single_year.index,
        color="Temperature (deg C)",
        color_continuous_scale="RdYlBu_r",
        range_color=(-10, 30),
        width=600,
        height=500,
        scope="europe",
        fitbounds="geojson",
    )

    fig.update_layout(
        title=f"Average Province Temperatures ({year})",
        title_x=0.5,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    # Removes the grey background map
    fig.update_geos(fitbounds="locations", visible=False)
    fig.show()
    fig.write_image(f"italy-{year}.png", scale=2)

# creating a GIF
images = []
for filename in [f"italy-{year}.png" for year in year_ids]:
    images.append(imageio.imread(filename))
imageio.mimsave("mygif.gif", images, duration=1)
