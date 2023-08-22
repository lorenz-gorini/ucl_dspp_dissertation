"""
Script to align the province/country names of the weather timeserie dataframe (coming
from administrative boundaries in shapefile) to the ones used in tourism data (coming
from tourism file)
"""
from pathlib import Path

import pandas as pd

from src.trip_dataset import TripDataset, TouristOrigin, VariableSubset

# ================== Countries ==================
timeseries_per_country_df = pd.read_csv(
    "/mnt/c/Users/loreg/Documents/dissertation_data/timeseries_per_country.csv"
)
# read tourism data
country_2019_dataset = TripDataset(
    variable_subset=VariableSubset.PRIMARY,
    tourist_origin=TouristOrigin.ITALIANS,
    year=2019,
    raw_folder=Path("/mnt/c/Users/loreg/Documents/dissertation_data/raw"),
    processed_folder=Path("/mnt/c/Users/loreg/Documents/dissertation_data/processed"),
)
country_column_map = {
    "ANDORRA": None,
    "AZORES ISLANDS": None,
    "BELARUS": "BIELORUSSIA",
    "BELGIUM": "BELGIO",
    "BOSNIA & HERZEGOVINA": "BOSNIA ERZEGOVINA",
    "CROATIA": "CROAZIA",
    "CZECH REPUBLIC": "CECA, REPUBBLICA",
    "DENMARK": "DANIMARCA",
    "FAROE ISLANDS": None,
    "FINLAND": "FINLANDIA",
    "FRANCE": "FRANCIA",
    "GERMANY": "GERMANIA",
    "GIBRALTAR": None,
    "GREECE": "GRECIA",
    "GUERNSEY": None,
    "HOLY SEE": None,
    "HUNGARY": "UNGHERIA",
    "ICELAND": "ISLANDA",
    "IRELAND": "IRLANDA",
    "ISLE OF MAN": None,
    "ITALY": "ITALIA",
    "JERSEY": None,
    "LATVIA": "LETTONIA",
    "LITHUANIA": "LITUANIA",
    "LUXEMBOURG": "LUSSEMBURGO",
    "MADEIRA ISLANDS": None,
    "MOLDOVA, REPUBLIC OF": "MOLDAVIA",
    "MONACO": "PRINCIPATO DI MONACO",
    "NETHERLANDS": "PAESI BASSI",
    "NORWAY": "NORVEGIA",
    "POLAND": "POLONIA",
    "PORTUGAL": "PORTOGALLO",
    "RUSSIAN FEDERATION": "RUSSIA, FEDERAZIONE DI",
    "SLOVAKIA": "SLOVACCA, REPUBBLICA",
    "SLOVENIA": "SLOVENIA",
    "SAN MARINO": None,
    "SPAIN": "SPAGNA",
    "SVALBARD AND JAN MAYEN ISLANDS": None,
    "SWEDEN": "SVEZIA",
    "SWITZERLAND": "SVIZZERA",
    "THE FORMER YUGOSLAV REPUBLIC OF MACEDONIA": "NORD MACEDONIA",
    "UKRAINE": "UCRAINA",
    "U.K. OF GREAT BRITAIN AND NORTHERN IRELAND": "REGNO UNITO",
}
# 1. Turn country names to uppercase
timeseries_per_country_df_mapped = timeseries_per_country_df.rename(
    columns={col: col.upper() for col in timeseries_per_country_df.columns}
)
# 2. The small countries are not included as "STATO_VISITATO" option in the tourism
# file, so we can just drop the related columns from the timeseries dataframe
timeseries_per_country_df_mapped = timeseries_per_country_df_mapped.drop(
    columns=[k for k, v in country_column_map.items() if v is None]
)

# 3. Map country names that are still mismatched by translating them
# + special cases (e.g. "Holy See" -> "Italy")
timeseries_per_country_df_mapped = timeseries_per_country_df_mapped.rename(
    columns=country_column_map
)
# Check that all the European countries have been mapped
tour_countries = set(
    country_2019_dataset.df["STATO_VISITATO_mapped"].str.upper().unique()
)
ts_countries = set(s for s in timeseries_per_country_df_mapped.columns)
ts_countries - tour_countries

# Check how many countries are in the tourism dataset but not in
# the timeseries (European countries only)
country_2019_dataset.df[
    country_2019_dataset.df["STATO_VISITATO_mapped"].isin(ts_countries)
].shape
country_2019_dataset.df.shape

# Save the timeseries dataframe with the mapped columns
timeseries_per_country_df_mapped.to_csv(
    "/mnt/c/Users/loreg/Documents/dissertation_data/timeseries_per_country_mapped.csv",
    index=False,
)


# ================== Provinces ==================
timeseries_per_prov_df = pd.read_csv(
    "/mnt/c/Users/loreg/Documents/dissertation_data/timeseries_per_province.csv"
)

# read tourism data
province_2019_dataset = TripDataset(
    variable_subset=VariableSubset.PRIMARY,
    tourist_origin=TouristOrigin.FOREIGNERS,
    year=2019,
    raw_folder=Path("/mnt/c/Users/loreg/Documents/dissertation_data/raw"),
    processed_folder=Path("/mnt/c/Users/loreg/Documents/dissertation_data/processed"),
)

# 1. Turn province names to uppercase
timeseries_per_prov_df_mapped = timeseries_per_prov_df.rename(
    columns={col: col.upper() for col in timeseries_per_prov_df.columns}
)
# 3. Map province names that are still mismatched
prov_column_map = {
    "MONZA E DELLA BRIANZA": "MONZA-BRIANZA",
    "REGGIO DI CALABRIA": "REGGIO CALABRIA",
    "REGGIO NELL'EMILIA": "REGGIO EMILIA",
    "VERBANO-CUSIO-OSSOLA": "VERBANO CUSIO OSSOLA",
}
timeseries_per_prov_df_mapped = timeseries_per_prov_df_mapped.rename(
    columns=prov_column_map
)
# Check that all the provinces have been mapped
tour_provinces = set(
    province_2019_dataset.df["PROVINCIA_VISITATA_mapped"].str.upper().unique()
)
ts_provinces = set(s for s in timeseries_per_prov_df_mapped.columns)
ts_provinces - tour_provinces
tour_provinces - ts_provinces

timeseries_per_prov_df_mapped.to_csv(
    "/mnt/c/Users/loreg/Documents/dissertation_data/timeseries_per_province_mapped.csv",
    index=False,
)

# Check how many provinces are NaN in the tourism dataset
province_2019_dataset.df["PROVINCIA_VISITATA_mapped"].isna().sum()
