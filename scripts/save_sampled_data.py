"""
Script to geocode the locations in the dataset and add the coordinates to the dataset.

First I map "PROVINCIA_VISITATA" column to "PROVINCIA_VISITATA_mapped" column
and then I geocode it to get the coordinates.

I will geolocate the province for the center+south of Italy, but I will geolocate the
city for the north of Italy ("PROVINCIA_VISITATA" column), because there is more
variance in snow coverage and altitude in the north of Italy, and there are only few
extended provinces.
"""
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.trip_dataset import TouristOrigin, TripDataset, VariableSubset


for tourist_origin in [
    TouristOrigin.ITALIANS,
    TouristOrigin.FOREIGNERS,
]:
    sampled_dfs = []
    print("tourist_origin ", tourist_origin)
    for year in tqdm(range(1997, 2020, 1)):
        dataset = TripDataset(
            variable_subset=VariableSubset.PRIMARY,
            tourist_origin=tourist_origin,
            year=year,
            raw_folder=Path("/mnt/c/Users/loreg/Documents/dissertation_data/raw"),
            processed_folder=Path(
                "/mnt/c/Users/loreg/Documents/dissertation_data/processed"
            ),
            force_raw=False,
            column_to_dtype_map={"CHIAVE": str},
        )
        print("year ", year)

        sampled_dfs.append(dataset.df.sample(n=5000, random_state=42))
        print("OK")

    sampled_df = pd.concat(sampled_dfs)
    # HACK: Copy the operations column from the last of the sampled datasets, then
    # restore them
    operations = dataset.operations

    sampled_df.reset_index(inplace=True, drop=True)
    sampled_df["operations"] = None
    sampled_df.loc[0 : len(operations) - 1, "operations"] = operations

    sampled_df.to_csv(
        f"/mnt/c/Users/loreg/Documents/dissertation_data/raw/sample_5K_{tourist_origin.value}.csv",
        index=False,
    )
