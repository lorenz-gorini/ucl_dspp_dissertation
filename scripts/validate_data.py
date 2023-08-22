from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.trip_dataset import TripDataset, TouristOrigin, VariableSubset

# %%
our_total_df = []
for origin in TouristOrigin:
    for year in tqdm(range(1997, 2020, 1)):
        df_primary = TripDataset(
            variable_subset=VariableSubset.PRIMARY,
            tourist_origin=origin,
            year=year,
            raw_folder=Path("/mnt/c/Users/loreg/Documents/dissertation_data/raw"),
        ).df

        df_exp_factor = TripDataset(
            variable_subset=VariableSubset.EXPANSION_FACTORS,
            tourist_origin=origin,
            year=year,
            raw_folder=Path("/mnt/c/Users/loreg/Documents/dissertation_data/raw"),
        ).df

        if "CHIAVE" in df_primary.columns:
            df_merge = pd.merge(df_primary, df_exp_factor, on="CHIAVE", how="inner")
        elif "chiave" in df_primary.columns:
            df_merge = pd.merge(df_primary, df_exp_factor, on="chiave", how="inner")
        else:
            raise KeyError("No key 'CHIAVE/chiave' found")

        missing_keys = df_primary.shape[0] - df_merge.shape[0]
        print(f"({year}, {origin}): During the merge we lost {missing_keys} rows")

        try:
            FPD_SPESA_FMI_computed = (
                df_merge["SPESA_FMI"]
                * df_merge["FATTORE_PONDER_DES"]
                * df_merge["COEFF_CODE"]
            ).sum()
        except KeyError:
            FPD_SPESA_FMI_computed = None

        # TODO: Fix computation of FPD_ variables
        our_total_df.append(
            {
                "tourist_origin": origin,
                "version": 1 if origin == TouristOrigin.ITALIANS else 2,
                "year": year,
                "primary_samples": df_primary.shape[0],
                "exp_factor_samples": df_exp_factor.shape[0],
                "missing_keys": missing_keys,
                "FPD_NOTTI": df_primary["FPD_NOTTI"].sum(),
                "FPD_NOTTI_computed": (
                    df_merge["NR_NOTTI"]
                    * df_merge["NR_TOT_VIAGG_1"]
                    * df_merge["COEFF_CODE"]
                    * df_merge["FATTORE_PONDER_DES"]
                ).sum(),
                "FPD_SPESA_FMI": df_primary["FPD_SPESA_FMI"].sum(),
                "FPD_SPESA_FMI_computed": FPD_SPESA_FMI_computed,
                "FP_VIAG": df_primary["FP_VIAG"].sum(),
                "FP_VIAG_computed": (
                    df_merge["NR_TOT_VIAGG_1"]
                    * df_merge["FATTORE_PONDER"]
                    * df_merge["COEFF_CODE"]
                ).sum(),
                "FPD_VIAG": df_primary["FPD_VIAG"].sum(),
                "FPD_VIAG_computed": (
                    df_merge["NR_TOT_VIAGG_1"]
                    * df_merge["FATTORE_PONDER_DES"]
                    * df_merge["COEFF_CODE"]
                ).sum(),
            }
        )

our_total_df = pd.DataFrame(our_total_df)
our_total_df.head()

our_total_df.to_csv(
    "/mnt/c/Users/loreg/Documents/dissertation_data/total_computed_for_data_validation.csv",
    index=False,
)
control_total_df = pd.read_excel(
    "/mnt/c/Users/loreg/Documents/dissertation_data/TOTALI_PER_CONTROLLO.xls"
)

# Merge the two dataframes
merge_df = pd.merge(
    control_total_df,
    our_total_df,
    left_on=["edizione", "Anno"],
    right_on=["version", "year"],
    how="inner",
)
df_control_to_df_total_column_tuples = [
    ("NUMERO RECORD PRINCIPALI", "primary_samples"),
    ("NUMERO RECORD FP", "exp_factor_samples"),
    ("FPD NOTTI Sum", "FPD_NOTTI"),
    ("FPD SPESA FMI Sum", "FPD_SPESA_FMI"),
    ("FP VIAG Sum", "FP_VIAG"),
    ("FPD VIAG Sum", "FPD_VIAG"),
]
discrepancies_df = []
for control_column, our_column in df_control_to_df_total_column_tuples:
    discrepancies_df.append(
        {
            "control_column": control_column,
            "total_column": our_column,
            "percentage_difference": np.sum(
                np.abs(merge_df[control_column] - merge_df[our_column])
            )
            / merge_df[control_column].sum(),
            "variance": (merge_df[control_column] - merge_df[our_column]).std(),
            "min": merge_df[control_column].min(),
            "max": merge_df[control_column].max(),
        }
    )

discrepancies_df = pd.DataFrame(discrepancies_df)
discrepancies_df.to_csv(
    "/mnt/c/Users/loreg/Documents/dissertation_data/discrepancies.csv", index=False
)
