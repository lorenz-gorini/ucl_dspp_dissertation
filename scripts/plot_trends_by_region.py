# %%
from src.dataset import MicroDataset, TouristOrigin, VariableSubset
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from bokeh.palettes import Category10
import bokeh.plotting as bk

# %%
region_code_map = pd.read_csv(
    "/mnt/c/Users/loreg/Documents/dissertation_data/Metadati-CSV/TRegioni.csv", sep=";"
)

# ["Piemonte","Valle d' Aosta","Liguria","Lombardia","Trentino Alto Adige","Veneto","Friuli Venezia Giulia","Emilia Romagna"]
north_region_codes = [10, 12, 14, 16, 18, 20, 22, 24]
# ["Toscana","Umbria","Marche","Lazio"]
center_region_codes = [26, 28, 30, 32]
# ["Abruzzi","Molise","Campania","Puglia","Basilicata","Calabria","Sicilia","Sardegna"]
south_region_codes = [34, 36, 38, 40, 42, 44, 46, 48]
macro_area_to_region_codes = {
    "North": north_region_codes,
    "Center": center_region_codes,
    "South": south_region_codes,
}
region_code_to_macro_area_map = {}
for macro_area, region_codes_single_area in macro_area_to_region_codes.items():
    for region_code in region_codes_single_area:
        region_code_to_macro_area_map[region_code] = macro_area

# Draw two bokeh plots for italian and foreigners to show how their expenses
# and number of trips changed over the years divided by country or by italian region
our_total_df = []

total_dfs_by_year = []
for year in tqdm(range(1997, 2023, 1)):
    df_primary = MicroDataset(
        variable_subset=VariableSubset.PRIMARY,
        tourist_origin=TouristOrigin.FOREIGNERS,
        year=year,
        destination_folder=Path("/mnt/c/Users/loreg/Documents/dissertation_data/raw"),
    ).df

    df_primary["macroarea_visited"] = df_primary["REGIONE_VISITATA"].map(
        region_code_to_macro_area_map
    )
    print(
        f"The region codes not associated to a macroarea are: {df_primary['macroarea_visited'].isna().sum()}"
    )
    totals_single_area = df_primary.groupby("macroarea_visited")[
        ["FPD_NOTTI", "FPD_SPESA_FMI"]
    ].sum()
    totals_single_area.reset_index(drop=False, inplace=True)
    totals_single_area["year"] = year

    total_dfs_by_year.append(totals_single_area)

totals_by_year = pd.concat(total_dfs_by_year, axis=0)
totals_by_year.sort_values(by=["year", "macroarea_visited"], inplace=True)
totals_by_year.head()
# %%
f = bk.figure(
    title=f"Total nights spent by foreigners in Italy by macroarea",
    x_axis_label="Year",
    y_axis_label="Total nights spent",
    x_range=(1997, 2023),
    y_range=(0, 2.1e8),
    tooltips=[("Year", "@year"), ("Nights", "@FPD_NOTTI")],
)
# Line plot for the total nights spent by foreigners in Italy colored by macroarea
for i, macro_area in enumerate(["North", "Center", "South"]):
    f.line(
        x="year",
        y="FPD_NOTTI",
        color=Category10[3][i],
        source=totals_by_year[totals_by_year["macroarea_visited"] == macro_area],
        line_width=2,
        legend_label=macro_area,
    )
f.legend.location = "bottom_left"
f.legend.click_policy = "hide"
bk.output_file(
    "/mnt/c/Users/loreg/Documents/dissertation_data/figures/nights_by_area.html"
)
bk.show(f)

# %%
f = bk.figure(
    title=f"Total amount spent by foreigners in Italy by macroarea",
    x_axis_label="Year",
    y_axis_label="Total Expenses",
    x_range=(1997, 2023),
    y_range=(0, max(totals_by_year["FPD_SPESA_FMI"]) * 1.1),
    tooltips=[("Year", "@year"), ("Expenses", "@FPD_SPESA_FMI")],
)
# Line plot for the total nights spent by foreigners in Italy colored by macroarea
for i, macro_area in enumerate(["North", "Center", "South"]):
    f.line(
        x="year",
        y="FPD_SPESA_FMI",
        color=Category10[3][i],
        source=totals_by_year[totals_by_year["macroarea_visited"] == macro_area],
        line_width=2,
        legend_label=macro_area,
    )
f.legend.location = "top_left"
f.legend.click_policy = "hide"
bk.output_file(
    "/mnt/c/Users/loreg/Documents/dissertation_data/figures/expenses_by_area.html"
)
bk.show(f)

# %%
