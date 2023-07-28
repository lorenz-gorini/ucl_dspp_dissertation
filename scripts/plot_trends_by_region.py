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

# Draw two bokeh plots for italian and foreigners to show how their expenses
# and number of trips changed over the years divided by country or by italian region
our_total_df = []

nights_by_area_year = []
for year in tqdm(range(1997, 2023, 1)):
    df_primary = MicroDataset(
        variable_subset=VariableSubset.PRIMARY,
        tourist_origin=TouristOrigin.FOREIGNERS,
        year=year,
        destination_folder=Path("/mnt/c/Users/loreg/Documents/dissertation_data/raw"),
    ).df
    for macro_area, region_codes_single_area in [
        ("North", north_region_codes),
        ("Center", center_region_codes),
        ("South", south_region_codes),
    ]:
        nights_single_area = 0
        expenses_single_area = 0
        for region_code in region_codes_single_area:
            df_single_region = df_primary[
                df_primary["REGIONE_VISITATA"] == region_code
            ]
            nights_single_area += df_single_region["FPD_NOTTI"].sum()
            expenses_single_area += df_single_region["FPD_SPESA_FMI"].sum()
        nights_by_area_year.append(
            {"year": year, "macro_area": macro_area, "nights": nights_single_area, "expenses": expenses_single_area}
        )
    # "FPD_SPESA_FMI": df_primary["FPD_SPESA_FMI"].sum(),

nights_by_area_year = pd.DataFrame(nights_by_area_year)
nights_by_area_year.head()
# %%
f = bk.figure(
    title=f"Total nights spent by foreigners in Italy by macroarea",
    x_axis_label="Year",
    y_axis_label="Total nights spent",
    x_range=(1997, 2023),
    y_range=(0, 2.1e8),
    tooltips=[("Year", "@year"), ("Nights", "@nights")],
)
# Line plot for the total nights spent by foreigners in Italy colored by macroarea
for i, macro_area in enumerate(["North", "Center", "South"]):
    f.line(
        x="year",
        y="nights",
        color=Category10[3][i],
        source=nights_by_area_year[nights_by_area_year["macro_area"] == macro_area],
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
