# %%
from pathlib import Path

import bokeh.plotting as bk
import pandas as pd
from bokeh.models import Legend
from bokeh.palettes import Category20, Turbo256
from tqdm import tqdm

from src.dataset import MicroDataset, TouristOrigin, VariableSubset
from src.operations import CodeToLocationMapperFromCSV

# %%
# Plot the same time series after grouping the travellers by their country of origin


code_mapper = CodeToLocationMapperFromCSV(
    dataset_column="STATO_RESIDENZA",
    destination_column="origin_country",
    code_map_csv="/mnt/c/Users/loreg/Documents/dissertation_data/Metadati-CSV/TSTATI.csv",
    code_column="ELEM_DOMINIO",
    location_name_column="D_EL_DOMINIO",
    separator=";",
)

# 2. Load the data by year and compute the aggregated values by year and
# by country of origin
total_dfs_by_year = []
for year in tqdm(range(1997, 2023, 1)):
    dataset = MicroDataset(
        variable_subset=VariableSubset.PRIMARY,
        tourist_origin=TouristOrigin.FOREIGNERS,
        year=year,
        raw_folder=Path("/mnt/c/Users/loreg/Documents/dissertation_data/raw"),
        processed_folder=Path(
            "/mnt/c/Users/loreg/Documents/dissertation_data/processed"
        ),
    )

    df_primary = dataset.apply(code_mapper).df
    
    totals_single_area = df_primary.groupby("origin_country")[
        ["FPD_NOTTI", "FPD_SPESA_FMI", "FPD_VIAG"]
    ].sum()
    totals_single_area.reset_index(drop=False, inplace=True)
    totals_single_area["year"] = year

    total_dfs_by_year.append(totals_single_area)

totals_by_year = pd.concat(total_dfs_by_year, axis=0)
totals_by_year.sort_values(by=["year", "origin_country"], inplace=True)
totals_by_year.head()

# %%
# Draw two bokeh plots for foreigners to show how their expenses
# and number of trips changed over the years divided by country of origin
f = bk.figure(
    title=f"Total amount spent by foreigners in Italy by country of origin",
    height=500,
    width=800,
    x_axis_label="Year",
    y_axis_label="Total Expenses",
    x_range=(1997, 2023),
    y_range=(0, max(totals_by_year["FPD_SPESA_FMI"]) * 1.1),
    tooltips=[
        ("Year", "@year"),
        ("Expenses", "@FPD_SPESA_FMI"),
        ("Country", "@origin_country"),
    ],
)
# Filter on the top 20 countries to improve readability
top_20_series = (
    totals_by_year.groupby("origin_country")["FPD_SPESA_FMI"]
    .sum()
    .sort_values(ascending=False)
    .head(20)
    .index
)

palette = Category20[len(top_20_series)] if len(top_20_series) <= 20 else Turbo256
# Line plot for the total nights spent by foreigners in Italy colored by macroarea
legend_it = []
for i, country in enumerate(top_20_series):
    country_serie = f.line(
        x="year",
        y="FPD_SPESA_FMI",
        color=palette[i],
        source=totals_by_year[totals_by_year["origin_country"] == country],
        line_width=2,
    )
    legend_it.append((country, [country_serie]))
legend = Legend(items=legend_it)
legend.click_policy = "mute"
legend.label_text_font_size = "8pt"
f.add_layout(legend, "right")

bk.output_file(
    "/mnt/c/Users/loreg/Documents/dissertation_data/figures/expenses_timeseries_by_origin.html"
)
# bk.output_notebook()
bk.show(f)

# %%
# Create a stacked bar chart to show the share of expenses spent by foreigners by country of origin

# Compute the share of total expenses by year and country of origin
totals_by_year["share"] = (
    totals_by_year["FPD_SPESA_FMI"]
    / totals_by_year.groupby("year")["FPD_SPESA_FMI"].transform("sum")
    * 100  # to get the percentage
)
totals_by_year.head()
f = bk.figure(
    title="Percentage share of the amount spent by foreigners in Italy by country"
    " of origin",
    height=500,
    width=800,
    x_axis_label="Year",
    y_axis_label="Expenses Share (%)",
    x_range=(1997, 2023),
    y_range=(0, 104),
    tooltips=[("Year", "@year")],
)
# Plot stacked bar chart
# Filter on the top 19 countries to improve readability (then add the "OTHER" category)
top_20_series = list(
    totals_by_year.groupby("origin_country")["FPD_SPESA_FMI"]
    .sum()
    .sort_values(ascending=False)
    .head(19)
    .index
)
# Create category "other" for the remaining countries in order to fill up the chart
# to 100% each year
other_countries = totals_by_year[~totals_by_year["origin_country"].isin(top_20_series)]
other_countries = other_countries.groupby("year")["share"].sum().reset_index()
other_countries["origin_country"] = "OTHER"
top_20_series.append("OTHER")
totals_by_year_w_other = pd.concat([totals_by_year, other_countries], axis=0)

palette = Category20[len(top_20_series)] if len(top_20_series) <= 20 else Turbo256
pivot_df = pd.pivot_table(
    totals_by_year_w_other,
    index="year",
    columns="origin_country",
    values="share",
    aggfunc="sum",
)
pivot_df.reset_index(drop=False, inplace=True)
f.add_layout(Legend(), "right")
country_serie = f.varea_stack(
    stackers=top_20_series,
    x="year",
    color=palette,
    source=pivot_df,
    legend_label=top_20_series,
)
f.legend.label_text_font_size = "8pt"
f.legend.click_policy = "mute"
# bk.output_notebook()
bk.output_file(
    "/mnt/c/Users/loreg/Documents/dissertation_data/figures/expenses_timeseries_percentage_share_stackedareas_by_origin.html"
)
bk.show(f)
