# %%
from pathlib import Path

import bokeh.plotting as bk
from bokeh.layouts import row
import pandas as pd
import math
from bokeh.models import Legend, Label, Circle
from sklearn.linear_model import LinearRegression
from bokeh.palettes import Category20, Turbo256
from tqdm import tqdm

from src.trip_dataset import TouristOrigin, TripDataset, VariableSubset
from src.trip_operations import CodeToLocationMapperFromCSV

# %%
# Plot the same time series after grouping the travellers by their country of origin


residence_country_mapper = CodeToLocationMapperFromCSV(
    input_column="STATO_RESIDENZA",
    output_column="STATO_RESIDENZA_mapped",
    code_map_csv=Path(
        "/mnt/c/Users/loreg/Documents/dissertation_data/Metadati-CSV/TSTATI.csv"
    ),
    code_column_csv="ELEM_DOMINIO",
    location_name_column="D_EL_DOMINIO",
    separator=";",
    nan_codes=[0, 99999],
    force_repeat=False,
)
dataset = TripDataset(
    variable_subset=VariableSubset.PRIMARY,
    tourist_origin=TouristOrigin.FOREIGNERS,
    year=1997,
    raw_folder=Path("/mnt/c/Users/loreg/Documents/dissertation_data/raw"),
    processed_folder=Path("/mnt/c/Users/loreg/Documents/dissertation_data/processed"),
    column_to_dtype_map={"CHIAVE": str},
    force_raw=False,
)
european_countries = dataset.df["STATO_RESIDENZA_mapped"].unique()

# 2. Load the data by year and compute the aggregated values by year and
# by country of origin
total_dfs_by_year = []
for year in tqdm(range(1997, 2023, 1)):
    dataset = TripDataset(
        variable_subset=VariableSubset.PRIMARY,
        tourist_origin=TouristOrigin.FOREIGNERS,
        year=year,
        raw_folder=Path("/mnt/c/Users/loreg/Documents/dissertation_data/raw"),
        processed_folder=Path(
            "/mnt/c/Users/loreg/Documents/dissertation_data/processed"
        ),
        column_to_dtype_map={"CHIAVE": str},
        force_raw=False,
    )

    df_primary = dataset.apply(residence_country_mapper).df
    df_primary = df_primary[
        df_primary["STATO_RESIDENZA_mapped"].isin(european_countries)
    ]

    totals_single_area = df_primary.groupby("STATO_RESIDENZA_mapped")[
        ["FPD_NOTTI", "FPD_SPESA_FMI", "FPD_VIAG"]
    ].sum()
    totals_single_area.reset_index(drop=False, inplace=True)
    totals_single_area["year"] = year

    total_dfs_by_year.append(totals_single_area)

totals_by_year = pd.concat(total_dfs_by_year, axis=0)
totals_by_year.sort_values(by=["year", "STATO_RESIDENZA_mapped"], inplace=True)
totals_by_year.head()

totals_by_year.to_csv(
    "/mnt/c/Users/loreg/Documents/dissertation_data/processed/foreigners_by_origin_by_year.csv",
    index=False,
)

# %%
totals_by_year = pd.read_csv(
    "/mnt/c/Users/loreg/Documents/dissertation_data/processed/foreigners_by_origin_by_year.csv",
)

# Filter on the top 20 countries to improve readability
top_20_series = list(
    totals_by_year.groupby("STATO_RESIDENZA_mapped")["FPD_VIAG"]
    .sum()
    .sort_values(ascending=False)
    .head(19)
    .index
)

# Create category "other" for the remaining countries in order to fill up the chart
# to 100% each year
totals_by_year_top20_map = totals_by_year["STATO_RESIDENZA_mapped"].isin(top_20_series)
other_countries = totals_by_year[~totals_by_year_top20_map]

other_countries = (
    other_countries[["year","FPD_NOTTI", "FPD_SPESA_FMI", "FPD_VIAG"]]
    .groupby("year")
    .sum()
    .reset_index()
)
other_countries["STATO_RESIDENZA_mapped"] = "OTHER"
totals_by_year = pd.concat(
    [totals_by_year[totals_by_year_top20_map], other_countries], axis=0
)

totals_by_year["FPD_SPESA_FMI"] = totals_by_year["FPD_SPESA_FMI"] / 1e9
totals_by_year["FPD_VIAG"] = totals_by_year["FPD_VIAG"] / 1e6
totals_by_year["FPD_NOTTI"] = totals_by_year["FPD_NOTTI"] / 1e6

totals_by_year["expenses_per_night"] = (
    totals_by_year["FPD_SPESA_FMI"] / totals_by_year["FPD_NOTTI"] * 1e3
)

top_20_series.append("OTHER")
palette = Category20[20]

# %%
COMBINE_PLOTS = True
SAVE_PLOTS = True
VARIABLE_NAME = "FPD_VIAG"

# ========================= PLOT NUMBER OF TRIPS =========================
# Draw two bokeh plots for foreigners to show how their expenses
# and number of trips changed over the years divided by country of origin
fig1 = bk.figure(
    title="Foreigner tourists in Italy by country of origin",
    height=500,
    width=600,
    x_axis_label="Year",
    y_axis_label="Number of visitors (in millions)",
    x_range=(1997, 2023),
    # y_range=(0, max(totals_by_year[VARIABLE_NAME]) * 1.1),
    # y_axis_type="log",
    tooltips=[
        ("Year", "@year"),
        ("Tourists", f"@{VARIABLE_NAME}"),
        ("Country", "@STATO_RESIDENZA_mapped"),
    ],
    toolbar_location=None,
)
# Line plot for the total nights spent by foreigners in Italy colored by macroarea
legend_it = []
for i, country in enumerate(top_20_series):
    country_serie = fig1.line(
        x="year",
        y=VARIABLE_NAME,
        color=palette[i],
        source=totals_by_year[totals_by_year["STATO_RESIDENZA_mapped"] == country],
        line_width=2,
    )
    legend_it.append((country, [country_serie]))

fig1.line(
    x=[2019, 2019],
    y=[0, max(totals_by_year[VARIABLE_NAME]) * 1.05],
    line_width=3,
    line_dash="dashed",
    line_color="black",
)
# Rotate label by 90 degrees
label = Label(
    x=2019 - 0.5,
    y=max(totals_by_year[VARIABLE_NAME]) * 0.62,
    angle=math.pi / 2,
    text="COVID-19",
)
fig1.add_layout(label)

legend = Legend(items=legend_it, location=(10, -30))
legend.click_policy = "mute"
legend.label_text_font_size = "8pt"
fig1.add_layout(legend, "right")

if COMBINE_PLOTS:
    fig1.legend.visible = False
elif SAVE_PLOTS:
    bk.output_file(
        "/mnt/c/Users/loreg/OneDrive/UCL/Dissertation/ucl-dissertation-tex/Figures/foreigner_visits_timeseries_by_origin.html"
    )
    bk.save(fig1)
else:
    bk.output_notebook()
    bk.show(fig1)

# ==================Plot stacked bar chart ==============
# Create a stacked bar chart to show the share of expenses spent by foreigners by country of origin

# Compute the share of total expenses by year and country of origin
totals_by_year["share"] = (
    totals_by_year[VARIABLE_NAME]
    / totals_by_year.groupby("year")[VARIABLE_NAME].transform("sum")
    * 100  # to get the percentage
)
totals_by_year.head()
fig2 = bk.figure(
    title="Foreigner tourists in Italy by country of origin",
    height=500,
    width=750,
    x_axis_label="Year",
    y_axis_label="Share of visitors (%)",
    x_range=(1997, 2023),
    y_range=(0, 104),
    tooltips=[("Year", "@year")],
    toolbar_location=None,
)

pivot_df = pd.pivot_table(
    totals_by_year,
    index="year",
    columns="STATO_RESIDENZA_mapped",
    values="share",
    aggfunc="sum",
)
pivot_df.reset_index(drop=False, inplace=True)

legend = Legend(location=(10, -30))
fig2.add_layout(legend, "right")

country_serie = fig2.varea_stack(
    stackers=top_20_series,
    x="year",
    color=palette,
    source=pivot_df,
    legend_label=top_20_series,
)
# Line and label for Covid19 pandemic
fig2.line(
    x=[2019, 2019],
    y=[0, 100],
    line_width=3,
    line_dash="dashed",
    line_color="black",
)
label = Label(
    x=2019 - 0.5,
    y=100 * 0.62,
    angle=math.pi / 2,
    text="COVID-19",
)
fig2.add_layout(label)

fig2.legend.label_text_font_size = "8pt"
fig2.legend.click_policy = "mute"

# show the results
if COMBINE_PLOTS:
    if SAVE_PLOTS:
        bk.output_file(
            "/mnt/c/Users/loreg/OneDrive/UCL/Dissertation/ucl-dissertation-tex/Figures/foreigner_visits_timeseries_by_origin_multiplot.html"
        )
        bk.save(row(fig1, fig2))
    else:
        bk.output_notebook()
        bk.show(row(fig1, fig2))
else:
    if SAVE_PLOTS:
        bk.output_file(
            "/mnt/c/Users/loreg/OneDrive/UCL/Dissertation/ucl-dissertation-tex/Figures/foreigner_visits_percentage_stackedareas_by_origin.html"
        )
        bk.save(fig2)
    else:
        bk.output_notebook()
        bk.show(fig2)

# %%
# ====================== PLOT MONEY SPENT ======================

VARIABLE_NAME = "FPD_SPESA_FMI"
COMBINE_PLOTS = True
SAVE_PLOTS = True
# Draw two bokeh plots for foreigners to show how their expenses
# and number of trips changed over the years divided by country of origin
fig1 = bk.figure(
    title=f"Money spent by foreigners in Italy by country of origin",
    height=500,
    width=600,
    x_axis_label="Year",
    y_axis_label="Total Expenses (in billions of euros)",
    x_range=(1997, 2023),
    y_range=(0, max(totals_by_year[VARIABLE_NAME]) * 1.1),
    tooltips=[
        ("Year", "@year"),
        ("Expenses", f"@{VARIABLE_NAME}"),
        ("Country", "@STATO_RESIDENZA_mapped"),
    ],
    toolbar_location=None,
)

# Line plot for the total nights spent by foreigners in Italy colored by macroarea
legend_it = []
for i, country in enumerate(top_20_series):
    country_serie = fig1.line(
        x="year",
        y=VARIABLE_NAME,
        color=palette[i],
        source=totals_by_year[totals_by_year["STATO_RESIDENZA_mapped"] == country],
        line_width=2,
    )
    legend_it.append((country, [country_serie]))
fig1.line(
    x=[2019, 2019],
    y=[0, max(totals_by_year[VARIABLE_NAME]) * 1.05],
    line_width=3,
    line_dash="dashed",
    line_color="black",
    legend_label="COVID-19",
)
# Add Covid19 label
label = Label(
    x=2019 - 0.5,
    y=max(totals_by_year[VARIABLE_NAME]) * 0.62,
    angle=math.pi / 2,
    text="COVID-19",
)
fig1.add_layout(label)

legend = Legend(items=legend_it, location=(10, -30))
legend.click_policy = "mute"
legend.label_text_font_size = "8pt"
fig1.add_layout(legend, "right")

if COMBINE_PLOTS:
    fig1.legend.visible = False
elif SAVE_PLOTS:
    bk.output_file(
        "/mnt/c/Users/loreg/OneDrive/UCL/Dissertation/ucl-dissertation-tex/Figures/expenses_timeseries_by_origin.html"
    )
    bk.save(fig1)
else:
    bk.output_notebook()
    bk.show(fig1)

# Create a stacked bar chart to show the share of expenses spent by foreigners by country of origin

# Compute the share of total expenses by year and country of origin
totals_by_year["share"] = (
    totals_by_year[VARIABLE_NAME]
    / totals_by_year.groupby("year")[VARIABLE_NAME].transform("sum")
    * 100  # to get the percentage
)
totals_by_year.head()
fig2 = bk.figure(
    title="Money spent by foreigners in Italy by country of origin",
    height=500,
    width=750,
    x_axis_label="Year",
    y_axis_label="Percentage of revenues (%)",
    x_range=(1997, 2023),
    y_range=(0, 104),
    tooltips=[("Year", "@year")],
    toolbar_location=None,
)
# ==================== Plot stacked bar chart ===================

pivot_df = pd.pivot_table(
    totals_by_year,
    index="year",
    columns="STATO_RESIDENZA_mapped",
    values="share",
    aggfunc="sum",
)
pivot_df.reset_index(drop=False, inplace=True)

legend = Legend(location=(10, -30))
fig2.add_layout(legend, "right")

country_serie = fig2.varea_stack(
    stackers=top_20_series,
    x="year",
    color=palette,
    source=pivot_df,
    legend_label=top_20_series,
)
# Add Covid19 pandemic label
label = Label(
    x=2019 - 0.5,
    y=max(totals_by_year[VARIABLE_NAME]) * 0.62,
    angle=math.pi / 2,
    text="COVID-19",
)
fig1.add_layout(label)

fig2.legend.label_text_font_size = "8pt"
fig2.legend.click_policy = "mute"

# show the results
if COMBINE_PLOTS:
    if SAVE_PLOTS:
        bk.output_file(
            "/mnt/c/Users/loreg/OneDrive/UCL/Dissertation/ucl-dissertation-tex/Figures/expenses_timeseries_by_year_by_origin_multiplot.html"
        )
        bk.save(row(fig1, fig2))
    else:
        bk.output_notebook()
        bk.show(row(fig1, fig2))
else:
    if SAVE_PLOTS:
        bk.output_file(
            "/mnt/c/Users/loreg/OneDrive/UCL/Dissertation/ucl-dissertation-tex/Figures/expenses_timeseries_percentage_share_stackedareas_by_origin.html"
        )
        bk.save(fig2)
    else:
        bk.output_notebook()
        bk.show(fig2)

# %%

# ========================= PLOT Expenses per Night =========================
VARIABLE_NAME = "expenses_per_night"
SAVE_PLOTS = True

# Create a yearly average of the expenses per night weighted on the share of
# tourists from each country
totals_by_year["yearly_share_total_nights"] = totals_by_year[
    "FPD_NOTTI"
] / totals_by_year.groupby("year")["FPD_NOTTI"].transform("sum")
expenses_per_night_weighted = (
    (totals_by_year["expenses_per_night"] * totals_by_year["yearly_share_total_nights"])
    .groupby(totals_by_year["year"])
    .sum()
)

# Draw two bokeh plots for foreigners to show how their expenses
# and number of trips changed over the years divided by country of origin
fig1 = bk.figure(
    title="Foreign tourists' expenses per night in Italy by country of origin",
    height=500,
    width=800,
    x_axis_label="Year",
    y_axis_label="Expenses per night (euros)",
    x_range=(1997, 2023),
    # y_range=(0, max(totals_by_year_w_other[VARIABLE_NAME]) * 1.1),
    y_axis_type="log",
    tooltips=[
        ("Year", "@year"),
        ("Expenses", f"@{VARIABLE_NAME}"),
        ("Country", "@STATO_RESIDENZA_mapped"),
    ],
    toolbar_location=None,
)
# Line plot for the total nights spent by foreigners in Italy colored by macroarea
legend_it = []
for i, country in enumerate(top_20_series):
    country_serie = fig1.line(
        x="year",
        y=VARIABLE_NAME,
        color=palette[i],
        source=totals_by_year[totals_by_year["STATO_RESIDENZA_mapped"] == country],
        line_width=2,
    )
    legend_it.append((country, [country_serie]))

# Add Covid19 pandemic line and label
fig1.line(
    x=[2019, 2019],
    y=[0, math.log(max(totals_by_year[VARIABLE_NAME]))],
    line_width=3,
    line_dash="dashed",
    line_color="black",
)
# Rotate label by 90 degrees
label = Label(
    x=2019 - 0.5,
    y=max(totals_by_year[VARIABLE_NAME]) * 0.4,
    angle=math.pi / 2,
    text="COVID-19",
)
fig1.add_layout(label)

# Add weighted average line
circle = fig1.circle(
    x=expenses_per_night_weighted.index.tolist(),
    y=expenses_per_night_weighted.tolist(),
    size=5,
    color="black",
)
# Compute and plot the fitted line of expenses_per_night_weighted
# using a linear regression
X = expenses_per_night_weighted.index.values.reshape(-1, 1)
y = expenses_per_night_weighted.values.reshape(-1, 1)
reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)
fitted_line = fig1.line(
    x=expenses_per_night_weighted.index.tolist(),
    y=y_pred.tolist(),
    line_width=2,
    line_dash="dashed",
    line_color="black",
)
legend_weighted_circle = Legend(
    items=[
        ("Weighted Average", [circle]),
        (
            f"y = {round(reg.coef_[0][0], 2)} * x {round(reg.intercept_[0], 2)}",
            [fitted_line],
        ),
    ],
    location="bottom_right",
    border_line_color="black",
)
fig1.add_layout(legend_weighted_circle)  # "center"

legend = Legend(items=legend_it, location=(10, -30))
legend.click_policy = "mute"
legend.label_text_font_size = "8pt"
fig1.add_layout(legend, "right")

if SAVE_PLOTS:
    bk.output_file(
        "/mnt/c/Users/loreg/OneDrive/UCL/Dissertation/ucl-dissertation-tex/Figures/expenses_per_night_timeseries_by_origin.html"
    )
    bk.save(fig1)
else:
    bk.output_notebook()
    bk.show(fig1)

# %%
import statsmodels.api as sm


X_w_const = sm.add_constant(X.ravel())
results = sm.OLS(endog=y, exog=X_w_const).fit()
results.summary()
# Write the equation
print(f"y = {round(results.params[0], 4)} + {round(results.params[1], 4)} * x")
# %%
