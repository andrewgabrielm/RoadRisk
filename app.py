"""RoadRisk — retrospective collision hotspot prioritization dashboard."""

from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="RoadRisk", page_icon="🛣️", layout="wide")

ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "data" / "processed" / "test_predictions.csv"
GRID = ROOT / "data" / "processed" / "manhattan_grid.gpkg"

MODELS = {
    "Naive baseline": "baseline_pred",
    "Historical LightGBM": "model_b_pred",
    "Graph-enhanced LightGBM": "model_c_pred",
    "GCN": "model_d_pred",
}


@st.cache_data

def load_data():
    predictions = pd.read_csv(PREDICTIONS)
    grid = gpd.read_file(GRID)[["cell_id", "geometry"]].to_crs("EPSG:4326")
    return predictions, grid


def evaluate(quarter_data, prediction_column, k):
    """Compare the predicted top K with the observed top K for one quarter."""
    predicted = quarter_data.sort_values(
        [prediction_column, "cell_id"], ascending=[False, True]
    ).head(k).copy()
    actual = quarter_data.sort_values(
        ["collisions", "cell_id"], ascending=[False, True]
    ).head(k)

    precision = len(set(predicted["cell_id"]) & set(actual["cell_id"])) / k
    total = quarter_data["collisions"].sum()
    capture = predicted["collisions"].sum() / total if total else np.nan
    oracle = actual["collisions"].sum() / total if total else np.nan
    return predicted, precision, capture, oracle


st.title("RoadRisk")
st.caption("Predicting collision hotspots · Prioritizing safer streets")
st.write(
    "Explore how four forecasting strategies prioritize road-safety inspections "
    "across Manhattan's 500 m grid cells. Results are retrospective, not live predictions."
)

if not PREDICTIONS.exists() or not GRID.exists():
    st.error(
        "Required files are missing. Run this app from the RoadRisk project folder and "
        "check data/processed/test_predictions.csv and data/processed/manhattan_grid.gpkg."
    )
    st.stop()

predictions, grid = load_data()
required = {"cell_id", "quarter", "collisions"}
missing = required - set(predictions.columns)
if missing:
    st.error(f"Missing prediction-data columns: {', '.join(sorted(missing))}")
    st.stop()

available_models = {name: col for name, col in MODELS.items() if col in predictions.columns}
if not available_models:
    st.error("No model prediction columns were found. Rerun Notebook 05 to save predictions.")
    st.stop()

if "model_d_pred" not in predictions.columns:
    st.info("GCN predictions are not in this CSV yet. Rerun Notebook 05 to include Model D.")

if predictions.duplicated(["quarter", "cell_id"]).any() or grid["cell_id"].duplicated().any():
    st.error("Duplicate grid-cell IDs found. Please check the saved data.")
    st.stop()

quarter_options = sorted(predictions["quarter"].astype(str).unique())
model_names = list(available_models)
model_default = model_names.index("Graph-enhanced LightGBM") if "Graph-enhanced LightGBM" in model_names else 0

col1, col2, col3 = st.columns([1.5, 1, 1])
with col1:
    model_name = st.selectbox("Forecasting model", model_names, index=model_default)
with col2:
    quarter = st.selectbox("Forecast quarter", quarter_options, index=len(quarter_options) - 1)

quarter_data = predictions.loc[predictions["quarter"].astype(str) == quarter].copy()
quarter_data = quarter_data.sort_values("cell_id")

if quarter_data["cell_id"].nunique() != len(quarter_data):
    st.error("Duplicate cells in the selected quarter.")
    st.stop()
if set(quarter_data["cell_id"]) != set(grid["cell_id"]):
    st.error("Predictions and map grid do not have matching cell IDs.")
    st.stop()
if quarter_data[["collisions", *available_models.values()]].isna().any().any():
    st.error("Some observed or predicted collision counts are missing.")
    st.stop()

with col3:
    k = st.slider("Inspection capacity (cells)", min_value=5,
                  max_value=min(50, len(quarter_data)), value=min(20, len(quarter_data)))

prediction_col = available_models[model_name]
selected, precision, capture, oracle = evaluate(quarter_data, prediction_col, k)
_, _, baseline_capture, _ = evaluate(quarter_data, "baseline_pred", k) if "baseline_pred" in available_models.values() else (None, None, np.nan, None)
lift = capture / baseline_capture if pd.notna(baseline_capture) and baseline_capture > 0 else np.nan

m1, m2, m3 = st.columns(3)
m1.metric(f"Precision@{k}", f"{precision:.1%}")
m2.metric(f"Capture@{k}", f"{capture:.1%}" if pd.notna(capture) else "N/A")
m3.metric("Lift over naive", f"{lift:.2f}×" if pd.notna(lift) else "N/A")
st.caption(
    f"Selecting {k} of {len(quarter_data)} grid cells · "
    f"Perfect retrospective selection could capture {oracle:.1%} of collisions."
    if pd.notna(oracle) else f"Selecting {k} of {len(quarter_data)} grid cells."
)

# Map: all cells remain visible, selected cells are highlighted.
map_data = grid.merge(
    quarter_data[["cell_id", "collisions", prediction_col]],
    on="cell_id", how="inner", validate="one_to_one"
)
map_data["Priority"] = np.where(
    map_data["cell_id"].isin(selected["cell_id"]), "Selected for inspection", "Other cell"
)
map_data = map_data.rename(columns={
    prediction_col: "Predicted collisions", "collisions": "Observed collisions"
})

geojson = json.loads(map_data[["cell_id", "geometry"]].to_json())
fig = px.choropleth_map(
    map_data.drop(columns="geometry"),
    geojson=geojson,
    locations="cell_id",
    featureidkey="properties.cell_id",
    color="Priority",
    color_discrete_map={"Selected for inspection": "#E76F51", "Other cell": "#B8C8D7"},
    category_orders={"Priority": ["Selected for inspection", "Other cell"]},
    hover_name="cell_id",
    hover_data={"Predicted collisions": ":.1f", "Observed collisions": True,
                "Priority": False},
    map_style="carto-positron",
    center={"lat": 40.783, "lon": -73.966},
    zoom=10.5,
    opacity=0.7,
)
fig.update_traces(marker_line_width=0.6, marker_line_color="#FFFFFF")
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=530,
                  legend=dict(orientation="h", y=1.02, x=0))

st.subheader("Priority inspection locations")
st.plotly_chart(fig, use_container_width=True)

# Selected locations, in predicted-risk order.
priority = selected[["cell_id", prediction_col, "collisions"]].copy()
priority.insert(0, "Rank", range(1, len(priority) + 1))
priority.columns = ["Rank", "Cell ID", "Predicted collisions", "Observed collisions"]
st.dataframe(priority, hide_index=True, use_container_width=True, height=250)
st.download_button(
    "Download priority list (CSV)",
    data=priority.to_csv(index=False),
    file_name=f"roadrisk_{quarter}_{model_name.lower().replace(' ', '_')}_top{k}.csv",
    mime="text/csv",
)

st.subheader(f"Strategy comparison · {quarter}")
comparison = []
for name, col in available_models.items():
    _, p, c, _ = evaluate(quarter_data, col, k)
    comparison.append({"Model": name, "Capture@K (%)": 100 * c, "Precision@K (%)": 100 * p})
comparison_df = pd.DataFrame(comparison)
chart = px.bar(
    comparison_df, x="Model", y="Capture@K (%)", text_auto=".1f",
    color="Model", color_discrete_sequence=px.colors.qualitative.Safe,
)
chart.update_layout(showlegend=False, margin=dict(l=0, r=0, t=15, b=0), height=325)
st.plotly_chart(chart, use_container_width=True)

with st.expander("How to interpret this dashboard"):
    st.write(
        f"Each model ranks cells using its **forecast collision count**. We select the top {k} "
        "cells without consulting the future outcomes. **Precision@K** measures overlap "
        "with the actual top-K hotspots; **Capture@K** measures the share of observed "
        "collisions in selected cells. **Lift** compares capture with the previous-quarter "
        "naive strategy. The **oracle** uses observed collisions to show the theoretical "
        "maximum capture, so it is not a deployable forecast."
    )
    st.write(
        "Forecasts are one quarter ahead with previous-quarter observed counts available. "
        "The 2025 evaluation is exploratory because it was inspected during development. "
        "These metrics evaluate risk coverage, **not collisions prevented**. Equal inspection "
        "cost per cell is assumed. OSM road features represent a contemporary snapshot."
    )
