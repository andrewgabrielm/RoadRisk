"""RoadRisk: explore saved, retrospective Manhattan collision predictions.

Run from this directory with: python -m streamlit run app.py
No model checkpoints, training code, or network data queries are used here.
"""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

import altair as alt
import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent
PREDICTIONS_PATH = ROOT / "data/processed/test_predictions.csv"
GRID_PATH = ROOT / "data/processed/manhattan_grid.gpkg"
K = 20
MODELS = {
    "Naive Baseline": "baseline_pred",
    "Historical LightGBM": "model_b_pred",
    "Graph-enhanced LightGBM": "model_c_pred",
    "GCN": "model_d_pred",
}
MODEL_DESCRIPTIONS = {
    "Naive Baseline": "Carries the previous quarter’s collision count forward for each cell.",
    "Historical LightGBM": "Learns from recent collision history and seasonal patterns.",
    "Graph-enhanced LightGBM": "Adds OSM road-network characteristics to collision history.",
    "GCN": "Combines the same eight inputs with information from adjacent grid cells.",
}
METRICS = ["Precision@20", "Capture@20", "MAE", "R²"]
STATUS_COLORS = {
    "Correctly predicted hotspot": [13, 133, 120, 210],
    "False positive hotspot": [229, 128, 38, 210],
    "Missed hotspot": [126, 87, 180, 210],
    "Other grid cell": [208, 216, 225, 130],
}
RISK_COLORS = np.array([[235, 241, 245], [254, 217, 142], [242, 128, 47], [175, 41, 47]])


class DataValidationError(ValueError):
    """A saved input is absent or unsuitable for an honest comparison."""


def file_signature(path: Path) -> tuple[int, int]:
    """Invalidate cached data when an artifact is replaced or regenerated."""
    try:
        stat = path.stat()
    except OSError as exc:
        raise DataValidationError(f"Cannot access {path}: {exc.strerror or exc}") from exc
    return stat.st_mtime_ns, stat.st_size


def normalize_cell_ids(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    frame = frame.copy()
    ids = []
    for raw in frame["cell_id"]:
        # Decimal avoids rounding large string/integer IDs through a float.
        try:
            value = Decimal(str(raw))
        except InvalidOperation as exc:
            raise DataValidationError(f"{source}: cell_id must contain finite integer IDs without missing values.") from exc
        if not value.is_finite():
            raise DataValidationError(f"{source}: cell_id must contain finite integer IDs without missing values.")
        if value != value.to_integral_value():
            raise DataValidationError(f"{source}: cell_id contains non-integer values.")
        if not -(2**63) <= value < 2**63:
            raise DataValidationError(f"{source}: cell_id must fit a signed 64-bit integer.")
        ids.append(int(value))
    frame["cell_id"] = pd.Series(ids, index=frame.index, dtype="int64")
    return frame


def validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = ["cell_id", "quarter", "collisions", *MODELS.values()]
    missing = sorted(set(required) - set(predictions.columns))
    if missing:
        raise DataValidationError(f"test_predictions.csv is missing required columns: {', '.join(missing)}.")
    if predictions.empty:
        raise DataValidationError("test_predictions.csv contains no evaluation rows.")
    predictions = normalize_cell_ids(predictions[required], "test_predictions.csv")
    if predictions["quarter"].isna().any():
        raise DataValidationError("test_predictions.csv: quarter has missing values.")
    predictions["quarter"] = predictions["quarter"].astype(str)
    if not predictions["quarter"].str.fullmatch(r"\d{4}Q[1-4]").all():
        raise DataValidationError("test_predictions.csv: quarter must use YYYYQ1–YYYYQ4, for example 2025Q1.")
    if predictions.duplicated(["cell_id", "quarter"]).any():
        raise DataValidationError("test_predictions.csv has duplicate (cell_id, quarter) keys; each must be unique.")
    for column in ["collisions", *MODELS.values()]:
        values = pd.to_numeric(predictions[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise DataValidationError(f"test_predictions.csv: {column} contains missing or non-finite numeric values.")
        if (values < 0).any():
            raise DataValidationError(f"test_predictions.csv: {column} contains negative collision counts.")
        predictions[column] = values
    if (predictions["collisions"] % 1 != 0).any():
        raise DataValidationError("test_predictions.csv: observed collisions must be whole counts.")
    return predictions.sort_values(["quarter", "cell_id"]).reset_index(drop=True)


def validate_grid(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "cell_id" not in grid.columns:
        raise DataValidationError("manhattan_grid.gpkg is missing the required cell_id column.")
    if grid.empty:
        raise DataValidationError("manhattan_grid.gpkg contains no grid cells.")
    try:
        geometry = grid.geometry
        crs = grid.crs
    except AttributeError as exc:
        raise DataValidationError("manhattan_grid.gpkg is missing an active geometry column and its CRS.") from exc
    grid = normalize_cell_ids(grid, "manhattan_grid.gpkg")
    if grid["cell_id"].duplicated().any():
        raise DataValidationError("manhattan_grid.gpkg has duplicate cell_id keys.")
    if crs is None:
        raise DataValidationError("manhattan_grid.gpkg has no CRS; save its correct source CRS before mapping.")
    if geometry.isna().any() or geometry.is_empty.any() or not geometry.is_valid.all():
        raise DataValidationError("manhattan_grid.gpkg contains missing, empty, or invalid geometries.")
    if not grid.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise DataValidationError("manhattan_grid.gpkg must contain Polygon or MultiPolygon grid boundaries.")
    return grid[["cell_id", "geometry"]].to_crs("EPSG:4326")


def validate_coverage(grid: gpd.GeoDataFrame, predictions: pd.DataFrame) -> None:
    grid_ids = set(grid["cell_id"])
    for quarter, rows in predictions.groupby("quarter", sort=True):
        ids = set(rows["cell_id"])
        if ids != grid_ids:
            raise DataValidationError(
                f"{quarter}: grid/prediction cell_id mismatch "
                f"({len(grid_ids - ids)} grid cells without predictions; "
                f"{len(ids - grid_ids)} predictions without geometry)."
            )
        if len(rows) < K:
            raise DataValidationError(f"{quarter}: at least {K} cells are required for hotspot evaluation.")


@st.cache_data(show_spinner="Loading saved predictions and Manhattan grid…")
def load_data(
    predictions_path: str,
    grid_path: str,
    predictions_signature: tuple[int, int],
    grid_signature: tuple[int, int],
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    # Signatures are cache inputs, so regenerated files do not leave stale results.
    try:
        predictions = pd.read_csv(predictions_path)
    except Exception as exc:
        raise DataValidationError(f"Could not read predictions at {predictions_path}: {exc}") from exc
    try:
        grid = gpd.read_file(grid_path)
    except Exception as exc:
        raise DataValidationError(f"Could not read grid at {grid_path}: {exc}") from exc
    predictions = validate_predictions(predictions)
    grid = validate_grid(grid)
    validate_coverage(grid, predictions)
    return grid, predictions


def top_cell_ids(rows: pd.DataFrame, value_column: str) -> set[int]:
    """Notebook 05 ordering: value descending, numeric cell_id ascending."""
    return set(rows.sort_values([value_column, "cell_id"], ascending=[False, True]).head(K)["cell_id"])


def evaluate_model(rows: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    """Reproduce Notebook 05, including its mean-hotspot/pooled-count aggregate."""
    precisions, captures = [], []
    for _, quarter in rows.groupby("quarter", sort=True):
        predicted_top = top_cell_ids(quarter, prediction_column)
        actual_top = top_cell_ids(quarter, "collisions")
        precisions.append(len(predicted_top & actual_top) / K)
        total = quarter["collisions"].sum()
        captures.append(quarter.loc[quarter["cell_id"].isin(predicted_top), "collisions"].sum() / total if total else 0.0)
    actual = rows["collisions"].to_numpy(dtype=float)
    predicted = rows[prediction_column].to_numpy(dtype=float)
    residual = np.square(actual - predicted).sum()
    variation = np.square(actual - actual.mean()).sum()
    # Match sklearn.r2_score's default force_finite=True for constant targets.
    r_squared = 1.0 - residual / variation if variation else (1.0 if residual == 0 else 0.0)
    return {
        "Precision@20": float(np.mean(precisions)),
        "Capture@20": float(np.mean(captures)),
        "MAE": float(np.abs(actual - predicted).mean()),
        "R²": float(r_squared),
    }


@st.cache_data(show_spinner=False)
def compute_scores(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    quarterly = [
        {"Model": name, "quarter": quarter, **evaluate_model(rows, column)}
        for quarter, rows in predictions.groupby("quarter", sort=True)
        for name, column in MODELS.items()
    ]
    aggregate = [{"Model": name, **evaluate_model(predictions, column)} for name, column in MODELS.items()]
    return pd.DataFrame(quarterly), pd.DataFrame(aggregate).set_index("Model")


def prepare_map_data(grid: gpd.GeoDataFrame, rows: pd.DataFrame, prediction_column: str) -> gpd.GeoDataFrame:
    predicted_top = top_cell_ids(rows, prediction_column)
    actual_top = top_cell_ids(rows, "collisions")
    merged = grid.merge(rows[["cell_id", "collisions", prediction_column]], on="cell_id", how="left", validate="one_to_one")
    if merged[["collisions", prediction_column]].isna().any().any():
        raise DataValidationError("The map join produced missing collision values; check cell_id coverage.")
    merged["predicted_top"] = merged["cell_id"].isin(predicted_top)
    merged["actual_top"] = merged["cell_id"].isin(actual_top)
    merged["status"] = np.select(
        [merged.predicted_top & merged.actual_top, merged.predicted_top, merged.actual_top],
        list(STATUS_COLORS)[:3], default="Other grid cell",
    )
    merged["predicted_label"] = merged[prediction_column].map(lambda value: f"{value:.2f}")
    merged["actual_label"] = merged["collisions"].map(lambda value: f"{value:,.0f}")
    return merged


def make_deck(map_data: gpd.GeoDataFrame, prediction_column: str, mode: str, scale_max: float) -> pdk.Deck:
    data = map_data.copy()
    if mode == "Predicted risk":
        intensity = (data[prediction_column].to_numpy() / max(scale_max, 1)).clip(0, 1)
        colors = np.column_stack([np.interp(intensity, np.linspace(0, 1, len(RISK_COLORS)), RISK_COLORS[:, i]) for i in range(3)])
        data["fill_color"] = [list(map(int, color)) + [195] for color in colors]
    else:
        data["fill_color"] = data["status"].map(STATUS_COLORS)
    features = json.loads(data.to_json(drop_id=True))
    outlined = {"type": "FeatureCollection", "features": [f for f in features["features"] if f["properties"]["predicted_top"]]}
    layers = [
        pdk.Layer(
            "GeoJsonLayer", features, id="collision-grid", pickable=True, auto_highlight=True,
            get_fill_color="properties.fill_color", get_line_color=[255, 255, 255, 150],
            get_line_width=0.6, line_width_units='"pixels"', stroked=True, filled=True,
            highlight_color=[15, 23, 42, 65],
        ),
        pdk.Layer(
            "GeoJsonLayer", outlined, id="predicted-hotspot-outlines", pickable=False,
            filled=False, stroked=True, get_line_color=[24, 42, 58, 255],
            get_line_width=2.5, line_width_units='"pixels"',
        ),
    ]
    west, south, east, north = data.total_bounds
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=(south + north) / 2, longitude=(west + east) / 2, zoom=10.4, pitch=0),
        map_provider="carto", map_style="light",
        tooltip={
            "html": "<b>Cell {properties.cell_id}</b><br/>Predicted collisions: {properties.predicted_label}<br/>Actual collisions: {properties.actual_label}<br/>{properties.status}",
            "style": {"backgroundColor": "#182a3a", "color": "white", "fontSize": "13px", "padding": "12px"},
        },
    )


def add_styles() -> None:
    st.markdown("""
<style>
.block-container {max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem;}
h1, h2, h3 {letter-spacing: -.035em;}
h1 {font-size: 3.5rem !important; padding-bottom: .15rem !important;}
h2 {font-size: 1.65rem !important;}
.rr-eyebrow {color: #0d8578; font-size: .73rem; font-weight: 750; letter-spacing: .15em; text-transform: uppercase;}
.rr-subtitle {font-size: 1.35rem; font-weight: 550; letter-spacing: -.02em; margin-bottom: .5rem;}
.rr-intro {max-width: 780px; opacity: .75; line-height: 1.65; margin-bottom: 1.5rem;}
.rr-legend {display: flex; gap: .7rem 1.2rem; flex-wrap: wrap; font-size: .78rem; margin: .6rem 0; align-items: center;}
.rr-legend span {display: inline-flex; align-items: center; gap: .4rem;}
.rr-swatch {display: inline-block; width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0;}
.rr-outline {background: transparent; border: 2px solid #182a3a;}
.rr-gradient {width: 170px; height: 11px; border-radius: 4px; background: linear-gradient(90deg,#ebf1f5,#fed98e,#f2802f,#af292f);}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums; letter-spacing: -.04em;}
[data-testid="stMetric"] {padding: 1.1rem 1.2rem;}
[data-testid="stDataFrame"] {font-variant-numeric: tabular-nums;}
@media (max-width: 640px) {
 .block-container {padding: 1.2rem 1rem 2rem;}
 h1 {font-size: 2.8rem !important;}
 .rr-subtitle {font-size: 1.1rem;}
}
</style>
""", unsafe_allow_html=True)


def render_legend(mode: str, scale_max: float) -> None:
    if mode == "Predicted risk":
        content = f'<span>0</span><i class="rr-gradient"></i><span>{scale_max:.1f} predicted collisions</span>'
    else:
        content = "".join(
            f'<span><i class="rr-swatch" style="background:rgb({color[0]},{color[1]},{color[2]})"></i>{label}</span>'
            for label, color in STATUS_COLORS.items()
        )
    st.markdown(f'<div class="rr-legend">{content}<span><i class="rr-swatch rr-outline"></i>Predicted top 20</span></div>', unsafe_allow_html=True)


def render_comparison(quarterly: pd.DataFrame, aggregate: pd.DataFrame, quarter: str, selected_model: str) -> None:
    st.subheader("How the models compare")
    scope = st.radio("Comparison period", ["Selected quarter", "Quarterly average", "Notebook aggregate"], horizontal=True, key="comparison_scope")
    if scope == "Selected quarter":
        table = quarterly.loc[quarterly.quarter == quarter].set_index("Model")[METRICS]
        explanation = f"{quarter} · All models evaluated on the same grid cells."
    elif scope == "Quarterly average":
        table = quarterly.groupby("Model", sort=False)[METRICS].mean()
        explanation = f"Unweighted mean of each metric across all {quarterly.quarter.nunique()} saved quarters, {quarterly.quarter.min()}–{quarterly.quarter.max()}. R² is averaged quarter by quarter."
    else:
        table = aggregate[METRICS]
        explanation = "Notebook 05 aggregation: Precision@20 and Capture@20 are quarterly means; MAE and R² use all evaluation rows pooled together."
    st.caption(explanation)
    table = table.reindex(MODELS)

    def best_values(column: pd.Series) -> list[str]:
        best = column.min() if column.name == "MAE" else column.max()
        return ["background-color: #d9f1e9; color: #075c50; font-weight: 700" if np.isclose(value, best, rtol=1e-10, atol=1e-12) else "" for value in column]

    display_table = table.copy()
    display_table.index = [f"{name} · selected" if name == selected_model else name for name in table.index]
    display_table.index.name = "Model"
    styled = display_table.style.apply(best_values).format({"Precision@20": "{:.1%}", "Capture@20": "{:.1%}", "MAE": "{:.3f}", "R²": "{:.3f}"})
    st.dataframe(styled, width="stretch", height=212)
    st.caption("Green cells mark the best value in each column, including ties. Higher is better for Precision@20, Capture@20 and R²; lower is better for MAE.")

    metric = st.selectbox("Chart metric", METRICS, key="chart_metric")
    chart_data = table.reset_index()
    best = table[metric].min() if metric == "MAE" else table[metric].max()
    chart_data["Role"] = ["Best value" if np.isclose(value, best, rtol=1e-10, atol=1e-12) else "Selected model" if name == selected_model else "Other model" for name, value in zip(chart_data.Model, chart_data[metric])]
    value_format = ".1%" if metric in METRICS[:2] else ".3f"
    chart_data["Value label"] = chart_data[metric].map(lambda value: format(value, value_format))
    base = alt.Chart(chart_data).encode(y=alt.Y("Model:N", sort=list(MODELS), title=None, axis=alt.Axis(labelLimit=240)))
    bars = base.mark_bar(cornerRadiusEnd=4, size=24).encode(
        x=alt.X(f"{metric}:Q", title=f"{metric} · {'lower' if metric == 'MAE' else 'higher'} is better", axis=alt.Axis(format=value_format), scale=alt.Scale(zero=True, padding=42)),
        color=alt.Color("Role:N", scale=alt.Scale(domain=["Best value", "Selected model", "Other model"], range=["#0d8578", "#244661", "#bac8d5"]), legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[alt.Tooltip("Model:N"), alt.Tooltip(f"{metric}:Q", format=value_format)],
    )
    labels = base.mark_text(align="left", dx=7, fontSize=12).encode(x=alt.X(f"{metric}:Q"), text="Value label:N")
    st.altair_chart((bars + labels).properties(height=215).configure_view(strokeWidth=0), width="stretch")


def render_methodology() -> None:
    with st.expander("About the methodology & evaluation", expanded=False):
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("""
**Data & spatial units**  
Manhattan motor-vehicle collision records are assigned to a **500 m spatial grid** and aggregated into quarterly counts. Every grid cell is scored, including cells with zero observed collisions.

**History & roads**  
Historical inputs are the previous quarter, the same quarter one year earlier, the preceding four-quarter mean, and quarter of year. OSM road-network inputs are road length, major-road share, intersection density, and mean streets per node.

**Four architectures**  
The naive baseline repeats the previous quarter. Historical LightGBM uses four history inputs; Graph-enhanced LightGBM adds four road features. The GCN uses the same eight inputs with shared-edge cell adjacency, two graph-convolution layers (32 and 16 channels), and a linear output with softplus for nonnegative counts. Its scaler is fitted on training data only.
""")
        with right:
            st.markdown("""
**Chronological evaluation**  
Training uses 2020–2023; 2024 informs LightGBM tuning and GCN early stopping using validation MAE. For 2025 evaluation, LightGBM is refitted on 2020–2024, while the GCN retains its validation-selected 2020–2023 checkpoint. Each quarter uses only preceding collision history.

**Precision@20 vs. Capture@20**  
Precision@20 is the fraction of the predicted top 20 cells also in the observed top 20. Capture@20 is the fraction of **all observed collisions** occurring in the predicted top 20. Both rankings break ties by ascending numeric `cell_id`. MAE measures count error; R² measures count fit and can be negative.

**How to read these results**  
This is **retrospective evaluation, not a live traffic-risk forecast**. The **2025 period was previously inspected during development** and is exploratory, rather than an untouched final test. The different LightGBM and GCN fitting periods should be considered when comparing them. Saved predictions are loaded directly; models are never retrained by the dashboard.
""")


def main() -> None:
    st.set_page_config(page_title="RoadRisk | Manhattan collision hotspots", page_icon="🗺️", layout="wide")
    add_styles()
    st.markdown('<div class="rr-eyebrow">Manhattan · Spatial forecasting · Retrospective evaluation</div>', unsafe_allow_html=True)
    st.title("RoadRisk")
    st.markdown('<div class="rr-subtitle">Graph-Based Traffic Collision Hotspot Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<p class="rr-intro">Where are collisions most likely to concentrate next quarter? Explore how collision history and road-network structure help rank Manhattan’s 500 m grid cells, using saved forecasts and observed outcomes.</p>', unsafe_allow_html=True)
    try:
        grid, predictions = load_data(str(PREDICTIONS_PATH), str(GRID_PATH), file_signature(PREDICTIONS_PATH), file_signature(GRID_PATH))
    except (DataValidationError, ValueError) as exc:
        st.error(f"RoadRisk could not load its evaluation data. {exc}")
        st.info("Expected saved artifacts under RoadRisk/data/processed/: test_predictions.csv and manhattan_grid.gpkg. Restore the complete outputs from the existing pipeline; the dashboard does not train models.")
        st.stop()

    quarters = sorted(predictions.quarter.unique())
    with st.container(border=True):
        model_col, quarter_col = st.columns([2, 1], gap="large")
        model = model_col.selectbox("Forecast model", list(MODELS), index=2, key="model")
        quarter = quarter_col.selectbox("Evaluation quarter", quarters, index=len(quarters) - 1, key="quarter")
        st.caption(f"{MODEL_DESCRIPTIONS[model]} · {len(grid):,} cells · {len(quarters)} saved evaluation quarters")
    rows = predictions.loc[predictions.quarter == quarter]
    column = MODELS[model]
    quarterly, aggregate = compute_scores(predictions)
    metrics = quarterly.loc[(quarterly.Model == model) & (quarterly.quarter == quarter)].iloc[0]
    map_data = prepare_map_data(grid, rows, column)
    matched = int((map_data.predicted_top & map_data.actual_top).sum())
    captured = int(map_data.loc[map_data.predicted_top, "collisions"].sum())
    total = int(rows.collisions.sum())
    st.subheader(f"{quarter} at a glance")
    for slot, metric, value, help_text in zip(
        st.columns(4), METRICS,
        [f"{metrics['Precision@20']:.1%}", f"{metrics['Capture@20']:.1%}", f"{metrics['MAE']:.3f}", f"{metrics['R²']:.3f}"],
        [f"{matched} of the predicted top 20 cells are also observed top-20 hotspots. Higher is better.",
         f"{captured:,} of {total:,} observed collisions occurred in the predicted top 20. Higher is better.",
         "Mean absolute prediction error in collisions per cell, for this quarter. Lower is better.",
         "Coefficient of determination across cells for this quarter. Higher is better; negative values are possible."],
    ):
        slot.metric(metric, value, help=help_text, border=True)
    st.caption(f"Selected quarter only · {total:,} observed collisions across {len(rows):,} cells · Retrospective results")

    st.divider()
    st.subheader("Explore the Manhattan grid")
    mode = st.radio("Map view", ["Predicted risk", "Hotspot classification"], horizontal=True, key="map_mode")
    scale_max = max(float(predictions[column].max()), 1.0)
    map_col, detail_col = st.columns([3, 1], gap="large")
    with map_col:
        render_legend(mode, scale_max)
        deck = make_deck(map_data, column, mode, scale_max)
        st.pydeck_chart(deck, height=590, width="stretch", key="manhattan_map")
        st.caption("Drag to pan · Scroll to zoom · Hover for cell-level counts. Basemap: © CARTO, © OpenStreetMap contributors.")
        if mode == "Predicted risk":
            st.caption("Risk means predicted collision count, not a probability. The color scale stays fixed across quarters for the selected model. Dark outlines mark its top 20 cells.")
        else:
            st.caption("Observed and predicted hotspots are each the top 20 cells. Correct = in both; false positive = predicted only; missed = observed only. Dark outlines mark predicted hotspots.")
    with detail_col:
        st.markdown("#### Inside the top 20")
        st.markdown(f"**{matched} / {K}** observed hotspots found")
        st.progress(matched / K)
        st.caption(f"{captured:,} observed collisions captured out of {total:,} across Manhattan.")
        top = rows.sort_values([column, "cell_id"], ascending=[False, True]).head(K).copy()
        top["Observed hotspot"] = top.cell_id.isin(top_cell_ids(rows, "collisions")).map({True: "Yes", False: "No"})
        st.dataframe(
            top[["cell_id", column, "collisions", "Observed hotspot"]].rename(columns={"cell_id": "Cell", column: "Predicted", "collisions": "Actual"}),
            hide_index=True, height=365, width="stretch",
            column_config={"Predicted": st.column_config.NumberColumn(format="%.2f")},
        )
        with st.expander("Download this selection"):
            st.download_button("Download cell predictions", rows[["cell_id", "quarter", "collisions", column]].to_csv(index=False), file_name=f"roadrisk_{column}_{quarter}.csv", mime="text/csv")

    st.divider()
    render_comparison(quarterly, aggregate, quarter, model)
    render_methodology()
    st.caption(f"RoadRisk · {len(grid):,} Manhattan cells at 500 m · Saved evaluation: {quarters[0]}–{quarters[-1]} · Data checked for complete one-to-one grid coverage")


if __name__ == "__main__":
    main()
