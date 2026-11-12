# RoadRisk — Graph-Based Traffic Collision Hotspot Forecasting

RoadRisk forecasts quarterly collision hotspots across 338 Manhattan grid cells at 500 m resolution. The focused comparison is:

| Model | Inputs |
| --- | --- |
| A — Naive baseline | Previous-quarter collisions (`lag_1`) |
| B — Historical LightGBM | `lag_1`, `lag_4`, `rolling_mean_4`, `quarter_of_year` |
| C — Graph-enhanced LightGBM | Model B's four features plus `road_length_m`, `major_road_share`, `intersection_density`, `mean_streets_per_node` |
| D — Graph Convolutional Network | The same eight features as C, with shared-edge grid adjacency |

**Precision@20 and Capture@20 are the primary hotspot metrics.** MAE and R² provide supplementary count-fit measures. The original LightGBM search and GCN early stopping select models using validation MAE.

## Explore the dashboard

Use Python 3.11 or newer. From the parent `3.Collisionprediction` workspace:

```bash
python -m pip install -r requirements.txt
cd RoadRisk
python -m streamlit run app.py
```

With the existing workspace environment, the equivalent launch command from that parent directory is:

```bash
cd RoadRisk
../.col/bin/python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). Launching from `RoadRisk/` also loads the included `.streamlit/config.toml` theme. Input paths are resolved relative to `app.py`, so the app also runs when launched from another working directory.

The dashboard reads only `data/processed/test_predictions.csv` and `data/processed/manhattan_grid.gpkg`. Data loading is cached, with file modification time and size used to refresh regenerated artifacts. No model checkpoints are loaded and no training runs in the app. The CARTO/OpenStreetMap basemap requires internet access; no API key is required.

Choose **Naive Baseline**, **Historical LightGBM**, **Graph-enhanced LightGBM** (default), or **GCN**, then a saved evaluation quarter (the latest is selected initially). Four cards show Precision@20, Capture@20, MAE, and R² for that model and quarter.

The map offers **Predicted risk** and **Hotspot classification** views. Dark outlines mark the predicted top 20 cells; the classification view distinguishes correctly predicted, false positive, and missed hotspots. Hover for predicted and observed collision counts. Beside the map, a table lists the predicted top 20, and a download provides the selected model's predictions for every cell in the chosen quarter.

The model comparison table and chart support the selected quarter, quarterly averages, and Notebook 05's aggregate. Choose any of the four metrics for the chart. The methodology section explains the inputs, chronological evaluation, and aggregation rules.

- **Precision@20:** predicted top-20 cells that also belong to the observed top 20, divided by 20.
- **Capture@20:** observed collisions in predicted top-20 cells, divided by all observed collisions that quarter.
- **MAE:** mean absolute prediction error in collisions per cell; lower is better.
- **R²:** coefficient of determination; higher is better, and negative values are possible.

Rankings sort collision counts descending, then numeric `cell_id` ascending. The metrics reproduce Notebook 05. Its aggregate averages Precision@20 and Capture@20 across quarters and calculates MAE and R² over pooled rows; the quarterly-average view averages all four metrics across quarters. This is a retrospective evaluation using a previously inspected 2025 period. It does not estimate collisions prevented or provide live risk predictions.

The required prediction columns are `cell_id`, `quarter`, `collisions`, `baseline_pred`, `model_b_pred`, `model_c_pred`, and `model_d_pred`. The grid needs unique numeric IDs, valid polygon boundaries, and a declared CRS; the app converts the saved EPSG:32618 grid to EPSG:4326. Each saved quarter must cover every grid cell with finite, nonnegative counts and unique keys. At least 20 cells are required for hotspot evaluation. Invalid files or joins produce an informative error.

The existing root `requirements.txt` supplies the dashboard runtime; model-training dependencies are not needed. To run the basic regression checks from the parent workspace:

```bash
.col/bin/python -m unittest discover -s RoadRisk/tests -v
```

These check metrics against Notebook 05’s extracted evaluator without running training, map hotspot classifications and outlines, tie-breaking, zero-collision handling, and app controls. The notebook comparison uses scikit-learn from the existing modeling environment and is skipped if that optional test dependency is absent.

## Reproduce the pipeline

Run the existing notebooks in order, using `notebooks/` as the working directory:

1. `01_data_aquisition.ipynb`: acquire collision records.
2. `02_eda.ipynb`: explore and clean collision data.
3. `03_spatial_grid_graph.ipynb`: construct the grid, assign collisions, build the quarterly panel, and save six road-network features in `graph_features.csv` (338 × 7 including `cell_id`).
4. `04_feature_engineering.ipynb`: calculate history using shifts before rolling, merge all six road features, and save `model_data.csv` (8,112 × 13). Only four road features enter training.
5. `05_modeling_evaluation.ipynb`: tune LightGBM, train the eight-input two-layer GCN from scratch, evaluate all four models, and save artifacts and fresh predictions.

With existing cleaned data and boundaries, rerun **03 → 04 → 05** after changing features. Notebook 03 includes network requests for borough boundaries and OSMnx roads. Existing Python dependencies include pandas, NumPy, requests, GeoPandas, Shapely, OSMnx, Matplotlib, LightGBM, scikit-learn, joblib, IPython, PyTorch, and PyTorch Geometric. The root `requirements.txt` specifies the dashboard runtime, not the full notebook-training environment.

The chronological split is **2020Q1–2023Q4 for training**, **2024Q1–2024Q4 for validation**, and **2025Q1–2025Q4 for exploratory evaluation**. Each forecast uses only history preceding that quarter. Fixed seeds are used; the GCN selects MPS when available and otherwise CPU. Its scaler is fitted only on training data. Notebook 05 sets `OMP_NUM_THREADS=1` before native-library imports to avoid a reproduced local LightGBM/PyTorch reload crash. Use the same setting before importing these libraries in a separate inference process.

The 2025 results have already been inspected during development, so they are **not an untouched final test**. For this exploratory period, selected LightGBM models are refitted on 2020–2024; the GCN retains its validation-selected checkpoint fitted on 2020–2023. These fitting-period differences should accompany comparisons.

Notebook 05 writes `models/model_b.pkl`, `model_c.pkl`, `model_config.pkl`, and `gnn_model.pth`. The GCN checkpoint includes CPU weights, eight feature names, scaler statistics, node order, adjacency, architecture, and training metadata; the notebook verifies reloading both weights and scaler. `data/processed/test_predictions.csv` is regenerated with predictions for all four models and retains the columns consumed by `app.py`.

## Experimental Findings

Development explored additional NetworkX degree and betweenness features, enhanced graph-feature LightGBM, neighboring-cell collision history, and GCN training and validation. Historical 2024 validation results were:

| Development configuration | MAE | Precision@20 | Capture@20 |
| --- | ---: | ---: | ---: |
| Enhanced graph LightGBM | 2.119 | 0.650 | 0.196 |
| With spatial lags | 2.138 | 0.638 | approximately 0.190 |

These results motivated simplifying the final feature set while retaining the GCN extension. They belong to historical development configurations and **are not the performance of the restored four-graph-feature Model C**. Current results are in notebook 05; rerun it to regenerate them.
