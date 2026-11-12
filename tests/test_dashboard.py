"""Saved-data dashboard checks; run: python -m unittest discover -s tests -v.

Only Notebook 05's evaluator definition is executed, never its modeling cells.
The optional scikit-learn reference dependency belongs to the notebook environment.
"""

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(app.PREDICTIONS_PATH)
        cls.predictions = app.validate_predictions(cls.raw)
        cls.grid = app.validate_grid(gpd.read_file(app.GRID_PATH))
        try:
            from sklearn.metrics import mean_absolute_error, r2_score
        except ImportError:
            cls.reference = None
        else:
            notebook = json.loads((ROOT / "notebooks/05_modeling_evaluation.ipynb").read_text())
            source = next("".join(c.get("source", [])) for c in notebook["cells"] if "def evaluate_model(" in "".join(c.get("source", [])))
            function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_model")
            scope = dict(pd=pd, np=np, mean_absolute_error=mean_absolute_error, r2_score=r2_score)
            exec(compile(ast.Module(body=[function], type_ignores=[]), "notebook_evaluator", "exec"), scope)
            cls.reference = staticmethod(scope["evaluate_model"])

    def test_metrics_match_notebook_for_each_quarter_and_pooled_rows(self):
        if self.reference is None:
            self.skipTest("Notebook reference requires optional scikit-learn")
        scopes = [*self.predictions.groupby("quarter"), ("pooled", self.predictions)]
        for quarter, rows in scopes:
            for model, column in app.MODELS.items():
                with self.subTest(quarter=quarter, model=model):
                    expected = self.reference(rows, column)
                    actual = app.evaluate_model(rows, column)
                    for metric in app.METRICS:
                        self.assertAlmostEqual(actual[metric], expected[metric], places=12)

    def test_numeric_tie_breaks_and_zero_collision_metrics(self):
        rows = pd.DataFrame({"cell_id": [str(i) for i in range(60)], "quarter": "2025Q1", "collisions": 0, **{c: 1.0 for c in app.MODELS.values()}})
        rows = app.validate_predictions(rows)
        for seed in [5, 20, 50]:
            shuffled = rows.sample(frac=1, random_state=seed)
            self.assertEqual(app.top_cell_ids(shuffled, "baseline_pred"), set(range(20)))
            score = app.evaluate_model(shuffled, "model_c_pred")
            self.assertEqual(score, {"Precision@20": 1, "Capture@20": 0, "MAE": 1, "R²": 0})
        rows["model_c_pred"] = 0
        score = app.evaluate_model(rows, "model_c_pred")
        self.assertEqual(score, {"Precision@20": 1, "Capture@20": 0, "MAE": 0, "R²": 1})
        rows.loc[rows.cell_id >= 40, "collisions"] = 1
        rows["model_c_pred"] = rows["collisions"]
        score = app.evaluate_model(rows, "model_c_pred")
        self.assertEqual(score, {"Precision@20": 1, "Capture@20": 1, "MAE": 0, "R²": 1})

    def test_map_hotspot_classes_and_both_views_use_all_cells(self):
        self.assertEqual(self.grid.crs.to_epsg(), 4326)
        app.validate_coverage(self.grid, self.predictions)
        for quarter, rows in self.predictions.groupby("quarter"):
            for column in app.MODELS.values():
                with self.subTest(quarter=quarter, column=column):
                    mapped = app.prepare_map_data(self.grid, rows, column)
                    predicted = set(rows.sort_values([column, "cell_id"], ascending=[False, True]).head(20).cell_id)
                    actual = set(rows.sort_values(["collisions", "cell_id"], ascending=[False, True]).head(20).cell_id)
                    self.assertEqual(set(mapped.loc[mapped.predicted_top, "cell_id"]), predicted)
                    self.assertEqual(set(mapped.loc[mapped.actual_top, "cell_id"]), actual)
                    expected_classes = {
                        "Correctly predicted hotspot": predicted & actual,
                        "False positive hotspot": predicted - actual,
                        "Missed hotspot": actual - predicted,
                        "Other grid cell": set(self.grid.cell_id) - (predicted | actual),
                    }
                    for status, ids in expected_classes.items():
                        self.assertEqual(set(mapped.loc[mapped.status == status, "cell_id"]), ids)
                    self.assertEqual(len(mapped), len(self.grid))
                    self.assertFalse(mapped[["predicted_label", "actual_label"]].isna().any().any())
                    for mode in ["Predicted risk", "Hotspot classification"]:
                        deck = json.loads(app.make_deck(mapped, column, mode, max(self.predictions[column].max(), 1)).to_json())
                        features = deck["layers"][0]["data"]["features"]
                        self.assertEqual(len(features), len(self.grid))
                        self.assertEqual(sum(f["properties"]["predicted_top"] for f in features), 20)
                        outlines = deck["layers"][1]["data"]["features"]
                        self.assertEqual({f["properties"]["cell_id"] for f in outlines}, predicted)

    def test_validation_rejects_missing_counts_duplicate_keys_and_bad_joins(self):
        invalid = [self.raw.iloc[:0], self.raw.drop(columns="model_d_pred"), pd.concat([self.raw, self.raw.iloc[:1]])]
        for value in [np.nan, np.inf, -1]:
            frame = self.raw.copy()
            frame.loc[0, "model_c_pred"] = value
            invalid.append(frame)
        for frame in invalid:
            with self.assertRaises(app.DataValidationError):
                app.validate_predictions(frame)
        with self.assertRaises(app.DataValidationError):
            app.validate_coverage(self.grid, self.predictions.iloc[1:])
        with self.assertRaises(app.DataValidationError):
            app.validate_grid(gpd.GeoDataFrame({"cell_id": [0]}))
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "predictions.csv"
            with self.assertRaises(app.DataValidationError):
                app.file_signature(missing)
            with patch.object(app, "PREDICTIONS_PATH", missing):
                at = AppTest.from_string("from app import main\nmain()", default_timeout=15).run()
                self.assertFalse(at.exception)
                self.assertEqual(len(at.error), 1)
                self.assertFalse(at.metric)

    def test_dashboard_controls_metrics_and_comparison_views(self):
        at = AppTest.from_file(ROOT / "app.py", default_timeout=15).run()
        self.assertFalse(at.exception)
        self.assertEqual(at.selectbox(key="model").options, list(app.MODELS))
        self.assertEqual(at.selectbox(key="model").value, "Graph-enhanced LightGBM")
        self.assertFalse(at.slider)
        quarters = sorted(self.predictions.quarter.unique())
        self.assertEqual(at.selectbox(key="quarter").options, quarters)
        self.assertEqual(at.selectbox(key="quarter").value, quarters[-1])
        self.assertEqual(at.radio(key="map_mode").options, ["Predicted risk", "Hotspot classification"])
        self.assertEqual(at.radio(key="comparison_scope").options, ["Selected quarter", "Quarterly average", "Notebook aggregate"])
        self.assertEqual(at.selectbox(key="chart_metric").options, app.METRICS)
        for model, column in app.MODELS.items():
            for quarter in quarters:
                with self.subTest(model=model, quarter=quarter):
                    at.selectbox(key="model").select(model)
                    at.selectbox(key="quarter").select(quarter).run()
                    self.assertFalse(at.exception)
                    self.assertFalse(at.error)
                    self.assertEqual(len(at.metric), 4)
                    expected = app.evaluate_model(self.predictions[self.predictions.quarter == quarter], column)
                    for metric, key in zip(at.metric, app.METRICS):
                        value = float(metric.value.rstrip("%"))
                        if key in app.METRICS[:2]:
                            value /= 100
                        self.assertAlmostEqual(value, expected[key], delta=.00051)
                    self.assertEqual(len(at.get("deck_gl_json_chart")), 1)
                    self.assertEqual(len(at.get("vega_lite_chart")), 1)
                    self.assertEqual(len(at.dataframe), 2)
                    self.assertEqual(len(at.dataframe[0].value), 20)
        at.radio(key="map_mode").set_value("Hotspot classification").run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.get("deck_gl_json_chart")), 1)
        quarterly, aggregate = app.compute_scores(self.predictions)
        quarter = at.selectbox(key="quarter").value
        expected_tables = {
            "Selected quarter": quarterly.loc[quarterly.quarter == quarter].set_index("Model")[app.METRICS],
            "Quarterly average": quarterly.groupby("Model", sort=False)[app.METRICS].mean(),
            "Notebook aggregate": aggregate[app.METRICS],
        }
        for scope, table in expected_tables.items():
            with self.subTest(comparison_scope=scope):
                at.radio(key="comparison_scope").set_value(scope).run()
                self.assertFalse(at.exception)
                np.testing.assert_allclose(at.dataframe[1].value[app.METRICS], table.reindex(app.MODELS))
        for metric in app.METRICS:
            with self.subTest(chart_metric=metric):
                at.selectbox(key="chart_metric").select(metric).run()
                self.assertFalse(at.exception)
                self.assertFalse(at.error)
                self.assertEqual(len(at.get("vega_lite_chart")), 1)


if __name__ == "__main__":
    unittest.main()
