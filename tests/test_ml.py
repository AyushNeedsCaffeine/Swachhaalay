"""
Smart Washroom — Unit Tests for ML Pipeline

Tests data loading, model inference, feature engineering, and edge cases.
Run: python -m pytest tests/test_ml.py -v
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = "data/washroom_dataset_multi_cubicle.csv"
MODELS_DIR = "ml/models"


class TestDataLoading:
    """Test dataset loading and integrity."""

    def test_csv_exists(self):
        assert os.path.exists(DATA_PATH), f"Dataset not found: {DATA_PATH}"

    def test_csv_shape(self):
        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
        assert df.shape[0] == 25056 * 4, f"Expected 100224 rows (25056 per cubicle x 4), got {df.shape[0]}"
        assert df.shape[1] == 18, f"Expected 18 columns, got {df.shape[1]}"

    def test_no_nulls(self):
        df = pd.read_csv(DATA_PATH)
        nulls = df.isnull().sum().sum()
        assert nulls == 0, f"Found {nulls} null values"

    def test_four_cubicles(self):
        df = pd.read_csv(DATA_PATH)
        cubicles = df["cubicle_id"].unique()
        assert len(cubicles) == 4, f"Expected 4 cubicles, got {len(cubicles)}"

    def test_timestamp_range(self):
        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
        assert df["timestamp"].min() == pd.Timestamp("2026-08-01 00:00:00")
        assert df["timestamp"].max() == pd.Timestamp("2026-10-26 23:55:00")

    def test_value_ranges(self):
        df = pd.read_csv(DATA_PATH)
        assert df["mq135_gas_ppm"].min() >= 100.0
        assert df["hygiene_score"].min() >= 0.0
        assert df["hygiene_score"].max() <= 100.0
        assert df["water_level_cm"].min() >= 0.0
        assert df["water_level_cm"].max() <= 20.0
        assert df["disinfectant_level_virtual_pct"].min() >= 0.0
        assert df["disinfectant_level_virtual_pct"].max() <= 100.0

    def test_binary_columns(self):
        df = pd.read_csv(DATA_PATH)
        binary_cols = ["occupancy_ld2410", "motion_ir", "mist_maker_status",
                       "water_refill_status", "disinfectant_refill_status",
                       "needs_manual_checkup", "needs_cleaning", "room_available"]
        for col in binary_cols:
            unique_vals = set(df[col].unique())
            assert unique_vals.issubset({0, 1}), f"{col} has non-binary values: {unique_vals}"

    def test_room_available_respects_cooldown(self):
        """room_available must track the post-spray cooldown state machine exactly:
        decrement once per 5-min row, block availability while >0, restart on spray."""
        df = pd.read_csv(DATA_PATH)
        df = df.sort_values(["cubicle_id", "timestamp"]).reset_index(drop=True)
        dry_steps = 0
        for _, row in df.iterrows():
            dry_steps = max(0, dry_steps - 1)
            assert int(dry_steps == 0) == row["room_available"], (
                f"{row['cubicle_id']}: room_available={row['room_available']} at "
                f"{row['timestamp']} but cooldown implies {int(dry_steps == 0)}"
            )
            if row["mist_maker_status"] == 1:
                dry_steps = 2  # DRY_STEPS (10 minutes at 5-min steps)


class TestModels:
    """Test that trained models exist and load correctly."""

    def test_anomaly_model_exists(self):
        path = os.path.join(MODELS_DIR, "anomaly_detection.joblib")
        assert os.path.exists(path), "Anomaly detection model not found"

    def test_virtual_sensing_model_exists(self):
        path = os.path.join(MODELS_DIR, "virtual_sensing.joblib")
        assert os.path.exists(path), "Virtual sensing model not found"

    def test_clustering_model_exists(self):
        path = os.path.join(MODELS_DIR, "usage_clustering.joblib")
        assert os.path.exists(path), "Usage clustering model not found"

    def test_metadata_exists(self):
        for name in ["anomaly_detection", "virtual_sensing", "usage_clustering"]:
            path = os.path.join(MODELS_DIR, f"{name}_meta.joblib")
            assert os.path.exists(path), f"{name} metadata not found"

    def test_anomaly_metadata(self):
        meta = joblib.load(os.path.join(MODELS_DIR, "anomaly_detection_meta.joblib"))
        assert meta["model_type"] == "IsolationForest"
        assert meta["anomaly_rate_test"] > 0
        assert meta["anomaly_rate_test"] < 20  # Should be around 5%

    def test_virtual_sensing_metadata(self):
        meta = joblib.load(os.path.join(MODELS_DIR, "virtual_sensing_meta.joblib"))
        assert meta["model_type"] == "RandomForest"
        res = meta["results"]

        # The LEAKY number is a reference for pipeline sanity only (R² should stay
        # ~1.0 because it cheats by feeding the true previous level), never a claim
        # about deployed performance.
        assert "RandomForest_LEAKY_reference_only" in res
        assert res["RandomForest_LEAKY_reference_only"]["R2"] > 0.9

        # The HONEST headline is the unsuffixed walk-forward result (model's own
        # predictions fed forward, reset only at observed refills). No R²>0 gate:
        # on this synthetic data multi-step prediction is genuinely hard; the gate
        # is protocol integrity -- the honest number must be present AND beat the
        # fixed naive baseline (which blindly assumes the level never changes
        # between refills).
        assert res["RandomForest"]["R2"] <= res["RandomForest_LEAKY_reference_only"]["R2"]
        assert res["RandomForest"]["MAE"] < res["NaiveBaseline_fixed"]["MAE"]

        # Strict rollout (seed at refill, own-predictions only) is also reported,
        # with its excluded pre-first-refill rows accounted for.
        assert "RandomForest_Rollout_between_refills" in res
        assert res["RandomForest_Rollout_between_refills"]["excluded_rows"] > 0
        assert res["RandomForest_Rollout_between_refills"]["scored_rows"] > 0

        # Metadata must clearly document that only the honest numbers are claims.
        note = meta.get("evaluation_note", "")
        assert "LEAKY_reference_only" in note and "walk-forward" in note

    def test_clustering_metadata(self):
        meta = joblib.load(os.path.join(MODELS_DIR, "usage_clustering_meta.joblib"))
        assert meta["model_type"] == "KMeans"
        assert meta["k"] >= 2
        assert len(meta["cluster_labels"]) == 4


class TestFeatureEngineering:
    """Test feature engineering pipeline."""

    def test_anomaly_features(self):
        from ml.anomaly_detection import engineer_features
        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
        # Use a small sample
        df_small = df[df["cubicle_id"] == "Cubicle_A_Office"].head(100)
        df_feat, feature_cols = engineer_features(df_small)
        assert len(feature_cols) >= 10, f"Expected >=10 features, got {len(feature_cols)}"
        assert df_feat.shape[0] == 100

    def test_virtual_sensing_features(self):
        from ml.virtual_sensing import prepare_features
        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
        df_small = df[df["cubicle_id"] == "Cubicle_A_Office"].head(100)
        df_feat, feature_cols = prepare_features(df_small)
        assert len(feature_cols) >= 8, f"Expected >=8 features, got {len(feature_cols)}"
        assert "prev_disinfectant_level" in feature_cols


class TestSafetyRules:
    """Test safety-critical occupancy rules."""

    def test_no_spray_while_occupied(self):
        """Verify the dataset has zero spray events while occupied."""
        df = pd.read_csv(DATA_PATH)
        occupied_mask = (df["occupancy_ld2410"] == 1) | (df["motion_ir"] == 1)
        spray_while_occupied = df[occupied_mask & (df["mist_maker_status"] == 1)]
        assert len(spray_while_occupied) == 0, (
            f"Found {len(spray_while_occupied)} spray events while occupied!"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
