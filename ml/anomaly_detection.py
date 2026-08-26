"""
Smart Washroom — Anomaly Detection (Feature 1)

Detects unusual air quality patterns using Isolation Forest.
Uses cuML (GPU) when available, falls back to scikit-learn (CPU).

The model learns "normal" MQ135 behaviour across hours/usage patterns
and flags unusual spikes (chemical spill, blocked drain, failing sensor).
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.gpu_utils import GPU_AVAILABLE, load_data, to_numpy, to_cudf, train_test_split_time


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ML features from raw sensor data."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["cubicle_id", "timestamp"]).reset_index(drop=True)

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    for cubicle in df["cubicle_id"].unique():
        mask = df["cubicle_id"] == cubicle
        gas = df.loc[mask, "mq135_gas_ppm"]

        # Rolling statistics (window = 6 readings = 30 minutes)
        df.loc[mask, "gas_rolling_mean_30m"] = gas.rolling(6, min_periods=1).mean()
        df.loc[mask, "gas_rolling_std_30m"] = gas.rolling(6, min_periods=1).std().fillna(0)
        df.loc[mask, "gas_rolling_max_30m"] = gas.rolling(6, min_periods=1).max()

        # Rolling statistics (window = 12 readings = 1 hour)
        df.loc[mask, "gas_rolling_mean_1h"] = gas.rolling(12, min_periods=1).mean()
        df.loc[mask, "gas_rolling_std_1h"] = gas.rolling(12, min_periods=1).std().fillna(0)

        # Rate of change
        df.loc[mask, "gas_diff"] = gas.diff().fillna(0)
        df.loc[mask, "gas_diff_abs"] = gas.diff().abs().fillna(0)

        # Rolling rate of change
        df.loc[mask, "gas_roc_30m"] = gas.diff(6).fillna(0) / 30.0

        # Spray-related features
        df.loc[mask, "spray_count_1h"] = df.loc[mask, "mist_maker_status"].rolling(12, min_periods=1).sum()

        # Combined occupancy signal
        df.loc[mask, "is_occupied_combined"] = (
            (df.loc[mask, "occupancy_ld2410"] == 1) | (df.loc[mask, "motion_ir"] == 1)
        ).astype(int)

    feature_cols = [
        "mq135_gas_ppm", "hour_sin", "hour_cos", "day_of_week",
        "gas_rolling_mean_30m", "gas_rolling_std_30m", "gas_rolling_max_30m",
        "gas_rolling_mean_1h", "gas_rolling_std_1h",
        "gas_diff", "gas_diff_abs", "gas_roc_30m",
        "spray_count_1h", "is_occupied_combined",
        "entry_count", "hours_since_deep_clean",
    ]
    return df, feature_cols


def build_model():
    """Build Isolation Forest model (GPU or CPU)."""
    if GPU_AVAILABLE:
        from cuml.ensemble import IsolationForest
        model = IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination=0.05,
            random_state=42,
            output_type="numpy",
        )
        print("[GPU] Using cuML Isolation Forest")
    else:
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination=0.05,
            random_state=42,
            n_jobs=-1,
        )
        print("[CPU] Using scikit-learn Isolation Forest")
    return model


def train_anomaly_model(csv_path: str, output_dir: str = "ml/models"):
    """Train and save the anomaly detection model."""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading dataset...")
    df = load_data(csv_path)

    print("Engineering features...")
    df_feat, feature_cols = engineer_features(df)

    print("Splitting data (time-based)...")
    train_df, test_df = train_test_split_time(df_feat)

    X_train_np = to_numpy(train_df[feature_cols])
    X_test_np = to_numpy(test_df[feature_cols])

    print(f"Training set: {X_train_np.shape}, Test set: {X_test_np.shape}")

    model = build_model()
    print("Fitting model...")
    if GPU_AVAILABLE:
        X_train_gpu = to_cudf(train_df[feature_cols])
        model.fit(X_train_gpu)
    else:
        model.fit(X_train_np)
    print("Model fitted.")

    # Predict on test set
    if GPU_AVAILABLE:
        X_test_gpu = to_cudf(test_df[feature_cols])
        predictions = to_numpy(model.predict(X_test_gpu))
        scores = to_numpy(model.decision_function(X_test_gpu))
    else:
        predictions = model.predict(X_test_np)
        scores = model.decision_function(X_test_np)

    # -1 = anomaly, 1 = normal (sklearn convention)
    # Convert to 0 = normal, 1 = anomaly for consistency
    anomaly_labels = (predictions == -1).astype(int)
    n_anomalies = int(anomaly_labels.sum())
    print(f"Anomalies detected in test set: {n_anomalies}/{len(test_df)} ({n_anomalies/len(test_df)*100:.2f}%)")

    # Save model and metadata
    model_path = os.path.join(output_dir, "anomaly_detection.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")

    metadata = {
        "model_type": "IsolationForest",
        "features": feature_cols,
        "contamination": 0.05,
        "n_estimators": 200,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "n_anomalies_test": n_anomalies,
        "anomaly_rate_test": round(n_anomalies / len(test_df) * 100, 2),
        "gpu_accelerated": GPU_AVAILABLE,
    }
    joblib.dump(metadata, os.path.join(output_dir, "anomaly_detection_meta.joblib"))
    print(f"Metadata saved.")

    # Save predictions for evaluation
    results = test_df[["timestamp", "cubicle_id", "mq135_gas_ppm"]].copy()
    results["anomaly_label"] = anomaly_labels
    results["anomaly_score"] = scores
    results.to_csv(os.path.join(output_dir, "anomaly_predictions.csv"), index=False)
    print(f"Predictions saved: {output_dir}/anomaly_predictions.csv")

    return model, metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/washroom_dataset_multi_cubicle.csv")
    parser.add_argument("--output-dir", default="ml/models")
    args = parser.parse_args()
    train_anomaly_model(args.input, args.output_dir)
