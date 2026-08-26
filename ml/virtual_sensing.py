"""
Smart Washroom — Virtual Sensing + Predictive Refill (Feature 2, Core Contribution)

Estimates the disinfectant tank level from indirect signals (spray events,
timing, usage history) — no physical sensor exists on this tank.

Compares:
  1. Primary model: Random Forest / Gradient Boosting regressor
  2. Baseline: naive spray_count × avg_volume_per_spray counter

The gap between the two is the actual result worth reporting.

Uses cuML (GPU) when available, falls back to scikit-learn (CPU).
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.gpu_utils import GPU_AVAILABLE, load_data, to_numpy, to_cudf, train_test_split_time


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for virtual disinfectant level prediction."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["cubicle_id", "timestamp"]).reset_index(drop=True)

    # Time encoding
    df["hour"] = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute"] = df["timestamp"].dt.minute
    df["day_of_year"] = df["timestamp"].dt.dayofyear

    # Key insight: the target IS in the dataset (disinfectant_level_virtual_pct).
    # We predict from features available BEFORE the current reading to avoid leakage.
    # Shift the target back by 1 step so we predict the NEXT level from current features.

    for cubicle in df["cubicle_id"].unique():
        mask = df["cubicle_id"] == cubicle
        sub = df.loc[mask].copy()

        # Cumulative spray count since last refill
        refill_mask = sub["disinfectant_refill_status"] == 1
        group_ids = (~refill_mask).cumsum()
        df.loc[mask, "sprays_since_refill"] = sub.groupby(group_ids)["mist_maker_status"].cumsum()

        # Time since last spray (from the data column)
        df.loc[mask, "hours_since_last_spray"] = sub["hours_since_seat_spray"]

        # Cumulative usage
        df.loc[mask, "cumulative_entry_count"] = sub["entry_count"]

        # Rolling spray intensity
        df.loc[mask, "spray_rate_6h"] = sub["mist_maker_status"].rolling(72, min_periods=1).sum() / 6.0
        df.loc[mask, "spray_rate_24h"] = sub["mist_maker_status"].rolling(288, min_periods=1).sum() / 24.0

        # Gas dynamics (proxy for usage intensity)
        df.loc[mask, "gas_rolling_mean_1h"] = sub["mq135_gas_ppm"].rolling(12, min_periods=1).mean()
        df.loc[mask, "gas_rolling_max_6h"] = sub["mq135_gas_ppm"].rolling(72, min_periods=1).max()

        # Occupancy cumulative
        df.loc[mask, "occ_rate_1h"] = sub["occupancy_ld2410"].rolling(12, min_periods=1).mean()

        # Deep clean recency
        df.loc[mask, "hours_since_deep_clean_val"] = sub["hours_since_deep_clean"]

        # Previous disinfectant level (lagged by 1 step, but within same cubicle)
        # This creates a "walk-forward" prediction: predict level[t] from features at [t-1]
        df.loc[mask, "prev_disinfectant_level"] = sub["disinfectant_level_virtual_pct"].shift(1)
        df.loc[mask, "prev_disinfectant_level"] = df.loc[mask, "prev_disinfectant_level"].ffill().fillna(100.0)

        # Change in disinfectant level (lagged)
        df.loc[mask, "disinfectant_level_diff"] = sub["disinfectant_level_virtual_pct"].diff().fillna(0)

    feature_cols = [
        "prev_disinfectant_level",
        "sprays_since_refill",
        "hours_since_last_spray",
        "cumulative_entry_count",
        "spray_rate_6h",
        "spray_rate_24h",
        "gas_rolling_mean_1h",
        "gas_rolling_max_6h",
        "occ_rate_1h",
        "hours_since_deep_clean_val",
        "hour_sin",
        "hour_cos",
        "mist_maker_status",
    ]
    return df, feature_cols


TARGET = "disinfectant_level_virtual_pct"


def build_models():
    """Build primary and ensemble models (GPU or CPU)."""
    models = {}

    if GPU_AVAILABLE:
        from cuml.ensemble import RandomForestRegressor
        from cuml.linear_model import LinearRegression
        models["RandomForest"] = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            random_state=42, n_streams=2,
        )
        print("[GPU] Using cuML Random Forest Regressor")
    else:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        models["RandomForest"] = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            random_state=42, n_jobs=-1,
        )
        models["GradientBoosting"] = GradientBoostingRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )
        print("[CPU] Using scikit-learn RF + GBR")

    return models


def naive_baseline(y_true, spray_counts_per_row, avg_volume_per_spray=0.25):
    """
    Naive baseline: estimate = 100 - (cumulative_sprays × avg_volume).
    avg_volume_per_spray = 0.25% per spray event (tunable).
    """
    cumulative_sprays = spray_counts_per_row.cumsum()
    estimate = np.clip(100.0 - cumulative_sprays * avg_volume_per_spray, 0, 100)
    return estimate


def evaluate(y_true, y_pred, model_name="model"):
    """Compute regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  {model_name}: MAE={mae:.3f} | RMSE={rmse:.3f} | R²={r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def train_virtual_sensing(csv_path: str, output_dir: str = "ml/models"):
    """Train and save virtual sensing models."""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading dataset...")
    df = load_data(csv_path)

    print("Preparing features...")
    df_feat, feature_cols = prepare_features(df)

    print("Splitting data (time-based)...")
    train_df, test_df = train_test_split_time(df_feat)

    X_train_np = to_numpy(train_df[feature_cols])
    y_train_np = to_numpy(train_df[TARGET])
    X_test_np = to_numpy(test_df[feature_cols])
    y_test_np = to_numpy(test_df[TARGET])

    print(f"Training: {X_train_np.shape}, Test: {X_test_np.shape}")

    # Train primary models
    models = build_models()
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        if GPU_AVAILABLE:
            X_train_gpu = to_cudf(train_df[feature_cols])
            y_train_gpu = to_cudf(train_df[[TARGET]])[TARGET]
            model.fit(X_train_gpu, y_train_gpu)
            X_test_gpu = to_cudf(test_df[feature_cols])
            y_pred = to_numpy(model.predict(X_test_gpu))
        else:
            model.fit(X_train_np, y_train_np)
            y_pred = model.predict(X_test_np)
        results[name] = evaluate(y_test_np, y_pred, name)

    # Naive baseline
    print("\nNaive baseline...")
    y_test_sprays = to_numpy(test_df[["mist_maker_status"]])
    y_baseline = naive_baseline(y_test_np, y_test_sprays)
    results["NaiveBaseline"] = evaluate(y_test_np, y_baseline, "NaiveBaseline")

    # Save best model
    best_name = min(results, key=lambda k: results[k]["MAE"])
    best_model = models[best_name]
    print(f"\nBest model: {best_name}")

    model_path = os.path.join(output_dir, "virtual_sensing.joblib")
    joblib.dump(best_model, model_path)

    metadata = {
        "model_type": best_name,
        "features": feature_cols,
        "target": TARGET,
        "results": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in results.items()},
        "train_size": len(train_df),
        "test_size": len(test_df),
        "gpu_accelerated": GPU_AVAILABLE,
    }
    joblib.dump(metadata, os.path.join(output_dir, "virtual_sensing_meta.joblib"))

    # Save predictions for dashboard
    preds_df = test_df[["timestamp", "cubicle_id", TARGET]].copy()
    preds_df["predicted_level"] = y_pred
    preds_df["naive_baseline"] = y_baseline
    preds_df.to_csv(os.path.join(output_dir, "virtual_sensing_predictions.csv"), index=False)

    print(f"\nAll models saved to {output_dir}/")
    for name, res in results.items():
        print(f"  {name}: MAE={res['MAE']:.3f} R²={res['R2']:.4f}")

    return best_model, metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/washroom_dataset_multi_cubicle.csv")
    parser.add_argument("--output-dir", default="ml/models")
    args = parser.parse_args()
    train_virtual_sensing(args.input, args.output_dir)
