"""
Smart Washroom — Virtual Sensing + Predictive Refill (Feature 2, Core Contribution)

Estimates the disinfectant tank level from indirect signals (spray events,
timing, usage history) — no physical sensor exists on this tank.

Compares:
  1. Primary model: Random Forest / Gradient Boosting regressor
  2. Baseline: naive spray_count x avg_volume_per_spray counter (per-cubicle,
     resets at refill events)

The gap between the two is the actual result worth reporting.

*** CORRECTED VERSION - two bugs fixed, both confirmed by Claude via code
    review + empirical replication on an equivalent dataset: ***

1. `prev_disinfectant_level` was populated from the TRUE historical value
   (ground truth) for both training AND testing. In real deployment there is
   no physical sensor, so this value would never be available -- only the
   model's OWN prior prediction would be. This inflated the reported R2.
   Fixed by adding `walk_forward_predict()`, which recursively feeds the
   model's own previous prediction forward at test time. The batch-mode
   evaluation is KEPT (as `evaluate_batch_leaky`) for comparison, but must
   never be reported as the deployment-realistic number on its own.

2. `naive_baseline()` took a raw cumsum of mist_maker_status with no groupby
   on cubicle_id and no reset at refill events -- so it drifted unbounded
   over a multi-refill, multi-cubicle test period and produced a misleadingly
   catastrophic score regardless of how good the trained model is. Fixed to
   reset at each `disinfectant_refill_status` transition, per cubicle, reusing
   the same grouping logic `sprays_since_refill` already uses.

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

    df["hour"] = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute"] = df["timestamp"].dt.minute
    df["day_of_year"] = df["timestamp"].dt.dayofyear

    for cubicle in df["cubicle_id"].unique():
        mask = df["cubicle_id"] == cubicle
        sub = df.loc[mask].copy()

        refill_mask = sub["disinfectant_refill_status"] == 1
        group_ids = (~refill_mask).cumsum()
        df.loc[mask, "sprays_since_refill"] = sub.groupby(group_ids)["mist_maker_status"].cumsum()

        df.loc[mask, "hours_since_last_spray"] = sub["hours_since_seat_spray"]
        df.loc[mask, "cumulative_entry_count"] = sub["entry_count"]
        df.loc[mask, "spray_rate_6h"] = sub["mist_maker_status"].rolling(72, min_periods=1).sum() / 6.0
        df.loc[mask, "spray_rate_24h"] = sub["mist_maker_status"].rolling(288, min_periods=1).sum() / 24.0
        df.loc[mask, "gas_rolling_mean_1h"] = sub["mq135_gas_ppm"].rolling(12, min_periods=1).mean()
        df.loc[mask, "gas_rolling_max_6h"] = sub["mq135_gas_ppm"].rolling(72, min_periods=1).max()
        df.loc[mask, "occ_rate_1h"] = sub["occupancy_ld2410"].rolling(12, min_periods=1).mean()
        df.loc[mask, "hours_since_deep_clean_val"] = sub["hours_since_deep_clean"]

        # NOTE: this lag feature is still computed from the TRUE value here, because
        # it is needed for TRAINING (the model must learn from real history) and for
        # seeding the walk-forward loop's first row. It must NOT be used this way at
        # test-time prediction -- see walk_forward_predict() below, which overwrites
        # this column with the model's own recursive predictions instead.
        df.loc[mask, "prev_disinfectant_level"] = sub["disinfectant_level_virtual_pct"].shift(1)
        df.loc[mask, "prev_disinfectant_level"] = df.loc[mask, "prev_disinfectant_level"].ffill().fillna(100.0)

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


def naive_baseline(df: pd.DataFrame, avg_volume_per_spray: float = 0.25) -> np.ndarray:
    """
    FIXED naive baseline: 100 - (sprays since last refill x avg_volume),
    reset to 100 at each observed refill event, computed PER CUBICLE.

    The original version took a raw, ungrouped cumsum of mist_maker_status
    across the whole (possibly multi-cubicle) test set with no refill reset,
    so it drifted unbounded and produced a misleadingly catastrophic score
    regardless of the trained model's quality. This version reuses the same
    "reset at refill" grouping that `sprays_since_refill` already computes,
    so the comparison against the trained model is actually fair.
    """
    df = df.sort_values(["cubicle_id", "timestamp"])
    estimate = np.zeros(len(df))
    for cubicle in df["cubicle_id"].unique():
        mask = (df["cubicle_id"] == cubicle).values
        sub = df.loc[mask]
        refill_mask = sub["disinfectant_refill_status"] == 1
        group_ids = (~refill_mask).cumsum()
        sprays_since_refill = sub.groupby(group_ids)["mist_maker_status"].cumsum()
        estimate[mask] = np.clip(100.0 - sprays_since_refill.values * avg_volume_per_spray, 0, 100)
    return estimate


def evaluate(y_true, y_pred, model_name="model"):
    """Compute regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  {model_name}: MAE={mae:.3f} | RMSE={rmse:.3f} | R2={r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def evaluate_batch_leaky(model, test_df, feature_cols):
    """
    Original batch-mode prediction. KEPT for comparison only -- this feeds the
    model the TRUE prev_disinfectant_level at every row, which is not available
    in real deployment. Never report this number alone as the headline result;
    always report it alongside walk_forward_predict()'s honest number.
    """
    return model.predict(test_df[feature_cols])


def walk_forward_predict(model, test_df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """
    Honest evaluation: prev_disinfectant_level is seeded ONCE per cubicle from
    the last known true value at the train/test boundary (reasonable -- a real
    system knows the level at commissioning/restart), then the model's OWN
    prediction is fed forward as "prev_disinfectant_level" for every
    subsequent row. This is what the model will actually have access to in
    the field, since there is no physical sensor for ground truth.

    Slower than batch prediction (one predict() call per row) -- for a large
    test set, predict in per-cubicle chunks as done here, and consider
    vectorizing further (e.g. re-fit-free tree traversal) if this becomes a
    bottleneck on the full dataset.
    """
    test_df = test_df.sort_values(["cubicle_id", "timestamp"]).reset_index(drop=True)
    preds = np.zeros(len(test_df))
    prev_idx = feature_cols.index("prev_disinfectant_level")

    for cubicle in test_df["cubicle_id"].unique():
        idx = test_df.index[test_df["cubicle_id"] == cubicle].tolist()
        X_cub = test_df.loc[idx, feature_cols].to_numpy(dtype=float)
        prev_level = float(test_df.loc[idx[0], "prev_disinfectant_level"])  # one-time true seed
        for row_i, global_i in enumerate(idx):
            X_cub[row_i, prev_idx] = prev_level
            pred = float(model.predict(X_cub[row_i:row_i + 1])[0])
            preds[global_i] = pred
            prev_level = pred  # KEY: feed the model's OWN prediction forward
    return preds


def train_virtual_sensing(csv_path: str, output_dir: str = "ml/models", walk_forward_sample: int = None):
    """
    Train and save virtual sensing models.

    walk_forward_sample: if set, run the honest walk-forward evaluation on
    only this many rows per cubicle (for speed) instead of the full test set.
    Set to None to run walk-forward on the entire test set (slow, but this is
    the number that should ultimately be reported).
    """
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

    models = build_models()
    results = {}
    all_preds = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        if GPU_AVAILABLE:
            X_train_gpu = to_cudf(train_df[feature_cols])
            y_train_gpu = to_cudf(train_df[[TARGET]])[TARGET]
            model.fit(X_train_gpu, y_train_gpu)
        else:
            model.fit(X_train_np, y_train_np)

        # 1) Batch/leaky number -- kept ONLY as a reference point, never the headline
        y_pred_leaky = evaluate_batch_leaky(model, test_df, feature_cols)
        results[f"{name}_LEAKY_reference_only"] = evaluate(y_test_np, y_pred_leaky, f"{name} (leaky, reference only)")

        # 2) Honest walk-forward number -- THIS is the one to report
        wf_df = test_df if walk_forward_sample is None else (
            test_df.groupby("cubicle_id", group_keys=False).head(walk_forward_sample)
        )
        y_pred_wf = walk_forward_predict(model, wf_df, feature_cols)
        y_true_wf = wf_df.sort_values(["cubicle_id", "timestamp"])[TARGET].to_numpy()
        results[name] = evaluate(y_true_wf, y_pred_wf, f"{name} (walk-forward, HONEST)")
        all_preds[name] = y_pred_wf.copy()

    # Naive baseline -- FIXED (per-cubicle, refill-reset)
    print("\nNaive baseline (fixed: per-cubicle, resets at refill)...")
    y_baseline = naive_baseline(test_df)
    results["NaiveBaseline_fixed"] = evaluate(y_test_np, y_baseline, "NaiveBaseline (fixed)")

    best_name = min(
        (k for k in results if not k.endswith("_LEAKY_reference_only") and k != "NaiveBaseline_fixed"),
        key=lambda k: results[k]["MAE"],
    )
    best_model = models[best_name]
    print(f"\nBest model (by honest walk-forward MAE): {best_name}")

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
        "evaluation_note": (
            "'*_LEAKY_reference_only' uses true prev_disinfectant_level at test time and "
            "should never be reported alone -- it is not achievable in real deployment. "
            "The unsuffixed model result is honest walk-forward evaluation."
        ),
    }
    joblib.dump(metadata, os.path.join(output_dir, "virtual_sensing_meta.joblib"))

    preds_df = test_df[["timestamp", "cubicle_id", TARGET]].copy()
    preds_df["naive_baseline_fixed"] = y_baseline
    preds_df.to_csv(os.path.join(output_dir, "virtual_sensing_predictions.csv"), index=False)

    print(f"\nAll models saved to {output_dir}/")
    for name, res in results.items():
        print(f"  {name}: MAE={res['MAE']:.3f} R2={res['R2']:.4f}")

    return best_model, metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/washroom_dataset_multi_cubicle.csv")
    parser.add_argument("--output-dir", default="ml/models")
    parser.add_argument("--walk-forward-sample", type=int, default=None,
                         help="Rows per cubicle for walk-forward eval (omit for full test set -- slow but correct)")
    args = parser.parse_args()
    train_virtual_sensing(args.input, args.output_dir, args.walk_forward_sample)