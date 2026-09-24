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
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  {model_name}: MAE={mae:.3f} | RMSE={rmse:.3f} | R2={r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def _predict_row(model, row_np, columns):
    """Predict on a single feature vector, handling cuML (GPU) input format."""
    return _predict_batch(model, row_np.reshape(1, -1), columns)[0]


def _predict_batch(model, X_batch, columns):
    """Predict on a 2D feature matrix, handling cuML (GPU) input format.

    cuML/scikit-learn single-row predict has large fixed per-call overhead
    (~30-200ms), so all honest sequential evaluations feed BATCHES of rows
    (one row per cubicle at each aligned time step) into a single predict()
    call -- this keeps the evaluation honest (never teacher-forced) while
    dropping per-row overhead by orders of magnitude.
    """
    X_batch = np.ascontiguousarray(X_batch, dtype=float)
    if GPU_AVAILABLE:
        from ml.gpu_utils import to_cudf
        import pandas as pd
        return to_numpy(model.predict(to_cudf(pd.DataFrame(X_batch, columns=columns))))
    return model.predict(X_batch)


def persistence_baseline(df: pd.DataFrame) -> np.ndarray:
    """
    Plain persistence baseline: prediction = the last known level, unchanged,
    until an observed refill event resets it to 100. Per cubicle. Seeded once
    from the last known true value at the train/test boundary (the same
    commissioning/restart assumption the walk-forward evaluation makes), then
    carries the last known value forward row by row.
    """
    df = df.sort_values(["cubicle_id", "timestamp"])
    estimate = np.zeros(len(df))
    for cubicle in df["cubicle_id"].unique():
        mask = (df["cubicle_id"] == cubicle).values
        sub = df.loc[mask]
        last_known = float(sub.iloc[0]["prev_disinfectant_level"])
        out = np.empty(mask.sum())
        for i, (_, row) in enumerate(sub.iterrows()):
            if int(row["disinfectant_refill_status"]) == 1:
                last_known = 100.0
            out[i] = last_known
        estimate[mask] = out
    return estimate


def evaluate_batch_leaky(model, test_df, feature_cols):
    """
    Original batch-mode prediction. KEPT for comparison only -- this feeds the
    model the TRUE prev_disinfectant_level at every row, which is not available
    in real deployment. Never report this number alone as the headline result;
    always report it alongside walk_forward_predict()'s honest number.
    """
    return model.predict(test_df[feature_cols])


def walk_forward_predict(model, test_df: pd.DataFrame, feature_cols: list,
                         reset_at_refill: bool = True) -> np.ndarray:
    """
    Honest deployment-style evaluation: `prev_disinfectant_level` is seeded
    ONCE per cubicle from the last known true value at the train/test boundary
    (a real system knows the level at commissioning/restart), then the model's
    OWN prediction is fed forward as `prev_disinfectant_level` for every
    subsequent row.

    `reset_at_refill=True` (default): whenever the dataset records a refill
    event (`disinfectant_refill_status == 1`), the seed is reset to 100.0 --
    a refill is an observed event (staff tops the tank up), so this is
    legitimate, non-teacher-forced information. Kept toggleable so the effect
    of the reset on the numbers can be measured directly.

    All cubicles share the same 5-minute timestamp grid, so this runs one
    aligned time-step at a time: at each step it predicts a batch of one row
    per cubicle whose prev_disinfectant_level comes from that cubicle's own
    prediction chain (never the ground-truth column). This keeps the recursion
    honest while avoiding the ~30-200ms fixed cost of per-row predict calls.

    Returns predictions in the SAME row order as the input `test_df`.
    """
    test_df = test_df.sort_values(["timestamp", "cubicle_id"])  # keep original index
    preds = np.zeros(len(test_df))
    prev_idx = feature_cols.index("prev_disinfectant_level")
    prev_levels = {}
    first_ts = True

    for _, grp in test_df.groupby("timestamp", sort=False):
        X = grp[feature_cols].to_numpy(dtype=float).copy()
        cubs = list(grp["cubicle_id"])
        refills = grp["disinfectant_refill_status"].to_numpy(dtype=int)
        for j, cub in enumerate(cubs):
            if first_ts:
                prev_levels[cub] = float(grp["prev_disinfectant_level"].iloc[j])
            elif reset_at_refill and refills[j] == 1:
                prev_levels[cub] = 100.0  # observed refill -> tank known full
            X[j, prev_idx] = prev_levels[cub]
        batch = _predict_batch(model, X, feature_cols)
        for j, cub in enumerate(cubs):
            row_global = grp.index[j]
            preds[row_global] = batch[j]
            prev_levels[cub] = float(batch[j])  # KEY: feed the model's OWN prediction forward
        first_ts = False
    return preds


def rollout_between_refills_predict(model, test_df: pd.DataFrame, feature_cols: list) -> dict:
    """
    STRICT multi-step rollout: exactly the protocol described in the external
    review. For each cubicle in the test period, seed `prev_disinfectant_level`
    with the true level at each refill event (100.0 after a top-up), then
    predict forward using ONLY the model's own previous predictions (never the
    ground-truth column) until the next refill resets it.

    Rows before a cubicle's FIRST refill inside the test window have no seed
    and are EXCLUDED from this metric (reported as `excluded_rows`). A cubicle
    with no refill at all in the window is dropped from the per-segment totals
    and reported separately.

    Aligned time-step batched (see `_predict_batch`): at each 5-minute step a
    batch of one row per in-segment cubicle is predicted in a single call.

    Returns per-segment, per-cubicle, and overall metrics. This is reported
    SEPARATELY from the one-step (leaky) and walk-forward numbers -- never
    merged with them.
    """
    test_df = test_df.sort_values(["timestamp", "cubicle_id"])  # keep original index
    prev_idx = feature_cols.index("prev_disinfectant_level")
    full_pred = np.full(len(test_df), np.nan)
    prev_levels = {}
    active = {}  # cubicle -> True once inside a scored segment

    for _, grp in test_df.groupby("timestamp", sort=False):
        X = grp[feature_cols].to_numpy(dtype=float)
        cubs = list(grp["cubicle_id"])
        refills = grp["disinfectant_refill_status"].to_numpy(dtype=int)
        positions = []
        keep = np.zeros(len(cubs), dtype=bool)

        for j, cub in enumerate(cubs):
            is_refill = int(refills[j]) == 1
            if is_refill:
                active[cub] = True
                prev_levels[cub] = 100.0  # seed from the true post-refill level (observed event)
            if active.get(cub, False):
                keep[j] = True
                positions.append((j, cub))

        if keep.any():
            batch = _predict_batch(model, X[keep], feature_cols)
            for (j, cub), pred in zip(positions, batch):
                row_global = grp.index[j]
                full_pred[row_global] = pred
                prev_levels[cub] = float(pred)  # only the model's own prediction is fed forward

    # Scored spans = each cubicle's rows from its first test-window refill onward.
    spans = []
    for cubicle in test_df["cubicle_id"].unique():
        sub = test_df[test_df["cubicle_id"] == cubicle]
        refs = sub["disinfectant_refill_status"].to_numpy(dtype=int)
        if refs.sum() == 0:
            continue
        first = int(np.argmax(np.cumsum(refs) > 0))
        spans.append({"cubicle_id": cubicle, "indices": sub.index.to_numpy()[first:]})

    per_cubicle = {}
    all_y, all_p = [], []
    for cubicle in test_df["cubicle_id"].unique():
        mask = (test_df["cubicle_id"] == cubicle).values
        scored = mask & ~np.isnan(full_pred)
        per_cubicle[cubicle] = {
            "n_refills": int((test_df["disinfectant_refill_status"] & mask).sum()),
            "scored_rows": int(scored.sum()),
            "excluded_rows": int(mask.sum() - scored.sum()),
        }
        if scored.any():
            per_cubicle[cubicle].update(evaluate(
                test_df.loc[scored, TARGET].to_numpy(), full_pred[scored],
                f"{cubicle} (rollout between refills)",
            ))
            all_y.append(test_df.loc[scored, TARGET].to_numpy())
            all_p.append(full_pred[scored])

    if not all_y:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "scored_rows": 0,
                "excluded_rows": int(len(test_df)), "n_segments": 0,
                "per_cubicle": per_cubicle, "segments": [], "pred_all": full_pred}

    y_true_all = np.concatenate(all_y)
    y_pred_all = np.concatenate(all_p)
    overall = evaluate(y_true_all, y_pred_all, "ALL (rollout between refills)")
    overall["scored_rows"] = len(y_true_all)
    overall["excluded_rows"] = int(len(test_df) - len(y_true_all))
    overall["n_segments"] = len(spans)
    overall["per_cubicle"] = per_cubicle
    overall["segments"] = spans
    overall["pred_all"] = full_pred  # aligned to input test_df row order
    return overall


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
        all_preds[f"{name}_leaky"] = y_pred_leaky.copy()

        # 2) Honest walk-forward number (own predictions fed forward; refill-reset) -- headline
        wf_df = test_df if walk_forward_sample is None else (
            test_df.groupby("cubicle_id", group_keys=False).head(walk_forward_sample)
        )
        y_pred_wf = walk_forward_predict(model, wf_df, feature_cols, reset_at_refill=True)
        y_true_wf = wf_df.sort_values(["timestamp", "cubicle_id"])[TARGET].to_numpy()
        results[name] = evaluate(y_true_wf, y_pred_wf, f"{name} (walk-forward, HONEST)")
        all_preds[name] = y_pred_wf.copy()

        # 3) STRICT multi-step rollout between refills -- reported separately, never merged
        print(f"  Running strict rollout-between-refills on {name}...")
        wf_full = test_df if walk_forward_sample is None else (
            test_df.groupby("cubicle_id", group_keys=False).head(walk_forward_sample)
        )
        rollout = rollout_between_refills_predict(model, wf_full, feature_cols)
        rollout_flat = {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                        for kk, vv in rollout.items() if kk in ("MAE", "RMSE", "R2")}
        rollout_flat.update({kk: vv for kk, vv in rollout.items()
                             if kk in ("scored_rows", "excluded_rows", "n_segments")})
        rollout_flat["per_cubicle"] = rollout["per_cubicle"]
        results[f"{name}_Rollout_between_refills"] = rollout_flat
        all_preds[f"{name}_rollout"] = rollout

    # Naive baseline -- FIXED (per-cubicle, refill-reset) and persistence baseline
    # (needs test_df feature columns because persistence seeds from prev_disinfectant_level)
    print("\nNaive baseline (fixed: per-cubicle, resets at refill)...")
    y_baseline = naive_baseline(test_df)
    results["NaiveBaseline_fixed"] = evaluate(y_test_np, y_baseline, "NaiveBaseline (fixed)")

    print("Persistence baseline (last known value, reset to 100 at refill)...")
    y_persist = persistence_baseline(test_df)
    results["PersistenceBaseline"] = evaluate(y_test_np, y_persist, "PersistenceBaseline")

    best_name = min(
        (k for k in results if not k.endswith("_LEAKY_reference_only")
         and not k.endswith("_Rollout_between_refills")
         and k not in ("NaiveBaseline_fixed", "PersistenceBaseline")),
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
        "results": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                        for kk, vv in v.items() if kk not in ("per_cubicle",)}
                    for k, v in results.items()},
        "rollout_per_cubicle": {
            best_name: results[f"{best_name}_Rollout_between_refills"].get("per_cubicle", {})
        },
        "train_size": len(train_df),
        "test_size": len(test_df),
        "gpu_accelerated": GPU_AVAILABLE,
        "evaluation_note": (
            "'*_LEAKY_reference_only' feeds the model the TRUE prev_disinfectant_level at "
            "every test row -- not achievable in real deployment -- and must never be "
            "reported alone. The unsuffixed model result is honest walk-forward evaluation "
            "(recursive use of the model's own predictions, reset to 100 only at observed "
            "refill events). '*_Rollout_between_refills' is the strictest number: for each "
            "cubicle, seeded at the true post-refill level and predicted using ONLY the "
            "model's own predictions until the next refill resets it; rows before the first "
            "test-window refill are excluded from it."
        ),
    }
    joblib.dump(metadata, os.path.join(output_dir, "virtual_sensing_meta.joblib"))

    preds_out = test_df[["timestamp", "cubicle_id", TARGET]].copy()
    preds_out["predicted_level"] = all_preds[best_name]
    preds_out["leaky_predicted_level"] = all_preds[f"{best_name}_leaky"]
    preds_out["naive_baseline_fixed"] = y_baseline
    preds_out["persistence_baseline"] = y_persist

    rollout_result = all_preds[f"{best_name}_rollout"]
    preds_out["rollout_between_refills"] = rollout_result["pred_all"]
    preds_out.to_csv(os.path.join(output_dir, "virtual_sensing_predictions.csv"), index=False)

    print(f"\nAll models saved to {output_dir}/")
    for name, res in results.items():
        print(f"  {name}: MAE={res.get('MAE', float('nan')):.3f} R2={res.get('R2', float('nan')):.4f}")

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