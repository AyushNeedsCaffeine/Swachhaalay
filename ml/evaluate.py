"""
Smart Washroom — Model Evaluation

Generates comparison tables, performance metrics, and figures
for the research paper and patent documentation.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.gpu_utils import GPU_AVAILABLE


def evaluate_anomaly_detection(models_dir: str = "ml/models", output_dir: str = "outputs/figures"):
    """Evaluate and visualize anomaly detection performance."""
    os.makedirs(output_dir, exist_ok=True)
    meta = joblib.load(os.path.join(models_dir, "anomaly_detection_meta.joblib"))
    preds = pd.read_csv(os.path.join(models_dir, "anomaly_predictions.csv"))

    print("\n--- Anomaly Detection ---")
    print(f"Model: {meta['model_type']}")
    print(f"Anomaly rate in test set: {meta['anomaly_rate_test']}%")

    # Per-cubicle anomaly distribution
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for i, cubicle in enumerate(sorted(preds["cubicle_id"].unique())):
        sub = preds[preds["cubicle_id"] == cubicle]
        ax = axes[i]
        ax.hist(sub[sub["anomaly_label"] == 0]["mq135_gas_ppm"], bins=30, alpha=0.6, label="Normal", color="steelblue")
        ax.hist(sub[sub["anomaly_label"] == 1]["mq135_gas_ppm"], bins=30, alpha=0.6, label="Anomaly", color="red")
        ax.set_title(cubicle.replace("Cubicle_", ""), fontsize=9)
        ax.set_xlabel("Gas PPM")
        if i == 0:
            ax.set_ylabel("Count")
        ax.legend(fontsize=7)
    plt.suptitle("Anomaly Detection — Gas PPM Distribution", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "anomaly_detection.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Anomaly score distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(preds["anomaly_score"], bins=50, color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.5)
    threshold = preds[preds["anomaly_label"] == 1]["anomaly_score"].max()
    ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold: {threshold:.3f}")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title("Anomaly Score Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "anomaly_scores.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return meta


def evaluate_virtual_sensing(models_dir: str = "ml/models", output_dir: str = "outputs/figures"):
    """Evaluate and visualize virtual sensing performance."""
    os.makedirs(output_dir, exist_ok=True)
    meta = joblib.load(os.path.join(models_dir, "virtual_sensing_meta.joblib"))
    preds = pd.read_csv(os.path.join(models_dir, "virtual_sensing_predictions.csv"))

    print("\n--- Virtual Sensing ---")
    print(f"Best model: {meta['model_type']}")
    for name, metrics in meta["results"].items():
        print(f"  {name}: MAE={metrics['MAE']:.3f} RMSE={metrics['RMSE']:.3f} R²={metrics['R2']:.4f}")

    # Model vs Naive comparison table
    results_df = pd.DataFrame(meta["results"]).T
    results_df.to_csv(os.path.join(output_dir, "virtual_sensing_results.csv"))
    print(f"\nResults table saved: {output_dir}/virtual_sensing_results.csv")

    # Predicted vs Actual scatter plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for i, cubicle in enumerate(sorted(preds["cubicle_id"].unique())):
        sub = preds[preds["cubicle_id"] == cubicle]
        ax = axes[i]
        ax.scatter(sub["disinfectant_level_virtual_pct"], sub["predicted_level"],
                   s=2, alpha=0.3, color="steelblue")
        lims = [0, 100]
        ax.plot(lims, lims, "r--", linewidth=1, label="Perfect")
        ax.set_title(cubicle.replace("Cubicle_", ""), fontsize=9)
        ax.set_xlabel("Actual %")
        if i == 0:
            ax.set_ylabel("Predicted %")
        ax.legend(fontsize=7)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    plt.suptitle(f"Virtual Sensing — Predicted vs Actual ({meta['model_type']})", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "virtual_sensing_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Model vs Naive time series (first cubicle)
    first_cubicle = preds["cubicle_id"].unique()[0]
    sub = preds[preds["cubicle_id"] == first_cubicle].head(500)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(len(sub)), sub["disinfectant_level_virtual_pct"], label="Actual", linewidth=1.5)
    ax.plot(range(len(sub)), sub["predicted_level"], label=f"{meta['model_type']}", linewidth=1.5, alpha=0.8)
    ax.plot(range(len(sub)), sub["naive_baseline"], label="Naive Baseline", linewidth=1.5, alpha=0.8, linestyle="--")
    ax.set_xlabel("Reading Index")
    ax.set_ylabel("Disinfectant Level (%)")
    ax.set_title(f"Virtual Sensing — {first_cubicle} (First 500 readings)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "virtual_sensing_timeseries.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return meta


def evaluate_clustering(models_dir: str = "ml/models", output_dir: str = "outputs/figures"):
    """Evaluate and visualize clustering results."""
    os.makedirs(output_dir, exist_ok=True)
    meta = joblib.load(os.path.join(models_dir, "usage_clustering_meta.joblib"))

    print("\n--- Usage Clustering ---")
    print(f"K = {meta['k']}")
    for name, label in meta["cluster_labels"].items():
        print(f"  {name} → Cluster {label}")

    return meta


def generate_comparison_table(models_dir: str = "ml/models", output_dir: str = "outputs/figures"):
    """Generate a summary comparison table for the paper."""
    os.makedirs(output_dir, exist_ok=True)

    try:
        anomaly_meta = joblib.load(os.path.join(models_dir, "anomaly_detection_meta.joblib"))
        virtual_meta = joblib.load(os.path.join(models_dir, "virtual_sensing_meta.joblib"))
        clustering_meta = joblib.load(os.path.join(models_dir, "usage_clustering_meta.joblib"))
    except FileNotFoundError as e:
        print(f"Missing model metadata: {e}")
        return

    rows = []
    for name, metrics in virtual_meta["results"].items():
        rows.append({
            "Feature": "Virtual Sensing",
            "Model": name,
            "Metric": "MAE",
            "Value": f"{metrics['MAE']:.3f}",
        })
        rows.append({
            "Feature": "Virtual Sensing",
            "Model": name,
            "Metric": "R²",
            "Value": f"{metrics['R2']:.4f}",
        })

    rows.append({
        "Feature": "Anomaly Detection",
        "Model": anomaly_meta["model_type"],
        "Metric": "Anomaly Rate",
        "Value": f"{anomaly_meta['anomaly_rate_test']}%",
    })
    rows.append({
        "Feature": "Usage Clustering",
        "Model": f"K-Means (k={clustering_meta['k']})",
        "Metric": "Silhouette",
        "Value": f"{max(clustering_meta['silhouettes']):.4f}",
    })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    print(f"\nComparison table saved: {output_dir}/model_comparison.csv")
    return df


def main():
    print("=" * 60)
    print("SMART WASHROOM — MODEL EVALUATION")
    print("=" * 60)

    evaluate_anomaly_detection()
    evaluate_virtual_sensing()
    evaluate_clustering()
    generate_comparison_table(models_dir="ml/models")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
