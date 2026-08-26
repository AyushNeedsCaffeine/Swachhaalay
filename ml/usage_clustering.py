"""
Smart Washroom — Usage Clustering (Feature 4)

K-Means clustering on hourly traffic profiles to identify
different usage patterns across cubicles.

Uses cuML (GPU) when available, falls back to scikit-learn (CPU).
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.gpu_utils import GPU_AVAILABLE, load_data, to_numpy, to_cudf


def build_hourly_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-cubicle hourly traffic profiles.
    Returns a DataFrame with shape (n_cubicles, 24) — one row per cubicle,
    each cell = average occupancy rate for that hour.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    profiles = df.groupby(["cubicle_id", "hour"])["occupancy_ld2410"].mean().unstack()
    profiles.columns = [f"hour_{h}" for h in range(24)]
    return profiles


def find_optimal_k(X_np: np.ndarray, k_range: range = range(2, 8)):
    """Find optimal k using elbow method + silhouette score."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_np)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_np, labels)
        silhouettes.append(sil)
        print(f"  k={k}: inertia={km.inertia_:.2f}, silhouette={sil:.4f}")

    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"  Best k by silhouette: {best_k}")
    return best_k, inertias, silhouettes


def train_usage_clustering(csv_path: str, output_dir: str = "ml/models"):
    """Train and save the usage clustering model."""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading dataset...")
    df = load_data(csv_path)

    print("Building hourly traffic profiles...")
    profiles = build_hourly_profiles(df)
    X_np = to_numpy(profiles)
    cubicle_names = profiles.index.tolist()

    print(f"Profile matrix: {X_np.shape} (cubicles × hours)")

    # Find optimal k
    print("\nFinding optimal k...")
    k_range = range(2, min(4, len(cubicle_names)))
    best_k, inertias, silhouettes = find_optimal_k(X_np, k_range)

    # Train final model
    print(f"\nTraining K-Means with k={best_k}...")
    if GPU_AVAILABLE:
        from cuml.cluster import KMeans
        model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        X_gpu = to_cudf(profiles)
        labels = to_numpy(model.fit_predict(X_gpu))
        print("[GPU] Using cuML K-Means")
    else:
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = model.fit_predict(X_np)
        print("[CPU] Using scikit-learn K-Means")
    print(f"Cluster assignments: {dict(zip(cubicle_names, labels))}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow plot
    axes[0].plot(list(k_range), inertias, "bo-", linewidth=2)
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")
    axes[0].grid(True, alpha=0.3)

    # Silhouette plot
    axes[1].plot(list(k_range), silhouettes, "rs-", linewidth=2)
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Analysis")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    elbow_path = os.path.join(output_dir, "..", "outputs", "eda", "kmeans_elbow.png")
    os.makedirs(os.path.dirname(elbow_path), exist_ok=True)
    plt.savefig(elbow_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {elbow_path}")

    # Cluster profile visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    hours = list(range(24))
    colors = plt.cm.Set2(np.linspace(0, 1, best_k))

    for i, (name, row) in enumerate(profiles.iterrows()):
        cluster = labels[i]
        row_np = to_numpy(row)
        ax.plot(hours, row_np, marker="o", markersize=4, linewidth=2,
                label=f"{name} (Cluster {cluster})", color=colors[cluster])

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Occupancy Rate")
    ax.set_title("Hourly Traffic Profiles by Cubicle & Cluster")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(hours)
    plt.tight_layout()
    cluster_path = os.path.join(output_dir, "..", "outputs", "eda", "cluster_profiles.png")
    plt.savefig(cluster_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {cluster_path}")

    # Save model and metadata
    model_path = os.path.join(output_dir, "usage_clustering.joblib")
    joblib.dump(model, model_path)

    metadata = {
        "model_type": "KMeans",
        "k": best_k,
        "cubicle_names": cubicle_names,
        "cluster_labels": {name: int(label) for name, label in zip(cubicle_names, labels)},
        "inertias": [round(x, 2) for x in inertias],
        "silhouettes": [round(x, 4) for x in silhouettes],
        "gpu_accelerated": GPU_AVAILABLE,
    }
    joblib.dump(metadata, os.path.join(output_dir, "usage_clustering_meta.joblib"))
    print(f"Model saved: {model_path}")

    return model, metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/washroom_dataset_multi_cubicle.csv")
    parser.add_argument("--output-dir", default="ml/models")
    args = parser.parse_args()
    train_usage_clustering(args.input, args.output_dir)
