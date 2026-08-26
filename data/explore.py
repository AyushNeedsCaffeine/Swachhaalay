"""
Smart Washroom — Exploratory Data Analysis (EDA)

Generates statistics, distribution plots, correlation heatmaps,
and class-balance reports from the washroom dataset.

Usage:
    python data/explore.py
    python data/explore.py --input data/washroom_dataset_multi_cubicle.csv --output-dir outputs/
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load and validate the washroom dataset."""
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"Cubicles: {df['cubicle_id'].unique().tolist()}")
    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic statistics per column."""
    stats = df.describe(include="all").T
    stats["dtype"] = df.dtypes
    stats["nulls"] = df.isnull().sum()
    stats["unique"] = df.nunique()
    return stats


def class_balance(df: pd.DataFrame) -> dict:
    """Analyze needs_cleaning class distribution overall and per cubicle."""
    results = {}
    for cubicle in sorted(df["cubicle_id"].unique()):
        sub = df[df["cubicle_id"] == cubicle]
        counts = sub["needs_cleaning"].value_counts()
        total = len(sub)
        pct_1 = counts.get(1, 0) / total * 100
        results[cubicle] = {
            "total_rows": total,
            "needs_cleaning=0": int(counts.get(0, 0)),
            "needs_cleaning=1": int(counts.get(1, 0)),
            "positive_pct": round(pct_1, 2),
        }
    overall = df["needs_cleaning"].value_counts()
    results["OVERALL"] = {
        "total_rows": len(df),
        "needs_cleaning=0": int(overall.get(0, 0)),
        "needs_cleaning=1": int(overall.get(1, 0)),
        "positive_pct": round(overall.get(1, 0) / len(df) * 100, 2),
    }
    return results


def occupancy_stats(df: pd.DataFrame) -> dict:
    """Analyze occupancy patterns per cubicle."""
    results = {}
    for cubicle in sorted(df["cubicle_id"].unique()):
        sub = df[df["cubicle_id"] == cubicle]
        occ_pct = sub["occupancy_ld2410"].mean() * 100
        motion_pct = sub["motion_ir"].mean() * 100
        total_entries = sub["entry_count"].max()
        results[cubicle] = {
            "occupancy_pct": round(occ_pct, 2),
            "motion_pct": round(motion_pct, 2),
            "max_daily_entries": int(total_entries),
            "avg_gas_ppm": round(sub["mq135_gas_ppm"].mean(), 1),
            "max_gas_ppm": round(sub["mq135_gas_ppm"].max(), 1),
            "avg_hygiene_score": round(sub["hygiene_score"].mean(), 1),
            "total_sprays": int(sub["mist_maker_status"].sum()),
            "total_checkups": int(sub["needs_manual_checkup"].sum()),
        }
    return results


def plot_distributions(df: pd.DataFrame, output_dir: str):
    """Plot feature distributions per cubicle."""
    numeric_cols = [
        "mq135_gas_ppm", "hygiene_score", "entry_count",
        "hours_since_seat_spray", "hours_since_deep_clean",
        "water_level_cm", "disinfectant_level_virtual_pct",
    ]
    cubicles = sorted(df["cubicle_id"].unique())
    n_cubicles = len(cubicles)

    fig, axes = plt.subplots(len(numeric_cols), n_cubicles, figsize=(4 * n_cubicles, 3 * len(numeric_cols)))
    if len(numeric_cols) == 1:
        axes = axes.reshape(1, -1)
    if n_cubicles == 1:
        axes = axes.reshape(-1, 1)

    for i, col in enumerate(numeric_cols):
        for j, cubicle in enumerate(cubicles):
            ax = axes[i, j]
            data = df[df["cubicle_id"] == cubicle][col]
            ax.hist(data, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
            if i == 0:
                ax.set_title(cubicle.replace("Cubicle_", "").replace("_", "\n"), fontsize=9)
            if j == 0:
                ax.set_ylabel(col, fontsize=8)
            ax.tick_params(labelsize=7)

    plt.suptitle("Feature Distributions by Cubicle", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str):
    """Plot correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    # Drop timestamp-based if any slipped in
    corr = numeric_df.corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                annot_kws={"size": 7})
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_timeseries(df: pd.DataFrame, output_dir: str):
    """Plot time series of key features for each cubicle (first 7 days)."""
    cubicles = sorted(df["cubicle_id"].unique())
    week_one = df[df["timestamp"] < df["timestamp"].min() + pd.Timedelta(days=7)]

    fig, axes = plt.subplots(len(cubicles), 1, figsize=(16, 4 * len(cubicles)), sharex=True)
    if len(cubicles) == 1:
        axes = [axes]

    for i, cubicle in enumerate(cubicles):
        ax = axes[i]
        data = week_one[week_one["cubicle_id"] == cubicle].set_index("timestamp")
        ax.plot(data.index, data["mq135_gas_ppm"], label="Gas PPM", alpha=0.8, linewidth=0.8)
        ax.plot(data.index, data["hygiene_score"], label="Hygiene Score", alpha=0.8, linewidth=0.8)
        ax.fill_between(data.index, 0, data["occupancy_ld2410"] * 100, alpha=0.15, label="Occupied")
        spray_times = data[data["mist_maker_status"] == 1].index
        ax.scatter(spray_times, [50] * len(spray_times), c="red", s=3, zorder=5, label="Spray")
        ax.set_ylabel(cubicle.replace("Cubicle_", ""), fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_ylim(0, 360)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Time Series — First 7 Days", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "timeseries_first_week.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_hourly_heatmap(df: pd.DataFrame, output_dir: str):
    """Plot hourly usage heatmap per cubicle."""
    df_copy = df.copy()
    df_copy["hour"] = df_copy["timestamp"].dt.hour
    df_copy["day_of_week"] = df_copy["timestamp"].dt.dayofweek

    cubicles = sorted(df_copy["cubicle_id"].unique())
    fig, axes = plt.subplots(1, len(cubicles), figsize=(5 * len(cubicles), 5))
    if len(cubicles) == 1:
        axes = [axes]

    for i, cubicle in enumerate(cubicles):
        sub = df_copy[df_copy["cubicle_id"] == cubicle]
        pivot = sub.groupby(["day_of_week", "hour"])["occupancy_ld2410"].mean().unstack()
        sns.heatmap(pivot, ax=axes[i], cmap="YlOrRd", vmin=0, vmax=0.8,
                    cbar_kws={"label": "Occupancy Rate"})
        axes[i].set_title(cubicle.replace("Cubicle_", ""), fontsize=10)
        axes[i].set_xlabel("Hour of Day")
        axes[i].set_ylabel("Day of Week")
        axes[i].set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fontsize=8)

    plt.suptitle("Hourly Occupancy Heatmap by Cubicle", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "hourly_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def print_report(df: pd.DataFrame):
    """Print a comprehensive EDA report to stdout."""
    print("\n" + "=" * 70)
    print("SMART WASHROOM — EXPLORATORY DATA ANALYSIS REPORT")
    print("=" * 70)

    print(f"\nDataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    print(f"Unique cubicles: {df['cubicle_id'].nunique()}")
    print(f"Rows per cubicle: {df.groupby('cubicle_id').size().to_dict()}")

    print("\n--- Class Balance (needs_cleaning) ---")
    balance = class_balance(df)
    for name, info in balance.items():
        print(f"  {name}: {info['needs_cleaning=1']}/{info['total_rows']} ({info['positive_pct']}% positive)")

    print("\n--- Occupancy & Usage Statistics ---")
    occ = occupancy_stats(df)
    for name, info in occ.items():
        print(f"\n  {name}:")
        for k, v in info.items():
            print(f"    {k}: {v}")

    print("\n--- Null Check ---")
    nulls = df.isnull().sum()
    if nulls.any():
        print(nulls[nulls > 0])
    else:
        print("  No null values found.")

    print("\n--- Value Ranges ---")
    for col in ["mq135_gas_ppm", "hygiene_score", "water_level_cm",
                 "disinfectant_level_virtual_pct", "entry_count"]:
        print(f"  {col}: [{df[col].min()}, {df[col].max()}]")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Smart Washroom EDA")
    parser.add_argument("--input", default="data/washroom_dataset_multi_cubicle.csv")
    parser.add_argument("--output-dir", default="outputs/eda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_dataset(args.input)
    print_report(df)

    print("\nGenerating plots...")
    plot_distributions(df, args.output_dir)
    plot_correlation_heatmap(df, args.output_dir)
    plot_timeseries(df, args.output_dir)
    plot_hourly_heatmap(df, args.output_dir)

    # Save stats to CSV
    stats = basic_stats(df)
    stats.to_csv(os.path.join(args.output_dir, "basic_stats.csv"))
    print(f"Saved: {args.output_dir}/basic_stats.csv")

    # Save balance report
    balance = class_balance(df)
    pd.DataFrame(balance).T.to_csv(os.path.join(args.output_dir, "class_balance.csv"))
    print(f"Saved: {args.output_dir}/class_balance.csv")

    print("\nEDA complete!")


if __name__ == "__main__":
    main()
