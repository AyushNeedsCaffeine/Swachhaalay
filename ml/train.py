"""
Smart Washroom — Unified Training Script

Trains all 3 ML models in sequence and saves them for the dashboard.
Supports GPU acceleration via cuML.

Usage:
    python ml/train.py
    python ml/train.py --input data/washroom_dataset_multi_cubicle.csv --output-dir ml/models
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml.gpu_utils import GPU_AVAILABLE, get_device_info


def main():
    parser = argparse.ArgumentParser(description="Train all Smart Washroom ML models")
    parser.add_argument("--input", default="data/washroom_dataset_multi_cubicle.csv")
    parser.add_argument("--output-dir", default="ml/models")
    parser.add_argument("--skip-anomaly", action="store_true")
    parser.add_argument("--skip-virtual", action="store_true")
    parser.add_argument("--skip-clustering", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("SMART WASHROOM — ML MODEL TRAINING PIPELINE")
    print("=" * 60)
    print(f"Compute device: {get_device_info()}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}/")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    total_start = time.time()

    # 1. Anomaly Detection
    if not args.skip_anomaly:
        print("\n" + "=" * 40)
        print("FEATURE 1: ANOMALY DETECTION")
        print("=" * 40)
        t0 = time.time()
        from ml.anomaly_detection import train_anomaly_model
        train_anomaly_model(args.input, args.output_dir)
        print(f"  Time: {time.time() - t0:.1f}s")
    else:
        print("\nSkipping anomaly detection...")

    # 2. Virtual Sensing
    if not args.skip_virtual:
        print("\n" + "=" * 40)
        print("FEATURE 2: VIRTUAL SENSING (Core Contribution)")
        print("=" * 40)
        t0 = time.time()
        from ml.virtual_sensing import train_virtual_sensing
        train_virtual_sensing(args.input, args.output_dir)
        print(f"  Time: {time.time() - t0:.1f}s")
    else:
        print("\nSkipping virtual sensing...")

    # 3. Usage Clustering
    if not args.skip_clustering:
        print("\n" + "=" * 40)
        print("FEATURE 4: USAGE CLUSTERING")
        print("=" * 40)
        t0 = time.time()
        from ml.usage_clustering import train_usage_clustering
        train_usage_clustering(args.input, args.output_dir)
        print(f"  Time: {time.time() - t0:.1f}s")
    else:
        print("\nSkipping usage clustering...")

    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"ALL MODELS TRAINED in {total_time:.1f}s")
    print(f"Models saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
