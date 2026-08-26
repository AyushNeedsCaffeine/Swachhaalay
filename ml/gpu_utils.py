"""
GPU Acceleration Utilities for Smart Washroom ML Pipeline.

Provides a unified interface that uses GPU (cuML/cuDF/CuPy) when available,
falling back to CPU (scikit-learn/pandas/NumPy) automatically.
"""

import warnings

GPU_AVAILABLE = False
_CUML_IMPORT_ERROR = None

try:
    import cuml
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
    warnings.filterwarnings("ignore", category=UserWarning, module="cuml")
except ImportError as e:
    _CUML_IMPORT_ERROR = str(e)
    import numpy as cp  # fallback: NumPy acts as cupy
    import pandas as pd


def is_gpu_available() -> bool:
    """Check if GPU libraries are loaded."""
    return GPU_AVAILABLE


def get_device_info() -> str:
    """Return a string describing the active compute device."""
    if GPU_AVAILABLE:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            gpu_info = result.stdout.strip() if result.returncode == 0 else "Unknown GPU"
            return f"GPU: {gpu_info} (cuML {cuml.__version__})"
        except Exception:
            return f"GPU: cuML {cuml.__version__}"
    return f"CPU: scikit-learn fallback ({_CUML_IMPORT_ERROR or 'cuML not installed'})"


def load_data(filepath: str):
    """
    Load CSV data into a DataFrame.
    Uses cuDF on GPU, pandas on CPU.
    Returns a pandas-compatible DataFrame (cuDF DataFrame is pandas-compatible).
    """
    if GPU_AVAILABLE:
        df = cudf.read_csv(filepath)
        print(f"[GPU] Loaded {len(df):,} rows into cuDF DataFrame")
    else:
        df = pd.read_csv(filepath)
        print(f"[CPU] Loaded {len(df):,} rows into pandas DataFrame")
    return df


def to_numpy(df_or_series):
    """Convert cuDF/pandas object to NumPy array."""
    if GPU_AVAILABLE:
        return cp.asnumpy(df_or_series.values) if hasattr(df_or_series, 'values') else cp.asnumpy(df_or_series)
    return df_or_series.values if hasattr(df_or_series, 'values') else df_or_series


def train_test_split_time(df, time_col="timestamp", train_ratio=0.69):
    """
    Time-based train/test split.
    First `train_ratio` fraction of time range = train, rest = test.
    This prevents data leakage from adjacent 5-minute readings.
    """
    if GPU_AVAILABLE:
        df_pd = df.to_pandas()
    else:
        df_pd = df.copy() if hasattr(df, 'copy') else df

    df_pd[time_col] = pd.to_datetime(df_pd[time_col])
    t_min = df_pd[time_col].min()
    t_max = df_pd[time_col].max()
    t_split = t_min + (t_max - t_min) * train_ratio

    train_mask = df_pd[time_col] <= t_split
    test_mask = df_pd[time_col] > t_split

    if GPU_AVAILABLE:
        train_df = cudf.from_pandas(df_pd[train_mask])
        test_df = cudf.from_pandas(df_pd[test_mask])
    else:
        train_df = df_pd[train_mask]
        test_df = df_pd[test_mask]

    print(f"Time-based split at {t_split}: train={len(train_df):,}, test={len(test_df):,}")
    return train_df, test_df
