"""
GPU Acceleration Utilities for Smart Washroom ML Pipeline.

Provides a unified interface that uses GPU (cuML/cuDF/CuPy) when available,
falling back to CPU (scikit-learn/pandas/NumPy) automatically.
"""

import warnings
import numpy as np
import pandas as pd

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
    """Load CSV data into a pandas DataFrame."""
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    print(f"Loaded {len(df):,} rows into pandas DataFrame")
    return df


def to_numpy(df_or_series):
    """Convert DataFrame/Series to NumPy array."""
    if hasattr(df_or_series, 'values'):
        vals = df_or_series.values
        return vals.get() if hasattr(vals, 'get') else vals  # cupy -> numpy
    if hasattr(df_or_series, 'to_numpy'):
        return df_or_series.to_numpy()
    return np.asarray(df_or_series)


def to_cudf(df: pd.DataFrame):
    """Convert pandas DataFrame to cuDF (for GPU model training)."""
    if GPU_AVAILABLE:
        return cudf.from_pandas(df)
    return df


def train_test_split_time(df, time_col="timestamp", train_ratio=0.69):
    """
    Time-based train/test split.
    First `train_ratio` fraction of time range = train, rest = test.
    This prevents data leakage from adjacent 5-minute readings.
    """
    df_pd = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df_pd[time_col]):
        df_pd[time_col] = pd.to_datetime(df_pd[time_col])

    t_min = df_pd[time_col].min()
    t_max = df_pd[time_col].max()
    t_split = t_min + (t_max - t_min) * train_ratio

    train_mask = df_pd[time_col] <= t_split
    test_mask = df_pd[time_col] > t_split

    train_df = df_pd[train_mask].reset_index(drop=True)
    test_df = df_pd[test_mask].reset_index(drop=True)

    print(f"Time-based split at {t_split}: train={len(train_df):,}, test={len(test_df):,}")
    return train_df, test_df
