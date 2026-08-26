"""
Smart Washroom — Streamlit Dashboard Utilities

Shared helpers for data loading, model loading, and formatting.
"""

import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np

DATA_PATH = "data/washroom_dataset_multi_cubicle.csv"
MODELS_DIR = "ml/models"


@st.cache_data(ttl=3600)
def load_dataset():
    """Load the washroom dataset (cached for performance)."""
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df


def load_model(name: str):
    """Load a trained model by name."""
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None


def load_metadata(name: str):
    """Load model metadata."""
    path = os.path.join(MODELS_DIR, f"{name}_meta.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None


def get_latest_reading(df: pd.DataFrame, cubicle: str) -> dict:
    """Get the most recent reading for a cubicle."""
    sub = df[df["cubicle_id"] == cubicle]
    if sub.empty:
        return {}
    return sub.iloc[-1].to_dict()


def get_today_data(df: pd.DataFrame, cubicle: str) -> pd.DataFrame:
    """Get today's data for a cubicle (simulated from last 24h of dataset)."""
    sub = df[df["cubicle_id"] == cubicle].copy()
    if sub.empty:
        return sub
    last_date = sub["timestamp"].dt.date.max()
    today = sub[sub["timestamp"].dt.date == last_date]
    return today


def format_percentage(value: float) -> str:
    """Format a value as percentage."""
    return f"{value:.1f}%"


def format_ppm(value: float) -> str:
    """Format gas PPM with color coding."""
    if value < 150:
        return f"🟢 {value:.0f} ppm"
    elif value < 250:
        return f"🟡 {value:.0f} ppm"
    else:
        return f"🔴 {value:.0f} ppm"


def get_health_color(score: float) -> str:
    """Get color for hygiene score."""
    if score >= 70:
        return "normal"
    elif score >= 45:
        return "warning"
    else:
        return "inverse"
