"""
Smart Washroom — Tab 2: Predictive Maintenance & ML Insights

Disinfectant forecast, anomaly alerts, sensor agreement panel.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from ml.gpu_utils import GPU_AVAILABLE
from dashboard.utils import load_dataset, load_metadata, get_today_data

MODELS_DIR = "ml/models"


def render():
    """Render the Predictive Maintenance & ML tab."""
    df = load_dataset()
    cubicles = sorted(df["cubicle_id"].unique())

    selected = st.selectbox("Select Cubicle", cubicles, key="pred_cubicle")
    today = get_today_data(df, selected)

    # === Disinfectant Forecast Card ===
    st.subheader("Disinfectant Forecast (Virtual Sensing)")
    vs_meta = load_metadata("virtual_sensing")
    if vs_meta:
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"**Model**: {vs_meta['model_type']}")
            st.info(f"**R² Score**: {vs_meta['results'][vs_meta['model_type']]['R2']:.4f}")
            st.info(f"**MAE**: {vs_meta['results'][vs_meta['model_type']]['MAE']:.3f}%")
        with c2:
            latest = df[df["cubicle_id"] == selected].iloc[-1]
            current_level = latest["disinfectant_level_virtual_pct"]
            spray_rate = today["mist_maker_status"].sum() if not today.empty else 0
            hours_of_data = len(today) * 5 / 60 if not today.empty else 24
            avg_spray_per_hour = spray_rate / hours_of_data if spray_rate > 0 and hours_of_data > 0 else 0.5

            if avg_spray_per_hour > 0:
                hours_remaining = (current_level - 15) / (avg_spray_per_hour * 0.25)
                hours_remaining = max(0, hours_remaining)
            else:
                hours_remaining = 999

            st.metric("Current Level", f"{current_level:.1f}%")
            st.metric("Est. Time to Manual Refill", f"~{hours_remaining:.0f} hours")

        # Model vs Baseline comparison
        st.markdown("**Model vs Naive Baseline Comparison:**")
        results_df = pd.DataFrame(vs_meta["results"]).T
        st.dataframe(results_df.style.format("{:.4f}"), use_container_width=True)
    else:
        st.warning("Virtual sensing model not found. Run `python ml/train.py` first.")

    st.divider()

    # === Anomaly Detection Section ===
    st.subheader("Anomaly Detection (Isolation Forest)")
    ad_meta = load_metadata("anomaly_detection")
    if ad_meta:
        ad_preds_path = os.path.join(MODELS_DIR, "anomaly_predictions.csv")
        if os.path.exists(ad_preds_path):
            ad_preds = pd.read_csv(ad_preds_path, parse_dates=["timestamp"])
            ad_preds = ad_preds[ad_preds["cubicle_id"] == selected]

            c1, c2, c3 = st.columns(3)
            with c1:
                device_label = "GPU" if GPU_AVAILABLE else "CPU"
                st.metric("Model", f"Isolation Forest ({device_label})")
            with c2:
                n_anomalies = ad_preds["anomaly_label"].sum()
                st.metric("Anomalies Detected", n_anomalies)
            with c3:
                rate = n_anomalies / len(ad_preds) * 100 if len(ad_preds) > 0 else 0
                st.metric("Anomaly Rate", f"{rate:.2f}%")

            # Anomaly timeline
            fig = go.Figure()
            normal = ad_preds[ad_preds["anomaly_label"] == 0]
            anomaly = ad_preds[ad_preds["anomaly_label"] == 1]

            fig.add_trace(go.Scatter(
                x=normal["timestamp"], y=normal["mq135_gas_ppm"],
                mode="markers", name="Normal",
                marker=dict(color="steelblue", size=3, opacity=0.5),
            ))
            fig.add_trace(go.Scatter(
                x=anomaly["timestamp"], y=anomaly["mq135_gas_ppm"],
                mode="markers", name="Anomaly",
                marker=dict(color="red", size=5),
            ))
            fig.update_layout(
                xaxis_title="Time", yaxis_title="Gas PPM",
                title=f"Anomaly Detection — {selected}",
                height=350, margin=dict(t=40, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Anomaly detection model not found. Run `python ml/train.py` first.")

    st.divider()

    # === needs_manual_checkup Events ===
    st.subheader("Manual Checkup Events")
    today_data = df[df["cubicle_id"] == selected].tail(288)  # Last day
    checkup_events = today_data[today_data["needs_manual_checkup"] == 1]

    if not checkup_events.empty:
        st.warning(f"⚠️ {len(checkup_events)} checkup events in the last 24h")
        for _, row in checkup_events.head(10).iterrows():
            ts = row["timestamp"].strftime("%Y-%m-%d %H:%M")
            st.markdown(f"- 🔧 **{ts}** — Gas: {row['mq135_gas_ppm']:.0f} ppm, "
                       f"Sprays since refill: {row.get('hours_since_seat_spray', 0):.1f}h")
    else:
        st.success("✅ No manual checkup events in the last 24h.")

    st.divider()

    # === Sensor Agreement Panel ===
    st.subheader("Sensor Agreement (LD2410 vs IR)")
    sub_df = df[df["cubicle_id"] == selected]
    disagree = (
        ((sub_df["occupancy_ld2410"] == 1) & (sub_df["motion_ir"] == 0)) |
        ((sub_df["occupancy_ld2410"] == 0) & (sub_df["motion_ir"] == 1))
    )

    n_disagree = disagree.sum()
    n_total = len(sub_df[sub_df["occupancy_ld2410"] == 1])
    disagree_pct = (n_disagree / n_total * 100) if n_total > 0 else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Occupied Readings", n_total)
    with c2:
        st.metric("Sensor Disagreements", n_disagree)
    with c3:
        st.metric("Disagreement Rate", f"{disagree_pct:.1f}%")

    if disagree_pct > 20:
        st.warning("⚠️ High sensor disagreement rate — possible sensor drift.")

    # Show disagreement examples
    disagree_data = sub_df[disagree].tail(5)
    if not disagree_data.empty:
        st.markdown("**Recent Disagreement Events:**")
        for _, row in disagree_data.iterrows():
            ts = row["timestamp"].strftime("%H:%M")
            ld = "Occupied" if row["occupancy_ld2410"] == 1 else "Vacant"
            ir = "Motion" if row["motion_ir"] == 1 else "No Motion"
            st.markdown(f"- **{ts}** — LD2410: {ld}, IR: {ir}")
