"""
Smart Washroom — Tab 1: Live Monitoring

Real-time status display: occupancy, air quality, tank levels,
mist status, and activity feed.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from dashboard.utils import (
    load_dataset, get_latest_reading, get_today_data,
    format_ppm, get_health_color,
)


def render():
    """Render the Live Monitoring tab."""
    df = load_dataset()
    cubicles = sorted(df["cubicle_id"].unique())

    # Cubicle selector
    selected = st.selectbox("Select Cubicle", cubicles, key="live_cubicle")

    latest = get_latest_reading(df, selected)
    today = get_today_data(df, selected)

    if not latest:
        st.warning("No data available for this cubicle.")
        return

    # === Metrics Row ===
    st.subheader("Current Status")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        today_entries = int(today["entry_count"].max()) if not today.empty else 0
        st.metric("People Count Today", today_entries)

    with c2:
        gas = latest.get("mq135_gas_ppm", 0)
        color = "normal" if gas < 200 else "warning" if gas < 300 else "inverse"
        st.metric("Air Quality", f"{gas:.0f} ppm", delta=None)

    with c3:
        water = latest.get("water_level_cm", 0)
        water_pct = (water / 20.0) * 100
        st.metric("Water Tank", f"{water_pct:.0f}%", delta=None)

    with c4:
        disinfectant = latest.get("disinfectant_level_virtual_pct", 0)
        delta_color = "normal" if disinfectant > 30 else "inverse"
        st.metric("Disinfectant (Virtual)", f"{disinfectant:.1f}%", delta=None,
                   delta_color=delta_color)

    with c5:
        mist = latest.get("mist_maker_status", 0)
        st.metric("Mist Maker", "ACTIVE" if mist else "OFF")

    # === Occupancy Row ===
    st.subheader("Occupancy Detection")
    occ_col, motion_col, combined_col = st.columns(3)

    with occ_col:
        ld2410 = int(latest.get("occupancy_ld2410", 0))
        st.metric("LD2410 (mmWave)", "Occupied" if ld2410 else "Vacant")

    with motion_col:
        ir = int(latest.get("motion_ir", 0))
        st.metric("IR (PIR Motion)", "Motion Detected" if ir else "No Motion")

    with combined_col:
        combined = "Occupied" if (ld2410 or ir) else "Vacant"
        st.metric("Combined Status", combined)

    # === Tank Visuals ===
    st.subheader("Tank Levels")
    tank_col1, tank_col2 = st.columns(2)

    with tank_col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=water_pct,
            title={"text": "Water Tank Level (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4CAF50"},
                "steps": [
                    {"range": [0, 25], "color": "#ffcdd2"},
                    {"range": [25, 50], "color": "#fff9c4"},
                    {"range": [50, 100], "color": "#c8e6c9"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 25,
                },
            },
        ))
        fig.update_layout(height=250, margin=dict(t=40, b=10, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

        refill = int(latest.get("water_refill_status", 0))
        if refill:
            st.success("🔄 Water refill pump ACTIVE")

    with tank_col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=disinfectant,
            title={"text": "Disinfectant Level (%) — Virtual Estimate"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2196F3"},
                "steps": [
                    {"range": [0, 15], "color": "#ffcdd2"},
                    {"range": [15, 40], "color": "#fff9c4"},
                    {"range": [40, 100], "color": "#bbdefb"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 15,
                },
            },
        ))
        fig.update_layout(height=250, margin=dict(t=40, b=10, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

        disinf_refill = int(latest.get("disinfectant_refill_status", 0))
        if disinf_refill:
            st.warning("⚠️ Disinfectant needs manual refill!")
        elif disinfectant < 25:
            st.warning("⚠️ Disinfectant low — staff alert")

    # === Activity Feed ===
    st.subheader("Recent Activity (Last 20 Readings)")
    if not today.empty:
        recent = today.tail(20).iloc[::-1]
        events = []
        for _, row in recent.iterrows():
            ts = row["timestamp"].strftime("%H:%M")
            if row.get("mist_maker_status", 0) == 1:
                events.append(f"🕐 **{ts}** — Mist spray fired")
            if row.get("water_refill_status", 0) == 1:
                events.append(f"💧 **{ts}** — Water refill pump ON")
            if row.get("needs_manual_checkup", 0) == 1:
                events.append(f"🔧 **{ts}** — Manual checkup flagged")
            if row.get("occupancy_ld2410", 0) == 1 and row.get("motion_ir", 0) == 1:
                events.append(f"👤 **{ts}** — Occupied (LD2410 + IR)")

        if events:
            for event in events[:15]:
                st.markdown(f"- {event}")
        else:
            st.info("No notable events in recent readings.")
    else:
        st.info("No activity data available for today.")

    # === Hygiene Score Trend ===
    st.subheader("Hygiene Score Trend (Today)")
    if not today.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=today["timestamp"],
            y=today["hygiene_score"],
            mode="lines",
            name="Hygiene Score",
            line=dict(color="steelblue", width=1.5),
        ))
        fig.add_hline(y=45, line_dash="dash", line_color="red",
                      annotation_text="Cleaning Threshold")
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Hygiene Score",
            yaxis=dict(range=[0, 105]),
            height=300,
            margin=dict(t=20, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)
