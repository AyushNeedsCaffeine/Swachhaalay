"""
Smart Washroom — Tab 3: Historical Analytics & Traffic Patterns

Peak-hour heatmap, air quality trends, resource consumption.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from dashboard.utils import load_dataset


def render():
    """Render the Historical Analytics tab."""
    df = load_dataset()
    cubicles = sorted(df["cubicle_id"].unique())

    # Date range filter
    col1, col2 = st.columns(2)
    with col1:
        min_date = df["timestamp"].min().date()
        max_date = df["timestamp"].max().date()
        date_range = st.date_input("Date Range",
                                    value=(min_date, max_date),
                                    min_value=min_date,
                                    max_value=max_date)
    with col2:
        selected_cubicles = st.multiselect("Cubicles", cubicles, default=cubicles, key="hist_cubicles")

    if len(date_range) == 2:
        mask = (df["timestamp"].dt.date >= date_range[0]) & (df["timestamp"].dt.date <= date_range[1])
        df_filtered = df[mask & df["cubicle_id"].isin(selected_cubicles)].copy()
    else:
        df_filtered = df[df["cubicle_id"].isin(selected_cubicles)].copy()

    if df_filtered.empty:
        st.warning("No data for selected filters.")
        return

    # === Peak Traffic Heatmap ===
    st.subheader("Peak Traffic Heatmap")
    df_filtered["hour"] = df_filtered["timestamp"].dt.hour
    df_filtered["day_of_week"] = df_filtered["timestamp"].dt.dayofweek

    for cubicle in selected_cubicles:
        sub = df_filtered[df_filtered["cubicle_id"] == cubicle]
        pivot = sub.groupby(["day_of_week", "hour"])["occupancy_ld2410"].mean().unstack(fill_value=0)

        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fig = px.imshow(
            pivot.values,
            labels=dict(x="Hour of Day", y="Day of Week", color="Occupancy Rate"),
            x=[f"{h:02d}" for h in range(24)],
            y=day_labels[:len(pivot)],
            color_continuous_scale="YlOrRd",
            aspect="auto",
        )
        fig.update_layout(
            title=f"{cubicle}",
            height=250,
            margin=dict(t=40, b=30, l=50, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # === Air Quality Trends ===
    st.subheader("Air Quality Trends (Before vs After Sprays)")
    for cubicle in selected_cubicles:
        sub = df_filtered[df_filtered["cubicle_id"] == cubicle].copy()
        sub = sub.set_index("timestamp")

        # Rolling average
        sub["gas_rolling"] = sub["mq135_gas_ppm"].rolling(12, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sub.index, y=sub["mq135_gas_ppm"],
            mode="lines", name="Raw Gas PPM",
            line=dict(color="lightblue", width=0.5),
            opacity=0.5,
        ))
        fig.add_trace(go.Scatter(
            x=sub.index, y=sub["gas_rolling"],
            mode="lines", name="1h Rolling Avg",
            line=dict(color="steelblue", width=1.5),
        ))

        # Mark spray events
        sprays = sub[sub["mist_maker_status"] == 1]
        fig.add_trace(go.Scatter(
            x=sprays.index, y=sprays["mq135_gas_ppm"],
            mode="markers", name="Spray Event",
            marker=dict(color="red", size=3, symbol="triangle-up"),
        ))

        fig.update_layout(
            title=f"Air Quality — {cubicle}",
            xaxis_title="Time", yaxis_title="Gas PPM",
            height=300, margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # === Resource Consumption ===
    st.subheader("Daily Resource Consumption")

    df_daily = df_filtered.copy()
    df_daily["date"] = df_daily["timestamp"].dt.date

    daily_stats = df_daily.groupby(["date", "cubicle_id"]).agg(
        total_sprays=("mist_maker_status", "sum"),
        total_water_refills=("water_refill_status", "sum"),
        total_checkups=("needs_manual_checkup", "sum"),
        avg_gas=("mq135_gas_ppm", "mean"),
        avg_hygiene=("hygiene_score", "mean"),
        max_entry_count=("entry_count", "max"),
    ).reset_index()

    # Sprays per day
    fig = px.bar(daily_stats, x="date", y="total_sprays", color="cubicle_id",
                 title="Daily Spray Count by Cubicle",
                 labels={"total_sprays": "Number of Sprays", "date": "Date"})
    fig.update_layout(height=300, margin=dict(t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)

    # Average gas and hygiene
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(daily_stats, x="date", y="avg_gas", color="cubicle_id",
                      title="Daily Average Gas PPM",
                      labels={"avg_gas": "Avg Gas PPM", "date": "Date"})
        fig.update_layout(height=280, margin=dict(t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(daily_stats, x="date", y="avg_hygiene", color="cubicle_id",
                      title="Daily Average Hygiene Score",
                      labels={"avg_hygiene": "Avg Score", "date": "Date"})
        fig.update_layout(height=280, margin=dict(t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    st.divider()
    st.subheader("Summary Statistics")
    summary = df_filtered.groupby("cubicle_id").agg(
        total_readings=("timestamp", "count"),
        avg_gas_ppm=("mq135_gas_ppm", "mean"),
        avg_hygiene_score=("hygiene_score", "mean"),
        total_sprays=("mist_maker_status", "sum"),
        total_entries=("entry_count", "max"),
        pct_occupied=("occupancy_ld2410", "mean"),
        pct_needs_cleaning=("needs_cleaning", "mean"),
    ).round(2)

    st.dataframe(summary, use_container_width=True)
