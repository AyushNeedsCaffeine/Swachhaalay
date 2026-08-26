"""
Smart Washroom — Streamlit Dashboard

Live monitoring, predictive maintenance, and historical analytics
for the Smart Washroom Hygiene Monitoring System.

Usage:
    streamlit run dashboard/app.py
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ml.gpu_utils import is_gpu_available, get_device_info

st.set_page_config(
    page_title="Smart Washroom Dashboard",
    page_icon="🚿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
with st.sidebar:
    st.title("🚿 Smart Washroom")
    st.caption("Hygiene Monitoring & Auto-Disinfection System")
    st.divider()

    device = get_device_info()
    st.markdown(f"**Compute**: {device}")

    st.divider()
    st.markdown("""
    **ML Features:**
    - 🔍 Anomaly Detection (Isolation Forest)
    - 🧪 Virtual Sensing (Random Forest)
    - 📊 Usage Clustering (K-Means)
    - 🛡️ Occupancy-Gated Control (Rule-based)
    """)

    st.divider()
    st.markdown("**ESP32 Sensors:**")
    st.markdown("- LD2410 mmWave (presence)")
    st.markdown("- IR/PIR (motion)")
    st.markdown("- MQ135 (air quality)")
    st.markdown("- HC-SR04 (water level)")

    st.divider()
    st.caption("Version 1.0 — Revision 2")

# Main content with tabs
st.title("🚿 Smart Washroom Dashboard")
st.caption("Hygiene Monitoring & Automatic Disinfection System — ESP32 + ML + Streamlit")

tab1, tab2, tab3 = st.tabs([
    "📡 Live Monitoring",
    "🔮 Predictive Maintenance & ML",
    "📊 Historical Analytics",
])

with tab1:
    from dashboard.tabs.live_monitoring import render as render_live
    render_live()

with tab2:
    from dashboard.tabs.predictive_maintenance import render as render_pred
    render_pred()

with tab3:
    from dashboard.tabs.historical_analytics import render as render_hist
    render_hist()
