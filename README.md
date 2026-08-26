# Smart Washroom

**Hygiene Monitoring & Automatic Disinfection System Using ESP32**

A low-cost, retrofit-friendly IoT system that monitors air quality, detects occupancy, and automatically disinfects toilet seats — without requiring washroom reconstruction.

## Architecture

```
Sensors (LD2410 + IR + MQ135 + HC-SR04)
        │
        ▼
   ESP32 Microcontroller
        │
        ├─► Mist Pump (disinfectant spray)
        ├─► Water Refill Pump
        └─► Wi-Fi → Streamlit Dashboard
                     │
                     ├─► Live Monitoring (Tab 1)
                     ├─► Predictive Maintenance + ML (Tab 2)
                     └─► Historical Analytics (Tab 3)
```

## ML Features

| Feature | Model | Purpose |
|---------|-------|---------|
| Anomaly Detection | Isolation Forest (GPU: cuML) | Detect unusual air quality patterns |
| Virtual Sensing | Random Forest / Gradient Boosting | Estimate disinfectant level without a physical sensor |
| Disinfection Control | Rule-based, occupancy-gated | Never spray while occupied |
| Usage Clustering | K-Means (GPU: cuML) | Identify traffic patterns across cubicles |

## Quick Start

```bash
# 1. Clone the repo
git clone <repo-url>
cd Smart-Washroom

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install GPU acceleration
pip install cuml-cu12 cudf-cu12 cupy-cuda12x

# 5. Generate dataset (if not present)
python data/generate_washroom_data.py

# 6. Train ML models
python ml/train.py

# 7. Launch dashboard
streamlit run dashboard/app.py
```

## Project Structure

```
Smart-Washroom/
├── data/
│   ├── generate_washroom_data.py    # Synthetic data generator
│   ├── washroom_dataset_multi_cubicle.csv
│   ├── explore.py                   # EDA & statistics
│   └── augment.py                   # Data augmentation (anomalies)
├── ml/
│   ├── anomaly_detection.py         # Isolation Forest
│   ├── virtual_sensing.py           # RF/GBR + naive baseline
│   ├── usage_clustering.py          # K-Means clustering
│   ├── train.py                     # Unified training script
│   └── evaluate.py                  # Metrics & comparison
├── dashboard/
│   ├── app.py                       # Streamlit entry point
│   └── tabs/
│       ├── live_monitoring.py       # Tab 1: Real-time status
│       ├── predictive_maintenance.py # Tab 2: ML insights
│       └── historical_analytics.py  # Tab 3: Trends & heatmaps
├── docs/                            # Project documentation
├── paper/                           # Research paper & patent drafts
└── tests/                           # Unit tests
```

## Dataset

- **100,224 rows** across 4 simulated cubicles (87 days, 5-minute resolution)
- **17 columns**: sensor readings, actuator states, derived features, labels
- Cubicle profiles: Office, Busy Station, Quiet Floor, Lunch Spike

## Hardware (Partial Build)

| Component | Purpose |
|-----------|---------|
| ESP32 | Main controller |
| LD2410 mmWave | Occupancy (stationary person detection) |
| IR/PIR | Occupancy (motion-based, cross-checks LD2410) |
| MQ135 | Air quality / odor |
| HC-SR04 | Water tank level |
| 4-Ch Relay | CH1: mist pump, CH2: refill pump |
| 2x 12V Pumps | Disinfectant spray + water auto-refill |

**Estimated cost**: ₹2,300–2,600

## License

MIT
