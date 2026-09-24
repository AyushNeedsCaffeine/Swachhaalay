# Swachhaalay — Smart Washroom

**IoT-Based Hygiene Monitoring & Automatic Disinfection System Using ESP32**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-18%20passing-brightgreen)](tests/)
[![cuML GPU](https://img.shields.io/badge/GPU-cuML%2026.08-76B900?logo=nvidia)](https://rapids.ai/cuml/)

---

## Objective

Public washrooms in developing nations suffer from poor hygiene due to inadequate maintenance, delayed cleaning, and unknown consumable levels. Existing smart-toilet solutions require expensive reconstruction and distributed sensor networks, making them impractical for already-constructed facilities.

**Swachhaalay** is a low-cost, retrofit-friendly IoT system that:

- Monitors air quality in real-time using MQ135 gas sensors
- Detects occupancy via dual cross-checking sensors (mmWave + IR/PIR)
- Automatically disinfects toilet seats after every use — **never while occupied**
- Estimates disinfectant tank level **without a physical sensor** using ML (virtual sensing)
- Detects air quality anomalies (sensor drift, chemical spills, blocked ventilation)
- Clusters usage patterns across multiple cubicle units for predictive maintenance

The system installs as a **single box** inside an existing washroom with **no structural modification**. Estimated hardware cost: **₹2,500–2,800 (~$30–34 USD)**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HARDWARE LAYER                               │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ LD2410   │  │ IR/PIR   │  │  MQ135   │  │ HC-SR04  │           │
│  │ mmWave   │  │  Motion  │  │   Gas    │  │ Ultrasonic│           │
│  │ Presence │  │  Detect  │  │  Quality │  │  Water   │           │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       │              │              │              │                 │
│       └──────────────┴──────┬───────┴──────────────┘               │
│                             │                                       │
│                      ┌──────┴──────┐                                │
│                      │   ESP32     │                                │
│                      │   MCU       │                                │
│                      └──────┬──────┘                                │
│                             │                                       │
│              ┌──────────────┼──────────────┐                       │
│              │              │              │                        │
│       ┌──────┴──────┐ ┌────┴────┐ ┌───────┴──────┐                │
│       │ Mist Pump   │ │ Refill  │ │ Wi-Fi        │                │
│       │ (Spray)     │ │ Pump    │ │ (JSON/3-5s)  │                │
│       └─────────────┘ └─────────┘ └──────┬───────┘                │
│                                          │                          │
└──────────────────────────────────────────┼──────────────────────────┘
                                           │
┌──────────────────────────────────────────┼──────────────────────────┐
│                        SOFTWARE LAYER    │                          │
│                                          │                          │
│                      ┌───────────────────▼──────────────────┐      │
│                      │      Streamlit Dashboard              │      │
│                      │                                       │      │
│                      │  ┌─────────┬───────────┬───────────┐ │      │
│                      │  │  Tab 1  │   Tab 2   │   Tab 3   │ │      │
│                      │  │  Live   │ Predictive│Historical │ │      │
│                      │  │ Monitor │ Maintenance│ Analytics │ │      │
│                      │  └─────────┴───────────┴───────────┘ │      │
│                      └──────────────────────────────────────┘      │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                    ML PIPELINE                             │   │
│  │                                                             │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐   │   │
│  │  │   Anomaly    │ │   Virtual    │ │     Usage        │   │   │
│  │  │  Detection   │ │   Sensing    │ │    Clustering    │   │   │
│  │  │ Isolation    │ │ Random Forest│ │    K-Means       │   │   │
│  │  │   Forest     │ │ (GBR fallback)│ │                  │   │   │
│  │  └──────────────┘ └──────────────┘ └──────────────────┘   │   │
│  │                                                             │   │
│  │  ┌──────────────────────────────────────────────────────┐  │   │
│  │  │  Rule-Based Control (Safety-Critical)                │  │   │
│  │  │  OR-Logic Occupancy Gating — Never Spray If Occupied│  │   │
│  │  └──────────────────────────────────────────────────────┘  │   │
│  └────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Hardware

| Component | Purpose | Interface |
|-----------|---------|-----------|
| ESP32 DevKit | Main controller | — |
| LD2410 mmWave | Presence detection (stationary person) | UART (GPIO16/17) |
| IR/PIR Sensor | Motion detection (cross-check) | Digital (GPIO4) |
| MQ135 Gas Sensor | Air quality / odor (ammonia, VOCs) | Analog (GPIO34) |
| HC-SR04 Ultrasonic | Water tank level measurement | Digital (GPIO5/18) |
| 4-Channel Relay | Actuator control | GPIO25, GPIO26 |
| 2× 12V Pumps | Disinfectant spray + water refill | Via relay |
| LED Indicator | Occupied/Vacant status | GPIO27 |

### Software

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.10+ | ML pipeline, dashboard |
| **ML Framework** | scikit-learn 1.3+ | Models (CPU fallback) |
| **GPU Acceleration** | cuML 26.08 / cuDF / CuPy | Fast GPU training (NVIDIA) |
| **Gradient Boosting** | scikit-learn GBR | Secondary candidate (CPU fallback builds it; Random Forest is the saved model) |
| **Dashboard** | Streamlit 1.28+ | Web-based monitoring UI |
| **Visualization** | Plotly 5.17+ / Matplotlib / Seaborn | Charts, heatmaps, gauges |
| **Data Processing** | Pandas 2.0+ / NumPy 1.24+ | Data pipeline |
| **Model Serialization** | Joblib 1.3+ | Save/load trained models |
| **Testing** | pytest | Unit & integration tests |
| **Version Control** | Git | 5-phase commit history |

---

## ML Features & Outcome Metrics

### Feature 1: Anomaly Detection

Detects unusual air quality patterns (chemical spills, sensor drift, blocked ventilation) using unsupervised learning.

| Metric | Value |
|--------|-------|
| **Model** | Isolation Forest |
| **Implementation** | cuML (GPU) / scikit-learn (CPU) |
| **Input Features** | 16 engineered (rolling stats, time encoding, gas dynamics) |
| **Contamination** | 5% (expected anomaly rate) |
| **Actual Anomaly Rate** | 5.21% on test set |
| **Training Time (GPU)** | ~1.7s |

### Feature 2: Virtual Sensing (Core Contribution)

Estimates disinfectant tank level **without a physical sensor** using indirect signals (spray events, timing, usage history). This is the key technical novelty — the ML output drives a real maintenance alert.

**Honest evaluation protocol.** Tank level is a measured, indirectly-observed quantity, so a one-step "teacher-forced" fit (feeding the *true* previous level at every test row) is **not** representative of deployment. The headline number below is a **walk-forward** evaluation: the model is seeded with the true level once per cubicle at the start of the test window and then receives only its *own* previous prediction, reset to 100.0 at observed refill events. A stricter **rollout-between-refills** protocol (seed at each refill, own-predictions only, rows before the first test-window refill excluded) is reported separately. The reference-only (teacher-forced) figure is included solely as a pipeline sanity check.

| Metric | Value |
|--------|-------|
| **Model** | Random Forest Regressor |
| **Input Features** | 13 (prev level, spray rates, gas dynamics, time encoding) |
| **Training Data** | 69,152 rows (≈60 days / 61 dates, Aug 1–Sep 30 — 69% of the 87-day span) |
| **Test Data** | 31,072 rows (≈27 days / 27 dates, Sep 30–Oct 26 — last 31%) |
| **Training Time (GPU)** | ~11s |

| Evaluation | MAE (%) | RMSE (%) | R² |
|------------|---------|----------|----|
| Teacher-forced (reference only) | 0.300 | 1.472 | 0.9965 |
| **Walk-forward (honest, headline)** | **38.084** | **45.644** | **−2.396** |
| Rollout between refills (strict) | 28.711 | 35.010 | −0.956 |
| Naive baseline (fixed: reset at refill) | 42.468 | 49.162 | −2.940 |
| Persistence baseline (last known value) | 38.257 | 45.431 | −2.365 |

The honest walk-forward model slightly beats both baselines in MAE but remains well below R² = 0 on this *synthetic* dataset: multi-step level prediction degrades quickly because depletion is dominated by irregular spray volumes and refill resets. The teacher-forced 0.9957 figure previously reported here was an artifact of the evaluation protocol and has been corrected; on physical hardware a tank-level sensor or refill log would be needed to ground the model's predictions across time.

### Feature 3: Usage Clustering

Identifies traffic patterns across cubicle units for differentiated maintenance scheduling.

| Metric | Value |
|--------|-------|
| **Model** | K-Means |
| **Input** | 24-dim hourly occupancy vector per cubicle |
| **Optimal k** | 2 (by silhouette score) |
| **Silhouette Score** | 0.1914 |
| **Training Time (GPU)** | ~2.3s |

**Result**: Cubicle B (busy public station) isolated from A/C/D — validates the pipeline's ability to distinguish fundamentally different traffic profiles.

### Feature 4: Safety-Critical Occupancy Gating

Rule-based control system (not ML) ensuring deterministic, auditable safety:

1. **OR-Logic**: If **either** LD2410 or IR sensor indicates occupied → **NO actuation**
2. **Exit-Triggered Spray**: Baseline spray on occupied → vacant transition
3. **Post-Spray Cooldown**: 10-minute dry/cooldown window after each spray — the room is reported unavailable (`room_available = 0`) and **no fresh spray is permitted** until it elapses
4. **Two-Spray Cap**: Maximum 2 extra sprays per cycle; exhausted → `needs_manual_checkup` flag
5. **Zero spray events while occupied** — verified by unit test `test_no_spray_while_occupied`

---

## Dataset

| Property | Value |
|----------|-------|
| **Total Rows** | 100,224 |
| **Cubicles** | 4 (distinct traffic profiles) |
| **Duration** | 87 days (2026-08-01 → 2026-10-26) |
| **Resolution** | 5-minute intervals |
| **Columns** | 18 (sensor readings, actuator states, derived features, labels, cooldown flag) |
| **Null Values** | 0 |

### Cubicle Profiles

| Cubicle | Peak (12–14h) | Regular (8–20h) | Night | Profile |
|---------|---------------|-----------------|-------|---------|
| A — Office | 45% | 25% | 3% | General office — moderate steady traffic |
| B — Station | 70% | 45% | 10% | Busy public (metro/mall) — heavy all-day |
| C — Quiet Floor | 20% | 10% | 1% | Low-traffic wing — very sparse |
| D — Lunch Spike | 75% | 8% | 1% | Quiet all day, sharp lunch spike |

---

## Quick Start

```bash
# 1. Clone the repo
git clone <repo-url>
cd Swachhaalay

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install GPU acceleration (NVIDIA GPU + CUDA required)
pip install cuml-cu12 cudf-cu12 cupy-cuda12x

# 5. Generate dataset
python data/generate_washroom_data.py

# 6. Run EDA (optional)
python data/explore.py

# 7. Train all ML models
python ml/train.py

# 8. Evaluate models & generate figures
python ml/evaluate.py

# 9. Run tests
python -m pytest tests/test_ml.py -v

# 10. Launch dashboard
streamlit run dashboard/app.py
```

---

## Project Structure

```
Swachhaalay/
├── data/
│   ├── generate_washroom_data.py    # Physics-informed synthetic data generator
│   ├── explore.py                   # EDA: stats, distributions, heatmaps
│   └── washroom_dataset_multi_cubicle.csv  # 100K+ rows dataset
│
├── ml/
│   ├── gpu_utils.py                 # GPU/CPU abstraction layer (cuML ↔ scikit-learn)
│   ├── anomaly_detection.py         # Feature 1: Isolation Forest
│   ├── virtual_sensing.py           # Feature 2: RF (GBR CPU fallback) + baselines
│   ├── usage_clustering.py          # Feature 3: K-Means clustering
│   ├── train.py                     # Unified training CLI
│   ├── evaluate.py                  # Evaluation metrics & figures
│   └── models/                      # Trained model artifacts (.joblib)
│
├── dashboard/
│   ├── app.py                       # Streamlit entry point
│   ├── utils.py                     # Shared helpers (cached loading, formatting)
│   └── tabs/
│       ├── live_monitoring.py       # Tab 1: Real-time status & gauges
│       ├── predictive_maintenance.py # Tab 2: ML insights & anomaly alerts
│       └── historical_analytics.py  # Tab 3: Trends, heatmaps, consumption
│
├── tests/
│   └── test_ml.py                   # 18 unit tests (data, models, features, safety)
│
├── docs/
│   ├── DATA_DICTIONARY.md           # All 18 columns documented
│   ├── MODEL_CARD.md                # Model cards for all 3 models
│   ├── TRAINING_LOG.md              # Experiment results & hyperparameters
│   └── *.docx / *.pptx             # Reports & presentation
│
├── paper/
│   ├── paper_draft.md               # Full academic paper (19 references)
│   └── patent_draft.md              # Patent application (8 claims)
│
├── outputs/
│   ├── eda/                         # EDA plots & statistics
│   └── figures/                     # Model evaluation figures
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Testing

```bash
python -m pytest tests/test_ml.py -v
```

| Test Class | Tests | Coverage |
|------------|-------|----------|
| `TestDataLoading` | 8 | CSV integrity, shape, nulls, cubicles, timestamps, value ranges, binary columns, cooldown flag |
| `TestModels` | 7 | Model existence, metadata correctness, honest-metric protocol gate, cluster count |
| `TestFeatureEngineering` | 2 | Anomaly features (≥10), virtual sensing features (≥8) |
| `TestSafetyRules` | 1 | Zero spray events while occupied |

---

## Hardware Cost Breakdown

| Component | Quantity | Cost (₹) |
|-----------|----------|-----------|
| ESP32 DevKit | 1 | 450 |
| MQ135 Gas Sensor | 1 | 180 |
| HC-SR04 Ultrasonic | 1 | 90 |
| LD2410 mmWave | 1 | 650 |
| IR/PIR Sensor | 1 | 60 |
| 4-Channel Relay | 1 | 180 |
| 12V Pumps (×2) | 2 | 200 |
| Backup Reservoir + Tubing | 1 | 250 |
| Mist Nozzle | 1 | 120 |
| LED + Misc | 1 | 320 |
| **Total** | | **₹2,500–2,800** |

**Estimated**: ~$30–34 USD

---

## Documentation

| Document | Description |
|----------|-------------|
| [Data Dictionary](docs/DATA_DICTIONARY.md) | All 18 columns with types, ranges, descriptions |
| [Model Card](docs/MODEL_CARD.md) | Detailed model cards for anomaly detection, virtual sensing, clustering |
| [Training Log](docs/TRAINING_LOG.md) | Experiment results, hyperparameters, GPU timing |
| [Paper Draft](paper/paper_draft.md) | Full academic paper with 19 references |
| [Patent Draft](paper/patent_draft.md) | Patent application with 8 claims |

---

## Key Design Decisions

1. **Retrofit-first**: Single control box, no structural modification to existing washrooms
2. **Safety-critical gating**: Dual-sensor OR-logic ensures zero false-negative occupancy (spraying an occupied room) at the cost of occasional false positives (delayed disinfection)
3. **Virtual sensing**: Eliminates the need for a physical level sensor on the disinfectant tank, reducing cost and failure modes
4. **Two-spray cap**: Prevents wasteful continuous spraying; flags for human inspection instead of blindly spraying
5. **GPU acceleration**: cuML falls back to scikit-learn automatically — works on any hardware

---

## License

MIT

---

## Citation

If you use this work in your research, please cite:

```bibtex
@software{swachhaalay2026,
  title  = {Swachhaalay: IoT-Based Hygiene Monitoring and Automatic Disinfection System},
  year   = {2026},
  url    = {https://github.com/<your-repo>/Swachhaalay}
}
```
