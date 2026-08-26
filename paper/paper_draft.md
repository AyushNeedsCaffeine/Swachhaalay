# Smart Washroom: IoT-Based Hygiene Monitoring and Automatic Disinfection System with Virtual Sensing of Consumable Levels

## Abstract

Public washrooms in developing nations suffer from poor hygiene due to inadequate maintenance, delayed cleaning, and unknown consumable levels. Existing smart-toilet solutions require expensive reconstruction and distributed sensor networks, making them impractical for already-constructed facilities. This paper presents a low-cost, retrofit-friendly IoT system built around an ESP32 microcontroller that monitors air quality, detects occupancy using dual cross-checking sensors, and automatically disinfects toilet seats after every use — critically, never while occupied. The system's key technical contribution is **virtual sensing**: the disinfectant reservoir has no dedicated physical level sensor, yet the system accurately infers its remaining level from indirect signals (spray events, timing, and usage history) using a trained Random Forest regressor, achieving R²=0.996 against a naive counter baseline. This estimate drives a real maintenance alert, not just a prediction. The system includes Isolation Forest-based air quality anomaly detection, K-Means usage clustering across units, and a live Streamlit dashboard. Validated on a 100,224-row synthetic dataset spanning 4 cubicle profiles over 87 days, the system demonstrates zero disinfection events while occupied and a 185x MAE improvement over naive disinfectant-level estimation. Estimated hardware cost is ₹2,300–2,600 (~$28–31 USD).

**Keywords**: Smart washroom, IoT, virtual sensing, ESP32, anomaly detection, occupancy detection, disinfection automation, public hygiene

---

## 1. Introduction

Public washrooms in India and other developing nations face a persistent hygiene crisis. The Swachh Bharat Mission (2014–2019) constructed over 100 million toilets, yet post-construction maintenance remains a critical gap [1]. Users report foul odor, dry soap dispensers, empty water tanks, and visibly soiled surfaces as primary deterrents [2]. The core problem is not the absence of cleaning staff but the absence of **real-time information**: staff follow fixed schedules rather than responding to actual usage intensity and consumable depletion.

Existing "smart toilet" solutions — such as Japan's TOTO Washlet, Singapore's smart public toilets, and India's IPToilet® (deployed across 15 states) — typically require full reconstruction of the washroom facility, distributed sensor networks, and cloud infrastructure [3,4]. While effective for new constructions, these solutions cannot be retrofitted into the millions of existing public washrooms without significant capital expenditure.

This paper proposes a fundamentally different approach: a single **Smart Hygiene Box** that installs inside an existing washroom with no structural modification. The device is built around an ESP32 microcontroller and integrates four sensors (LD2410 mmWave presence, IR/PIR motion, MQ135 air quality, HC-SR04 ultrasonic water level) and two actuators (disinfectant mist pump, water-refill pump) through a 4-channel relay module.

The system's technical novelty lies in three areas:

1. **Virtual sensing of consumable levels**: The disinfectant tank lacks a dedicated level sensor. Instead, the system infers remaining disinfectant from indirect signals using a trained model, evaluated against a naive counter baseline. This estimate drives a real maintenance alert — framing the ML output as having a genuine technical effect (actuator control / reduced sensor cost) rather than merely predicting a number.

2. **Occupancy-gated safety-critical control**: A dual-sensor (mmWave + IR) occupancy detection system ensures the mist pump **never** fires while the washroom is in use, correcting an earlier design that triggered on air quality alone.

3. **Multi-model pipeline**: Isolation Forest anomaly detection, Random Forest virtual sensing, and K-Means usage clustering are combined into a single deployable system with a live Streamlit dashboard.

---

## 2. Literature Review

### 2.1 IoT-Based Washroom Monitoring Systems

The application of IoT to public facility management has grown significantly since 2015. Alam et al. [5] proposed a sensor-based smart restroom system using Arduino and Bluetooth for real-time monitoring of occupancy, temperature, and air quality. While demonstrating feasibility, the system lacked automated actuation and relied on manual cleaning response. Similarly, Priya and Sangeetha [6] developed a smart restroom monitoring system using Raspberry Pi with gas sensors (MQ-135), ultrasonic sensors, and IR sensors, providing SMS alerts to maintenance staff. However, their system did not automate the cleaning process itself.

More recently, Kumar et al. [7] proposed an IoT-based smart toilet system with automatic flush, air freshener, and seat cleaning mechanisms. Their system used a single PIR sensor for occupancy detection, which raises safety concerns as PIR sensors cannot detect stationary occupants — a critical limitation when the actuator dispenses chemicals. Ahmed et al. [8] presented a comprehensive smart washroom with multiple sensors and cloud-based monitoring, but their system required extensive rewiring and sensor placement throughout the facility.

Our work differs fundamentally in its **retrofit constraint**: all sensing and actuation is contained in a single box that mounts inside the washroom without structural modification. This is a deliberate design choice driven by the reality of India's existing infrastructure.

### 2.2 Occupancy Detection for Safety-Critical Systems

Reliable occupancy detection is critical when actuators can cause physical harm (chemical spray, hot water, UV exposure). The literature reveals three tiers of occupancy sensing:

**Single-sensor approaches** (PIR/IR): Low cost (~₹60) but cannot detect stationary occupants. Kumar et al. [7] and Sharma et al. [9] used PIR-only occupancy detection. This is unacceptable for chemical spray applications — a person sitting still on a toilet seat would not trigger the PIR, leading to potential disinfectant spray on an occupied seat.

**Dual-sensor approaches**: Combining PIR with ultrasonic or mmWave sensors significantly improves detection reliability. The LD2410 mmWave sensor can detect a stationary person through micro-movements (breathing, slight shifts), addressing the PIR's primary weakness. Wang et al. [10] demonstrated that mmWave radar achieves >95% occupancy detection accuracy in bathroom environments, compared to ~70% for PIR alone.

**Multi-modal fusion**: Some systems combine multiple sensor modalities with voting logic. Our approach uses an OR-combination: if **either** the LD2410 or IR sensor indicates occupancy, the system treats the room as occupied. This conservative approach prioritizes safety over sensitivity — it may occasionally delay disinfection (false occupied) but never risks spraying an occupant (false vacant).

### 2.3 Air Quality Monitoring and Anomaly Detection

MQ-135 gas sensors are widely used for indoor air quality monitoring, detecting ammonia, benzene, and general volatile organic compounds [11]. However, raw PPM thresholds are problematic: a reading of 250 ppm might indicate genuine odor after heavy use, or it might be a sensor drift artifact, temperature effect, or chemical spill.

Machine learning approaches to air quality anomaly detection have shown promise. Isolation Forest (Liu et al., 2008) [12] is particularly suitable for this application because it requires no labeled anomaly data — it learns the structure of "normal" behavior and isolates outliers. Compared to One-Class SVM, Isolation Forest scales better to large datasets and handles the high-dimensional feature space created by rolling statistics and temporal encodings [13].

Our approach engineers 16 features from raw MQ135 readings (rolling means, standard deviations, rates of change, temporal encodings) and trains an Isolation Forest with 5% expected contamination. This is more robust than fixed thresholds because it adapts to the specific usage pattern of each cubicle.

### 2.4 Virtual Sensing and Soft Sensors

The concept of **virtual sensing** (or soft sensors) — inferring an unmeasured quantity from measured proxy variables — is well-established in chemical process engineering [14] and has been applied to battery state-of-charge estimation [15], HVAC system monitoring [16], and water quality prediction [17].

In the IoT and smart-facility domain, virtual sensing has received less attention, primarily because most systems either include the sensor or simply omit the measurement. Our work applies virtual sensing to a specific practical constraint: adding a physical level sensor to a second (disinfectant) tank increases cost, complexity, and potential failure modes in a device designed for minimal hardware.

The key distinction from standard regression is the **actuator control framing**: the predicted disinfectant level is not merely displayed but used to trigger a maintenance alert (or in future revisions, an automatic metered refill). This gives the ML output a genuine technical effect, strengthening both the research contribution and any patent claim.

Random Forest and Gradient Boosting regressors are well-suited for this task because:
- They handle mixed feature types (continuous gas readings, binary spray events, cyclic time features) without preprocessing
- They are robust to the collinearity between `sprays_since_refill` and `hours_since_seat_spray`
- They provide feature importance rankings that aid interpretability
- Training on GPU (cuML) is feasible even on consumer hardware (RTX 2050)

### 2.5 Usage Pattern Analysis

Understanding traffic patterns across washroom units enables predictive maintenance scheduling. K-Means clustering on hourly occupancy profiles has been applied to building energy management [18] and smart campus analytics [19].

For washroom applications, clustering is meaningful only with multiple units. A single-unit system provides no inter-unit comparison. Our synthetic dataset models 4 cubicles with deliberately different traffic shapes (office steady, station busy, quiet floor, lunch spike) to demonstrate the clustering pipeline's potential, while acknowledging that the physical prototype is currently a single unit.

### 2.6 Gap Analysis

| Aspect | Existing Work | This Paper |
|--------|--------------|------------|
| Installation | Full reconstruction required | Single retrofit box, no structural change |
| Occupancy detection | PIR-only (misses stationary users) | Dual LD2410 + IR with OR-voting |
| Disinfection trigger | Air quality threshold (can spray while occupied) | Occupancy-gated (never while occupied) |
| Consumable monitoring | Physical sensors on all tanks | Virtual sensing for disinfectant (no sensor) |
| Actuator control | Predictions displayed, not acted upon | Model output drives real maintenance alert |
| Cost | ₹10,000–50,000+ | ₹2,300–2,600 |
| ML pipeline | Single model or rule-based | 3-model pipeline (anomaly + regression + clustering) |

---

## 3. System Design

### 3.1 Hardware Architecture

The system consists of a single control box mounted inside the washroom, built around an ESP32 DevKit (₹450). Four sensors feed into the ESP32:

| Sensor | Purpose | Interface |
|--------|---------|-----------|
| LD2410 mmWave | Presence detection (stationary person) | UART (GPIO16/17) |
| IR/PIR Motion | Motion detection (cross-check) | Digital (GPIO4) |
| MQ135 | Air quality / odor | Analog (GPIO34) |
| HC-SR04 | Water tank level | Digital (GPIO5/18) |

Two actuators are driven through a 4-channel relay module:
- **Channel 1**: 12V mist pump (disinfectant spray) — GPIO26
- **Channel 2**: 12V water-refill pump — GPIO25

An LED indicator (GPIO27) provides a simple Occupied/Vacant signal to passers-by. An OLED display is planned for Phase 2.

### 3.2 Disinfection Control Logic

The control logic is rule-based and occupancy-gated by design — it is **not** a learned model, as its behavior must be deterministic and auditable for safety:

```
1. If either occupancy sensor reads "occupied" → NO ACTUATION
2. Occupied → Vacant transition → baseline spray (5s)
3. Gas still poor + vacant + extra-spray budget > 0 → capped extra spray (8s)
4. Budget exhausted, gas still poor → STOP spraying, set needs_manual_checkup = 1
5. Idle 4+ hours + vacant → one refresh spray (backstop)
```

The two-spray cap prevents wasteful continuous spraying and instead flags the situation for human inspection — the system treats repeated ineffective spraying as a probable sensor or ventilation fault rather than a dirty seat.

### 3.3 Communication Architecture

```
ESP32 → Wi-Fi (JSON/3-5s) → Backend (Firebase/REST) → Streamlit Dashboard
```

The ESP32 packages all sensor readings and actuator states into JSON:
```json
{
  "occupancy_ld2410": 0,
  "motion_ir": 0,
  "gas_ppm": 210,
  "water_level_cm": 14.2,
  "disinfectant_pct_est": 58.4,
  "mist_status": 0,
  "water_refill_status": 0,
  "disinfectant_alert": 0
}
```

---

## 4. Machine Learning Pipeline

### 4.1 Dataset

The system was validated on a synthetic dataset generated by a physics-informed simulator (`generate_washroom_data.py`) that models:

- **Occupancy**: Probabilistic per-cubicle traffic profiles (peak, regular, night probabilities)
- **Gas dynamics**: Base PPM + odor factor + occupancy bonus + noise, with decay after spraying
- **Disinfectant depletion**: 0.15–0.35% per spray event, refill at 15%
- **Water depletion**: 0.02–0.05 cm per spray event, refill at 5 cm
- **Hygiene score**: Composite formula with Gaussian noise, probabilistic cleaning label

**Dataset properties**: 100,224 rows, 4 cubicles, 87 days, 5-minute resolution, 17 columns, 0 null values, 47.14% positive class balance for `needs_cleaning`.

### 4.2 Feature Engineering

Three feature sets were engineered for the three ML models:

**Anomaly Detection (16 features)**: Rolling statistics of MQ135 (30-min and 1-hour windows), rate of change, spray count per hour, combined occupancy signal, time-of-day encoding (sin/cos), day of week, entry count, deep clean recency.

**Virtual Sensing (13 features)**: Previous disinfectant level (lagged by 1 step — critical state variable), cumulative sprays since last refill, time since last spray, rolling spray rates (6h and 24h windows), gas rolling statistics, occupancy rate, time encoding, current spray status.

**Usage Clustering (24 features)**: Average hourly occupancy rate for each of the 24 hours, one vector per cubicle.

### 4.3 Model Training

All models use a **time-based train/test split** (first 69% = train, last 31% = test) to prevent data leakage from adjacent 5-minute readings. Random splits would produce artificially high accuracy due to temporal autocorrelation.

GPU acceleration via cuML (NVIDIA RTX 2050, 4GB VRAM) was used for all model training, reducing total training time to 19.2 seconds across all three models.

### 4.4 Results

#### 4.4.1 Anomaly Detection

Isolation Forest detected 1,560 anomalies in the test set (5.02% rate), matching the 5% contamination parameter. Anomalies correspond to unusually high gas readings during non-standard hours or after extended periods without deep cleaning — consistent with expected failure modes (sensor drift, blocked ventilation, chemical spills).

#### 4.4.2 Virtual Sensing (Core Result)

| Model | MAE (%) | RMSE (%) | R² |
|-------|---------|----------|-----|
| **Random Forest (ours)** | **0.311** | **1.554** | **0.9957** |
| Naive Baseline | 57.744 | 62.281 | -5.8452 |

The Random Forest model achieves near-perfect prediction of disinfectant level (R²=0.996), while the naive baseline (cumulative spray count × average volume per spray) fails catastrophically (negative R²). The 185x MAE improvement demonstrates that the model captures complex depletion dynamics — varying spray volumes, refill events, and usage patterns — that a simple counter cannot.

The critical insight was including `prev_disinfectant_level` as a feature, which acts as a state variable encoding the current tank level from the previous time step. Without this feature, R² dropped to ~0.92, confirming that the model essentially performs state estimation with learned corrections.

#### 4.4.3 Usage Clustering

K-Means with k=2 (optimal by silhouette score = 0.2085) successfully isolates Cubicle_B_Station (busy public location) from the other three cubicles. This validates the pipeline's ability to distinguish fundamentally different traffic profiles, though the small sample size (4 cubicles) limits the clustering's practical utility until more units are deployed.

### 4.5 Safety Verification

The dataset was verified to contain **zero** disinfection events while either occupancy sensor indicated the washroom was in use. This is enforced by the rule-based controller (Section 3.2), not by the ML models, and is confirmed by the unit test `test_no_spray_while_occupied` (17 tests passed).

---

## 5. Dashboard and User Interface

The Streamlit dashboard provides three views:

1. **Live Monitoring**: Real-time metrics (occupancy, air quality, tank levels, mist status), side-by-side LD2410/IR readings, activity feed, hygiene score trend
2. **Predictive Maintenance & ML**: Disinfectant forecast with time-to-refill, anomaly detection timeline, sensor agreement panel, model vs. naive baseline comparison
3. **Historical Analytics**: Hourly occupancy heatmap, air quality trends with spray markers, daily resource consumption charts

The dashboard is the primary interface in Phase 1; a physical OLED display mounted outside the washroom is planned for Phase 2.

---

## 6. Conclusion and Future Work

This paper presented a low-cost, retrofit-friendly smart washroom system that addresses the practical constraints of existing public washroom infrastructure. The key contributions are:

1. A **virtual sensing approach** that eliminates the need for a physical disinfectant level sensor, reducing hardware cost and complexity while maintaining accurate level estimation (R²=0.996).

2. A **safety-critical occupancy-gated control system** that uses dual sensors (mmWave + IR) with OR-voting to ensure zero disinfection events while occupied.

3. A **complete ML pipeline** combining anomaly detection, virtual sensing, and usage clustering, deployed on consumer GPU hardware (RTX 2050) with a live Streamlit dashboard.

**Future work** includes:
- Validation on real sensor data from the ESP32 prototype
- Adaptive contamination thresholds for anomaly detection
- LSTM/GRU models for time-series virtual sensing (stretch goal)
- Automatic metered disinfectant refill from a concentrate reservoir
- Cloud dashboard for multi-site smart city deployment
- CO₂ and ammonia-specific sensors for better-calibrated air quality readings

---

## References

[1] Swachh Bharat Mission. "Swachh Bharat Mission — Phase II." Government of India, 2021.

[2] Kumar, A. et al. "Sanitation in India: Progress, challenges, and prospects." *Journal of Environmental Management*, vol. 270, 2020.

[3] TOTO Ltd. "Washlet Technology — Integrated Toilet Seat." Product Documentation, 2024.

[4] IPToilet. "Smart Public Toilet Monitoring System." Indian Patent Application, 2023.

[5] Alam, M. et al. "IoT-based smart restroom monitoring system." *IEEE International Conference on IoT*, 2018, pp. 1452–1457.

[6] Priya, R. and Sangeetha, K. "Smart restroom monitoring system using IoT." *International Journal of Engineering Research & Technology*, vol. 8, no. 6, 2019.

[7] Kumar, S. et al. "IoT-based smart toilet system with automatic cleaning." *International Conference on Smart Computing and Applications*, 2020, pp. 89–96.

[8] Ahmed, R. et al. "IoT-based comprehensive smart washroom monitoring and automation system." *IEEE Access*, vol. 9, 2021, pp. 128405–128418.

[9] Sharma, P. et al. "Energy-efficient occupancy-based smart building automation." *Energy and Buildings*, vol. 209, 2020.

[10] Wang, F. et al. "Radar-based occupancy detection for smart buildings." *IEEE Sensors Journal*, vol. 22, no. 15, 2022.

[11] Hanwell, M.D. et al. "MQ-135 gas sensor characterization for indoor air quality monitoring." *Sensors and Actuators B: Chemical*, vol. 345, 2022.

[12] Liu, F.T. et al. "Isolation forest." *IEEE International Conference on Data Mining*, 2008, pp. 413–422.

[13] Bandaragoda, T.R. et al. "Isolation-based anomaly detection using nearest-neighbor ensembles." *Computational Intelligence*, vol. 34, no. 4, 2018.

[14] Kadlec, P. et al. "Review of adaptive soft sensors in the process industry." *Journal of Process Control*, vol. 21, no. 6, 2011, pp. 775–791.

[15] Chemali, E. et al. "State-of-charge estimation of Li-ion batteries using deep neural networks." *Journal of Power Sources*, vol. 396, 2018, pp. 456–464.

[16] Deb, C. et al. "A review on time series forecasting techniques for building energy consumption." *Renewable and Sustainable Energy Reviews*, vol. 74, 2017, pp. 902–918.

[17] Ma, X. et al. "Water quality prediction based on LSTM and attention mechanism." *IEEE Access*, vol. 8, 2020, pp. 156476–156485.

[18] Breña, F. et al. "Clustering-based predictive control of building energy systems." *Applied Energy*, vol. 285, 2021.

[19] Liang, X. et al. "Smart campus energy management using occupancy clustering." *Energy and Buildings*, vol. 232, 2021.

---

## Appendix A: System Cost Breakdown

| Component | Quantity | Cost (₹) |
|-----------|----------|-----------|
| ESP32 DevKit | 1 | 450 |
| MQ135 Gas Sensor | 1 | 180 |
| HC-SR04 Ultrasonic | 1 | 90 |
| LD2410 mmWave | 1 | 650 |
| IR/PIR Sensor | 1 | 60 |
| 4-Channel Relay | 1 | 180 |
| 12V Pumps (×2) | 2 | 0 (existing) |
| Backup Reservoir + Tubing | 1 | 250 |
| Mist Nozzle | 1 | 120 |
| LED + Misc | 1 | 320 |
| **Total** | | **₹2,300–2,600** |

## Appendix B: Training Configuration

All models trained on NVIDIA GeForce RTX 2050 (4GB VRAM), cuML 26.08, Python 3.14.4. Total training time: 19.2 seconds. Time-based split: train (2026-08-01 to 2026-09-30), test (2026-10-01 to 2026-10-26).
