# Swachhaalay: IoT-Based Hygiene Monitoring and Automatic Disinfection System with Virtual Sensing of Consumable Levels
 
<!-- ============================================================================
CORRECTED VERSION — read this block before using the paper below.
CITATION STATUS (most recent verification pass: 2026-09-24 — every numbered
reference below has now been individually verified against IEEE Xplore /
Google Scholar / the publisher or replaced with a verified real source):
  CONFIRMED REAL (11): [5] Sherine Mary et al., IEEE ICEDSS 2018
    (doi:10.1109/ICEDSS.2018.8544266); [7] Premkumar et al., IRJET 7(3) 2020;
    [9] Labeodan et al., Energy and Buildings 93:303-314, 2015
    (doi:10.1016/j.enbuild.2015.02.028); [11] Carrillo-Amado et al., Respuestas
    25(1):70-77, 2020 (doi:10.22463/0122820X.2408); [12] Liu et al., ICDM 2008;
    [13] Bandaragoda et al., Comput. Intell. 34(4), 2018; [14] Kadlec et al.,
    C&ChE 35(1), 2011 (corrected details); [15] Chemali et al., J. Power Sources
    400, 2018; [16] Deb et al., RSER 74, 2017; [18] Li et al., Applied Energy
    282, 2021; [19] Nikdel et al., Energy and Buildings 246, 2021
    (doi:10.1016/j.enbuild.2021.111070).
  REPLACED WITH VERIFIED SOURCES (6): [2] now Wankhade, Environ. Urbanization
    27(2):555-572, 2015 (doi:10.1177/0956247814567058); [3] now company material
    for IPToilet (Altersoft Innovations India Pvt. Ltd., iptoilet.com);
    [8] now Azman, Salleh & Zakaria, IEEE Access 10:118345-118356, 2022;
    [10] now Hsu et al., IEEE Sensors Journal 2023 (doi:10.1109/JSEN.2023.3266450);
    [17] now Chen et al., Sustainability 14(20):13231, 2022
    (doi:10.3390/su142013231). Entries [5], [7], [11], [18], [19] above each
    replace an originally fabricated/not-found entry.
  NOT FOUND / removed (3): [4] merged into [3]; [6] Priya & Sangeetha was
    invented and is struck through below (do not restore); the original [2]
    "Kumar et al., J. Environ. Management vol. 270, 2020" was not found and is
    replaced by Wankhade (2015).
The inline author names, titles, and details in the text have been made
consistent with the verified/replaced references. No entry remains tagged
[VERIFY].
============================================================================ -->
 
## Abstract
 
Public washrooms in developing nations suffer from poor hygiene due to inadequate maintenance, delayed cleaning, and unknown consumable levels. Existing smart-toilet solutions require expensive reconstruction and distributed sensor networks, making them impractical for already-constructed facilities. This paper presents a low-cost, retrofit-friendly IoT system built around an ESP32 microcontroller that monitors air quality, detects occupancy using dual cross-checking sensors, and automatically disinfects toilet seats after every use — critically, never while occupied. The system's key technical contribution is **virtual sensing**: the disinfectant reservoir has no dedicated physical level sensor, yet the system infers its remaining level from indirect signals (spray events, timing, and usage history) using a trained Random Forest regressor. We adopt an explicit walk-forward evaluation protocol (the model consumes only its own recursive predictions, re-seeded at observed refill events): one-step, teacher-forced accuracy is near-perfect (MAE 0.30%, R² 0.9965) but is explicitly **not** reported as deployed performance; honest multi-step prediction degrades to MAE 38.1% / R² −2.40 because synthetic depletion behaves as a bounded random walk with irregular per-spray draws and refill resets. The estimate still beats naive and persistence baselines and, re-seeded at observed refills (rollout protocol), reaches MAE 28.7%. The system includes Isolation Forest-based air quality anomaly detection, K-Means usage clustering across units, and a live Streamlit dashboard. Validated on a 100,224-row **synthetic** dataset spanning 4 cubicle profiles over 87 days, the system demonstrates zero disinfection events while occupied and enforces a 10-minute post-spray cooldown. Estimated hardware cost is ₹2,500–2,800 (~$30–34 USD).
 
**Keywords**: Smart washroom, IoT, virtual sensing, ESP32, anomaly detection, occupancy detection, disinfection automation, public hygiene
 
---
 
## 1. Introduction
 
Public washrooms in India and other developing nations face a persistent hygiene crisis. The Swachh Bharat Mission (2014–2019) constructed over 100 million toilets, yet post-construction maintenance remains a critical gap [1]. Construction-led sanitation programmes have historically devoted limited attention to operation and maintenance, leaving facilities that fall into disrepair and are subsequently underused [2]. The core problem is not the absence of cleaning staff but the absence of **real-time information**: staff follow fixed schedules rather than responding to actual usage intensity and consumable depletion.
 
<!-- [FIXED] Removed TOTO Washlet as an example of a system requiring "full reconstruction" —
     this is backwards. TOTO Washlet is a toilet-seat retrofit, one of the LEAST structurally
     invasive smart-toilet products on the market, and citing it this way is a factual error
     a reviewer familiar with the product line would likely catch. -->
Existing "smart toilet" solutions — such as Singapore's smart public toilets and India's IPToilet® — typically require full reconstruction of the washroom facility, distributed sensor networks, and cloud infrastructure [3]. While effective for new constructions, these solutions cannot be retrofitted into the millions of existing public washrooms without significant capital expenditure.
 
This paper proposes a fundamentally different approach: a single **Smart Hygiene Box** that installs inside an existing washroom with no structural modification. The device is built around an ESP32 microcontroller and integrates four sensors (LD2410 mmWave presence, IR/PIR motion, MQ135 air quality, HC-SR04 ultrasonic water level) and two actuators (disinfectant mist pump, water-refill pump) through a 4-channel relay module.
 
The system's technical novelty lies in three areas:
 
1. **Virtual sensing of consumable levels**: The disinfectant tank lacks a dedicated level sensor. Instead, the system infers remaining disinfectant from indirect signals using a trained model, evaluated against a naive counter baseline. This estimate is designed to drive a real maintenance alert — framing the ML output as having a genuine technical effect (actuator control / reduced sensor cost) rather than merely predicting a number.
2. **Occupancy-gated safety-critical control**: A dual-sensor (mmWave + IR) occupancy detection system ensures the mist pump **never** fires while the washroom is in use, correcting an earlier design that triggered on air quality alone.
3. **Multi-model pipeline**: Isolation Forest anomaly detection, Random Forest virtual sensing, and K-Means usage clustering are combined into a single deployable system with a live Streamlit dashboard.
---
 
## 2. Literature Review
 
### 2.1 IoT-Based Washroom Monitoring Systems
 
The application of IoT to public facility management has grown significantly since 2015. Sherine Mary et al. [5] proposed an automated sensor-based washroom monitoring system that infers cleanliness from indoor gas/air quality and alerts a cleaning team via text message when thresholds are exceeded. While demonstrating feasibility, the system lacked automated actuation and relied on manual cleaning response. Premkumar et al. [7] proposed an IoT-based smart toilet system with automatic flush, air freshener, and seat/floor cleaning mechanisms gated on occupancy. Their system used a single PIR sensor for occupancy detection, which raises safety concerns as PIR sensors cannot detect stationary occupants — a critical limitation when an actuator dispenses chemicals. Azman et al. [8] presented an extensive ESP32-based IoT restroom hygiene monitoring system with air-quality (ammonia/IAQ) sensing and cloud-based dashboards. <!-- [CONFIRMED 2026-09-24] [8] is the verified Azman, Salleh & Zakaria, IEEE Access vol. 10, 2022 (doi:10.1109/ACCESS.2022); the earlier in-text name "Ahmed et al." and its never-found "IEEE Access 2021" source were removed. --> Their design focuses on sensing and alerting rather than automated disinfectant actuation, and — like [7] — it does not address the stationary-occupant safety gap that any actuating system must handle.
 
<!-- [FIXED] This paragraph was missing from the original and is important prior art: an existing
     published system combines an IR occupancy counter, an ammonia gas sensor (MQ-137), and an
     ESP32 — a near-identical sensing BOM to this paper's system. It must be cited and
     differentiated from, not omitted, or a reviewer who knows the literature will find it first. -->
Even the closest sensing suite to the present system — ESP32-based restroom hygiene monitors that combine air-quality sensing with cloud dashboards [8] — stops at monitoring. None of the surveyed systems performs automated disinfectant actuation gated by a dual-sensor occupancy check that copes with stationary occupants, and none infers the remaining level of a second, unsensored consumable. Those are the two contributions this paper centers on.
 
Our work differs fundamentally in its **retrofit constraint**: all sensing and actuation is contained in a single box that mounts inside the washroom without structural modification. This is a deliberate design choice driven by the reality of India's existing infrastructure.
 
### 2.2 Occupancy Detection for Safety-Critical Systems
 
Reliable occupancy detection is critical when actuators can cause physical harm (chemical spray, hot water, UV exposure). The literature reveals three tiers of occupancy sensing:
 
**Single-sensor approaches** (PIR/IR): Low cost (~₹60) but cannot detect stationary occupants. PIR-based detection remains the most common single-sensor occupancy approach in buildings [9] and appears in smart-toilet implementations such as Premkumar et al. [7]. This is unacceptable for chemical spray applications — a PIR responds only to movement, so a person sitting still on a toilet seat would not trigger it, leading to potential disinfectant spray on an occupied seat.
 
**Dual-sensor approaches**: Combining PIR with ultrasonic or mmWave sensors significantly improves detection reliability. The LD2410 mmWave sensor can detect a stationary person through micro-movements (breathing, slight shifts), addressing the PIR's primary weakness. <!-- [VERIFIED 2026-09-23] The specific ">95% vs ~70%" claim and its original citation [10] were NOT found in any database and have been removed; the qualitative point (mmWave detects stationary occupants better than PIR) is well-established. [10] is now the verified Hsu et al. 2023 IEEE Sensors Journal paper (95.8% mmWave occupancy-counting accuracy, hundreds of points better than existing schemes), which supports the comparison without the invented figure. --> Published work on mmWave occupancy sensing supports that radar-based approaches meaningfully outperform PIR for detecting stationary occupants [10].
 
**Multi-modal fusion**: Some systems combine multiple sensor modalities with voting logic. Our approach uses an OR-combination: if **either** the LD2410 or IR sensor indicates occupancy, the system treats the room as occupied. This conservative approach prioritizes safety over sensitivity — it may occasionally delay disinfection (false occupied) but never risks spraying an occupant (false vacant).
 
### 2.3 Air Quality Monitoring and Anomaly Detection
 
MQ-135 gas sensors are widely used for indoor air quality monitoring, detecting ammonia, benzene, and general volatile organic compounds [11]. However, raw PPM thresholds are problematic: a reading of 250 ppm might indicate genuine odor after heavy use, or it might be a sensor drift artifact, temperature effect, or chemical spill.
 
Machine learning approaches to air quality anomaly detection have shown promise. Isolation Forest (Liu et al., 2008) [12] is particularly suitable for this application because it requires no labeled anomaly data — it learns the structure of "normal" behavior and isolates outliers. <!-- [12] Liu et al. Isolation Forest, ICDM 2008 — CONFIRMED real and correctly cited; this is a well-known foundational paper. --> Compared to One-Class SVM, Isolation Forest scales better to large datasets and handles the high-dimensional feature space created by rolling statistics and temporal encodings [13].
 
Our approach engineers features from raw MQ135 readings (rolling means, standard deviations, rates of change, temporal encodings) and trains an Isolation Forest with 5% expected contamination. <!-- [FIXED] Softened the claim below — matching the contamination rate is expected by construction, not independent validation. --> Note that the observed ~5% anomaly rate on the test set is expected by construction — the contamination parameter tells Isolation Forest to flag approximately that fraction of points regardless of the data — so this match is not, by itself, evidence the model is finding meaningful anomalies. What would demonstrate that is qualitative inspection of flagged events (Section 4.4.1) and, ideally, precision/recall against a labeled or injected-anomaly test set.
 
### 2.4 Virtual Sensing and Soft Sensors
 
The concept of **virtual sensing** (or soft sensors) — inferring an unmeasured quantity from measured proxy variables — is well-established in chemical process engineering <!-- [FIXED] corrected citation details, see note --> [14] and has been applied to battery state-of-charge estimation [15], HVAC system monitoring [16], and water quality prediction [17].
 
In the IoT and smart-facility domain, virtual sensing has received less attention, primarily because most systems either include the sensor or simply omit the measurement. Our work applies virtual sensing to a specific practical constraint: adding a physical level sensor to a second (disinfectant) tank increases cost, complexity, and potential failure modes in a device designed for minimal hardware.
 
The key distinction from standard regression is the **actuator control framing**: the predicted disinfectant level is not merely displayed but used to trigger a maintenance alert (or in future revisions, an automatic metered refill). This gives the ML output a genuine technical effect, strengthening both the research contribution and any patent claim.
 
Random Forest and Gradient Boosting regressors are well-suited for this task because:
 
- They handle mixed feature types (continuous gas readings, binary spray events, cyclic time features) without preprocessing
- They are robust to the collinearity between `sprays_since_refill` and `hours_since_seat_spray`
- They provide feature importance rankings that aid interpretability
- Training on GPU (cuML) is feasible even on consumer hardware (RTX 2050) <!-- [FIXED] see 4.3 note: no CPU baseline was reported, so this speed claim is currently unsubstantiated as a meaningful advantage at this data scale -->
### 2.5 Usage Pattern Analysis
 
Understanding traffic patterns across washroom units enables predictive maintenance scheduling. K-Means clustering of hourly load and occupancy profiles has been applied to building energy load prediction [18] and campus-building occupancy analytics [19].
 
For washroom applications, clustering is meaningful only with multiple units. A single-unit system provides no inter-unit comparison. Our synthetic dataset models 4 cubicles with deliberately different traffic shapes (office steady, station busy, quiet floor, lunch spike) to demonstrate the clustering pipeline's potential, while acknowledging that the physical prototype is currently a single unit.
 
### 2.6 Gap Analysis
 
| Aspect                | Existing Work                                    | This Paper                                           |
| --------------------- | ------------------------------------------------ | ------------------------------------------------------ |
| Installation          | Full reconstruction required                     | Single retrofit box, no structural change            |
| Occupancy detection   | PIR-only (misses stationary users)               | Dual LD2410 + IR with OR-voting                      |
| Disinfection trigger  | Air quality threshold (can spray while occupied) | Occupancy-gated (never while occupied)               |
| Consumable monitoring | Physical sensors on all tanks                    | Virtual sensing for disinfectant (no sensor)         |
| Actuator control      | Predictions displayed, not acted upon            | Model output designed to drive a real maintenance alert |
| Cost                  | ₹10,000–50,000+                                  | ₹2,500–2,800                                         |
| ML pipeline           | Single model or rule-based                       | 3-model pipeline (anomaly + regression + clustering) |
 
---
 
## 3. System Design

---
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
2. Post-spray cooldown: for 10 minutes after any spray, the room reports unavailable (`room_available = 0`) and NO fresh spray is permitted (`dry_steps_remaining` starts at 2 and ticks down once per 5-min step)
3. Occupied → Vacant transition → baseline spray (5s)
4. Gas still poor + vacant + extra-spray budget > 0 → capped extra spray (8s)
5. Budget exhausted, gas still poor → STOP spraying, set needs_manual_checkup = 1
6. Idle 4+ hours + vacant → one refresh spray (backstop)
```

The two-spray cap prevents wasteful continuous spraying and instead flags the situation for human inspection — the system treats repeated ineffective spraying as a probable sensor or ventilation fault rather than a dirty seat. The cooldown period is currently enforced in the data generator and reflected in the dataset (`room_available`); the equivalent lockout timer on the ESP32 firmware is outstanding work (see Section 6).

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
 
The system was validated on a **synthetic** dataset generated by a physics-informed simulator (`generate_washroom_data.py`). Dataset properties: 100,224 rows, 4 cubicles, 87 days, 5-minute resolution, 18 columns, 0 null values.
 
<!-- [FIXED] Added explicit reminder — this belongs in every section that reports a metric, not
     just the abstract, since results sections are often read in isolation. -->
**This is simulated data, not field-collected sensor logs.** Every result in this section should be read as "achieved on synthetic validation data" until real-hardware validation is complete.
 
### 4.2 Feature Engineering
 
Three feature sets were engineered for the three ML models:

**Anomaly Detection (16 features)**: Rolling statistics of MQ135 (30-min and 1-hour windows), rate of change, spray count per hour, combined occupancy signal, time-of-day encoding (sin/cos), day of week, entry count, deep clean recency.

**Virtual Sensing (13 features)**: Previous disinfectant level (lagged by 1 step — critical state variable), cumulative sprays since last refill, time since last spray, rolling spray rates (6h and 24h windows), gas rolling statistics, occupancy rate, time encoding, current spray status.

**Usage Clustering (24 features)**: Average hourly occupancy rate for each of the 24 hours, one vector per cubicle.


### 4.3 Model Training
 
All models use a time-based train/test split (first 69% = train, last 31% = test) to prevent data leakage from adjacent 5-minute readings — this part of the methodology is sound and should be kept.
 
<!-- [FIXED] Added caveat: a timing claim with no comparison point isn't very informative. -->
GPU acceleration via cuML was used for training. No CPU-only timing was reported for comparison, so the practical speedup this provides at this dataset size (100K rows, low-dimensional features — well within what scikit-learn typically handles quickly on CPU) is currently unclear. Recommend reporting both CPU and GPU times if this is kept as a result.
 
### 4.4 Results
 
#### 4.4.1 Anomaly Detection
 
Isolation Forest detected anomalies at a rate close to the 5% contamination parameter. <!-- [FIXED] see 2.3 note --> As noted in Section 2.3, this match is expected by construction and is not independent evidence of detection quality; a precision/recall evaluation against labeled or injected anomalies would substantiate this claim.
 
#### 4.4.2 Virtual Sensing (Core Result)

**Resolution of prior methodology concerns.** Two methodological questions previously flagged for this result are now resolved by changing the evaluation itself rather than qualifying the numbers:

1. **Walk-forward evaluation** — the train pipeline now performs an honest walk-forward: the model is seeded once per cubicle at the start of the test window with the true level, then receives only its *own* recursive predictions, reset to 100.0 at observed refill events (`disinfectant_refill_status = 1`). The teacher-forced protocol that originally produced R² = 0.9957 is retained only as a labeled reference/sanity check (`*_LEAKY_reference_only`) and is not reported as a performance claim.
2. **Fair baselines** — the naive baseline is now fixed to reset at each refill event exactly like the physical tank, and a persistence baseline (last known value, reset at refill) was added.

**Results (honest protocols, full test set, 4 cubicles):**

| Evaluation protocol | MAE (%) | RMSE (%) | R² |
| ------------------- | ------- | -------- | -- |
| Teacher-forced (reference only; not a claim) | 0.300 | 1.472 | 0.9965 |
| **Walk-forward (honest, refill-reset)** | **38.084** | **45.644** | **−2.396** |
| Rollout between refills (strict, seeded per refill) | 28.711 | 35.010 | −0.956 |
| Naive baseline (fixed, resets at refill) | 42.468 | 49.162 | −2.940 |
| Persistence baseline (last known value) | 38.257 | 45.431 | −2.365 |

**Framing for the paper.** The original narrative — a trained model dramatically outperforming a simple counter — does **not** survive honest evaluation on the synthetic dataset. Two defensible claims do:

1. **Virtual sensing is tractable as a short-horizon state estimate, not as a long multi-step forecast.** With the true previous level fed in (which a real system *does* have through its own prior estimate plus occasional refill observations), per-row error is small (MAE 0.30). When forced to bootstrap forward unobserved, error compounds because synthetic depletion is dominated by irregular per-spray draws (partial/missed deliveries, nozzle drift) and refill resets — a bounded random walk that resists long-horizon prediction. This is an honest, publishable negative-result framing with a clear motivation for sequence models (Section 5.5).
2. **The model's learned signal is real but weak.** Honest walk-forward MAE (38.1) beats the fixed naive baseline (42.5) and persistence (38.3), and the strict refill-seeded rollout (MAE 28.7) is the best of all honest protocols — because observed refills re-ground the state every segment. In deployment, a refill log (a staff-toggle at top-up time) plus per-segment prediction is the realistic mode, not open-loop bootstrapping.

The synthetic nature of the data is itself a limitation: the depletion physics is a simulator, and the honest R² < 0 motivates validating against real deployment data before relying on the estimate for maintenance alerts.
 
#### 4.4.3 Usage Clustering
 
<!-- [FIXED] Softened from "successfully isolates... validates the pipeline's ability to distinguish
     fundamentally different traffic profiles" — a silhouette score of ~0.19 is a weak clustering
     signal by standard interpretation (scores below ~0.25-0.5 are generally considered weak/no
     substantial structure), and k=2 means three of the four deliberately-different profiles were
     NOT distinguished from one another. -->
K-Means with k=2 (by silhouette score = 0.1914) separates Cubicle_B_Station (the busiest simulated profile) from the other three. The silhouette score is low by standard interpretation guidelines, indicating a fairly weak clustering structure overall — three of the four deliberately different traffic profiles (office-steady, quiet-floor, lunch-spike) were not distinguished from each other despite being designed with different shapes, not just different volumes. This should be reported as a modest, honest result (one clear high-traffic outlier detected) rather than validation of the clustering approach's ability to separate all traffic types; the small sample size (4 cubicles) limits any stronger claim.
 
### 4.5 Safety Verification
 
The dataset was verified to contain **zero** disinfection events while either occupancy sensor indicated the washroom was in use, and `room_available` correctly tracks the 10-minute post-spray cooldown for every spray row. Both are enforced by the rule-based controller (Section 3.2), not by the ML models, and are confirmed by unit tests (`test_no_spray_while_occupied`, `test_room_available_respects_cooldown`). The full suite is **18 tests**, all passing.
 
---
 
## 5. Dashboard and User Interface
 
*(Unchanged from the original draft.)*
 
---
The Streamlit dashboard provides three views:

1. **Live Monitoring**: Real-time metrics (occupancy, air quality, tank levels, mist status), side-by-side LD2410/IR readings, activity feed, hygiene score trend
2. **Predictive Maintenance & ML**: Disinfectant forecast with time-to-refill, anomaly detection timeline, sensor agreement panel, model vs. naive baseline comparison
3. **Historical Analytics**: Hourly occupancy heatmap, air quality trends with spray markers, daily resource consumption charts

The dashboard is the primary interface in Phase 1; a physical OLED display mounted outside the washroom is planned for Phase 2.

---
 
## 6. Conclusion and Future Work
 
This paper presented a low-cost, retrofit-friendly smart washroom system that addresses the practical constraints of existing public washroom infrastructure. The key contributions are:
  
1. A **virtual sensing approach** designed to eliminate the need for a physical disinfectant level sensor, evaluated with an honest walk-forward protocol: we report that one-step (teacher-forced) accuracy does **not** transfer to multi-step deployment performance on synthetic data (honest R² < 0) and quantify the refill-seeded rollout regime that is the realistic field mode.
2. A **safety-critical occupancy-gated control system** that uses dual sensors (mmWave + IR) with OR-voting to ensure zero disinfection events while occupied, plus a post-spray 10-minute cooldown (`room_available`) that blocks occupancy entry and fresh spraying — both verified by test on the synthetic dataset.
3. A **complete ML pipeline** combining anomaly detection, virtual sensing, and usage clustering, with a live Streamlit dashboard.
**Future work** includes:
  
- Validation on real sensor data from the ESP32 prototype (the most important open item — every headline number in this paper is currently synthetic-only)
- Implementing the post-spray cooldown lockout timer in the ESP32 firmware and publishing it to the repo (currently simulation + dashboard only)
- Refill log integration (staff toggle at top-up) so the deployment re-seeds prediction segments per the rollout protocol
- Adaptive contamination thresholds for anomaly detection
- LSTM/GRU models for time-series virtual sensing (motivated directly by the honest R² < 0 result)
- Automatic metered disinfectant refill from a concentrate reservoir
- Cloud dashboard for multi-site smart city deployment
- CO₂ and ammonia-specific sensors for better-calibrated air quality readings
---
 
## References
 
<!-- [VERIFY-ALL] Verification completed 2026-09-24: every numbered reference below has been individually
     confirmed against IEEE Xplore / Google Scholar / the publisher, or replaced with a verified real source.
     No reference remains tagged [VERIFY]. -->

[1] Swachh Bharat Mission. "Swachh Bharat Mission — Phase II." Government of India, 2021. — *low-risk, government program citation*

[2] Wankhade, K. "Urban sanitation in India: Key shifts in the national policy frame." *Environment and Urbanization*, vol. 27, no. 2, pp. 555–572, 2015. doi:10.1177/0956247814567058. — **[REPLACED + CONFIRMED 2026-09-24: original "Kumar, A. et al., J. Environmental Management vol. 270, 2020" was NOT found. This verified peer-reviewed paper documents the operation-and-maintenance gap in Indian urban sanitation that the [2]-marked sentence relies on.]**

[3] IPToilet (Altersoft Innovations India Pvt. Ltd.). "Digitalized smart toilet with IoT sensors and automated self-cleaning." Company website, https://www.iptoilet.com/. — **[CONFIRMED 2026-09-24: verified real as company material (IoT-enabled tenant-marketed smart public toilet with patented self-cleaning technology); cite as company website, not as a patent/paper.]**

[4] *(merged with [3] above; original [4] removed)*

[5] Sherine Mary, W., Muthukumar, S., Manisha, A., Nandhini, K., Vanitha, R. "Sensor based automated washroom monitoring system." *Proc. IEEE International Conference on Emerging Devices and Smart Systems (ICEDSS)*, 2018, pp. 247–249. doi:10.1109/ICEDSS.2018.8544266. — **[REPLACED + CONFIRMED 2026-09-24: original "Alam, M. et al., IEEE ICC IoT 2018" was NOT found; this verified paper describes a sensor-based washroom monitoring system that infers cleanliness from indoor air quality and alerts a cleaning team — matching the sentence it supports.]**

[6] ~~Priya, R. and Sangeetha, K.~~ — **[NOT FOUND — do not restore; removed from text and not cited anywhere.]**

[7] Premkumar, M., Kanimozhi, S., Krishika, V., Sindhupriya, K. "Smart Lavatory with Automatic Cleansing System and Live Hygiene Maintenance for Public." *International Research Journal of Engineering and Technology (IRJET)*, vol. 7, no. 3, pp. 1239–1245, 2020. — **[REPLACED + CONFIRMED 2026-09-24: original "Kumar, S. et al., 'IoT-based smart toilet system with automatic cleaning'" was NOT found. This verified paper describes an IoT smart-toilet system with automatic flush, seat/floor cleansing, perfume dispensing, and single-PIR occupancy detection — matching the [7]-marked sentences.]**

[8] Azman, F.I., Salleh, N.L., Zakaria, M.A. "IoT-based smart hygiene monitoring system." *IEEE Access*, vol. 10, pp. 118345–118356, 2022. — **[REPLACED + CONFIRMED 2026-09-24: original "Ahmed et al., IEEE Access vol. 9, 2021" was NOT found on IEEE Xplore/Scholar (likely fabricated). This verified Azman et al. paper covers an ESP32-based IoT restroom hygiene monitoring system (ammonia/IAQ sensing, MQTT/InfluxDB for cloud monitoring) and is cited in-text at Sections 2.1 (Paragraph 2) and 2.1 (Paragraph 4).]**

[9] Labeodan, T., Zeiler, W., Boxem, G., Zhao, Y. "Occupancy measurement in commercial office buildings for demand-driven control applications—A survey and detection system evaluation." *Energy and Buildings*, vol. 93, pp. 303–314, 2015. doi:10.1016/j.enbuild.2015.02.028. — **[REPLACED + CONFIRMED 2026-09-24: original "Sharma, P. et al., Energy and Buildings vol. 209, 2020" was NOT found. This verified survey documents that PIR sensors are the most common single-sensor occupancy approach and cannot detect stationary occupants — matching the [9]-marked sentence.]**

[10] Hsu, P., Liu, G., Fang, S.-H., Wu, H.-C., Yan, K. "Novel robust on-line indoor occupancy counting system using mmWave radar." *IEEE Sensors Journal*, 2023. doi:10.1109/JSEN.2023.3266450. — **[REPLACED 2026-09-23: original "Wang, F. et al., IEEE Sensors Journal 2022" with the ">95% vs ~70%" statistic was NOT found (the specific mmWave-vs-PIR accuracy comparison could not be traced to a real source). This verified Hsu et al. paper reports 95.8% mmWave occupancy-counting accuracy and notes it "greatly outperforms other existing schemes," which supports the qualitative dual-sensor sentence at Section 2.2 without the invented >95%-vs-70% figure.]**

[11] Carrillo-Amado, Y.R., Califa-Urquiza, M.A., Ramón-Valencia, J.A. "Calibration and standardization of air quality measurements using MQ sensors." *Respuestas*, vol. 25, no. 1, pp. 70–77, 2020. doi:10.22463/0122820X.2408. — **[REPLACED + CONFIRMED 2026-09-24: original "Hanwell, M.D. et al., MQ-135 gas sensor characterization" was NOT found. This verified paper calibrates and standardizes MQ-135 (and other MQ-series) sensors against datasheet models with a public Arduino library, supporting the MQ-135 usage and calibration-caveats sentences at Section 2.3.]**

[12] Liu, F.T., Ting, K.M., Zhou, Z.-H. "Isolation forest." *IEEE International Conference on Data Mining (ICDM)*, 2008, pp. 413–422. — **CONFIRMED real and correctly cited.**
 
[13] Bandaragoda, T.R., Ting, K.M., Albrecht, D., Liu, F.T., Zhu, Y., Wells, J.R. "Isolation-based anomaly detection using nearest-neighbor ensembles." *Computational Intelligence*, vol. 34, no. 4, pp. 968–998, 2018. — **CONFIRMED 2026-09-23: real and correctly cited (doi:10.1111/coin.12156).**
  
[14] Kadlec, P., Grbić, R., Gabrys, B. "Review of adaptation mechanisms for data-driven soft sensors." *Computers & Chemical Engineering*, vol. 35, no. 1, 2011, pp. 1–24. — **[FIXED: this is the correct title/journal/volume/pages for the real 2011 Kadlec review; the original draft had the right author and year but the wrong title ("adaptive soft sensors in the process industry"), wrong journal (cited as Journal of Process Control), and wrong volume/pages.]**
  
[15] Chemali, E., Kollmeyer, P.J., Preindl, M., Emadi, A. "State-of-charge estimation of Li-ion batteries using deep neural networks: a machine learning approach." *Journal of Power Sources*, vol. 400, pp. 242–255, 2018. — **CONFIRMED 2026-09-23: real and correctly cited (doi:10.1016/j.jpowsour.2018.06.104).**

[16] Deb, C., Zhang, F., Yang, J., Lee, S.E., Shah, K.W. "A review on time series forecasting techniques for building energy consumption." *Renewable and Sustainable Energy Reviews*, vol. 74, pp. 902–924, 2017. — **CONFIRMED 2026-09-23: real and correctly cited (doi:10.1016/j.rser.2017.02.085).**
  
[17] Chen, H., Yang, J., Fu, X., et al. "Water quality prediction based on LSTM and attention mechanism: a case study of the Burnett River, Australia." *Sustainability*, vol. 14, no. 20, art. 13231, 2022. doi:10.3390/su142013231. — **[REPLACED 2026-09-23: original "Ma, X. et al., IEEE Access, vol. 8, 2020" was NOT found. The verified Chen et al. paper is the widely-cited LSTM+attention water-quality-prediction work and supports the same sentence in Section 2.4.]**
  
[18] Li, W., Gong, G., Fan, H., Peng, P., Chun, L., Fang, X. "A clustering-based approach for 'cross-scale' load prediction on building level in HVAC systems." *Applied Energy*, vol. 282, 2021. — **[REPLACED + CONFIRMED 2026-09-24: original "Breña, F. et al., Applied Energy vol. 285, 2021" was NOT found. This verified Applied Energy paper applies K-Means clustering to hourly building load profiles for load prediction, supporting the [18]-marked sentence at Section 2.5.]**
  
[19] Nikdel, L., Schay, A., Hou, D., Powers, S.E. "Data-driven occupancy profiles for apartment-style student housing." *Energy and Buildings*, vol. 246, art. 111070, 2021. doi:10.1016/j.enbuild.2021.111070. — **[REPLACED + CONFIRMED 2026-09-24: original "Liang, X. et al., Energy and Buildings vol. 232, 2021" was NOT found. This verified Energy and Buildings paper derives campus-building occupancy profiles via K-Means clustering, supporting the [19]-marked sentence at Section 2.5.]**
 
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
| 12V Pumps (×2) | 2 | 200 |
| Backup Reservoir + Tubing | 1 | 250 |
| Mist Nozzle | 1 | 120 |
| LED + Misc | 1 | 320 |
| **Total** | | **₹2,500–2,800** |


## Appendix B: Training Configuration

All models trained on NVIDIA GeForce RTX 2050 (4GB VRAM), cuML 26.08, Python 3.14.4. Training time: ~5 s for anomaly detection + usage clustering; virtual sensing fit plus its honest walk-forward/rollout evaluation completes in a few minutes on GPU. Time-based split: train (2026-08-01 → 2026-09-30, 69,152 rows / 61 dates) and test (2026-09-30 00:40 → 2026-10-26, 31,072 rows / 27 dates).
 






