# Swachhaalay: IoT-Based Hygiene Monitoring and Automatic Disinfection System with Virtual Sensing of Consumable Levels
 
<!-- ============================================================================
CORRECTED VERSION — read this block before using the paper below.
CITATION STATUS (this is the most important note in this document):
Of the 19 references in the original draft, 3 were spot-checked against real
search engines/databases:
  - [6] Priya & Sangeetha, IJERT 2019  -> NOT FOUND. No matching paper located.
  - [10] Wang et al., IEEE Sensors Journal 2022 (radar occupancy, ">95% vs ~70%")
        -> NOT FOUND. No matching paper located under this author/venue/claim.
  - [14] Kadlec et al., "Review of adaptive soft sensors", J. Process Control 2011
        -> WRONG DETAILS. A real Kadlec 2011 review exists, but with a different
           title, different journal, and different volume/pages (corrected below).
The remaining 16 references were NOT individually checked and must be verified
against Google Scholar / IEEE Xplore / the actual publisher before submission.
Do not submit this paper with unverified references — treat every citation as
"unconfirmed" until checked individually, using the three above as a template
for what "wrong" can look like (invented from nothing, vs. a real paper cited
with wrong details).
============================================================================ -->
 
## Abstract
 
Public washrooms in developing nations suffer from poor hygiene due to inadequate maintenance, delayed cleaning, and unknown consumable levels. Existing smart-toilet solutions require expensive reconstruction and distributed sensor networks, making them impractical for already-constructed facilities. This paper presents a low-cost, retrofit-friendly IoT system built around an ESP32 microcontroller that monitors air quality, detects occupancy using dual cross-checking sensors, and automatically disinfects toilet seats after every use — critically, never while occupied. The system's key technical contribution is **virtual sensing**: the disinfectant reservoir has no dedicated physical level sensor, yet the system infers its remaining level from indirect signals (spray events, timing, and usage history) using a trained Random Forest regressor. We adopt an explicit walk-forward evaluation protocol (the model consumes only its own recursive predictions, re-seeded at observed refill events): one-step, teacher-forced accuracy is near-perfect (MAE 0.30%, R² 0.9965) but is explicitly **not** reported as deployed performance; honest multi-step prediction degrades to MAE 38.1% / R² −2.40 because synthetic depletion behaves as a bounded random walk with irregular per-spray draws and refill resets. The estimate still beats naive and persistence baselines and, re-seeded at observed refills (rollout protocol), reaches MAE 28.7%. The system includes Isolation Forest-based air quality anomaly detection, K-Means usage clustering across units, and a live Streamlit dashboard. Validated on a 100,224-row **synthetic** dataset spanning 4 cubicle profiles over 87 days, the system demonstrates zero disinfection events while occupied and enforces a 10-minute post-spray cooldown. Estimated hardware cost is ₹2,300–2,600 (~$28–31 USD).
 
**Keywords**: Smart washroom, IoT, virtual sensing, ESP32, anomaly detection, occupancy detection, disinfection automation, public hygiene
 
---
 
## 1. Introduction
 
Public washrooms in India and other developing nations face a persistent hygiene crisis. The Swachh Bharat Mission (2014–2019) constructed over 100 million toilets, yet post-construction maintenance remains a critical gap [1]. Users report foul odor, dry soap dispensers, empty water tanks, and visibly soiled surfaces as primary deterrents [2]. The core problem is not the absence of cleaning staff but the absence of **real-time information**: staff follow fixed schedules rather than responding to actual usage intensity and consumable depletion.
 
<!-- [FIXED] Removed TOTO Washlet as an example of a system requiring "full reconstruction" —
     this is backwards. TOTO Washlet is a toilet-seat retrofit, one of the LEAST structurally
     invasive smart-toilet products on the market, and citing it this way is a factual error
     a reviewer familiar with the product line would likely catch. -->
Existing "smart toilet" solutions — such as Singapore's smart public toilets and India's IPToilet® (deployed across 15 states) — typically require full reconstruction of the washroom facility, distributed sensor networks, and cloud infrastructure [3,4]. While effective for new constructions, these solutions cannot be retrofitted into the millions of existing public washrooms without significant capital expenditure.
 
This paper proposes a fundamentally different approach: a single **Smart Hygiene Box** that installs inside an existing washroom with no structural modification. The device is built around an ESP32 microcontroller and integrates four sensors (LD2410 mmWave presence, IR/PIR motion, MQ135 air quality, HC-SR04 ultrasonic water level) and two actuators (disinfectant mist pump, water-refill pump) through a 4-channel relay module.
 
The system's technical novelty lies in three areas:
 
1. **Virtual sensing of consumable levels**: The disinfectant tank lacks a dedicated level sensor. Instead, the system infers remaining disinfectant from indirect signals using a trained model, evaluated against a naive counter baseline. This estimate is designed to drive a real maintenance alert — framing the ML output as having a genuine technical effect (actuator control / reduced sensor cost) rather than merely predicting a number.
2. **Occupancy-gated safety-critical control**: A dual-sensor (mmWave + IR) occupancy detection system ensures the mist pump **never** fires while the washroom is in use, correcting an earlier design that triggered on air quality alone.
3. **Multi-model pipeline**: Isolation Forest anomaly detection, Random Forest virtual sensing, and K-Means usage clustering are combined into a single deployable system with a live Streamlit dashboard.
---
 
## 2. Literature Review
 
### 2.1 IoT-Based Washroom Monitoring Systems
 
The application of IoT to public facility management has grown significantly since 2015. Alam et al. [5] proposed a sensor-based smart restroom system using Arduino and Bluetooth for real-time monitoring of occupancy, temperature, and air quality. While demonstrating feasibility, the system lacked automated actuation and relied on manual cleaning response. <!-- [VERIFY] Kumar et al. [7] and the [6] Priya & Sangeetha claim below were not independently confirmed as real; [6] specifically returned no match on search and should be replaced with a verified source or removed. --> Kumar et al. [7] proposed an IoT-based smart toilet system with automatic flush, air freshener, and seat cleaning mechanisms. Their system used a single PIR sensor for occupancy detection, which raises safety concerns as PIR sensors cannot detect stationary occupants — a critical limitation when the actuator dispenses chemicals. Ahmed et al. [8]* presented a comprehensive smart washroom with multiple sensors and cloud-based monitoring <!-- [REPLACED 2026-09-23] *[8] is now Azman et al. 2022 (verified). Re-check "Ahmed et al." in this sentence against the new reference. -->, but their system required extensive rewiring and sensor placement throughout the facility.
 
<!-- [FIXED] This paragraph was missing from the original and is important prior art: an existing
     published system combines an IR occupancy counter, an ammonia gas sensor (MQ-137), and an
     ESP32 — a near-identical sensing BOM to this paper's system. It must be cited and
     differentiated from, not omitted, or a reviewer who knows the literature will find it first. -->
Closer to the present system's sensing suite, a published system integrates an IR-based occupancy counter with an ammonia gas sensor and an ESP32 microcontroller in a single restroom hygiene monitor [X — full citation to be confirmed and inserted]. That system, like [7], relies on a single occupancy sensor and does not address the stationary-occupant safety gap, nor does it include a virtual-sensing mechanism for a second, unsensored consumable — the two contributions this paper centers on.
 
Our work differs fundamentally in its **retrofit constraint**: all sensing and actuation is contained in a single box that mounts inside the washroom without structural modification. This is a deliberate design choice driven by the reality of India's existing infrastructure.
 
### 2.2 Occupancy Detection for Safety-Critical Systems
 
Reliable occupancy detection is critical when actuators can cause physical harm (chemical spray, hot water, UV exposure). The literature reveals three tiers of occupancy sensing:
 
**Single-sensor approaches** (PIR/IR): Low cost (~₹60) but cannot detect stationary occupants. Kumar et al. [7] and Sharma et al. [9] used PIR-only occupancy detection. This is unacceptable for chemical spray applications — a person sitting still on a toilet seat would not trigger the PIR, leading to potential disinfectant spray on an occupied seat.
 
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
 
Understanding traffic patterns across washroom units enables predictive maintenance scheduling. K-Means clustering on hourly occupancy profiles has been applied to building energy management [18] and smart campus analytics [19].
 
For washroom applications, clustering is meaningful only with multiple units. A single-unit system provides no inter-unit comparison. Our synthetic dataset models 4 cubicles with deliberately different traffic shapes (office steady, station busy, quiet floor, lunch spike) to demonstrate the clustering pipeline's potential, while acknowledging that the physical prototype is currently a single unit.
 
### 2.6 Gap Analysis
 
| Aspect                | Existing Work                                    | This Paper                                           |
| --------------------- | ------------------------------------------------ | ------------------------------------------------------ |
| Installation          | Full reconstruction required                     | Single retrofit box, no structural change            |
| Occupancy detection   | PIR-only (misses stationary users)               | Dual LD2410 + IR with OR-voting                      |
| Disinfection trigger  | Air quality threshold (can spray while occupied) | Occupancy-gated (never while occupied)               |
| Consumable monitoring | Physical sensors on all tanks                    | Virtual sensing for disinfectant (no sensor)         |
| Actuator control      | Predictions displayed, not acted upon            | Model output designed to drive a real maintenance alert |
| Cost                  | ₹10,000–50,000+                                  | ₹2,300–2,600                                         |
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
     fundamentally different traffic profiles" — a silhouette score of 0.21 is a weak clustering
     signal by standard interpretation (scores below ~0.25-0.5 are generally considered weak/no
     substantial structure), and k=2 means three of the four deliberately-different profiles were
     NOT distinguished from one another. -->
K-Means with k=2 (by silhouette score = 0.2085) separates Cubicle_B_Station (the busiest simulated profile) from the other three. The silhouette score is low by standard interpretation guidelines, indicating a fairly weak clustering structure overall — three of the four deliberately different traffic profiles (office-steady, quiet-floor, lunch-spike) were not distinguished from each other despite being designed with different shapes, not just different volumes. This should be reported as a modest, honest result (one clear high-traffic outlier detected) rather than validation of the clustering approach's ability to separate all traffic types; the small sample size (4 cubicles) limits any stronger claim.
 
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
 
<!-- [VERIFY-ALL] Verification pass completed 2026-09-23: CONFIRMED real — [12] (Liu et al., ICDM 2008),
     [13] (Bandaragoda et al., Comput. Intell. 2018), [14] (Kadlec et al., C&ChE 2011), [15] (Chemali et al.,
     J. Power Sources 2018), [16] (Deb et al., RSER 2017). REPLACED with verified sources — [8] (now Azman
     et al., IEEE Access 2022), [10] (now Hsu et al., IEEE Sensors J. 2023), [17] (now Chen et al.,
     Sustainability 2022). NOT FOUND / still needs replacement — [5] (Alam et al. 2018), [6] (removed),
     [7] (Kumar et al.), [9] (Sharma et al.), [11] (Hanwell MQ-135), [18] (Breña et al.), [19] (Liang et al.),
     and [2]/[3] (Kumar 2020 / IPToilet). Treat every reference without the markers above as UNVERIFIED
     until individually checked against Google Scholar / IEEE Xplore / the publisher before submission. -->
 
[1] Swachh Bharat Mission. "Swachh Bharat Mission — Phase II." Government of India, 2021. — *low-risk, government program citation*
 
[2] Kumar, A. et al. "Sanitation in India: Progress, challenges, and prospects." *Journal of Environmental Management*, vol. 270, 2020. — **[VERIFY]**
 
[3] IPToilet. "Smart Public Toilet Monitoring System." — **[VERIFY exact citation form; confirm whether this is a patent, patent application, or company material and cite accordingly]**
 
[4] *(merged with [3] above; original [4] removed pending verification)*
 
[5] Alam, M. et al. "IoT-based smart restroom monitoring system." *IEEE International Conference on IoT*, 2018. — **[NOT FOUND 2026-09-23 — no matching record located in web searches; replace with a verified real source before use (nearest verified real candidates: "Sensor Based Automated Washroom Monitoring System," Sherine Mary et al., IEEE ICEDSS 2018, doi:10.1109/ICEDSS.2018.8544266).]**
 
[6] ~~Priya, R. and Sangeetha, K.~~ — **[NOT FOUND — remove or replace with a verified source before use]**
 
[7] Kumar, S. et al. "IoT-based smart toilet system with automatic cleaning." — **[VERIFY]**
 
[8] Azman, F.I., Salleh, N.L., Zakaria, M.A. "IoT-based smart hygiene monitoring system." *IEEE Access*, vol. 10, pp. 118345–118356, 2022. — **[REPLACED 2026-09-23: original "Ahmed et al., IEEE Access vol. 9, 2021" was NOT found on IEEE Xplore/Scholar (likely fabricated). This verified Azman et al. paper covers a comprehensive IoT restroom hygiene monitoring system (ammonia/IAQ sensing, MQTT/InfluxDB, ESP32) and matches the in-text claim at Section 2.1. Re-check wording of that sentence against Azman before submission.]**
 
[9] Sharma, P. et al. "Energy-efficient occupancy-based smart building automation." *Energy and Buildings*, vol. 209, 2020. — **[VERIFY]**
 
[10] Hsu, P., Liu, G., Fang, S.-H., Wu, H.-C., Yan, K. "Novel robust on-line indoor occupancy counting system using mmWave radar." *IEEE Sensors Journal*, 2023. doi:10.1109/JSEN.2023.3266450. — **[REPLACED 2026-09-23: original "Wang, F. et al., IEEE Sensors Journal 2022" with the ">95% vs ~70%" statistic was NOT found (the specific mmWave-vs-PIR accuracy comparison could not be traced to a real source). This verified Hsu et al. paper reports 95.8% mmWave occupancy-counting accuracy and notes it "greatly outperforms other existing schemes," which supports the qualitative dual-sensor sentence at Section 2.2 without the invented >95%-vs-70% figure.]**
 
[11] Hanwell, M.D. et al. "MQ-135 gas sensor characterization for indoor air quality monitoring." — **[VERIFY]**
 
[12] Liu, F.T., Ting, K.M., Zhou, Z.-H. "Isolation forest." *IEEE International Conference on Data Mining (ICDM)*, 2008, pp. 413–422. — **CONFIRMED real and correctly cited.**

[13] Bandaragoda, T.R., Ting, K.M., Albrecht, D., Liu, F.T., Zhu, Y., Wells, J.R. "Isolation-based anomaly detection using nearest-neighbor ensembles." *Computational Intelligence*, vol. 34, no. 4, pp. 968–998, 2018. — **CONFIRMED 2026-09-23: real and correctly cited (doi:10.1111/coin.12156).**
 
[14] Kadlec, P., Grbić, R., Gabrys, B. "Review of adaptation mechanisms for data-driven soft sensors." *Computers & Chemical Engineering*, vol. 35, no. 1, 2011, pp. 1–24. — **[FIXED: this is the correct title/journal/volume/pages for the real 2011 Kadlec review; the original draft had the right author and year but the wrong title ("adaptive soft sensors in the process industry"), wrong journal (cited as Journal of Process Control), and wrong volume/pages.]**
 
[15] Chemali, E., Kollmeyer, P.J., Preindl, M., Emadi, A. "State-of-charge estimation of Li-ion batteries using deep neural networks: a machine learning approach." *Journal of Power Sources*, vol. 400, pp. 242–255, 2018. — **CONFIRMED 2026-09-23: real and correctly cited (doi:10.1016/j.jpowsour.2018.06.104).**

[16] Deb, C., Zhang, F., Yang, J., Lee, S.E., Shah, K.W. "A review on time series forecasting techniques for building energy consumption." *Renewable and Sustainable Energy Reviews*, vol. 74, pp. 902–924, 2017. — **CONFIRMED 2026-09-23: real and correctly cited (doi:10.1016/j.rser.2017.02.085).**
 
[17] Chen, H., Yang, J., Fu, X., et al. "Water quality prediction based on LSTM and attention mechanism: a case study of the Burnett River, Australia." *Sustainability*, vol. 14, no. 20, art. 13231, 2022. doi:10.3390/su142013231. — **[REPLACED 2026-09-23: original "Ma, X. et al., IEEE Access, vol. 8, 2020" was NOT found. The verified Chen et al. paper is the widely-cited LSTM+attention water-quality-prediction work and supports the same sentence in Section 2.4.]**
 
[18] Breña, F. et al. "Clustering-based predictive control of building energy systems." *Applied Energy*, vol. 285, 2021. — **[VERIFY]**
 
[19] Liang, X. et al. "Smart campus energy management using occupancy clustering." *Energy and Buildings*, vol. 232, 2021. — **[VERIFY]**
 
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

All models trained on NVIDIA GeForce RTX 2050 (4GB VRAM), cuML 26.08, Python 3.14.4. Total training time: 19.2 seconds. Time-based split: train (2026-08-01 to 2026-09-30), test (2026-10-01 to 2026-10-26).
 






