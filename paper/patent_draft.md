# Patent Draft — Smart Hygiene Box
<!-- CORRECTED VERSION — changes from the original marked with [FIXED] comments below.
     This is a draft for attorney/agent review, not a filing-ready document. -->

## Title

Low-Cost Retrofit IoT System with Virtual Sensing for Automatic Washroom Disinfection and Consumable Level Estimation

## Field of Invention

The present invention relates to Internet of Things (IoT) systems for automated hygiene management in public washroom facilities, and more specifically to a single-unit retrofit device that monitors air quality and occupancy, automatically disinfects toilet seats after use, and estimates consumable reservoir levels using machine learning without dedicated physical sensors.

---

## Background

Public washroom facilities in developing nations suffer from inadequate hygiene due to delayed maintenance, fixed cleaning schedules that do not respond to actual usage, and lack of real-time information about consumable depletion.

<!-- [FIXED] Original text cited "TOTO Washlet systems" as an example requiring "full structural
     reconstruction." This is factually backwards — TOTO Washlet is a toilet-seat replacement,
     one of the LEAST structurally invasive smart-toilet products on the market. Removed it from
     this example and kept only the genuinely retrofit-heavy examples. -->
Existing solutions either require full structural reconstruction of the washroom facility (e.g., distributed multi-sensor smart-toilet installations, IPToilet-style prefabricated units) or provide monitoring without automated actuation. Furthermore, existing systems that automate cleaning face a fundamental safety problem: triggering chemical spray based on air quality alone risks spraying occupants.

The problem of monitoring consumable levels (disinfectant, soap, water) is typically addressed by adding dedicated level sensors to each reservoir. This increases cost, complexity, wiring, and potential failure points — particularly problematic for retrofit installations where the reservoir may be a simple open container without sensor mounting provisions.

<!-- [FIXED] Added: closer academic prior art was missing. A published system combining an IR
     occupancy counter, an ammonia gas sensor, and an ESP32 is a near-identical BOM to this
     invention's sensing suite and should be acknowledged and distinguished from, not omitted. -->
It is also noted that IoT hygiene-monitoring systems combining a gas sensor, an occupancy/IR sensor, and an ESP32-class microcontroller have been described in academic literature — notably Azman, Salleh & Zakaria, "IoT-based smart hygiene monitoring system," *IEEE Access*, vol. 10, pp. 118345–118356, 2022 (see also the companion paper draft, Section 2.1, reference [8]). The present invention differs from such systems specifically in (a) the dual-sensor occupancy safety gating described below, and (b) the virtual sensing method for a consumable with no dedicated physical sensor — neither of which is addressed by prior single-sensor or fully-instrumented designs.

---

## Summary of Invention

The present invention provides a single control box that mounts inside an existing washroom and provides three functions: (1) occupancy-gated automatic disinfection using dual-sensor safety verification, (2) virtual sensing of consumable reservoir levels using machine learning inference from indirect signals, and (3) air quality anomaly detection with automatic alert generation.

The key innovation is the virtual sensing approach: the disinfectant reservoir has no dedicated physical level sensor. Instead, a trained machine learning model (Random Forest regressor) estimates the remaining disinfectant level from indirect signals — spray event history, timing patterns, and usage statistics. This estimated level drives a real maintenance alert, not merely a displayed prediction, giving the ML output a genuine technical effect.

<!-- [FIXED] This is the most important correction in the whole document. The original text claimed
     ground truth came from "periodic manual measurement or a calibrated reference" — this did not
     happen. All current results are on a synthetic dataset. Filing with the original wording would
     misstate how the invention was validated. -->
<!-- [RESOLVED 2026-09-23] The R²=0.996/MAE=0.311 numbers below have been replaced. The previous
     figures came from a teacher-forced evaluation that fed the model the TRUE previous level at
     every test row (impossible in deployment) and from a naive baseline that never reset at refill.
     The pipeline now reports honest walk-forward/rollout metrics and a refill-resetting naive + a
     persistence baseline; the spec text reflects those corrected numbers. -->
At the current stage of development, the system has been validated on a physics-informed **synthetic dataset** (100,224 rows, 4 simulated cubicles, 87 days at 5-minute resolution) rather than field-collected sensor logs. Under an honest walk-forward evaluation (the model consumes only its own recursive predictions, re-seeded at observed refill events) accuracy is MAE ~38.1% / R² ~ −2.40, which is comparable to a persistence baseline and better than the fixed naive baseline; a stricter refill-seeded rollout reaches MAE ~28.7%. One-step teacher-forced accuracy (MAE 0.30%, R² 0.9965) is explicitly **not** a deployed-performance claim. Validation against real deployed sensors and human-judged ground truth is planned future work and should be completed, or the claims scoped to reflect simulation-only validation, before this application is finalized.

---

## Detailed Description

### System Architecture

The invention comprises a single enclosure containing:

1. **Processing unit**: ESP32 microcontroller with Wi-Fi connectivity
2. **Sensing suite**:
  - LD2410 mmWave radar sensor for presence detection of stationary occupants
  - IR/PIR motion sensor for cross-verification of occupancy
  - MQ135 metal-oxide gas sensor for air quality (ammonia, benzene, VOCs)
  - HC-SR04 ultrasonic sensor for water tank level measurement
3. **Actuation suite**:
  - 12V pump for disinfectant mist spraying (connected to the disinfectant reservoir)
  - 12V pump for water level refill (connected to a backup water reservoir)
4. **User interface**: LED indicator (simple Occupied/Vacant signal); OLED display planned for a later phase
5. **Power**: 5V DC adapter
6. **Connectivity**: Wi-Fi to cloud backend (Firebase or REST API)

Total hardware cost: approximately ₹2,500–2,800 (~$30–34 USD).

### Control Logic — Safety-Critical Occupancy Gating

The disinfection spray is controlled by a deterministic rule-based system, not by the ML model. The rule system:

1. Reads both occupancy sensors (LD2410 and IR/PIR)
2. Computes an OR-combination: if EITHER sensor indicates occupied, the room is treated as occupied
3. When occupied: NO actuation of any pump under any circumstance
4. After any spray, enforces a post-spray cooldown (~10 minutes) during which the room is reported unavailable (`room_available = 0`) and no fresh spray is permitted until the window elapses
5. On transition from occupied to vacant: initiates baseline spray
6. Monitors gas levels post-spray; if gas remains high and extra-spray budget is available, initiates capped additional spray
7. If extra-spray budget is exhausted, sets a `needs_manual_checkup` flag rather than continuing to spray

The two-spray cap prevents wasteful continuous spraying and instead flags the situation for human inspection, treating repeated ineffective spraying as a probable sensor or ventilation fault. The cooldown lockout timer is currently implemented in the simulation and dashboard; the equivalent firmware timer on the ESP32 is pending.

### Virtual Sensing Method

**a) Feature Engineering**: From the raw sensor stream, features are computed including a lagged version of the previously estimated disinfectant level, cumulative spray count since last refill, time since last spray event, rolling spray rates, air quality statistics, occupancy rate, and time-of-day encoding.

<!-- [RESOLVED 2026-09-23] The walk-forward protocol described as "should be confirmed" is now
     implemented. `walk_forward_predict()` in the repo seeds each cubicle once at the test-window
     boundary, feeds only the model's own previous predictions forward, and resets to 100.0 at
     observed refill events; rollouts are also computed per refill segment. -->
**Important methodological note**: one of the engineered features is a lagged version of the model's own target (the previous estimated disinfectant level). This is a legitimate and common technique in state-estimation problems, but it is **only valid when evaluation uses a true walk-forward protocol** (the model recursively consumes its own prior predictions at test time, with the state re-seeded at observed refill events). The implemented evaluation is such a protocol, and the resulting honest numbers are reported in Section 4.4.2 of the companion paper draft. The one-step teacher-forced result (R² 0.9965) is retained only as a pipeline sanity check and must not be cited as deployed performance.

**b) Model**: A Random Forest regressor trained on historical data. At the current stage this is synthetic simulation data; the model is intended to be retrained on field data as it becomes available.

**c) Actuator Control**: The predicted disinfectant level is compared against a low-level threshold. When the estimated level falls below the threshold, the system sets a `disinfectant_alert` flag transmitted to the cloud dashboard and sends a notification to maintenance staff. An automatic metered refill from a concentrate reservoir is noted as a possible future revision, not the current implementation.

<!-- [RESOLVED 2026-09-23] Baseline now resets at refill (like the physical tank) and a persistence
     baseline was added; the honest numbers from the corrected evaluation are given below. -->
**d) Evaluation**: On synthetic validation data, under a true **walk-forward** protocol the model reaches MAE ≈ 38.1%, RMSE ≈ 45.6%, R² ≈ −2.40; a stricter rollout protocol re-seeded at observed refills reaches MAE ≈ 28.7%, R² ≈ −0.96. Both honest protocols are comparable to or better than the refill-reset naive baseline (MAE 42.5%) and the persistence baseline (MAE 38.3%). The teacher-forced one-step result (MAE 0.30%, R² 0.9965) is a pipeline sanity check only. These figures were produced by the corrected codebase; the previously reported 0.311/0.9957/185× figures are withdrawn.

### Anomaly Detection Method

An Isolation Forest model trained on engineered features from the MQ135 air quality sensor detects unusual patterns: sharp gas spikes outside normal usage patterns, gradual sensor drift, and blocked ventilation or chemical spill events. The model runs inference on each new reading and flags anomalies with an anomaly score.

### Usage Pattern Analysis

A K-Means clustering model profiles each washroom unit based on hourly occupancy vectors, to enable differentiated maintenance scheduling across multiple installations. At the current stage of development (4 simulated cubicles), this distinguishes one high-traffic profile from the others; more installations would be needed to validate finer-grained clustering.

---

## Claims

### Claim 1 (Independent)

A retrofit IoT system for automated washroom hygiene management, comprising:

- A single control box mountable inside an existing washroom without structural modification
- A dual-sensor occupancy detection system combining a mmWave radar sensor and an infrared motion sensor, wherein the system employs OR-logic such that the washroom is treated as occupied if either sensor indicates presence
- A disinfectant spray actuator controlled by a rule-based controller that inhibits actuation whenever the occupancy detection system indicates the washroom is occupied
- A machine learning module that estimates a remaining consumable level from indirect signals without a dedicated physical level sensor for said consumable
- A wireless communication module that transmits sensor data, actuator states, and estimated consumable levels to a remote dashboard

### Claim 2

The system of Claim 1, wherein the machine learning module comprises a regression model trained on spray event history, timing patterns, air quality readings, and occupancy statistics, and wherein the estimated consumable level is used to generate a maintenance alert when the estimate falls below a configurable threshold.

### Claim 3

The system of Claim 1, wherein the rule-based controller implements a two-spray cap per occupancy cycle, such that after at most two spray events following a single vacancy event, the system sets a manual checkup flag instead of continuing to spray.

### Claim 4

The system of Claim 1, further comprising an anomaly detection model trained on engineered features from the air quality sensor, wherein detected anomalies trigger a maintenance alert indicating a potential sensor fault or ventilation failure.

### Claim 5

The system of Claim 1, wherein the dual-sensor occupancy detection employs a conservative safety policy: the OR-logic ensures zero false-negative occupancy events (spraying an occupied room) at the cost of occasional false-positive occupancy events (delaying disinfection when the room is actually vacant).

### Claim 6

A method for estimating consumable reservoir levels without a physical sensor, comprising:

- Receiving a stream of spray event timestamps and durations from an automated dispensing system
- Computing rolling statistics over configurable time windows including spray rates, cumulative event counts, and time-since-last-event
- Inputting the computed features into a trained regression model to produce a level estimate
- Using the level estimate to trigger a maintenance alert or automatic refill, thereby providing a technical effect from the machine learning inference

### Claim 7

The method of Claim 6, wherein the regression model includes as a feature a lagged version of its own previous prediction, acting as a state variable that encodes the current estimated level and enables the model to track depletion dynamics across spray events, and wherein said previous prediction — not a ground-truth measurement — is used at inference time.

### Claim 8

A method for safe automatic disinfection of a washroom, comprising:

- Simultaneously reading a mmWave presence sensor and an infrared motion sensor
- Applying OR-logic to determine an occupancy state: occupied if either sensor indicates presence
- Inhibiting all actuator operation while the occupancy state is "occupied"
- Initiating a disinfectant spray upon detection of a transition from "occupied" to "vacant"
- Monitoring air quality after the spray and optionally initiating additional capped sprays
- Setting a manual inspection flag if the air quality does not improve after the maximum allowed spray count

---

## Abstract of Disclosure

A low-cost, retrofit IoT system for automatic washroom disinfection and hygiene monitoring. The system mounts as a single unit inside an existing washroom without structural modification. Dual occupancy sensors (mmWave + IR) with OR-logic ensure the disinfectant spray never fires while the room is occupied. A trained regression model estimates the remaining disinfectant level from indirect signals (spray history, timing, air quality) without a dedicated physical level sensor, and that estimate drives real maintenance alerts. Validated to date on synthetic simulation data; field validation is ongoing. An anomaly detection model flags air quality irregularities. Total hardware cost: ~$30–34 USD.

---

## Reviewer Checklist Before Filing

- [x] **RESOLVED**: Virtual-sensing evaluation uses true walk-forward validation (own predictions fed forward, re-seeded at observed refill events; strict per-refill rollouts also computed). Class-level numbers updated in spec.
- [x] **RESOLVED**: Naive baseline now resets at refill events; persistence baseline added. The 0.311 / 0.9957 / 185× figures previously cited are withdrawn.
- [ ] Confirm cooldown-lockout firmware timer is implemented on ESP32 before making the cooldown a claim (currently simulation + dashboard only)
- [ ] Replace "Background" prior-art discussion with attorney-reviewed prior art search results (this draft's prior-art list is illustrative, not a substitute for a professional search)
- [ ] Decide whether to file now on synthetic-data validation (with claims/spec scoped accordingly) or wait for field data
- [ ] Have a registered patent agent review claim scope and language