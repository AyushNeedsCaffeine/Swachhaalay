# Patent Draft — Smart Hygiene Box

## Title
Low-Cost Retrofit IoT System with Virtual Sensing for Automatic Washroom Disinfection and Consumable Level Estimation

## Field of Invention
The present invention relates to Internet of Things (IoT) systems for automated hygiene management in public washroom facilities, and more specifically to a single-unit retrofit device that monitors air quality and occupancy, automatically disinfects toilet seats after use, and estimates consumable reservoir levels using machine learning without dedicated physical sensors.

---

## Background

Public washroom facilities in developing nations suffer from inadequate hygiene due to delayed maintenance, fixed cleaning schedules that do not respond to actual usage, and lack of real-time information about consumable depletion. Existing solutions either require full structural reconstruction of the washroom facility (e.g., TOTO Washlet systems, IPToilet installations) or provide monitoring without automated actuation. Furthermore, existing systems that automate cleaning face a fundamental safety problem: triggering chemical spray based on air quality alone risks spraying occupants.

The problem of monitoring consumable levels (disinfectant, soap, water) is typically addressed by adding dedicated level sensors to each reservoir. This increases cost, complexity, wiring, and potential failure points — particularly problematic for retrofit installations where the reservoir may be a simple open container without sensor mounting provisions.

---

## Summary of Invention

The present invention provides a single control box that mounts inside an existing washroom and provides three functions: (1) occupancy-gated automatic disinfection using dual-sensor safety verification, (2) virtual sensing of consumable reservoir levels using machine learning inference from indirect signals, and (3) air quality anomaly detection with automatic alert generation.

The key innovation is the virtual sensing approach: the disinfectant reservoir has no dedicated physical level sensor. Instead, a trained machine learning model (Random Forest regressor) estimates the remaining disinfectant level from indirect signals — spray event history, timing patterns, and usage statistics. This estimated level drives a real maintenance alert, not merely a displayed prediction, giving the ML output a genuine technical effect.

The system achieves near-perfect level estimation (R² = 0.996, MAE = 0.311%) while the naive approach of counting sprays and multiplying by average volume fails catastrophically (R² = -5.85, MAE = 57.74%), demonstrating that the trained model captures complex depletion dynamics that a simple counter cannot.

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
   - 12V peristaltic pump for disinfectant mist spraying (connected to backup reservoir)
   - 12V solenoid/pump for water level refill
4. **User interface**: RGB LED indicator (Red = Occupied, Green = Vacant)
5. **Power**: 5V DC adapter (USB-C or barrel jack)
6. **Connectivity**: Wi-Fi to cloud backend (Firebase or REST API)

Total hardware cost: approximately ₹2,300–2,600 (~$28–31 USD).

### Control Logic — Safety-Critical Occupancy Gating

The disinfection spray is controlled by a deterministic rule-based system, not by the ML model. The rule system:

1. Reads both occupancy sensors (LD2410 and IR/PIR) every 5 seconds
2. Computes an OR-combination: if EITHER sensor indicates occupied, the room is treated as occupied
3. When occupied: NO actuation of any pump under any circumstance
4. On transition from occupied to vacant (rising edge of vacant state): initiates baseline spray
5. Monitors gas levels post-spray; if gas remains high and extra-spray budget is available, initiates capped additional spray
6. If extra-spray budget is exhausted, sets a `needs_manual_checkup` flag rather than continuing to spray

The two-spray cap prevents wasteful continuous spraying and instead flags the situation for human inspection, treating repeated ineffective spraying as a probable sensor or ventilation fault.

### Virtual Sensing Method

The virtual sensing system comprises:

**a) Feature Engineering**: From the raw sensor stream, 13 features are computed:
- Lagged target variable (previous estimated disinfectant level) — acts as a state variable
- Cumulative spray count since last refill
- Time since last spray event
- Rolling spray rates over 6-hour and 24-hour windows
- Air quality statistics (rolling mean, max)
- Occupancy rate over the past hour
- Time-of-day encoding (sinusoidal)

**b) Model**: A Random Forest regressor trained on historical data where the ground truth was obtained from periodic manual measurement or a calibrated reference. The model is retrained periodically to account for changing usage patterns.

**c) Actuator Control**: The predicted disinfectant level is compared against a low-level threshold. When the estimated level falls below the threshold, the system:
- Sets a `disinfectant_alert` flag transmitted to the cloud dashboard
- Sends a notification to maintenance staff
- Optionally triggers an automatic metered refill from a concentrate reservoir (future revision)

**d) Evaluation**: The model achieves R² = 0.996 and MAE = 0.311% on a held-out test set (time-based split, no data leakage). The naive baseline (cumulative sprays × average volume per spray) achieves MAE = 57.74%, confirming that the trained model captures dynamics (varying spray volumes, refill events, usage patterns) that a simple counter cannot.

### Anomaly Detection Method

An Isolation Forest model trained on 16 engineered features from the MQ135 air quality sensor detects unusual patterns:
- Sharp gas spikes outside normal usage patterns
- Gradual sensor drift indicating MQ135 degradation
- Blocked ventilation or chemical spill events

The model runs inference on each new reading and flags anomalies with an anomaly score. Flagged events are logged to the dashboard and trigger a maintenance alert when the anomaly score exceeds a threshold.

### Usage Pattern Analysis

A K-Means clustering model profiles each washroom unit based on 24-dimensional hourly occupancy vectors. Units are clustered into usage profiles (e.g., "busy public", "office moderate", "lunch-spike") to enable differentiated maintenance scheduling across multiple installations.

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
The system of Claim 1, wherein the machine learning module comprises a Random Forest regressor trained on spray event history, timing patterns, air quality readings, and occupancy statistics, and wherein the estimated consumable level is used to generate a maintenance alert when the estimate falls below a configurable threshold.

### Claim 3
The system of Claim 1, wherein the rule-based controller implements a two-spray cap per occupancy cycle, such that after at most two spray events following a single vacancy event, the system sets a manual checkup flag instead of continuing to spray.

### Claim 4
The system of Claim 1, further comprising an Isolation Forest anomaly detection model trained on engineered features from the air quality sensor, wherein detected anomalies trigger a maintenance alert indicating a potential sensor fault or ventilation failure.

### Claim 5
The system of Claim 1, wherein the dual-sensor occupancy detection employs a conservative safety policy: the OR-logic ensures zero false-negative occupancy events (spraying an occupied room) at the cost of occasional false-positive occupancy events (delaying disinfection when the room is actually vacant).

### Claim 6
A method for estimating consumable reservoir levels without a physical sensor, comprising:
- Receiving a stream of spray event timestamps and durations from an automated dispensing system
- Computing rolling statistics over configurable time windows including spray rates, cumulative event counts, and time-since-last-event
- Inputting the computed features into a trained regression model to produce a level estimate
- Using the level estimate to trigger a maintenance alert or automatic refill, thereby providing a technical effect from the machine learning inference

### Claim 7
The method of Claim 6, wherein the regression model includes as a feature a lagged version of its own previous prediction, acting as a state variable that encodes the current estimated level and enables the model to track depletion dynamics across spray events.

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

A low-cost, retrofit IoT system for automatic washroom disinfection and hygiene monitoring. The system mounts as a single unit inside an existing washroom without structural modification. Dual occupancy sensors (mmWave + IR) with OR-logic ensure the disinfectant spray never fires while the room is occupied. A trained Random Forest model estimates the remaining disinfectant level from indirect signals (spray history, timing, air quality) without a dedicated physical level sensor, achieving R² = 0.996 and driving real maintenance alerts. An Isolation Forest model detects air quality anomalies. Total hardware cost: ~$30 USD.
