# Training Log — Smart Washroom ML Pipeline

## Experiment 1: Anomaly Detection
- **Date**: 2026-08-26 (re-run on new 18-column dataset: 2026-09-24)
- **Model**: Isolation Forest (cuML GPU)
- **Features**: 16 engineered (rolling stats, time encoding, gas dynamics)
- **Contamination**: 5%
- **n_estimators**: 200
- **Train set**: 69,152 rows | **Test set**: 31,072 rows
- **Result**: 5.21% anomaly rate on test set (matches contamination setting)
- **GPU training time**: 1.7s
- **Notes**: cuML IsolationForest serialization warning — model needs re-fitting after unpickling. The dashboard only reads stored `anomaly_predictions.csv` + meta (it never calls `.predict()`), so the unpickling re-fit issue does not affect the running app. If live inference is added later, save the training data alongside the artifact and re-fit on load (or persist a CPU sklearn version).

## Experiment 2: Virtual Sensing

### Run 2026-08-26 (SUPERSEDED — do not quote)
Original run reported teacher-forced (leaky) metrics MAE **0.311** / R² **0.9957** and a naive baseline MAE 57.744 (185× claim). That protocol fed the model the **true** previous tank level at every test row — impossible at deployment — and the naive baseline used was buggy (it never reset at refill). These numbers have been replaced.

### Run 2026-09-23 (current, honest)
- **Model**: Random Forest Regressor (cuML GPU)
- **Features**: 13 (prev_level, sprays_since_refill, rolling rates, gas dynamics, time encoding)
- **Target**: `disinfectant_level_virtual_pct`
- **Train set**: 69,152 rows | **Test set**: 31,072 rows
- **Generator**: v2 physics — per-spray draw `U(0.10, 0.55)`, partial-delivery (15%), missed-spray (5%), nozzle drift 0.006/day, post-spray 10-min cooldown (`room_available` column added, dataset now 18 columns)
- **GPU training time**: cuML fit ~11s + honest eval sweeps ~7 min

### Results (honest evaluation protocol)
| Evaluation | MAE | RMSE | R² |
|-----------|-----|------|-----|
| Teacher-forced (reference only) | 0.300 | 1.472 | 0.9965 |
| **Walk-forward (honest, refill-reset)** | **38.084** | **45.644** | **−2.396** |
| Rollout between refills (strict) | 28.711 | 35.010 | −0.956 |
| Naive baseline (fixed, resets at refill) | 42.468 | 49.162 | −2.940 |
| Persistence baseline (last known value) | 38.257 | 45.431 | −2.365 |

- **Key finding**: `prev_disinfectant_level` is an essential state feature — with it, one-step (teacher-forced) prediction is near-perfect (R² 0.9965). That number is **deployed-performance meaningless**: when the model must rely on its OWN previous predictions (walk-forward), error compounds and R² is strongly negative because synthetic depletion is dominated by irregular draw noise and refill resets.
- **New finding**: the fixed naive baseline (42.5 MAE) vs honest walk-forward (38.1 MAE) still shows a real learned signal, but the strict rollout (28.7 MAE over refill-bounded segments) is the realistic operating target once refills are observed in the field.
- **vs Naive**: honest walk-forward is 1.12× better than the fixed naive baseline in MAE, not 185×. The earlier 185× was the leaky protocol × the buggy baseline.

### Feature Importance Notes
The most important features (by permutation importance):
1. `prev_disinfectant_level` — current state (lagged)
2. `sprays_since_refill` — cumulative depletion
3. `spray_rate_6h` — recent usage intensity
4. `hours_since_last_spray` — time since last event

## Experiment 3: Usage Clustering
- **Date**: 2026-08-26
- **Model**: K-Means (cuML GPU)
- **Input**: 24-dim hourly occupancy profile per cubicle
- **k search range**: 2-3 (limited by 4 cubicles)

### Results
| k | Inertia | Silhouette |
|---|---------|------------|
| 2 | 0.44 | 0.1914 |
| 3 | 0.14 | 0.1336 |

- **Selected k=2**: Cubicle B isolated from A/C/D
- **GPU training time**: 2.3s
- **Limitation**: Only 4 cubicles means silhouette scores are inherently low. Need 10+ units for meaningful clustering.

## Hardware Used
- **GPU**: NVIDIA GeForce RTX 2050 (4GB VRAM)
- **CUDA**: 13.1
- **cuML**: 26.08
- **Python**: 3.14.4
- **Total training time**: anomaly ~1.7s + clustering ~2.3s; virtual sensing fit ~11s + honest walk-forward/rollout evaluation sweeps ~7 min (all 3 models)

## Next Steps
- [ ] Validate on real sensor data from ESP32 prototype
- [ ] Implement the post-spray cooldown lockout timer on the ESP32 firmware and publish it to this repo
- [ ] Add XGBoost/LightGBM as alternative regressors for benchmarking
- [ ] Explore sequence models (LSTM/GRU/transformer) for multi-step virtual sensing — the honest R² < 0 result is the motivation
- [ ] Field protocol: staff-topped refill log (switch/toggle) so every prediction segment is re-seeded per the rollout protocol
- [ ] Adaptive contamination threshold for anomaly detection
- [ ] Collect 10+ cubicle profiles for meaningful usage clustering
