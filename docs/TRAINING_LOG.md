# Training Log — Smart Washroom ML Pipeline

## Experiment 1: Anomaly Detection
- **Date**: 2026-08-26
- **Model**: Isolation Forest (cuML GPU)
- **Features**: 16 engineered (rolling stats, time encoding, gas dynamics)
- **Contamination**: 5%
- **n_estimators**: 200
- **Train set**: 69,152 rows | **Test set**: 31,072 rows
- **Result**: 5.02% anomaly rate on test set (matches contamination setting)
- **GPU training time**: 1.7s
- **Notes**: cuML IsolationForest serialization warning — model needs re-fitting after unpickling. Consider saving training data + re-fitting approach for production.

## Experiment 2: Virtual Sensing
- **Date**: 2026-08-26
- **Model**: Random Forest Regressor (cuML GPU)
- **Features**: 13 (prev_level, sprays_since_refill, rolling rates, gas dynamics, time encoding)
- **Target**: `disinfectant_level_virtual_pct`
- **Train set**: 69,152 rows | **Test set**: 31,072 rows

### Results
| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| RandomForest (GPU) | 0.311 | 1.554 | 0.9957 |
| Naive Baseline | 57.744 | 62.281 | -5.8452 |

- **Key finding**: Including `prev_disinfectant_level` as a feature was critical. Without it, R² dropped to ~0.92. The lagged target acts as a state variable that captures the current tank level, making the problem tractable.
- **GPU training time**: 15.3s
- **vs Naive**: 185x better MAE

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
| 2 | 0.66 | 0.2085 |
| 3 | 0.21 | 0.1255 |

- **Selected k=2**: Cubicle B isolated from A/C/D
- **GPU training time**: 2.3s
- **Limitation**: Only 4 cubicles means silhouette scores are inherently low. Need 10+ units for meaningful clustering.

## Hardware Used
- **GPU**: NVIDIA GeForce RTX 2050 (4GB VRAM)
- **CUDA**: 13.1
- **cuML**: 26.08
- **Python**: 3.14.4
- **Total training time**: 19.2s (all 3 models)

## Next Steps
- [ ] Validate on real sensor data from ESP32 prototype
- [ ] Add XGBoost/LightGBM as alternative regressors for benchmarking
- [ ] Explore LSTM/GRU for time-series virtual sensing (stretch goal)
- [ ] Adaptive contamination threshold for anomaly detection
- [ ] Collect 10+ cubicle profiles for meaningful usage clustering
