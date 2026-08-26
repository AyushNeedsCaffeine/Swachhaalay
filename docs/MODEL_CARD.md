# Model Card — Smart Washroom ML Pipeline

## Model 1: Anomaly Detection (Isolation Forest)

| Property | Value |
|----------|-------|
| **Model** | Isolation Forest |
| **Implementation** | cuML 26.08 (GPU) / scikit-learn 1.9 (CPU fallback) |
| **Task** | Unsupervised anomaly detection on air quality signals |
| **Input Features** | 16 engineered features from MQ135, occupancy, time |
| **Output** | Binary label: 0=Normal, 1=Anomaly + anomaly score |
| **Training Data** | 69,152 rows (first 69 days, 4 cubicles) |
| **Test Data** | 31,072 rows (last 18 days) |
| **Contamination** | 5% (expected anomaly rate) |
| **Actual Anomaly Rate** | 5.02% on test set |
| **n_estimators** | 200 |
| **Training Time (GPU)** | ~1.7s |

### Feature Set
`mq135_gas_ppm`, `hour_sin`, `hour_cos`, `day_of_week`, `gas_rolling_mean_30m`, `gas_rolling_std_30m`, `gas_rolling_max_30m`, `gas_rolling_mean_1h`, `gas_rolling_std_1h`, `gas_diff`, `gas_diff_abs`, `gas_roc_30m`, `spray_count_1h`, `is_occupied_combined`, `entry_count`, `hours_since_deep_clean`

### Use Case
Flags unusual air quality patterns (chemical spill, blocked drain, failing sensor) vs. normal high-usage spikes. Output feeds into the dashboard's anomaly alert panel and can trigger `needs_manual_checkup` confirmation.

---

## Model 2: Virtual Sensing (Random Forest Regressor) — Core Contribution

| Property | Value |
|----------|-------|
| **Model** | Random Forest Regressor |
| **Implementation** | cuML 26.08 (GPU) / scikit-learn 1.9 (CPU fallback) |
| **Task** | Regression: estimate disinfectant tank level from indirect signals |
| **Target** | `disinfectant_level_virtual_pct` (0-100%) |
| **Input Features** | 13 features from spray events, timing, usage history |
| **Training Data** | 69,152 rows (first 69 days, 4 cubicles) |
| **Test Data** | 31,072 rows (last 18 days) |
| **n_estimators** | 200 |
| **max_depth** | 15 |
| **Training Time (GPU)** | ~15s |

### Performance

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **Random Forest (ours)** | **0.311** | **1.554** | **0.9957** |
| Naive Baseline (spray × volume) | 57.744 | 62.281 | -5.8452 |

### Feature Set
`prev_disinfectant_level`, `sprays_since_refill`, `hours_since_last_spray`, `cumulative_entry_count`, `spray_rate_6h`, `spray_rate_24h`, `gas_rolling_mean_1h`, `gas_rolling_max_6h`, `occ_rate_1h`, `hours_since_deep_clean_val`, `hour_sin`, `hour_cos`, `mist_maker_status`

### Novelty Claim
The disinfectant tank has **no physical level sensor**. This model infers remaining level from indirect signals (spray counts, timing, occupancy history) and uses that estimate to drive a **real alert** (maintenance notification) — not just predict a number. The 185x MAE improvement over the naive baseline demonstrates that the learned model captures depletion dynamics that a simple counter cannot.

---

## Model 3: Usage Clustering (K-Means)

| Property | Value |
|----------|-------|
| **Model** | K-Means Clustering |
| **Implementation** | cuML 26.08 (GPU) / scikit-learn 1.9 (CPU fallback) |
| **Task** | Unsupervised clustering of cubicle traffic profiles |
| **Input** | 24-dimensional hourly occupancy rate vector per cubicle |
| **Optimal k** | 2 (by silhouette score) |
| **Silhouette Score** | 0.2085 |

### Cluster Assignments
| Cubicle | Cluster | Profile |
|---------|---------|---------|
| Cubicle_A_Office | 0 | Moderate traffic |
| **Cubicle_B_Station** | **1** | **High traffic (isolated)** |
| Cubicle_C_QuietFloor | 0 | Low traffic |
| Cubicle_D_LunchSpike | 0 | Moderate (lunch spike) |

### Interpretation
Cubicle B (busy public station) forms its own cluster due to consistently high occupancy across all hours. Cubicles A, C, and D cluster together despite different shapes — their average occupancy rates are more similar to each other than to B.

---

## Limitations

1. **Synthetic data**: All models trained on simulated data, not real sensor logs. Performance must be validated on real hardware.
2. **4 cubicles only**: Usage clustering is limited to 4 data points. More units needed for meaningful fleet analytics.
3. **Static contamination**: Anomaly detection uses fixed 5% contamination; real deployment may need adaptive thresholds.
4. **Time-based split caveat**: Train/test split is time-based (days 1-60 vs 61-87), but distribution shift between periods is not explicitly modeled.

## Hardware Requirements

- **Training**: NVIDIA GPU with CUDA support (RTX 2050 tested) for cuML acceleration
- **Inference**: CPU-only is sufficient; cuML models can fall back to scikit-learn
- **Dashboard**: Streamlit server (single-user or small team)
