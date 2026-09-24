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
| **Training Time (GPU)** | ~11s |

### Performance

**All reported numbers below are NEW, honest evaluations.** An earlier version of this card reported teacher-forced metrics (MAE 0.311 / R² 0.9957) — that protocol feeds the model the *true* previous tank level at every test row and is impossible in deployment; it has been corrected.

| Evaluation protocol | MAE (%) | RMSE (%) | R² |
|---------------------|---------|----------|----|
| Teacher-forced (reference only; pipeline sanity check) | 0.300 | 1.472 | 0.9965 |
| **Walk-forward (honest headline)** — seed once, model's own predictions fed forward, reset at observed refills | **38.084** | **45.644** | **−2.396** |
| Rollout between refills (strict) — seed at each refill, own-predictions only, pre-first-refill rows excluded | 28.711 | 35.010 | −0.956 |
| Naive baseline (fixed: level reset to 100 at refill, else unchanged) | 42.468 | 49.162 | −2.940 |
| Persistence baseline (last known value, reset at refill) | 38.257 | 45.431 | −2.365 |

### Interpretation

- The teacher-forced row is **not a claim**. It is computed only as a sanity check that the training/eval code path is intact.
- On the current *synthetic* dataset the honest walk-forward model is **not** predictive across multiple steps in the R² sense (R² < 0): once the model must rely on its own past predictions, error compounds because depletion is dominated by irregular spray draws and refill resets. It still beats the naive "level never changes" baseline in MAE (38.1 vs 42.5) — i.e., the learned relationship is real but too weak to carry multi-step forecasts on synthetic physics.
- **Deployment requirement**: because single-seed walk-forward degrades, a physical refill log (staff toggles a switch when topping up) is mandatory in the field so every segment is re-seeded like the rollout protocol; segment-level rollout (MAE 28.7, R² −0.956) is the realistic operating regime once refills are observed.

### Feature Set
`prev_disinfectant_level`, `sprays_since_refill`, `hours_since_last_spray`, `cumulative_entry_count`, `spray_rate_6h`, `spray_rate_24h`, `gas_rolling_mean_1h`, `gas_rolling_max_6h`, `occ_rate_1h`, `hours_since_deep_clean_val`, `hour_sin`, `hour_cos`, `mist_maker_status`

### Novelty Claim
The disinfectant tank has **no physical level sensor**. This model infers remaining level from indirect signals (spray counts, timing, occupancy history) and uses that estimate to drive a **real alert** (maintenance notification). Virtual sensing as a concept is already in the bibliography for mmWave/radar sensing; the Swachhaalay contribution is a hygiene-management application of it, not the regression technique itself. The honest walk-forward evaluation above shows multi-step accuracy on synthetic physics is limited (R² < 0) — an earlier "185× improvement over naive baseline" claim was an artifact of a leaky (teacher-forced) evaluation and has been removed.

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
4. **Multi-step virtual sensing is hard**: honest walk-forward/rollout R² is negative on synthetic data. In the field, observed refills (staff toggling a float/null switch on top-up) are required to re-seed per-segment prediction; the unfed-forward teacher-forced metric must never be quoted as expected performance.
5. **Cooldown is simulation-only for now**: `room_available` enforces the post-spray 10-minute dry period in the data generator and dashboard, but the ESP32 firmware implementing the cooldown lockout timer is not yet in this repo.
6. **Time-based split caveat**: Train/test split is time-based (days 1-60 vs 61-87), but distribution shift between periods is not explicitly modeled.

## Hardware Requirements

- **Training**: NVIDIA GPU with CUDA support (RTX 2050 tested) for cuML acceleration
- **Inference**: CPU-only is sufficient; cuML models can fall back to scikit-learn
- **Dashboard**: Streamlit server (single-user or small team)
