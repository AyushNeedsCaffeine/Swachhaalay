# Data Dictionary — washroom_dataset_multi_cubicle.csv

**Source**: `data/generate_washroom_data.py`
**Resolution**: 5-minute intervals
**Duration**: 87 days (2026-08-01 to 2026-10-26)
**Cubicles**: 4 (distinct traffic profiles)
**Total rows**: 100,224 (25,056 per cubicle)
**Columns**: 18

## Columns

| # | Column | Type | Range | Description |
|---|--------|------|-------|-------------|
| 1 | `timestamp` | datetime | 2026-08-01 00:00:00 → 2026-10-26 23:55:00 | 5-minute-resolution reading |
| 2 | `cubicle_id` | string | 4 values | Simulated cubicle identifier |
| 3 | `mq135_gas_ppm` | float | 100.0 → ~420 | Air quality / odor sensor (ammonia, smoke). Base ~110 ppm + noise + odor factor |
| 4 | `occupancy_ld2410` | int (0/1) | 0 or 1 | LD2410 mmWave presence sensor (catches stationary person) |
| 5 | `motion_ir` | int (0/1) | 0 or 1 | PIR motion sensor. If occupied, fires with 70% probability (30% miss rate). Always 0 when unoccupied |
| 6 | `water_level_cm` | float | 5.0 → 20.0 | Primary water tank level (HC-SR04 sensor). Refills to 20.0 when < 5.0 |
| 7 | `disinfectant_level_virtual_pct` | float | 15.0 → 100.0 | Disinfectant level — virtually sensed, no physical sensor. Refills to 100% when < 15% |
| 8 | `mist_maker_status` | int (0/1) | 0 or 1 | 1 = disinfectant spray firing this row |
| 9 | `needs_manual_checkup` | int (0/1) | 0 or 1 | 1 = repeated spraying failed to clear air; flagged for staff. Persists until gas ≤ 260 ppm |
| 10 | `water_refill_status` | int (0/1) | 0 or 1 | 1 = water refill pump active (automatic) |
| 11 | `disinfectant_refill_status` | int (0/1) | 0 or 1 | 1 = disinfectant crossed low threshold (manual staff top-up) |
| 12 | `refill_motor_status` | int (0/1) | 0 or 1 | 1 = any refill active (backward compat: water OR disinfectant) |
| 13 | `entry_count` | int | 0 → ~50 | Cumulative visits today. Resets at midnight |
| 14 | `hours_since_seat_spray` | float | 0.0 → ~9 | Hours since last mist spray. Increments 0.083/step (5 min). Long intervals occur when the gas-driven checkup state (extra_spray_streak ≥ 2) suppresses the 4-hour idle backstop |
| 15 | `hours_since_deep_clean` | float | 0.0 → ~24.0 | Hours since 05:00 janitorial reset |
| 16 | `hygiene_score` | float | 0.0 → 100.0 | Composite score: `100 - (gas×0.12) - (hrs_deep×1.8) - (entries×0.4) + N(0,8)` |
| 17 | `needs_cleaning` | int (0/1) | 0 or 1 | **Target label**. Probabilistic: `P(clean) = 1 / (1 + exp((score - 45) / 6))` |
| 18 | `room_available` | int (0/1) | 0 or 1 | Post-spray cooldown flag. 0 while the 10-minute dry period after a spray is running (`dry_steps_remaining > 0`); the room is reported unavailable and no fresh spray is permitted until it elapses |

## Depletion Physics (v2)

The disinfectant tank model is deliberately **irregular** (per-spray draw is a random variable, not a constant):

- **Per-spray draw** `~U(0.10, 0.55)`% per spray (nozzle-dependent)
- **Partial delivery** with 15% probability → draw × `U(0.20, 0.45)`
- **Missed spray** with 5% probability → draw = 0 (blocked/wet nozzle)
- **Nozzle drift**: draw scales by `1 + (0.006 × day_index) + nozzle_offset`, a slow per-cubicle wear that inflates consumption over the 87-day window
- **Refill**: level crosses below 15% → `disinfectant_refill_status = 1`, level reset to 100% (manual staff top-up event — treated as an *observed* event by the ML evaluation, never guessed)

## Cubicle Traffic Profiles

| Cubicle | Peak (12-14h) | Regular (8-20h) | Night | Profile |
|---------|---------------|-----------------|-------|---------|
| Cubicle_A_Office | 45% | 25% | 3% | General office/college — moderate steady traffic |
| Cubicle_B_Station | 70% | 45% | 10% | Busy public (metro/mall) — heavy all-day |
| Cubicle_C_QuietFloor | 20% | 10% | 1% | Low-traffic wing — very sparse |
| Cubicle_D_LunchSpike | 75% | 8% | 1% | Quiet all day, sharp lunch spike |

## Disinfection Control Logic

1. Either sensor "occupied" → **never spray**
2. **Post-spray cooldown**: for 10 minutes (2 rows) after any spray, `room_available = 0` and spray is suppressed (`dry_steps_remaining` set to 2 at the spray row, decremented once per row)
3. Occupied → Vacant → baseline spray (exit-triggered)
4. Gas still poor + vacant → up to 2 extra sprays (capped)
5. Still poor after 2 extra → stop + `needs_manual_checkup = 1`
6. Idle 4+ hours + vacant → one refresh spray

## Notes

- `hygiene_score` and `needs_cleaning` are **synthetic labels** for pipeline development, not real measurements. `needs_cleaning` is used only in the EDA (`data/explore.py`) and the historical-analytics dashboard tab — it is **not** an ML model target in this repo.
- Water refill is automatic; disinfectant refill is manual (staff alert only) — and, in the ML evaluation, an *observed* event that re-seeds the virtual-sensing prediction
- The `room_available` cooldown is enforced in the data generator and surfaced in the dashboard; the ESP32 firmware cooldown lockout timer is not yet in this repo
- The dataset models 4 hypothetical cubicles; physical prototype is a single unit
