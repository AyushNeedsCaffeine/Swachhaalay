import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set seed for reproducible synthetic generation
np.random.seed(42)

# --- CONFIGURATION ---
NUM_DAYS = 87                          # 87 days * 288 rows/day = 25,056 rows PER CUBICLE
TIME_STEP_MINUTES = 5
ROWS_PER_DAY = (24 * 60) // TIME_STEP_MINUTES
start_time = datetime(2026, 8, 1, 0, 0, 0)


def generate_cubicle_data(cubicle_id, peak_prob, regular_prob, night_prob, num_days=NUM_DAYS):
    """
    Simulates one washroom cubicle for `num_days` days at 5-minute resolution.
    peak_prob / regular_prob / night_prob control how busy THIS cubicle is during
    lunch hours / regular daytime hours / night hours, so different cubicles get
    genuinely different traffic SHAPES, not just a scaled copy of the same curve.
    """
    total_rows = num_days * ROWS_PER_DAY
    timestamps = [start_time + timedelta(minutes=i * TIME_STEP_MINUTES) for i in range(total_rows)]

    rows = []

    # --- per-cubicle state (each cubicle has its own physical tanks/timers) ---
    water_level_cm = 20.0
    disinfectant_pct = 100.0
    hours_since_seat_spray = 0.0
    hours_since_deep_clean = 0.0
    entry_count = 0
    odor_suppression = 0.0   # FIX: new state var so spraying actually lowers the smell for a while
    prev_occupied = False    # FIX v2: tracks previous row's occupancy to detect an EXIT transition
    extra_spray_streak = 0   # FIX v2: caps back-to-back "air still bad" sprays so it can't spray forever
    checkup_flag = 0         # NEW: stays 1 once repeated spraying fails to help, until air clears

    for ts in timestamps:
        hour, minute = ts.hour, ts.minute

        if hour == 0 and minute == 0:
            entry_count = 0
        if hour == 5 and minute == 0:
            hours_since_deep_clean = 0.0

        # 1. Occupancy simulation (peak vs regular vs night, per-cubicle profile)
        if 8 <= hour <= 20:
            occupancy_prob = peak_prob if (12 <= hour <= 14) else regular_prob
        else:
            occupancy_prob = night_prob

        is_occupied = np.random.choice([0, 1], p=[1 - occupancy_prob, occupancy_prob])

        # FIX v2: entry_count used to increment on a SEPARATE random filter (0.7/0.3)
        # from what counted as an "exit" for the spray trigger below — so a lot of
        # single-row occupancy blips triggered an exit-spray without ever being
        # counted as an entry, and the spray count came out far higher than the
        # entry count. Now both use the exact same 0->1 transition, so one real
        # visit = exactly one entry AND (later) exactly one exit-triggered spray.
        is_new_entry = is_occupied and not prev_occupied
        if is_new_entry:
            entry_count += 1

        if is_occupied:
            occupancy_ld2410 = 1
            motion_ir = np.random.choice([1, 0], p=[0.7, 0.3])  # PIR misses a stationary user ~30% of the time
        else:
            occupancy_ld2410 = 0
            motion_ir = 0

        # FIX: suppression decays every step, so the "just cleaned" smell reduction fades over time
        odor_suppression = max(0.0, odor_suppression - 6.0)

        # 2. MQ135 Gas PPM Dynamics
        base_ppm = 110.0 + np.random.normal(0, 5)
        odor_factor = (entry_count * 2.5) + (hours_since_deep_clean * 8.0)
        mq135_gas_ppm = max(
            100.0,
            base_ppm + odor_factor + (50.0 if is_occupied else 0.0) - odor_suppression
        )

        # 3. Actuators
        # FIX v2 (safety): the old trigger (gas>250 OR timeout) fired regardless of
        # occupancy, so it could spray while someone was still inside. New rule:
        #   - NEVER spray while occupied (either sensor seeing someone = treat as occupied)
        #   - baseline spray fires right when the room goes from occupied -> vacant
        #   - if air is STILL bad a bit after that, allow up to 2 extra sprays
        #   - CONFIRMED (team decision): if 2 extra sprays still haven't helped, stop
        #     spraying and raise needs_manual_checkup=1 instead of continuing to spray
        #     blind -- repeated ineffective spraying is treated as a probable sensor
        #     or ventilation fault, not a dirty seat, and needs a person to look at it
        #   - idle backstop if nobody has triggered a clean in a long time, still
        #     only when the room is currently empty
        is_occupied_now = (occupancy_ld2410 == 1) or (motion_ir == 1)
        just_exited = prev_occupied and not is_occupied_now
        needs_manual_checkup = 0

        if is_occupied_now:
            mist_maker_status = 0
        elif just_exited:
            mist_maker_status = 1
            extra_spray_streak = 0
        elif mq135_gas_ppm > 250 and hours_since_seat_spray > 0.25 and extra_spray_streak < 2:
            mist_maker_status = 1
            extra_spray_streak += 1
        elif mq135_gas_ppm > 250 and extra_spray_streak >= 2:
            mist_maker_status = 0
            needs_manual_checkup = 1   # 2 sprays didn't fix it -- flag for a person, don't waste more disinfectant
        elif hours_since_seat_spray > 4.0:
            mist_maker_status = 1
            extra_spray_streak = 0
        else:
            mist_maker_status = 0

        if mist_maker_status == 1:
            hours_since_seat_spray = 0.0
            disinfectant_pct -= np.random.uniform(0.15, 0.35)
            water_level_cm -= np.random.uniform(0.02, 0.05)
            odor_suppression = np.random.uniform(80, 100)
        else:
            hours_since_seat_spray += TIME_STEP_MINUTES / 60.0

        prev_occupied = is_occupied_now

        # NEW: manual checkup / fault flag. Once the extra-spray cap is used up and the
        # air is STILL bad, stop spraying (handled above) and raise this flag instead --
        # it covers both meanings AJ asked for: "send someone to manually check/clean it"
        # AND "flag this as a possible fault" for the anomaly-detection model to learn from.
        # It stays on until gas_ppm actually drops back down (someone dealt with it),
        # not just for one row.
        if extra_spray_streak >= 2 and mq135_gas_ppm > 300:
            checkup_flag = 1
        elif checkup_flag == 1 and mq135_gas_ppm <= 260:
            checkup_flag = 0

        hours_since_deep_clean += TIME_STEP_MINUTES / 60.0

        # FIX: water and disinfectant are separate physical tanks (per the hardware doc) —
        # they should refill independently, not both jump to full whenever either is low.
        water_refill_status = 0
        if water_level_cm < 5.0:
            water_refill_status = 1
            water_level_cm = 20.0

        disinfectant_refill_status = 0
        if disinfectant_pct < 15.0:
            disinfectant_refill_status = 1
            disinfectant_pct = 100.0

        # kept for backward compatibility with any dashboard code already reading this column
        refill_motor_status = 1 if (water_refill_status or disinfectant_refill_status) else 0

        water_level_cm = round(max(0.0, min(20.0, water_level_cm)), 2)
        disinfectant_pct = round(max(0.0, min(100.0, disinfectant_pct)), 2)

        # 4. Hygiene score with noise
        raw_score = 100.0 - (mq135_gas_ppm * 0.12) - (hours_since_deep_clean * 1.8) - (entry_count * 0.4)
        sensor_noise = np.random.normal(0, 8.0)  # FIX: nudged up from 4.5 so score isn't ~100% back-solvable
        hygiene_score = round(max(0.0, min(100.0, raw_score + sensor_noise)), 1)

        # Probabilistic (soft) label instead of a hard cutoff
        clean_prob = 1 / (1 + np.exp((hygiene_score - 45) / 6.0))
        needs_cleaning = int(np.random.rand() < clean_prob)

        rows.append({
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'cubicle_id': cubicle_id,
            'mq135_gas_ppm': round(mq135_gas_ppm, 1),
            'occupancy_ld2410': occupancy_ld2410,
            'motion_ir': motion_ir,
            'water_level_cm': water_level_cm,
            'disinfectant_level_virtual_pct': disinfectant_pct,
            'mist_maker_status': mist_maker_status,
            'needs_manual_checkup': int(needs_manual_checkup or checkup_flag),
            'water_refill_status': water_refill_status,
            'disinfectant_refill_status': disinfectant_refill_status,
            'refill_motor_status': refill_motor_status,
            'entry_count': entry_count,
            'hours_since_seat_spray': round(hours_since_seat_spray, 2),
            'hours_since_deep_clean': round(hours_since_deep_clean, 2),
            'hygiene_score': hygiene_score,
            'needs_cleaning': needs_cleaning,
        })

    return rows


# --- Cubicle profiles: different traffic SHAPES, not just different volumes ---
# (this is what makes "usage clustering across units" mean something later)
CUBICLES = [
    ('Cubicle_A_Office',     0.45, 0.25, 0.03),   # original profile — general office/college washroom
    ('Cubicle_B_Station',    0.70, 0.45, 0.10),   # busy public spot — railway/metro/mall style
    ('Cubicle_C_QuietFloor', 0.20, 0.10, 0.01),   # low-traffic floor/wing
    ('Cubicle_D_LunchSpike', 0.75, 0.08, 0.01),   # quiet all day except a sharp lunchtime spike
]

all_rows = []
for cubicle_id, peak_p, regular_p, night_p in CUBICLES:
    all_rows.extend(generate_cubicle_data(cubicle_id, peak_p, regular_p, night_p))

df_all = pd.DataFrame(all_rows)

csv_filename = 'washroom_dataset_multi_cubicle.csv'
df_all.to_csv(csv_filename, index=False)
print(f"Success! Dataset created with {len(df_all)} rows, {df_all.cubicle_id.nunique()} cubicles, {len(df_all.columns)} columns.")

# Auto-download only if running inside Google Colab; safe to run anywhere else too
try:
    from google.colab import files
    files.download(csv_filename)
except ImportError:
    print("(Not running in Colab -- file saved locally, skip the auto-download.)")
