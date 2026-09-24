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

# --- DISINFECTANT DEPLETION / DRY-TIME CONFIG ---
# The depletion process is deliberately NOT a clean linear function of the
# spray counter: per-spray draw is noisy, some sprays only partially deliver,
# some are effectively missed, and the nozzle slowly drifts over the 87 days as
# the orifice wears. This is what makes virtual sensing non-trivial.
SPRAY_DRAW_RANGE = (0.10, 0.55)          # base % of tank drawn per successful spray
PARTIAL_SPRAY_PROB = 0.15                # fraction of sprays that only partially deliver
PARTIAL_SPRAY_FACTOR = (0.20, 0.45)      # of a full draw, when partial
MISSED_SPRAY_PROB = 0.05                 # pump fires but ~no product (blocked/wet nozzle)
NOZZLE_DRIFT_PER_DAY = 0.006             # draw multiplier grows ~0.6%/day (orifice wear)

# --- POST-SPRAY DRY / COOLDOWN CONFIG (Item 5) ---
# Minimum interval after a spray completes during which the room is still
# reported UNAVAILABLE to a new entrant, regardless of sensor state. Set from
# the disinfectant's stated contact time (quaternary-ammonium products
# typically cite ~10 min). Must be a multiple of TIME_STEP_MINUTES.
MIN_SPRAY_DRY_MINUTES = 10
DRY_STEPS = MIN_SPRAY_DRY_MINUTES // TIME_STEP_MINUTES


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
    odor_suppression = 0.0   # spraying actually lowers the smell for a while
    prev_occupied = False    # tracks previous row's occupancy to detect an EXIT transition
    extra_spray_streak = 0   # caps back-to-back "air still bad" sprays so it can't spray forever
    checkup_flag = 0         # stays 1 once repeated spraying fails to help, until air clears
    dry_steps_remaining = 0  # post-spray cooldown; >0 -> room reported unavailable
    nozzle_drift_offset = np.random.uniform(-0.3, 0.3)  # slight per-cubicle drift variation

    for ts in timestamps:
        hour, minute = ts.hour, ts.minute

        if hour == 0 and minute == 0:
            entry_count = 0
        if hour == 5 and minute == 0:
            hours_since_deep_clean = 0.0

        # Cooldown ticks down one 5-min step per row. While >0 the room is
        # unavailable regardless of what the sensors say.
        dry_steps_remaining = max(0, dry_steps_remaining - 1)
        room_available = int(dry_steps_remaining == 0)

        # 1. Occupancy simulation (peak vs regular vs night, per-cubicle profile).
        # A would-be entrant is turned away during the post-spray dry period.
        if 8 <= hour <= 20:
            occupancy_prob = peak_prob if (12 <= hour <= 14) else regular_prob
        else:
            occupancy_prob = night_prob

        wants_occupied = bool(np.random.choice([0, 1], p=[1 - occupancy_prob, occupancy_prob]))
        is_occupied = wants_occupied and room_available == 1

        # entry_count uses the exact same 0->1 transition as exit-triggered spray,
        # so one real visit = exactly one entry AND (later) exactly one spray.
        is_new_entry = is_occupied and not prev_occupied
        if is_new_entry:
            entry_count += 1

        if is_occupied:
            occupancy_ld2410 = 1
            motion_ir = np.random.choice([1, 0], p=[0.7, 0.3])  # PIR misses a stationary user ~30% of the time
        else:
            occupancy_ld2410 = 0
            motion_ir = 0

        # suppression decays every step, so the "just cleaned" smell reduction fades over time
        odor_suppression = max(0.0, odor_suppression - 6.0)

        # 2. MQ135 Gas PPM Dynamics
        base_ppm = 110.0 + np.random.normal(0, 5)
        odor_factor = (entry_count * 2.5) + (hours_since_deep_clean * 8.0)
        mq135_gas_ppm = max(
            100.0,
            base_ppm + odor_factor + (50.0 if is_occupied else 0.0) - odor_suppression
        )

        # 3. Actuators — safety rules:
        #   - NEVER spray while occupied (either sensor seeing someone = occupied)
        #   - NEVER spray during the post-spray dry/cooldown period
        #   - baseline spray fires right when the room goes from occupied -> vacant
        #   - if air is STILL bad a bit after that, allow up to 2 extra sprays
        #   - if 2 extra sprays still haven't helped, stop and raise needs_manual_checkup
        #   - idle backstop if nobody has triggered a clean in a long time (room empty only)
        is_occupied_now = (occupancy_ld2410 == 1) or (motion_ir == 1)
        just_exited = prev_occupied and not is_occupied_now
        needs_manual_checkup = 0

        if dry_steps_remaining > 0:
            mist_maker_status = 0                       # room is drying; no fresh spray
        elif is_occupied_now:
            mist_maker_status = 0
        elif just_exited:
            mist_maker_status = 1
            extra_spray_streak = 0
        elif mq135_gas_ppm > 250 and hours_since_seat_spray > 0.25 and extra_spray_streak < 2:
            mist_maker_status = 1
            extra_spray_streak += 1
        elif mq135_gas_ppm > 250 and extra_spray_streak >= 2:
            mist_maker_status = 0
            needs_manual_checkup = 1   # 2 sprays didn't fix it -- flag, don't waste disinfectant
        elif hours_since_seat_spray > 4.0:
            mist_maker_status = 1
            extra_spray_streak = 0
        else:
            mist_maker_status = 0

        if mist_maker_status == 1:
            hours_since_seat_spray = 0.0
            water_level_cm -= np.random.uniform(0.02, 0.05)
            odor_suppression = np.random.uniform(80, 100)

            # Slow nozzle drift: the same "spray" draws more product as the
            # orifice wears over the 87-day window (per-cubicle variation).
            day_index = (ts - start_time).days
            drift = 1.0 + (NOZZLE_DRIFT_PER_DAY * day_index + max(0.0, nozzle_drift_offset))
            draw = np.random.uniform(*SPRAY_DRAW_RANGE)
            if np.random.rand() < MISSED_SPRAY_PROB:
                draw *= 0.0                              # blocked/wet nozzle delivered nothing
            elif np.random.rand() < PARTIAL_SPRAY_PROB:
                draw *= np.random.uniform(*PARTIAL_SPRAY_FACTOR)   # partial delivery
            disinfectant_pct -= draw * drift
            dry_steps_remaining = DRY_STEPS              # start the post-spray cooldown
        else:
            hours_since_seat_spray += TIME_STEP_MINUTES / 60.0

        prev_occupied = is_occupied_now

        # manual checkup / fault flag persists until gas drops back down
        if extra_spray_streak >= 2 and mq135_gas_ppm > 300:
            checkup_flag = 1
        elif checkup_flag == 1 and mq135_gas_ppm <= 260:
            checkup_flag = 0

        hours_since_deep_clean += TIME_STEP_MINUTES / 60.0

        # water and disinfectant are separate physical tanks; refill independently
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
        sensor_noise = np.random.normal(0, 8.0)
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
            'room_available': room_available,
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

csv_filename = 'data/washroom_dataset_multi_cubicle.csv'
df_all.to_csv(csv_filename, index=False)
print(f"Success! Dataset created with {len(df_all)} rows, {df_all.cubicle_id.nunique()} cubicles, {len(df_all.columns)} columns.")

# Auto-download only if running inside Google Colab; safe to run anywhere else too
try:
    from google.colab import files
    files.download(csv_filename)
except ImportError:
    print("(Not running in Colab -- file saved locally, skip the auto-download.)")
