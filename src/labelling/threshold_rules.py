"""
threshold_rules.py — Implements the threshold rules for each of the seven
detectable telemetry behaviours defined in proposal §1.6 / §3.5.1.

Each function takes the canonical telemetry DataFrame (output of the
LFS/AC adapter, see src.data.lfs_adapter) and returns a 0/1 Series flagging
the timesteps where that behaviour occurs. All thresholds are imported from
src.data.constants — the single source of truth.

Two spatial classes from the original eleven-behaviour proposal design
(apex_miss, track_limit_violation) ended up being learned visually by the
CNN module instead of threshold-detected here, since they depend on frame
content (track boundary, racing line) rather than telemetry alone.
mechanical_abuse and vehicle_instability were folded into
label_aggressive_downshift / label_rough_steering respectively during
implementation, since they shared the same underlying signals.

Ported from `final-lstm-module-codes.ipynb` ("LSTM Data - Labelling").
"""

import numpy as np
import pandas as pd

from src.data.constants import (
    SAMPLE_RATE_HZ, LOCKUP_MIN_SAMPLES, WHEEL_SPIN_SLIP_THRESH,
    WHEEL_SPIN_THROTTLE_MIN, WHEEL_SPIN_MIN_SAMPLES, TRAIL_BRAKE_THRESH,
    DOWNSHIFT_RPM_PCT, DOWNSHIFT_POST_SHIFT_WINDOW, LATE_UPSHIFT_MIN_SAMPLES,
    ROUGH_STEER_RATE_THRESH, ROUGH_STEER_HIGH_SPEED_THRESH,
    GENTLE_ACCEL_RATE_THRESH, GENTLE_ACCEL_MIN_SAMPLES,
    brake_activation_thresh, speed_gate_kmh, slip_lock_thresh,
    wheel_locked_speed_ratio,
)


# ── Core utility: vectorized consecutive-run check ────────────────────────────

def has_consecutive_true(series: pd.Series, min_count: int) -> pd.Series:
    """
    Returns a boolean Series where True means the input Series has been
    continuously True for AT LEAST min_count consecutive rows at that point.

    Uses the cumsum-groupby trick — O(n), no Python-level loops.

    Example: min_count=3, series=[F,F,T,T,T,F,T,T,F]
             output          =[F,F,F,F,T,F,F,F,F]
    """
    # Create run-group IDs: increments each time the value changes
    groups = (series != series.shift()).cumsum()

    # Within each run-group, count how many rows belong to it
    run_lengths = series.groupby(groups).transform("count")

    # Valid if: currently True AND run is long enough
    return series & (run_lengths >= min_count)


def _get_max_slip(df: pd.DataFrame) -> pd.Series:
    """Maximum slip ratio across all four wheels at each timestep."""
    slip_cols = ["slip_ratio_lf", "slip_ratio_rf", "slip_ratio_lr", "slip_ratio_rr"]
    available = [c for c in slip_cols if c in df.columns]
    if not available:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return df[available].max(axis=1)


def _get_max_wheel_spin(df: pd.DataFrame) -> pd.Series:
    """Maximum wheel spin across all four driven wheels at each timestep."""
    spin_cols = ["wheel_spin_lf", "wheel_spin_rf", "wheel_spin_lr", "wheel_spin_rr"]
    available = [c for c in spin_cols if c in df.columns]
    if not available:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return df[available].max(axis=1)


# FIX (see distribution check on raw CSV): wheel_spin_lf/rf/lr/rr are NOT
# angular velocity in rad/s and NOT a 0-1 ratio. describe() confirms they
# share the same mean/std/range as speed_kmh (mean≈102 vs speed mean≈104,
# same -40..180 span) — this is wheel SURFACE speed in km/h. Any threshold
# written as if it were a small rad/s or 0-1 value (wheel_spin_zero_thresh
# = 0.5, WHEEL_SPIN_SLIP_THRESH compared directly to the raw column) is a
# silent no-op or always-true condition against this column. Compare it to
# speed_kmh instead, the same way slip_ratio_* already does.

def _get_max_wheel_slip_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Per-wheel slip expressed as (wheel_surface_speed - car_speed) / car_speed,
    i.e. how much faster the wheel's contact patch is moving than the car
    itself — the same physical quantity slip_ratio_* already encodes, but
    derived independently from wheel_spin_* as a cross-check / fallback.
    Returns the max across all four wheels. Speed is floored at 1 km/h to
    avoid a divide-by-near-zero blowup at a dead stop.
    """
    spin_cols = ["wheel_spin_lf", "wheel_spin_rf", "wheel_spin_lr", "wheel_spin_rr"]
    available = [c for c in spin_cols if c in df.columns]
    if not available or "speed_kmh" not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index)
    speed_floor = df["speed_kmh"].clip(lower=1.0)
    ratios = df[available].sub(df["speed_kmh"], axis=0).div(speed_floor, axis=0)
    return ratios.max(axis=1)


def _get_min_wheel_speed_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Per-wheel (wheel_surface_speed / car_speed) ratio, minimum across all
    four wheels. A locked wheel has near-zero surface speed while the car
    is still moving, so this ratio collapses toward 0 regardless of what
    slip_ratio_* reports — used as the independent lockup fallback signal.
    """
    spin_cols = ["wheel_spin_lf", "wheel_spin_rf", "wheel_spin_lr", "wheel_spin_rr"]
    available = [c for c in spin_cols if c in df.columns]
    if not available or "speed_kmh" not in df.columns:
        return pd.Series(np.ones(len(df)), index=df.index)  # neutral -> never "locked"
    speed_floor = df["speed_kmh"].clip(lower=1.0)
    ratios = df[available].div(speed_floor, axis=0)
    return ratios.min(axis=1)


# ── Individual labelling functions ────────────────────────────────────────────

def label_brake_locked(df: pd.DataFrame) -> pd.Series:
    """
    The previous method of computing brake lock up was unreliable, dangours and unsafe
    thus this new throttle independent method.
    
    Race-engineer level brake lockup detection.
    Tracks individual wheel slip magnitudes and absolute wheel speed collapse
    to catch front and rear axle lockups under any braking pressure.
    """

    # <---old way
    #brake_high  = df["brake"] > BRAKE_PRESSURE_THRESH
    #slip_high   = _get_max_slip(df) >= SLIP_RATIO_THRESH
    #candidate   = brake_high & slip_high

    # <---new way
    # Ensure the driver is actually braking and the car is moving, prevents false positives while stationary in the pits
    braking_active = (df["brake"] > brake_activation_thresh) & (df["speed_kmh"] > speed_gate_kmh)

    # Extract absolute magnitudes of slip ratios, fixes the negative sign issue during deceleration
    lf_slip_lock = df["slip_ratio_lf"].abs() >= slip_lock_thresh
    rf_slip_lock = df["slip_ratio_rf"].abs() >= slip_lock_thresh
    lr_slip_lock = df["slip_ratio_lr"].abs() >= slip_lock_thresh
    rr_slip_lock = df["slip_ratio_rr"].abs() >= slip_lock_thresh

    # Wheel Spin Fallback
    # FIX: wheel_spin_xx is wheel SURFACE SPEED in km/h (confirmed against
    # speed_kmh's matching mean/std/range in describe()), not rad/s and not
    # a 0-1 value — comparing it to wheel_spin_zero_thresh=0.5 was a no-op
    # that never fired. A locked wheel's surface speed collapses toward 0
    # *relative to the car's speed*, so compare per-wheel speed against
    # car speed instead of against a fixed small constant.
    wheel_speed_ratio_lf = df["wheel_spin_lf"] / df["speed_kmh"].clip(lower=1.0)
    wheel_speed_ratio_rf = df["wheel_spin_rf"] / df["speed_kmh"].clip(lower=1.0)
    wheel_speed_ratio_lr = df["wheel_spin_lr"] / df["speed_kmh"].clip(lower=1.0)
    wheel_speed_ratio_rr = df["wheel_spin_rr"] / df["speed_kmh"].clip(lower=1.0)

    lf_spin_lock = wheel_speed_ratio_lf < wheel_locked_speed_ratio
    rf_spin_lock = wheel_speed_ratio_rf < wheel_locked_speed_ratio
    lr_spin_lock = wheel_speed_ratio_lr < wheel_locked_speed_ratio
    rr_spin_lock = wheel_speed_ratio_rr < wheel_locked_speed_ratio

    # Consolidate axle lockups
    # Front lockup matches the symptom of not able to steer at all
    front_locked = (lf_slip_lock | lf_spin_lock) | (rf_slip_lock | rf_spin_lock)
    rear_locked  = (lr_slip_lock | lr_spin_lock) | (rr_slip_lock | rr_spin_lock)
    
    any_wheel_locked = front_locked | rear_locked

    # Gate by active braking context
    candidate = braking_active & any_wheel_locked

    # Pass back into your consecutive frame window constraint (e.g., LOCKUP_MIN_SAMPLES)

    return has_consecutive_true(candidate, LOCKUP_MIN_SAMPLES).astype(int)


def label_wheel_spin(df: pd.DataFrame) -> pd.Series:
    """
    §1.6 definition: max driven wheel spin > 0.3 while throttle > 0.1
    for at least 2 consecutive samples.

    FIX: wheel_spin_xx is wheel surface speed in km/h, not a 0-1 slip
    fraction (see describe() — same scale as speed_kmh). The original
    code compared the raw ~100-scale column directly to
    WHEEL_SPIN_SLIP_THRESH=0.30, which is true almost any time the car
    is moving at all — that's why this label fired on 85.4% of rows.
    Use the wheel-speed-vs-car-speed ratio instead; slip_ratio_* (already
    a correctly-scaled dimensionless ratio) is checked too, and either
    signal firing counts as wheelspin — this makes the label robust even
    if one sensor channel is noisy.
    """
    wheel_slip_ratio = _get_max_wheel_slip_ratio(df)     # (wheel_speed - car_speed) / car_speed
    slip_ratio        = _get_max_slip(df)                # existing slip_ratio_* columns

    spin_high    = (wheel_slip_ratio > WHEEL_SPIN_SLIP_THRESH) | (slip_ratio > WHEEL_SPIN_SLIP_THRESH)
    accelerating = df["throttle"] > WHEEL_SPIN_THROTTLE_MIN
    candidate    = spin_high & accelerating
    return has_consecutive_true(candidate, WHEEL_SPIN_MIN_SAMPLES).astype(int)


def label_trail_brake(df: pd.DataFrame,
                       apex_reference: dict = None) -> pd.Series:
    """
    §1.6 definition: brake pressure > 20% after the geometric apex point.

    If apex_reference is provided (dict mapping track name to list of
    apex_dist_m values per corner), uses distance-based detection.
    Otherwise falls back to: brake > threshold while speed is rising
    (car is already past slowest point of corner).

    apex_reference format:
        {"blackwood_gp": [245.0, 612.0, 890.0, ...]}
    """
    brake_active = df["brake"] > TRAIL_BRAKE_THRESH

    if apex_reference is not None and "lap_dist_m" in df.columns:
        # Build a per-row boolean: True if we're in the post-apex zone of any corner
        post_apex_zone = pd.Series(False, index=df.index)
        for apex_dist in apex_reference:
            window_start = apex_dist
            window_end   = apex_dist + 150  # 150m past apex counts as "post apex"
            in_zone = (
                (df["lap_dist_m"] >= window_start) &
                (df["lap_dist_m"] <= window_end)
            )
            post_apex_zone = post_apex_zone | in_zone
        return (brake_active & post_apex_zone).astype(int)
    else:
        # Fallback: braking while speed_kmh is rising (past the slowest point)
        speed_rising = df["speed_kmh"].diff() > 0 if "speed_kmh" in df.columns \
                       else pd.Series(False, index=df.index)
        return (brake_active & speed_rising).astype(int)


def label_aggressive_downshift(df: pd.DataFrame,
                                vehicle_redline:        float = 8500.0,
                                rpm_danger_pct:         float = DOWNSHIFT_RPM_PCT, #<--- wired to constants cell (0.95, matches docstring)
                                slip_angle_threshold:   float = 4.0,
                                long_g_rate_threshold:  float = 0.40,
                                post_shift_window:      int   = DOWNSHIFT_POST_SHIFT_WINDOW) -> pd.Series: #<--- wired to its own constant (12 rows/200ms), NOT LATE_UPSHIFT_MIN_SAMPLES
    """
    Race-engineer definition of an aggressive downshift.
 
    An aggressive downshift is one whose CONSEQUENCES indicate mechanical
    stress or vehicle instability. It is NOT about what the RPM was
    BEFORE the shift — it is about what happens AFTER the gear engages.
 
    A race engineer watching data would flag a downshift as aggressive when
    any of the following appear in the data within ~200ms after the shift:
 
        a) Post-shift RPM overrev: the engine is pushed above 95% of
           redline by the downshift. This means the driver rev-matched
           too aggressively or downshifted too early in the braking zone,
           forcing the engine to spin faster than intended.
 
        b) Driveline shock: the rate of change of longitudinal deceleration
           spikes sharply immediately after the shift completes. This
           indicates the gearbox engaged abruptly rather than smoothly,
           sending a shock through the driveline (heard as a "clunk" and
           felt as a lurch). Measured as the absolute rate of change of
           longitudinal G exceeding a threshold.
 
        c) Rear instability: body slip angle spikes beyond a threshold
           immediately post-shift. In a rear-wheel-drive car, an aggressive
           downshift under braking can unsettle the rear axle via sudden
           engine braking torque, causing oversteer. Body slip angle
           increasing abruptly after a downshift event is the signature.
 
    ── WHY THE ORIGINAL WAS WRONG (51.7% label rate) ─────────────────────
    The original checked: RPM > 90% of redline AND a gear decrease happened
    in the last 5 rows. The problem: being at 90%+ of redline is completely
    NORMAL at the top of any gear's range. That is literally where you drive
    on a race track. With a 5-row (83ms) look-back window, almost any
    downshift near the top of a gear range was being flagged, regardless of
    whether anything problematic actually happened. The fix is to look at
    what happens AFTER the shift, not what the RPM was before it.
 
    Args:
        vehicle_redline:        Redline RPM of the current car.
        rpm_danger_pct:         Fraction of redline that post-shift RPM must
                                exceed to count as an overrev. Default 0.95.
        slip_angle_threshold:   Body slip angle (degrees) threshold for
                                instability detection post-shift. Default 4.0°.
        long_g_rate_threshold:  Rate of change of longitudinal G (G/sample)
                                threshold for driveline shock detection.
                                Default 0.40 G/sample.
        post_shift_window:      Number of samples (rows) after the gear-decrease
                                event to look for consequences. Default 12 rows
                                = 200ms at 60Hz.
    """
 
    # Step 1: Identify the downshift event row (the exact moment gear decreases)
    if "gear" not in df.columns:
        return pd.Series(0, index=df.index)
 
    gear_diff       = df["gear"].diff()
    downshift_event = gear_diff < 0   # True only at the exact row of the shift
 
    # Step 2: Create a "post-shift window" — the next N rows after the shift event.
    # This is where consequences manifest. Rolling max forward-propagates the flag.
    in_post_window = (
        downshift_event
        .rolling(post_shift_window, min_periods=1)
        .max()
        .astype(bool)
    )
 
    # Step 3a: Post-shift RPM overrev
    # The downshift drove the engine above 95% of redline.
    if "engine_rpm_norm" in df.columns:
        rpm_overrev = df["engine_rpm_norm"] > rpm_danger_pct
    elif "engine_rpm" in df.columns:
        rpm_overrev = df["engine_rpm"] > (vehicle_redline * rpm_danger_pct)
    else:
        rpm_overrev = pd.Series(False, index=df.index)
 
    # Step 3b: Driveline shock via longitudinal G rate of change
    # A smooth shift produces gradual G changes. A shock produces a spike.
    if "longitudinal_g" in df.columns:
        long_g_rate    = df["longitudinal_g"].diff().abs()
        driveline_shock = long_g_rate > long_g_rate_threshold
    else:
        driveline_shock = pd.Series(False, index=df.index)
 
    # Step 3c: Rear instability via body slip angle spike
    # If the car was stable going INTO the braking zone and the slip angle
    # jumps sharply within the post-shift window, the downshift unsettled the rear.
    if "body_slip_angle" in df.columns:
        slip_abs     = df["body_slip_angle"].abs()
        # Rate of slip angle change — a sudden jump signals instability
        slip_rate    = slip_abs.diff().abs()
        rear_unstable = (slip_abs > slip_angle_threshold) | (slip_rate > 1.5)
    else:
        rear_unstable = pd.Series(False, index=df.index)
 
    # Step 4: Combine — a downshift is aggressive if it happened recently
    # AND at least one danger sign is present in the post-shift window.
    aggressive = in_post_window & (rpm_overrev | driveline_shock | rear_unstable)
 
    # Require at least 2 consecutive rows to filter single-sample noise spikes
    return has_consecutive_true(aggressive, 2).astype(int)


def label_late_upshift(df: pd.DataFrame,
                       vehicle_power_peak_norm: float = 0.847,
                       throttle_threshold:      float = 0.70,
                       min_sustained_samples:   int   = LATE_UPSHIFT_MIN_SAMPLES) -> pd.Series: #<--- wired to constants cell (20 samples/0.33s, matches docstring)
    """
    Race-engineer definition of a late upshift.
 
    A late upshift is NOT simply "RPM above power peak." Every driver
    spends time above power peak on every lap — that is where peak power
    is delivered. What a race engineer flags is:
 
        The driver is on full (or near-full) throttle AND has allowed RPM
        to climb into the overrev zone past the power peak AND has not
        upshifted despite being there for long enough that power is being
        lost by staying in the current gear.
 
    ── WHY THE ORIGINAL WAS WRONG (100% label rate) ─────────────────────
    The original condition was: engine_rpm_norm > power_peak_norm sustained
    for 90 consecutive samples. There was NO requirement for the driver to
    be accelerating. During engine braking, trailing throttle, corner entry,
    and any coasting phase, RPM sits comfortably above power peak. Since the
    driver is always transitioning through the power-peak zone between shifts,
    almost every row qualified, giving 100%.
 
    ── WHAT A RACE ENGINEER ACTUALLY LOOKS FOR ──────────────────────────
    1. Driver is meaningfully ON the throttle (throttle > 0.70).
       Engine braking / coasting above power peak is NORMAL technique,
       not a mistake. Filtering on throttle eliminates those cases.
 
    2. RPM is above the power-peak norm.
       The driver is in the zone where staying longer costs them lap time.
 
    3. Gear is not neutral / reverse and a gear increase did not just happen
       (we don't want to double-flag the exact moment of a correct upshift).
 
    4. The condition is sustained for min_sustained_samples rows (default 20
       rows = 0.33 seconds at 60Hz). A skilled driver shifts within ~0.1–0.25
       seconds of passing the power peak. Staying there for 0.33 seconds on
       full throttle is a measurable and consistent late upshift.
 
    Args:
        vehicle_power_peak_norm: power_peak_rpm / redline_rpm.
                                 Default 0.847 ≈ 7200/8500 for a typical LFS GT car.
        throttle_threshold:      Minimum throttle input to count as accelerating.
                                 0.70 = 70% pedal travel — filters coasting / braking.
        min_sustained_samples:   How many consecutive qualifying rows before firing.
                                 20 samples = 0.33 s at 60Hz.
    """
 
    # Condition 1: RPM above power peak
    if "engine_rpm_norm" in df.columns:
        above_peak = df["engine_rpm_norm"] > vehicle_power_peak_norm
    elif "engine_rpm" in df.columns:
        above_peak = df["engine_rpm"] > (vehicle_power_peak_norm * 8500.0)
    else:
        return pd.Series(0, index=df.index)
 
    # Condition 2: Driver is meaningfully accelerating — the critical missing gate
    # that caused the 100% label rate in the original implementation.
    # Without this, engine braking at high RPM fires the label constantly.
    if "throttle" in df.columns:
        accelerating = df["throttle"] > throttle_threshold
    else:
        accelerating = pd.Series(True, index=df.index)
 
    # Condition 3: Not currently in the process of upshifting.
    # A gear increase in the last 3 rows means the driver IS shifting — do not
    # penalise them for the brief window while the shift is executing.
    if "gear" in df.columns:
        gear_increasing     = df["gear"].diff() > 0
        shift_in_progress   = gear_increasing.rolling(3, min_periods=1).max().astype(bool)
        not_shifting        = ~shift_in_progress
        # Also exclude neutral and reverse (gear <= 1) where "upshift" is meaningless
        in_valid_gear       = df["gear"] > 1
    else:
        not_shifting  = pd.Series(True, index=df.index)
        in_valid_gear = pd.Series(True, index=df.index)
 
    # Combine all gates, then apply the run-length check
    candidate = above_peak & accelerating & not_shifting & in_valid_gear
    return has_consecutive_true(candidate, min_sustained_samples).astype(int)

def label_rough_steering(df: pd.DataFrame,
                        lock_to_lock_deg: float = 540.0,
                        base_rough_thresh_deg_s: float = ROUGH_STEER_RATE_THRESH, #<--- wired to constants cell (180.0, low-speed endpoint)
                        high_speed_rough_thresh_deg_s: float = ROUGH_STEER_HIGH_SPEED_THRESH, #<--- wired to constants cell (90.0, high-speed endpoint)
                        speed_low_gate_kmh: float = 40.0,
                        speed_high_gate_kmh: float = 140.0,
                        min_duration_samples: int = 3) -> pd.Series:
    """
    Race-engineer grade rough steering and chassis instability detection.
    Converts normalized steering data to physical degrees/sec, applies a 
    dynamic, speed-sensitive threshold, and filters out high-frequency noise.
    """
    #<---old method
    #if "steer_rate" not in df.columns:
        # Derive from steering angle differences if rate not available
    #    steer_rate = df["steer"].diff().abs() * SAMPLE_RATE_HZ if "steer" in df.columns \
    #                 else pd.Series(np.zeros(len(df)), index=df.index)
    #else:
    #    steer_rate = df["steer_rate"].abs()

    #<---new method

    # Compute or extract steering rate, already in physical degrees/second
    # FIX: steer_rate (precomputed by the LFS telemetry script) is confirmed
    # via describe() to already be in degrees/second — mean≈0, std≈3.9,
    # range -185..328. That range only makes sense as deg/s (a fast wheel
    # flick can hit a few hundred deg/s); read as "normalized units/sec" it
    # would imply traversing the full steering range ~160 times a second,
    # which is physically absurd. The old code re-multiplied this already-
    # physical value by deg_per_norm_unit (270), inflating any nonzero
    # reading by 270x and pushing it past the 90-180 deg/s threshold almost
    # every time. Only the "steer" position-diff fallback (genuinely on the
    # -1..1 normalized scale) needs that conversion.
    deg_per_norm_unit = lock_to_lock_deg / 2.0   # used only by the fallback branch below
    if "steer_rate" in df.columns:
        steer_rate_deg_s = df["steer_rate"].abs()
    elif "steer" in df.columns:
        norm_steer_rate = df["steer"].diff().abs() * SAMPLE_RATE_HZ
        norm_steer_rate = norm_steer_rate.fillna(0.0)
        steer_rate_deg_s = norm_steer_rate * deg_per_norm_unit
    else:
        return pd.Series(np.zeros(len(df)), index=df.index).astype(int)

    # Dynamic Thresholding based on Vehicle Speed
    # Extract raw speed prioritizing non-normalized speed from the adapter stage
    speed = df["speed_kmh"] if "speed_kmh" in df.columns else (df["speed_kmh_norm"] * 250.0)
    
    # Linearly interpolate the threshold between low-speed handling and high-speed stability
    # At 40 km/h: threshold is 180 deg/s. At 140 km/h: threshold tightens to 90 deg/s.
    speed_fraction = ((speed - speed_low_gate_kmh) / (speed_high_gate_kmh - speed_low_gate_kmh)).clip(0.0, 1.0)
    dynamic_threshold = base_rough_thresh_deg_s - (speed_fraction * (base_rough_thresh_deg_s - high_speed_rough_thresh_deg_s))

    # Check if steering velocity violates dynamic safety envelope
    # Also ignore parking/pit maneuvers below 20 km/h entirely
    steering_violating = (steer_rate_deg_s > dynamic_threshold) & (speed > 20.0)

    # Cross-Reference with Chassis Instability
    # High steering inputs are especially "rough" if they cause sudden spikes in lateral loading
    if "lateral_g" in df.columns:
        lateral_jerk = df["lateral_g"].diff().abs() * SAMPLE_RATE_HZ
        # A jerk over 4.0 G/sec indicates violent platform snapping
        heavy_chassis_transient = (lateral_jerk > 4.0) & (speed > 20.0)
        # FIX: original was `steering_violating | (steering_violating & heavy_chassis_transient)`,
        # which is a tautology (A | (A & B) == A) — heavy_chassis_transient never
        # actually affected the label. Genuine OR: fire on either a pure steering-rate
        # overload, or a chassis-snap event, matching the docstring's stated intent.
        candidate = steering_violating | heavy_chassis_transient
    else:
        candidate = steering_violating

    # Apply persistence filter (e.g., 3 consecutive samples / 50ms)
    # Prevents sharp single-frame force feedback spikes from polluting labels
    return has_consecutive_true(candidate, min_duration_samples).astype(int)
#return (steer_rate > ROUGH_STEER_RATE_THRESH).astype(int)


def label_gentle_accel(df: pd.DataFrame,
                        apex_reference: dict = None) -> pd.Series:
    """
    §1.6 definition: throttle application RATE below 15%/sample from apex exit,
    indicating smooth and controlled power delivery. Positive label.

    Uses the same apex reference structure as trail_brake.
    Falls back to any sustained-throttle region if no apex reference provided.
    """
    # NOTE: throttle_rate (from the LFS telemetry script) is on a
    # percentage-points-per-sample scale (describe(): min=-117.7, max=100.2),
    # NOT a 0-1 fraction. GENTLE_ACCEL_RATE_THRESH is fixed at the constant
    # level (config cell) to match this scale — if throttle_rate is ever
    # re-derived from a differently-scaled column, re-check this threshold.
    if "throttle_rate" not in df.columns:
        throttle_rate = df["throttle"].diff().clip(lower=0) * 100.0 if "throttle" in df.columns \
                        else pd.Series(np.zeros(len(df)), index=df.index)
    else:
        throttle_rate = df["throttle_rate"].clip(lower=0)

    smooth_accel = throttle_rate < GENTLE_ACCEL_RATE_THRESH
    throttle_active = df["throttle"] > 0.1 if "throttle" in df.columns \
                      else pd.Series(True, index=df.index)

    candidate = smooth_accel & throttle_active

    if apex_reference is not None and "lap_dist_m" in df.columns:
        post_apex_zone = pd.Series(False, index=df.index)
        for apex_dist in apex_reference:
            in_zone = (
                (df["lap_dist_m"] >= apex_dist) &
                (df["lap_dist_m"] <= apex_dist + 200)  # 200m exit zone
            )
            post_apex_zone = post_apex_zone | in_zone
        candidate = candidate & post_apex_zone

    return has_consecutive_true(candidate, GENTLE_ACCEL_MIN_SAMPLES).astype(int)
