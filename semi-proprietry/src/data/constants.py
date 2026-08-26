"""
constants.py — Single source of truth for column names, thresholds, and
all shared constants used across the LSTM data/training pipeline.
Matches project proposal §1.6 (behaviour thresholds) and §3.5.4 (architecture).

Ported verbatim from `final-lstm-module-codes.ipynb` ("Constants").
"""

# ── Canonical feature columns fed to the LSTM ────────────────────────────────
# These are the column names AFTER the LFS/AC adapter has run.
# Order matters — model input tensors follow this order.
FEATURE_COLS = [
    "brake",            # pedal input 0–1
    "throttle",         # pedal input 0–1
    "steer",            # normalised steering -1 to 1
    "clutch",           # 0–1
    "engine_rpm_norm",  # RPM / redline  (computed in adapter)
    "gear",             # integer, kept as float for the model
    "speed_kmh_norm",   # speed / max_speed (computed in adapter)
    "slip_ratio_lf",
    "slip_ratio_rf",
    "slip_ratio_lr",
    "slip_ratio_rr",
    "wheel_spin_lf",
    "wheel_spin_rf",
    "wheel_spin_lr",
    "wheel_spin_rr",
    "body_slip_angle",
    "throttle_rate",    # pre-computed by LFS telemetry script
    "brake_rate",
    "steer_rate",
    "lateral_g",        # Accel_Y / 9.81
    "longitudinal_g",   # Accel_X / 9.81
    "susp_load_lf_norm",
    "susp_load_rf_norm",
    "susp_load_lr_norm",
    "susp_load_rr_norm",
]

INPUT_DIM = len(FEATURE_COLS)

# ── Label columns (7 behaviour classes) ──────────────────────────────────────
LABEL_COLS = [
    "label_brake_locked",
    "label_wheel_spin",
    "label_trail_brake",
    "label_aggressive_downshift",
    "label_late_upshift",
    "label_rough_steering",
    "label_gentle_accel",
]
NUM_CLASSES = len(LABEL_COLS)

CLASS_NAMES = [
    "Brake Locked",
    "Wheel Spin",
    "Trail Brake",
    "Aggressive Downshift",
    "Late Upshift",
    "Rough Steering",
    "Gentle Acceleration",
]

# ── Labelling thresholds (proposal §1.6) ─────────────────────────────────────
SAMPLE_RATE_HZ           = 60
BRAKE_PRESSURE_THRESH    = 0.68    # brake pedal input threshold for lockup
SLIP_RATIO_THRESH        = 0.90    # wheel slip ratio for lockup
LOCKUP_MIN_SAMPLES       = 3       # 50ms at 60Hz
WHEEL_SPIN_SLIP_THRESH   = 0.30    # driven wheel slip ratio (wheel_speed-car_speed)/car_speed
WHEEL_SPIN_THROTTLE_MIN  = 0.10    # must be accelerating
WHEEL_SPIN_MIN_SAMPLES   = 2
TRAIL_BRAKE_THRESH       = 0.15    # brake pressure past apex

# NOTE: these two ARE now read by label_aggressive_downshift and
# label_late_upshift (wired in below). Values corrected to match each
# function's own docstring reasoning — do not lower DOWNSHIFT_RPM_PCT
# toward 0.60 again: mean engine_rpm_norm across the dataset is ~0.72-0.87,
# so anything below ~0.90 makes "overrev" true on most driving rows, not
# just genuine overrevs (this is exactly the 51.7%-label-rate bug the
# function's rewrite was designed to fix).
DOWNSHIFT_RPM_PCT        = 0.95    # fraction of redline for post-shift overrev detection
# NEW: label_aggressive_downshift's post-shift consequence window needs its
# OWN constant — it was previously (incorrectly) sharing LATE_UPSHIFT_MIN_SAMPLES
# (90 samples / 1.5s), which stretched "did this happen because of the downshift"
# to 1.5s post-shift instead of the intended ~200ms.
DOWNSHIFT_POST_SHIFT_WINDOW = 12   # rows (200ms @ 60Hz) to look for post-shift consequences

LATE_UPSHIFT_MIN_SAMPLES = 20      # 0.33s @ 60Hz — how long above-power-peak+throttle must persist

# NOTE: base/high-speed thresholds for label_rough_steering are both wired
# in below. base = low-speed endpoint (looser), high_speed = high-speed
# endpoint (tighter) — dynamic_threshold interpolates DOWN from base to
# high_speed as speed rises. Do not set base below high_speed or the
# interpolation direction inverts (loosens at high speed instead of
# tightening) and low-speed driving becomes falsely flagged as "rough."
ROUGH_STEER_RATE_THRESH       = 180.00   # deg/s at speed_low_gate_kmh (40 km/h)
ROUGH_STEER_HIGH_SPEED_THRESH = 90.00    # deg/s at speed_high_gate_kmh (140 km/h)

# FIX: throttle_rate is on a percentage-points-per-sample scale (describe():
# min=-117.7, max=100.2), NOT a 0-1 fraction. The old value (0.10) was true on
# almost every row with any throttle motion at all, giving a 68.8% positive
# rate. 10.0 means "throttle moved by less than 10 percentage points in one
# 1/60s sample" — a much more realistic definition of "smooth."
GENTLE_ACCEL_RATE_THRESH = 10.0    # throttle_rate units/sample (below = smooth)
GENTLE_ACCEL_MIN_SAMPLES = 10      # must be sustained to count
brake_activation_thresh  = 0.10    # used to tell if actually braking or stop hold
speed_gate_kmh           = 10.0    # minimum speed to activating brake lock up tracking
slip_lock_thresh         = SLIP_RATIO_THRESH

# FIX: wheel_spin_lf/rf/lr/rr is wheel SURFACE SPEED in km/h (same scale as
# speed_kmh, confirmed via describe()), not rad/s and not a 0-1 value. The old
# wheel_spin_zero_thresh=0.5 compared against that ~100-scale column was a
# silent no-op (never true) in label_brake_locked's lockup fallback. Replaced
# with a dimensionless ratio: a wheel turning at < 20% of the car's ground
# speed while the car is moving is physically locked.
wheel_locked_speed_ratio = 0.20    # wheel_surface_speed / car_speed threshold for lockup

# ── LSTM architecture ─────────────────────────────────────────────────────────
# FIX: train_F1 (~0.86-0.96) vs val_F1 (~0.44-0.46) is a large, persistent gap —
# classic overfitting. 512-hidden x 3-layer LSTM+attention is a lot of capacity
# for 92 laps of data, especially combined with TRAIN_STRIDE=1 (windows only
# 1 row apart are near-duplicates, so the model can partially memorize rather
# than generalize). Cut capacity and raise dropout; this is the highest-leverage
# change available without more data.
WINDOW_SIZE   = 60       # timesteps — 1 second at 60Hz
HIDDEN_DIM    = 192      # was 512 — cut capacity to reduce overfitting on 92 laps
NUM_LAYERS    = 2        # was 3 — one fewer stacked layer, same reasoning
DROPOUT       = 0.4      # was 0.3 — slightly stronger regularization to match

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE       = 256
LEARNING_RATE    = 1e-3
FINETUNE_LR      = 1e-5
MAX_EPOCHS       = 512
PATIENCE         = 64    # early stopping
GRAD_CLIP        = 1.0
# FIX: stride=1 means adjacent training windows differ by only 1 of 60 rows —
# almost the same sequence seen thousands of times over, which inflates train
# metrics without adding real information and likely contributes to the
# overfitting gap above. A modest stride keeps plenty of training data (still
# far more windows than laps) while cutting near-duplicate redundancy.
TRAIN_STRIDE     = 3     # was 1
# FIX: EVAL_STRIDE=30 only samples a window's label every 0.5s. Brake Locked is
# a 0.5%-base-rate event with a 3-sample (50ms) minimum burst length — at stride
# 30 it's easy for a whole test split to contain zero sampled positives (which is
# what the 0/0 confusion matrix showed: 353 test windows, 0 True-1 either way).
# Lowered to 5 (~12ms overlap) so rare, short-duration events actually show up
# in val/test. This costs more eval compute but training (TRAIN_STRIDE) is unaffected.
EVAL_STRIDE      = 5

# ── Loss function ─────────────────────────────────────────────────────────────
# Set True to use focal loss instead of plain BCEWithLogitsLoss+pos_weight for
# the extreme-imbalance classes (see build_criterion below). Off by default —
# turn on and compare val macro-F1 against the BCE run before committing to it.
USE_FOCAL_LOSS   = False
FOCAL_GAMMA      = 2.0   # higher = more focus on hard/rare examples
FOCAL_ALPHA      = 0.25  # base weighting; combined with per-class pos_weight

# ── Inference ─────────────────────────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.50   # sigmoid threshold — tuned per class after training