"""
Project-wide configuration constants.

Referenced by: src/collection, src/labelling, src/preprocessing, src/app
"""

# ── Removable storage medium ────────────────────────────────────────────
# CHANGE THIS PATH to match the drive letter / mount point of the
# 128GB removable storage medium on the current machine.
# Windows example: "E:/bimodal_driver_coach_data"
# macOS/Linux example: "/Volumes/DRIVECOACH/bimodal_driver_coach_data"
STORAGE_ROOT = "E:/bimodal_driver_coach_data"  # <-- EDIT PER MACHINE

# ── Subpaths under STORAGE_ROOT ─────────────────────────────────────────
RAW_VIDEO_DIR = f"{STORAGE_ROOT}/raw/video"
RAW_TELEMETRY_DIR = f"{STORAGE_ROOT}/raw/telemetry"
LABELLED_DIR = f"{STORAGE_ROOT}/labelled"
PROCESSED_DIR = f"{STORAGE_ROOT}/processed"

# ── Telemetry / video capture parameters (per proposal §3.4) ───────────
TELEMETRY_SAMPLE_RATE_HZ = 60
VIDEO_FRAME_RATE_FPS = 60
VIDEO_RESOLUTION = (1280, 720)

# ── Labelling thresholds (per proposal §1.6 / §3.5.1) ───────────────────
BRAKE_LOCKUP_PRESSURE_THRESHOLD = 0.80
BRAKE_LOCKUP_SLIP_THRESHOLD = 1.0
BRAKE_LOCKUP_MIN_SAMPLES = 3  # 50ms at 60Hz

APEX_MISS_DEVIATION_METERS = 1.5

MECHANICAL_ABUSE_RPM_DURATION_SAMPLES = 120  # 2 seconds at 60Hz

WHEEL_SPIN_SLIP_THRESHOLD = 0.3
TRAIL_BRAKE_PRESSURE_THRESHOLD = 0.20
AGGRESSIVE_DOWNSHIFT_RPM_PCT = 0.90
LATE_UPSHIFT_DURATION_SECONDS = 1.5
ROUGH_STEERING_RATE_DEG_PER_SEC = 120
GENTLE_ACCEL_THROTTLE_RATE_PCT = 0.15
VEHICLE_INSTABILITY_LATERAL_G = 0.8

# ── Model input specifications (per proposal §3.5.2-§3.5.4) ────────────
CNN_INPUT_SIZE = (224, 224)
LSTM_WINDOW_SIZE = 60  # timesteps, equivalent to 1 second at 60Hz

# ── Fusion / feedback parameters (per proposal §3.5.5) ──────────────────
FEEDBACK_PROBABILITY_THRESHOLD = 0.65
MODALITY_DROPOUT_PROBABILITY = 0.2
