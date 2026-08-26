"""
lfs_adapter.py — Maps raw LFS telemetry CSV columns to the canonical schema
and resolves unit discrepancies (LFS vs Assetto Corsa), per proposal §3.5.2.

Run this first, before threshold_rules / annotation_pipeline, and before
temporal_alignment's resampling step.

Ported from `final-lstm-module-codes.ipynb` ("LFS Adapter").
"""

from pathlib import Path
from typing import Union

import pandas as pd

from src.preprocessing.temporal_alignment import drop_duplicates_and_resample

# ── Raw LFS column -> canonical column mapping ───────────────────────────────
LFS_TO_CANONICAL = {
    "Timestamp_MS": "timestamp_ms",
    "Session_Time_S": "session_time_s",
    "Lap": "lap",
    "Lap_Dist_M": "lap_dist_m",
    "Distance_M": "distance_m",
    "Speed_KMH": "speed_kmh",
    "Engine_RPM": "engine_rpm",
    "Gear": "gear",
    "Throttle": "throttle",
    "Brake": "brake",
    "Steer": "steer",
    "Clutch": "clutch",
    "Handbrake": "handbrake",
    "Throttle_Rate": "throttle_rate",
    "Brake_Rate": "brake_rate",
    "Steer_Rate": "steer_rate",
    "Slip_Ratio_LF": "slip_ratio_lf",
    "Slip_Ratio_RF": "slip_ratio_rf",
    "Slip_Ratio_LR": "slip_ratio_lr",
    "Slip_Ratio_RR": "slip_ratio_rr",
    "Body_Slip_Angle": "body_slip_angle",
    "AngVel_X": "angvel_x",
    "AngVel_Y": "angvel_y",
    "AngVel_Z": "angvel_z",
    "Susp_Load_LF": "susp_load_lf",
    "Susp_Load_RF": "susp_load_rf",
    "Susp_Load_LR": "susp_load_lr",
    "Susp_Load_RR": "susp_load_rr",
    "Wheel_Spin_LF": "wheel_spin_lf",
    "Wheel_Spin_RF": "wheel_spin_rf",
    "Wheel_Spin_LR": "wheel_spin_lr",
    "Wheel_Spin_RR": "wheel_spin_rr",
    "Accel_X": "accel_x_ms2",
    "Accel_Y": "accel_y_ms2",
    "Accel_Z": "accel_z_ms2",
    "Roll": "roll",
    "Pitch": "pitch",
    "Heading": "heading",
}

# Vehicle-specific constants — update per car used in LFS.
VEHICLE_CONFIG = {
    "default": {
        "redline_rpm": 7031,
        "power_peak_rpm": 5956,   # placeholder ratio-matched default — verify against real car
        "max_speed_kmh": 280,
        "susp_load_max": 5000,   # Newton-metres, normalise by this
    }
}


def load_raw_lfs_csv(csv_path: str) -> pd.DataFrame:
    """Load the raw LFS telemetry CSV (as produced by DataCatcher.py) and sanity-check columns."""
    df = pd.read_csv(csv_path)
    missing = [c for c in LFS_TO_CANONICAL if c not in df.columns]
    if missing:
        print(f"[WARN] Missing expected LFS columns: {missing}")
    return df


def rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw LFS columns to the canonical schema names."""
    rename_map = {k: v for k, v in LFS_TO_CANONICAL.items() if k in df.columns}
    return df.rename(columns=rename_map)


def convert_units(df: pd.DataFrame, vehicle: str = "default") -> pd.DataFrame:
    """
    Apply unit conversions that differ between LFS and Assetto Corsa:
      - Accel X/Y/Z: m/s^2 -> G  (AC gives G natively; LFS gives m/s^2)
      - RPM: normalize by redline
      - Speed: normalize by max speed
      - Suspension load: normalize by max load
    """
    cfg = VEHICLE_CONFIG.get(vehicle, VEHICLE_CONFIG["default"])

    axis_name = {"x": "longitudinal_g", "y": "lateral_g", "z": "vertical_g"}
    for axis, out_col in axis_name.items():
        col = f"accel_{axis}_ms2"
        if col in df.columns:
            df[out_col] = df[col] / 9.81

    if "engine_rpm" in df.columns:
        df["engine_rpm_norm"] = (df["engine_rpm"] / cfg["redline_rpm"]).clip(0, 1.2)

    if "speed_kmh" in df.columns:
        df["speed_kmh_norm"] = (df["speed_kmh"] / cfg["max_speed_kmh"]).clip(0, 1.2)

    for corner in ["lf", "rf", "lr", "rr"]:
        raw_col = f"susp_load_{corner}"
        if raw_col in df.columns:
            df[f"{raw_col}_norm"] = (df[raw_col] / cfg["susp_load_max"]).clip(0, 2)

    return df


def add_source_column(df: pd.DataFrame, source: str = "lfs") -> pd.DataFrame:
    """Tag every row with its data source for downstream traceability."""
    df["data_source"] = source
    return df


def adapt_lfs_telemetry(csv_path: Union[str, Path, pd.DataFrame],
                         vehicle: str = "default",
                         output_path: str = None) -> pd.DataFrame:
    """Full adapter pipeline: load -> rename -> convert units -> resample -> tag source."""
    if isinstance(csv_path, pd.DataFrame):
        print("[Adapter] Processing pre-compiled unified DataFrame...")
        df = csv_path.copy()
    else:
        print(f"[Adapter] Loading: {csv_path}")
        df = load_raw_lfs_csv(csv_path)

    print(f"  Raw rows: {len(df):,}  |  Columns: {len(df.columns)}")

    df = rename_to_canonical(df)
    df = convert_units(df, vehicle=vehicle)
    df = drop_duplicates_and_resample(df)
    df = add_source_column(df, source="lfs")

    print(f"  Canonical rows: {len(df):,}  |  Source: lfs")

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"  Saved -> {output_path}")

    return df


def adapt_ac_telemetry(csv_path: str, vehicle: str = "default", output_path: str = None) -> pd.DataFrame:
    """
    Adapter for Assetto Corsa / Telemetrick CSV output. AC already provides
    G-forces and hydraulic brake pressure natively, so unit conversion is
    lighter than the LFS path. Fill in AC_TO_CANONICAL once an AC CSV sample
    is available — not exercised in this project's dataset.
    """
    AC_TO_CANONICAL = {
        # e.g. "BrakePressure": "brake", "Speed": "speed_kmh", ...
    }
    print("[Adapter] AC adapter - column mapping not yet defined. "
          "Add AC->canonical mappings to AC_TO_CANONICAL.")
    df = pd.read_csv(csv_path)
    rename_map = {k: v for k, v in AC_TO_CANONICAL.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    df = add_source_column(df, source="ac")

    if output_path:
        df.to_csv(output_path, index=False)
    return df


def merge_sessions(csv_paths: list, vehicle: str = "default", output_path: str = None) -> pd.DataFrame:
    """Adapt and concatenate multiple LFS session CSVs, tagging each with a session_id."""
    frames = []
    for i, path in enumerate(csv_paths):
        df = adapt_lfs_telemetry(path, vehicle=vehicle)
        df["session_id"] = i
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"[Adapter] Merged {len(csv_paths)} sessions -> {len(combined):,} rows")

    if output_path:
        combined.to_csv(output_path, index=False)
        print(f"  Saved -> {output_path}")
    return combined
