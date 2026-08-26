"""
annotation_pipeline.py — End-to-end telemetry labelling pipeline.

Loads a canonical-schema telemetry CSV (post LFS/AC-adapter), applies all
seven threshold_rules label functions, and writes labelled_telemetry.csv —
the file the private repo's preprocessing/windowing stage consumes.

Designed to run unattended (overnight batch mode) over a folder of raw
session recordings, per proposal §3.5.1.

Ported from `final-lstm-module-codes.ipynb` ("LSTM Data - Labelling").
"""

import json
from pathlib import Path

import pandas as pd

from src.data.constants import LABEL_COLS
from src.labelling.threshold_rules import (
    label_brake_locked, label_wheel_spin, label_trail_brake,
    label_aggressive_downshift, label_late_upshift, label_rough_steering,
    label_gentle_accel,
)


# ── Apex reference loader ─────────────────────────────────────────────────────

def load_apex_reference(json_path: str, track_name: str) -> list:
    """
    Load apex distance values for a given track from a JSON reference file.

    JSON structure:
    {
        "blackwood_gp": [245.0, 612.0, 890.0, 1150.0, ...],
        "kyalami":      [310.0, 755.0, ...],
        ...
    }

    Build this file once per track by driving a clean reference lap and
    noting the Lap_Dist_M value at each corner's geometric apex.
    Returns a list of apex distances in metres, or empty list if not found.
    """
    path = Path(json_path)
    if not path.exists():
        print(f"[Label] No apex reference file found at {json_path}. "
              "Trail brake and gentle accel labels will use fallback logic.")
        return []

    with open(path) as f:
        refs = json.load(f)

    apexes = refs.get(track_name, [])
    if not apexes:
        print(f"[Label] Track '{track_name}' not found in apex reference. "
              f"Available: {list(refs.keys())}")
    return apexes


# ── Main labelling pipeline ───────────────────────────────────────────────────

def apply_all_labels(df: pd.DataFrame,
                      apex_reference: list = None,
                      vehicle_redline: float = 8500.0,
                      vehicle_power_peak_norm: float = 0.847) -> pd.DataFrame:
    """
    Apply all 7 behaviour labels to a canonical-schema DataFrame.

    Args:
        df:                     Canonical DataFrame (output of lfs_adapter)
        apex_reference:         List of apex_dist_m values for the current track
        vehicle_redline:        RPM redline for this car
        vehicle_power_peak_norm: Power peak RPM / redline for this car

    Returns:
        DataFrame with 7 new binary label columns appended.
    """
    df = df.copy()
    print(f"[Label] Applying labels to {len(df):,} rows...")

    df["label_brake_locked"] = label_brake_locked(df)
    df["label_wheel_spin"]   = label_wheel_spin(df)
    df["label_trail_brake"]  = label_trail_brake(df, apex_reference)
    df["label_aggressive_downshift"] = label_aggressive_downshift(
        df, vehicle_redline=vehicle_redline)
    df["label_late_upshift"] = label_late_upshift(
        df, vehicle_power_peak_norm=vehicle_power_peak_norm)
    df["label_rough_steering"] = label_rough_steering(df)
    df["label_gentle_accel"] = label_gentle_accel(df, apex_reference)

    # Summary
    for col in LABEL_COLS:
        count = df[col].sum()
        pct   = 100 * count / len(df)
        print(f"  {col:<30}: {count:>6,} positive rows ({pct:5.1f}%)")

    print(f"[Label] Done. {len(LABEL_COLS)} label columns added.")
    return df


def run_labelling_pipeline(canonical_csv_path: str,
                            output_csv_path: str,
                            apex_json_path: str = None,
                            track_name: str = None,
                            vehicle_redline: float = 8500.0,
                            vehicle_power_peak_norm: float = 0.847) -> str:
    """
    End-to-end labelling pipeline: load canonical CSV → label → save.
    Designed to run unattended (overnight batch mode).

    Returns:
        Path to the output labelled CSV.
    """
    df = pd.read_csv(canonical_csv_path)
    print(f"[Pipeline] Loaded {len(df):,} rows from {canonical_csv_path}")

    apex_ref = []
    if apex_json_path and track_name:
        apex_ref = load_apex_reference(apex_json_path, track_name)

    df = apply_all_labels(
        df,
        apex_reference=apex_ref if apex_ref else None,
        vehicle_redline=vehicle_redline,
        vehicle_power_peak_norm=vehicle_power_peak_norm,
    )

    df.to_csv(output_csv_path, index=False)
    print(f"[Pipeline] Labelled CSV saved → {output_csv_path}")
    return output_csv_path


def batch_process_all_sessions(raw_telemetry_dir: str, output_dir: str, apex_json_path: str = None, track_name: str = None) -> list:
    """
    Run run_labelling_pipeline across every canonical CSV found in
    raw_telemetry_dir. Intended for unattended overnight batch execution.

    Returns:
        List of output file paths for all newly labelled sessions.
    """
    raw_dir = Path(raw_telemetry_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for csv_path in sorted(raw_dir.glob("*.csv")):
        out_path = out_dir / f"labelled_{csv_path.stem}.csv"
        outputs.append(
            run_labelling_pipeline(
                str(csv_path), str(out_path),
                apex_json_path=apex_json_path, track_name=track_name,
            )
        )
    return outputs
