"""
temporal_alignment.py — Multi-file chronological ingestion and fixed-rate
resampling for LFS telemetry sessions, per proposal §3.5.2.

DataCatcher.py (src/collection/telemetry_recorder.py) logs at a nominal
60Hz but real capture jitter (observed: mean ~17.24ms, spikes to 156ms)
means a session is rarely on a perfectly regular grid, and long sessions
are often split across multiple CSV files. This module:
  1. Discovers and chronologically merges all CSVs for a session, shifting
     lap numbers across file boundaries so lap counting stays continuous.
  2. Resamples the merged stream onto a fixed time grid via interpolation,
     so every downstream window is a uniform ~17ms (58.8Hz) cadence.

Ported from `final-lstm-module-codes.ipynb` ("LFS Multistream Ingestion"
and the resampling half of "LFS Adapter").
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


def load_and_compile_telemetry_folder(input_source, fallback_engine: str = "c") -> pd.DataFrame:
    """
    Discover all LFS telemetry CSVs in a folder (or take an explicit list of
    paths), sort them chronologically by the numeric timestamp embedded in
    each DataCatcher.py filename, and merge them into one continuous stream
    with lap numbers shifted across file boundaries.
    """
    if isinstance(input_source, (str, Path)):
        folder = Path(input_source)
        csv_files = list(folder.glob("*.csv"))
    elif isinstance(input_source, list):
        csv_files = [Path(f) for f in input_source]
    else:
        raise ValueError("input_source must be a folder path string, Path object, or list of file paths.")

    if not csv_files:
        raise FileNotFoundError(f"No CSV telemetry files discovered for input: {input_source}")

    def extract_timestamp(path: Path) -> int:
        match = re.search(r'\d+', path.name)
        return int(match.group()) if match else 0

    sorted_files = sorted(csv_files, key=extract_timestamp)
    print(f"[Ingest] Found and sorted {len(sorted_files)} telemetry files chronologically")

    try:
        import pyarrow  # noqa: F401
        csv_engine = "pyarrow"
    except ImportError:
        csv_engine = fallback_engine
    print(f"[Ingest] Using CSV engine: '{csv_engine}'")

    compiled_dfs = []
    cumulative_lap_offset = 0

    for idx, file_path in enumerate(sorted_files):
        df = pd.read_csv(file_path, engine=csv_engine)
        if df.empty:
            continue
        if "Lap" not in df.columns:
            raise KeyError(f"'Lap' column missing in telemetry file: {file_path.name}")

        if idx > 0 and cumulative_lap_offset > 0:
            df["Lap"] += cumulative_lap_offset

        cumulative_lap_offset = int(df["Lap"].max())
        compiled_dfs.append(df)
        print(f"  Processed: {file_path.name} | rows: {len(df):,} | "
              f"laps: {df['Lap'].min()}-{df['Lap'].max()}")

    unified_df = pd.concat(compiled_dfs, ignore_index=True)
    print(f"[Ingest] Consolidated shape: {unified_df.shape}")
    return unified_df


def drop_duplicates_and_resample(df: pd.DataFrame, target_interval_ms: int = 17) -> pd.DataFrame:
    """
    Fix sampling-rate jitter in the raw telemetry stream:
      1. Drop exact-duplicate timestamps.
      2. Resample to a fixed target_interval_ms grid via linear interpolation.

    target_interval_ms=17 ~= 58.8Hz, matching the observed real capture rate.
    Residual timestamp error after resampling stays within the bound needed
    for the 50ms minimum label-event duration used throughout threshold_rules.
    """
    df = df.copy()
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    df = df.drop_duplicates(subset="timestamp_ms", keep="first")

    t_start = int(df["timestamp_ms"].iloc[0])
    t_end = int(df["timestamp_ms"].iloc[-1])
    regular_times = np.arange(t_start, t_end + 1, target_interval_ms)

    df = df.set_index("timestamp_ms")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    df = df.reindex(df.index.union(regular_times))
    df[numeric_cols] = df[numeric_cols].interpolate(method="index")
    df = df.reindex(regular_times)
    df.index.name = "timestamp_ms"
    df = df.reset_index()

    df["interpolated"] = False  # per-row flag placeholder; refined at labelling stage
    return df


def compute_jitter_statistics(df: pd.DataFrame, timestamp_col: str = "timestamp_ms") -> dict:
    """Compute inter-sample gap statistics, used to verify capture quality before resampling."""
    gaps = df[timestamp_col].diff().dropna()
    return {
        "mean_gap_ms": float(gaps.mean()),
        "max_gap_ms": float(gaps.max()),
        "min_gap_ms": float(gaps.min()),
        "std_gap_ms": float(gaps.std()),
    }
