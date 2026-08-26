"""
telemetry_normalization.py — Normalizes telemetry channels, handles missing
values, and provides session/lap-stratified train/val/test splitting so no
lap leaks across splits, per proposal §3.5.2.

Ported from `final-lstm-module-codes.ipynb` ("LSTM Data - Windowing").
"""

import json

import pandas as pd

from src.data.constants import FEATURE_COLS


def fit_normalizer(df: pd.DataFrame, feature_cols: list, save_path: str = "normalizer_params.json") -> dict:
    """
    Fit MinMax normalization on the TRAINING set only. Saves min/max per
    feature to JSON so inference can reproduce the exact same transform on
    never-seen data.
    """
    params = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        params[col] = {"min": float(df[col].min()), "max": float(df[col].max())}

    if save_path:
        with open(save_path, "w") as f:
            json.dump(params, f, indent=2)
        print(f"[Norm] Normalizer params saved -> {save_path}")

    return params


def apply_normalizer(df: pd.DataFrame, params: dict, feature_cols: list) -> pd.DataFrame:
    """
    Apply saved min-max normalization using pre-fitted params. Values are
    clipped to [0, 1] so out-of-range inputs (e.g. from a different
    simulator) can't blow up the model input.
    """
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns or col not in params:
            continue
        col_min, col_max = params[col]["min"], params[col]["max"]
        rng = col_max - col_min
        df[col] = 0.0 if rng == 0 else ((df[col] - col_min) / rng).clip(0, 1)
    return df


def load_normalizer(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def forward_fill_missing_values(df: pd.DataFrame, feature_cols: list = None) -> pd.DataFrame:
    """Fill missing values caused by session initialization lag using forward-fill."""
    cols = [c for c in (feature_cols or FEATURE_COLS) if c in df.columns]
    df = df.copy()
    df[cols] = df[cols].fillna(method="ffill").fillna(0.0)
    return df


def session_split(df: pd.DataFrame, train_sessions: list, val_sessions: list,
                   test_sessions: list, session_col: str = "session_id") -> tuple:
    """
    Split a multi-session DataFrame into train/val/test by session ID.
    Prevents data leakage — no frame from the same lap appears in both
    training and evaluation sets.
    """
    if session_col not in df.columns:
        n = len(df)
        t, v = int(n * 0.70), int(n * 0.85)
        print("[Split] No session_id column found — using 70/15/15 row split.")
        return df.iloc[:t], df.iloc[t:v], df.iloc[v:]

    train_df = df[df[session_col].isin(train_sessions)].reset_index(drop=True)
    val_df = df[df[session_col].isin(val_sessions)].reset_index(drop=True)
    test_df = df[df[session_col].isin(test_sessions)].reset_index(drop=True)

    print(f"[Split] Train: {len(train_df):,} rows | Val: {len(val_df):,} rows | Test: {len(test_df):,} rows")
    return train_df, val_df, test_df


def lap_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15) -> tuple:
    """Single-session fallback: split by lap number so consecutive laps stay together."""
    if "lap" not in df.columns:
        return session_split(df, [], [], [])

    laps = sorted(df["lap"].unique())
    n = len(laps)
    t_end, v_end = int(n * train_frac), int(n * (train_frac + val_frac))

    train_laps, val_laps, test_laps = laps[:t_end], laps[t_end:v_end], laps[v_end:]
    train_df = df[df["lap"].isin(train_laps)].reset_index(drop=True)
    val_df = df[df["lap"].isin(val_laps)].reset_index(drop=True)
    test_df = df[df["lap"].isin(test_laps)].reset_index(drop=True)

    print(f"[Split] Train laps: {train_laps} | Val laps: {val_laps} | Test laps: {test_laps}")
    print(f"        Rows -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    return train_df, val_df, test_df
