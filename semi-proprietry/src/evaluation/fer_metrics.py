"""
fer_metrics.py — Feedback Event Rate (FER): unique coaching items generated
per lap, the core metric for RQ2 (proposal §3.7).

Ported from `final-cnn-module-codes.ipynb` ("RQ2 — CELL 1: FER computation").
"""

import pandas as pd


def compute_fer(predictions_df: pd.DataFrame, n_laps: int) -> dict:
    """
    Compute Feedback Event Rate and per-class event counts from the
    DataFrame returned by an inference engine's predict_from_csv().

    FER = number of windows where at least one event fires / number of laps.

    Returns:
        dict with fer, total_events, per_class counts, and per_lap_fer
        (a list, one value per lap — required for the paired t-test).
    """
    pred_cols = [c for c in predictions_df.columns if c.startswith("pred_")]
    if not pred_cols:
        raise ValueError("[FER] No pred_ columns found. Run predict_from_csv() first.")

    any_event = predictions_df[pred_cols].any(axis=1)
    total_events = int(any_event.sum())
    fer = total_events / max(n_laps, 1)

    per_class = {}
    for col in pred_cols:
        cls_name = col.replace("pred_", "").replace("_", " ")
        per_class[cls_name] = int(predictions_df[col].sum())

    per_lap_fer = []
    if "lap" in predictions_df.columns:
        for lap_id in sorted(predictions_df["lap"].unique()):
            lap_mask = predictions_df["lap"] == lap_id
            per_lap_fer.append(int(any_event[lap_mask].sum()))
    else:
        n = len(predictions_df)
        lap_size = n // max(n_laps, 1)
        for i in range(n_laps):
            start = i * lap_size
            end = start + lap_size if i < n_laps - 1 else n
            per_lap_fer.append(int(any_event.iloc[start:end].sum()))

    return {"fer": round(fer, 2), "total_events": total_events, "per_class": per_class, "per_lap_fer": per_lap_fer}


def compare_sessions(normal_preds: pd.DataFrame, detuned_preds: pd.DataFrame,
                      normal_laps: int, detuned_laps: int) -> dict:
    """
    Compute and print FER for a normal-driving session and an intentionally
    detuned (impaired) session, for the RQ2 sensitivity comparison.
    """
    normal_fer_data = compute_fer(normal_preds, n_laps=normal_laps)
    detuned_fer_data = compute_fer(detuned_preds, n_laps=detuned_laps)

    print(f"Normal  session — FER: {normal_fer_data['fer']:.2f} events/lap  (total: {normal_fer_data['total_events']})")
    print(f"Detuned session — FER: {detuned_fer_data['fer']:.2f} events/lap  (total: {detuned_fer_data['total_events']})")
    pct = ((detuned_fer_data['fer'] / max(normal_fer_data['fer'], 0.01)) - 1) * 100
    print(f"\nFER increase: {detuned_fer_data['fer'] - normal_fer_data['fer']:.2f} ({pct:.1f}% higher)")

    return {"normal": normal_fer_data, "detuned": detuned_fer_data}
