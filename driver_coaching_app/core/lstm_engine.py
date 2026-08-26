"""
Telemetry (LSTM) engine — ported 1:1 from cnn-module-codes.ipynb
Section 16.2 (AttentionLSTM) and Section 16.2b (LFS adapter + LSTMInferenceEngine).

Any architecture change made in the notebook must be mirrored here or
torch.load(...).load_state_dict(...) will fail with a shape/key mismatch.
"""
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

# ── Canonical feature columns fed to the LSTM (order matters) ──────────────
FEATURE_COLS = [
    "brake", "throttle", "steer", "clutch", "engine_rpm_norm", "gear",
    "speed_kmh_norm", "slip_ratio_lf", "slip_ratio_rf", "slip_ratio_lr",
    "slip_ratio_rr", "wheel_spin_lf", "wheel_spin_rf", "wheel_spin_lr",
    "wheel_spin_rr", "body_slip_angle", "throttle_rate", "brake_rate",
    "steer_rate", "lateral_g", "longitudinal_g", "susp_load_lf_norm",
    "susp_load_rf_norm", "susp_load_lr_norm", "susp_load_rr_norm",
]
INPUT_DIM = len(FEATURE_COLS)  # 25

LABEL_COLS = [
    "label_brake_locked", "label_wheel_spin", "label_trail_brake",
    "label_aggressive_downshift", "label_late_upshift",
    "label_rough_steering", "label_gentle_accel",
]
CLASS_NAMES = [
    "Brake Locked", "Wheel Spin", "Trail Brake", "Aggressive Downshift",
    "Late Upshift", "Rough Steering", "Gentle Acceleration",
]
NUM_CLASSES = len(LABEL_COLS)  # 7

WINDOW_SIZE = 60
HIDDEN_DIM = 512
NUM_LAYERS = 3
DROPOUT = 0.3

VEHICLE_CONFIG = {
    "default": {
        "redline_rpm": 7031,
        "power_peak_rpm": 5956,
        "max_speed_kmh": 205,
        "susp_load_max": 8000,
    }
}

LFS_TO_CANONICAL = {
    "Timestamp_MS": "timestamp_ms", "Session_Time_S": "session_time_s",
    "Lap": "lap", "Lap_Dist_M": "lap_dist_m", "Distance_M": "distance_m",
    "Speed_KMH": "speed_kmh", "Engine_RPM": "engine_rpm", "Gear": "gear",
    "Throttle": "throttle", "Brake": "brake", "Steer": "steer",
    "Clutch": "clutch", "Handbrake": "handbrake",
    "Throttle_Rate": "throttle_rate", "Brake_Rate": "brake_rate",
    "Steer_Rate": "steer_rate", "Slip_Ratio_LF": "slip_ratio_lf",
    "Slip_Ratio_RF": "slip_ratio_rf", "Slip_Ratio_LR": "slip_ratio_lr",
    "Slip_Ratio_RR": "slip_ratio_rr", "Body_Slip_Angle": "body_slip_angle",
    "AngVel_X": "angvel_x", "AngVel_Y": "angvel_y", "AngVel_Z": "angvel_z",
    "Susp_Load_LF": "susp_load_lf", "Susp_Load_RF": "susp_load_rf",
    "Susp_Load_LR": "susp_load_lr", "Susp_Load_RR": "susp_load_rr",
    "Wheel_Spin_LF": "wheel_spin_lf", "Wheel_Spin_RF": "wheel_spin_rf",
    "Wheel_Spin_LR": "wheel_spin_lr", "Wheel_Spin_RR": "wheel_spin_rr",
    "Accel_X": "accel_x_ms2", "Accel_Y": "accel_y_ms2", "Accel_Z": "accel_z_ms2",
    "Roll": "roll", "Pitch": "pitch", "Heading": "heading",
}


if TORCH_AVAILABLE:

    class TemporalAttention(nn.Module):
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.attn_linear = nn.Linear(hidden_dim, 1, bias=False)

        def forward(self, lstm_out):
            scores = self.attn_linear(lstm_out)
            weights = torch.softmax(scores, dim=1)
            context = (lstm_out * weights).sum(dim=1)
            return context, weights.squeeze(-1)

    class AttentionLSTM(nn.Module):
        def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                     num_layers=NUM_LAYERS, num_classes=NUM_CLASSES,
                     dropout=DROPOUT, window_size=WINDOW_SIZE):
            super().__init__()
            self.input_norm = nn.BatchNorm1d(window_size)
            self.lstm = nn.LSTM(
                input_size=input_dim, hidden_size=hidden_dim,
                num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.attention = TemporalAttention(hidden_dim)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=False),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim // 2, num_classes),
            )

        def forward(self, x, return_attention: bool = False):
            x = self.input_norm(x)
            lstm_out, _ = self.lstm(x)
            context, attn_weights = self.attention(lstm_out)
            logits = self.classifier(context)
            if return_attention:
                return logits, attn_weights
            return logits


# ── LFS raw -> canonical adapter (ported from the notebook, unchanged) ─────

def rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {k: v for k, v in LFS_TO_CANONICAL.items() if k in df.columns}
    return df.rename(columns=rename_map)


def convert_units(df: pd.DataFrame, vehicle: str = "default") -> pd.DataFrame:
    cfg = VEHICLE_CONFIG.get(vehicle, VEHICLE_CONFIG["default"])
    for axis in ["x", "y", "z"]:
        col = f"accel_{axis}_ms2"
        if col in df.columns:
            name = "lateral_g" if axis == "y" else "longitudinal_g" if axis == "x" else "vertical_g"
            df[name] = df[col] / 9.81
    if "engine_rpm" in df.columns:
        df["engine_rpm_norm"] = (df["engine_rpm"] / cfg["redline_rpm"]).clip(0, 1.2)
    if "speed_kmh" in df.columns:
        df["speed_kmh_norm"] = (df["speed_kmh"] / cfg["max_speed_kmh"]).clip(0, 1.2)
    for corner in ["lf", "rf", "lr", "rr"]:
        raw_col = f"susp_load_{corner}"
        if raw_col in df.columns:
            df[f"{raw_col}_norm"] = (df[raw_col] / cfg["susp_load_max"]).clip(0, 2)
    return df


def drop_duplicates_and_resample(df: pd.DataFrame, target_interval_ms: int = 17) -> pd.DataFrame:
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
    df["interpolated"] = False
    return df


def adapt_lfs_telemetry(csv_path, vehicle: str = "default") -> pd.DataFrame:
    if isinstance(csv_path, pd.DataFrame):
        df = csv_path.copy()
    else:
        df = pd.read_csv(csv_path)
    df = rename_to_canonical(df)
    df = convert_units(df, vehicle=vehicle)
    df = drop_duplicates_and_resample(df)
    df["data_source"] = "lfs"
    return df


def apply_normalizer(df: pd.DataFrame, params: dict, feature_cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns or col not in params:
            continue
        col_min = params[col]["min"]
        col_max = params[col]["max"]
        rng = col_max - col_min
        df[col] = 0.0 if rng == 0 else ((df[col] - col_min) / rng).clip(0, 1)
    return df


def load_and_compile_telemetry_folder(input_source) -> pd.DataFrame:
    """Discovers + chronologically merges a folder of raw LFS telemetry CSVs."""
    folder = Path(input_source)
    csv_files = sorted(folder.glob("*.csv"),
                        key=lambda p: int(m.group()) if (m := re.search(r"\d+", p.name)) else 0)
    if not csv_files:
        raise FileNotFoundError(f"No telemetry CSVs found in {input_source}")
    dfs, offset = [], 0
    for idx, f in enumerate(csv_files):
        df = pd.read_csv(f)
        if df.empty or "Lap" not in df.columns:
            continue
        if idx > 0 and offset > 0:
            df["Lap"] += offset
        offset = int(df["Lap"].max())
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


class TelemetryEngine:
    """Runtime wrapper matching LSTMInferenceEngine from the notebook.
    If no checkpoint/normalizer is present, falls back to zero-probability
    output (never a fabricated heuristic — telemetry math is too specific
    to approximate safely)."""

    def __init__(self, weights_path: Optional[Path] = None,
                 normalizer_path: Optional[Path] = None,
                 threshold_path: Optional[Path] = None,
                 device: Optional[str] = None):
        self.available = TORCH_AVAILABLE and weights_path and Path(weights_path).exists() \
            and normalizer_path and Path(normalizer_path).exists()

        self.thresholds = [0.5] * NUM_CLASSES
        if threshold_path and Path(threshold_path).exists():
            with open(threshold_path) as f:
                thr = json.load(f)
            self.thresholds = [thr.get(col, 0.5) for col in LABEL_COLS]

        if not self.available:
            self.norm_params = {}
            return

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        with open(normalizer_path) as f:
            self.norm_params = json.load(f)

        ckpt = torch.load(weights_path, map_location=self.device)
        cfg = ckpt.get("model_config", {})
        self.model = AttentionLSTM(
            input_dim=cfg.get("input_dim", INPUT_DIM),
            hidden_dim=cfg.get("hidden_dim", HIDDEN_DIM),
            num_layers=cfg.get("num_layers", NUM_LAYERS),
            num_classes=cfg.get("num_classes", NUM_CLASSES),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def predict_from_csv(self, raw_csv_path, vehicle: str = "default", stride: int = 30) -> np.ndarray:
        """Returns (N_windows, 7) sigmoid probability array, LABEL_COLS order.
        Empty array if no checkpoint is loaded."""
        if not self.available:
            return np.zeros((0, NUM_CLASSES), dtype=np.float32)

        df = adapt_lfs_telemetry(raw_csv_path, vehicle=vehicle)
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
        if not feature_cols:
            return np.zeros((0, NUM_CLASSES), dtype=np.float32)

        df = apply_normalizer(df, self.norm_params, feature_cols)
        df[feature_cols] = df[feature_cols].fillna(0.0)

        features = df[feature_cols].reindex(columns=FEATURE_COLS, fill_value=0.0).values.astype(np.float32)
        windows = []
        for start in range(0, len(df) - WINDOW_SIZE + 1, stride):
            windows.append(features[start:start + WINDOW_SIZE])
        if not windows:
            return np.zeros((0, NUM_CLASSES), dtype=np.float32)

        x = torch.from_numpy(np.stack(windows)).to(self.device)
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(x), 512):
                logits = self.model(x[i:i + 512])
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(all_probs, axis=0)
