"""
Late Fusion engine — ported 1:1 from cnn-module-codes.ipynb Section 16.4
(LateFusionHead). Takes ONE concatenated 10-dim vector (3 CNN probs +
7 LSTM probs) and outputs 10 fused logits — this is "the framework's"
decision, not either sub-model's decision alone.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

CNN_COLS = ["cnn_off_track", "cnn_apex_miss", "cnn_sliding"]
LSTM_COLS = [
    "lstm_brake_locked", "lstm_wheel_spin", "lstm_trail_brake",
    "lstm_aggressive_downshift", "lstm_late_upshift",
    "lstm_rough_steering", "lstm_gentle_accel",
]
DEFAULT_INPUT_COLS = CNN_COLS + LSTM_COLS  # 10-dim, used if inference_config.json absent

FUSED_LABEL_COLS = [
    "label_off_track", "label_apex_miss", "label_sliding",
    "lstm_brake_locked", "lstm_wheel_spin", "lstm_trail_brake",
    "lstm_aggressive_downshift", "lstm_late_upshift",
    "lstm_rough_steering", "lstm_gentle_accel",
]
FUSED_CLASS_NAMES = [
    "Off-Track", "Apex Miss", "Sliding", "Brake Locked", "Wheel Spin",
    "Trail Brake", "Aggressive Downshift", "Late Upshift",
    "Rough Steering", "Gentle Accel",
]
INPUT_DIM = len(DEFAULT_INPUT_COLS)   # 10
NUM_CLASSES = len(FUSED_LABEL_COLS)   # 10


if TORCH_AVAILABLE:

    class LateFusionHead(nn.Module):
        """10-dim input -> Linear(64) -> ReLU -> Dropout(0.3)
                          -> Linear(32) -> ReLU -> Linear(num_classes)
        raw logits — sigmoid applied at inference."""

        def __init__(self, input_dim=INPUT_DIM, num_classes=NUM_CLASSES):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, num_classes),
            )

        def forward(self, x):
            return self.net(x)


class BimodalFusionEngine:
    """Runtime orchestrator for the trained fusion head. Loads
    inference_config.json (input_cols + per-label thresholds) if present,
    otherwise falls back to the notebook's default 10-class order with a
    flat 0.5 threshold."""

    def __init__(self, weights_path: Optional[Path] = None,
                 config_path: Optional[Path] = None,
                 device: Optional[str] = None):
        self.available = TORCH_AVAILABLE and weights_path and Path(weights_path).exists()

        self.input_cols: List[str] = list(DEFAULT_INPUT_COLS)
        self.label_cols: List[str] = list(FUSED_LABEL_COLS)
        self.thresholds: Dict[str, float] = {c: 0.5 for c in self.label_cols}

        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                cfg = json.load(f)
            self.input_cols = cfg.get("input_cols", self.input_cols)
            self.thresholds = cfg.get("thresholds", self.thresholds)

        if not self.available:
            return

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(weights_path, map_location=self.device)
        cfg = ckpt.get("model_config", {})
        self.model = LateFusionHead(
            input_dim=cfg.get("input_dim", len(self.input_cols)),
            num_classes=cfg.get("num_classes", len(self.label_cols)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def evaluate(self, cnn_probs: np.ndarray, lstm_probs: np.ndarray) -> pd.DataFrame:
        """cnn_probs: (N, 3), lstm_probs: (M, 7).

        Either modality may be absent (len 0) — the fusion head was
        trained with modality dropout specifically so it stays usable
        with only vision or only telemetry. When one is missing, that
        stream is zero-filled instead of collapsing the row count to 0.
        Only if BOTH are missing does this return empty.
        """
        n_cnn, n_lstm = len(cnn_probs), len(lstm_probs)
        if n_cnn == 0 and n_lstm == 0:
            return pd.DataFrame()

        n = max(n_cnn, n_lstm) if min(n_cnn, n_lstm) == 0 else min(n_cnn, n_lstm)

        if n_cnn == 0:
            cnn_probs = np.zeros((n, 3), dtype=np.float32)
        if n_lstm == 0:
            lstm_probs = np.zeros((n, 7), dtype=np.float32)

        combined = pd.DataFrame(cnn_probs[:n], columns=CNN_COLS)
        for i, col in enumerate(LSTM_COLS):
            combined[col] = lstm_probs[:n, i]
        combined = combined.reindex(columns=self.input_cols, fill_value=0.0)

        if not self.available:
            return pd.DataFrame()

        x = torch.tensor(combined.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            fused_probs = torch.sigmoid(self.model(x)).cpu().numpy()

        out = pd.DataFrame({"window_id": np.arange(n)})
        for i, col in enumerate(self.label_cols):
            thr = self.thresholds.get(col, 0.5)
            out[f"prob_{col}"] = fused_probs[:, i]
            out[f"pred_{col}"] = (fused_probs[:, i] > thr).astype(int)
        return out
