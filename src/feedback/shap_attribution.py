"""
shap_attribution.py — SHAP GradientExplainer on the fusion head, per
proposal §3.5.5.

Decomposes each fusion head prediction into contributions from its 10
input features (3 CNN probabilities + 7 LSTM probabilities), e.g.
"the Off-Track prediction was driven 70% by cnn_off_track and 30% by
lstm_rough_steering co-occurring." This is the SHAP explanation that
directly drives feedback_generator's confirmation gate — different from
CNN Grad-CAM (which pixels drove the CNN) or LSTM SHAP (which telemetry
features drove the LSTM); this one explains which SUB-MODEL drove the
final combined decision.

Requires `pip install shap`.

Ported from `final-cnn-module-codes.ipynb` ("16.4 Late Fusion Pipeline",
SECTION 5: SHAP on fusion head).
"""

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.fusion.fusion_constants import CLASS_NAMES, INPUT_COLS, DIAG_DIR


def run_fusion_shap(model, val_dl, device: str = None, output_dir: Path = DIAG_DIR,
                     n_bg: int = 200, n_test: int = 500) -> dict:
    """
    Run SHAP GradientExplainer over the trained LateFusionHead and save both
    a per-class importance bar-chart figure and a JSON summary consumed by
    FeedbackGenerator's SHAP-confirmation gate.
    """
    try:
        import shap
    except ImportError:
        print("[SHAP] Run: pip install shap")
        return {}

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_x, all_y = [], []
    for x, y in val_dl:
        all_x.append(x.numpy())
        all_y.append(y.numpy())
        if sum(a.shape[0] for a in all_x) >= n_bg + n_test:
            break
    all_x = np.concatenate(all_x, axis=0)

    bg_x = torch.tensor(all_x[:n_bg], dtype=torch.float32).to(device)
    test_x = torch.tensor(all_x[n_bg:n_bg + n_test], dtype=torch.float32).to(device)

    model.eval()
    explainer = shap.GradientExplainer(model, bg_x)
    shap_values = explainer.shap_values(test_x)  # list of arrays, one per output class

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        vals = np.abs(shap_values[cls_idx])
        mean_importance = vals.mean(axis=0)

        order = np.argsort(mean_importance)[::-1]
        features = [INPUT_COLS[i] for i in order]
        values = mean_importance[order]
        colors = ["#d62728" if "cnn" in f else "#1f77b4" for f in features]

        ax = axes[cls_idx]
        ax.barh(range(len(features)), values[::-1], color=colors[::-1], alpha=0.85)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([f.replace("cnn_", "CNN:").replace("lstm_", "LSTM:") for f in features[::-1]], fontsize=7)
        ax.set_title(cls_name, fontsize=9)
        ax.set_xlabel("|SHAP|", fontsize=7)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#d62728", label="CNN output"), Patch(color="#1f77b4", label="LSTM output")],
               loc="lower right", fontsize=9)
    fig.suptitle("Fusion Head SHAP — Which sub-model output drove each final prediction?", fontsize=12)
    plt.tight_layout()
    out = output_dir / "fusion_shap_importance.png"
    plt.savefig(str(out), dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Fusion SHAP saved -> {out}")

    shap_summary = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        vals = np.abs(shap_values[cls_idx]).mean(axis=0)
        shap_summary[cls_name] = {col: round(float(v), 6) for col, v in zip(INPUT_COLS, vals)}

    with open(output_dir / "fusion_shap_summary.json", "w") as f:
        json.dump(shap_summary, f, indent=2)
    print(f"[SHAP] SHAP summary JSON -> {output_dir}/fusion_shap_summary.json")

    return shap_summary


def load_shap_summary(json_path: str) -> dict:
    """Load a previously saved fusion_shap_summary.json for use in FeedbackGenerator."""
    with open(json_path) as f:
        return json.load(f)
