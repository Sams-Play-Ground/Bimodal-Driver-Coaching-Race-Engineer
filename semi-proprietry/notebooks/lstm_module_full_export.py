# LSTM Module — full notebook export
# Auto-exported from the original Kaggle .ipynb for reproducibility.
# Cell boundaries are marked; markdown cells are kept as comments.

# %% [markdown] cell 0
# # Libraries Imports

# %% cell 1
#lstm diagnostics.py
import json, shutil
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

#lstm evaluator.py
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve,
    classification_report,
)

#lstm inference.py

#lstm model.py
import torch.nn as nn
import torch.nn.functional as F

#augmentation.py

#distributions.py

#labelling.py

#lfs_adapter.py
from typing import Union, Optional

#windowing.py
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler

#lstm finetune.py

#lstm trainer.py
import time

#lstm pipeline.py
import os

# multistream injestion.py
import re


# %% [markdown] cell 2
# # LSTM Module - Diagnostics
# lstm_diagnostics.py — Explainability tools for the trained LSTM.
#
# Two complementary diagnostic methods:
#   1. Temporal Attention heatmaps  — WHICH timestep in the 1-second window
#                                     drove the prediction
#   2. SHAP DeepExplainer           — WHICH feature values drove the prediction
#
# Together they answer the full question: "At t-0.3 seconds, the high brake
# pressure value was what triggered the brake-locked classification."

# %% cell 3
# ── Attention visualization ───────────────────────────────────────────────────

@torch.no_grad()
def get_attention_weights(model: AttentionLSTM,
                           windows: torch.Tensor,
                           device: str) -> tuple:
    """
    Run a forward pass with attention weights returned.

    Args:
        windows: (N, T, F) tensor — already on CPU, will be moved to device
    Returns:
        probs:        (N, C) numpy — sigmoid probabilities
        attn_weights: (N, T) numpy — attention over 60 timesteps
    """
    model.eval()
    windows = windows.to(device)
    logits, attn = model(windows, return_attention=True)
    probs = torch.sigmoid(logits).cpu().numpy()
    attn  = attn.cpu().numpy()
    return probs, attn


def plot_attention_heatmap(attn_weights: np.ndarray,
                            probs: np.ndarray,
                            labels: np.ndarray,
                            class_idx: int,
                            n_samples: int   = 30,
                            output_path: str = "attention_heatmap.png") -> None:
    """
    Heatmap showing attention distribution across 60 timesteps.
    Each row = one test window. Colour = attention weight.
    X-axis goes from t-59 (1 second ago) to t-0 (now).

    High attention near t-0 means the model uses very recent inputs.
    High attention at t-30 suggests it finds context from 0.5s ago important.

    Only shows windows where the model predicted this class positively.
    """
    cls_name = CLASS_NAMES[class_idx]
    predicted_pos = probs[:, class_idx] > 0.5
    idx = np.where(predicted_pos)[0]

    if len(idx) == 0:
        print(f"[Diag] No positive predictions for {cls_name} — skipping heatmap.")
        return

    # Pick up to n_samples, sorted by confidence
    order   = np.argsort(-probs[idx, class_idx])
    idx     = idx[order[:n_samples]]
    weights = attn_weights[idx]        # (n_samples, T)
    true_lbl = labels[idx, class_idx] if labels is not None else None

    fig, ax = plt.subplots(figsize=(14, max(4, len(idx) * 0.28)))
    im = ax.imshow(weights, aspect="auto", cmap="YlOrRd",
                   extent=[-WINDOW_SIZE, 0, len(idx), 0])

    plt.colorbar(im, ax=ax, fraction=0.02, label="Attention weight")

    # Y-tick labels: show confidence + correct/wrong
    if true_lbl is not None:
        ylabels = [f"p={probs[i, class_idx]:.2f} "
                   f"{'✓' if true_lbl[j] > 0.5 else '✗'}"
                   for j, i in enumerate(idx)]
    else:
        ylabels = [f"p={probs[i, class_idx]:.2f}" for i in idx]

    ax.set_yticks(np.arange(len(idx)) + 0.5)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("Timestep (t-59 = 1 second ago  →  t-0 = now)")
    ax.set_ylabel("Test window (sorted by confidence)")
    ax.set_title(f"Temporal Attention — {cls_name}\n"
                 f"(where in the 1-second window did the model look?)")

    # Mark the halfway point
    ax.axvline(-WINDOW_SIZE // 2, color="white", lw=0.8, alpha=0.6,
               linestyle="--", label="0.5s ago")

    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[Diag] Attention heatmap saved → {output_path}")


def plot_attention_per_class(model: AttentionLSTM,
                              test_loader,
                              device: str,
                              labels_all: np.ndarray,
                              output_dir: str = "./diagnostics") -> None:
    """
    Generate one attention heatmap per behaviour class across the test set.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all windows from the loader
    all_windows = []
    for windows, _ in test_loader:
        all_windows.append(windows)
    all_windows = torch.cat(all_windows, dim=0)

    probs, attn = get_attention_weights(model, all_windows, device)

    for cls_idx in range(len(CLASS_NAMES)):
        out = output_dir / f"attention_{CLASS_NAMES[cls_idx].replace(' ', '_')}.png"
        plot_attention_heatmap(
            attn, probs, labels_all, cls_idx,
            n_samples=40, output_path=str(out)
        )

    # Also plot the mean attention profile across all windows
    _plot_mean_attention_profile(attn, probs, output_dir)


def _plot_mean_attention_profile(attn: np.ndarray,
                                  probs: np.ndarray,
                                  output_dir: Path) -> None:
    """
    Single line plot showing the average attention weight at each timestep
    across all test windows. Shows whether the model is generally
    recency-biased (attends to t-0) or context-sensitive (spreads attention).
    """
    mean_attn = attn.mean(axis=0)   # (T,)
    timesteps = np.arange(-WINDOW_SIZE + 1, 1)  # -59 to 0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timesteps, mean_attn, lw=2, color="#1f77b4")
    ax.fill_between(timesteps, mean_attn, alpha=0.2, color="#1f77b4")
    ax.axvline(0, color="red", lw=1, alpha=0.5, label="Now (t-0)")
    ax.axvline(-WINDOW_SIZE // 2, color="orange", lw=1, alpha=0.5,
               label="0.5s ago")
    ax.set_xlabel("Timestep (t-59 = 1s ago  →  t-0 = now)")
    ax.set_ylabel("Mean attention weight")
    ax.set_title("Mean Temporal Attention Profile (all test windows)\n"
                 "High value = model uses that timestep more often")
    ax.legend()
    plt.tight_layout()
    out = output_dir / "mean_attention_profile.png"
    plt.savefig(str(out), dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[Diag] Mean attention profile saved → {out}")


# ── SHAP attribution ──────────────────────────────────────────────────────────

class LSTMWrapper(torch.nn.Module):
    """
    Thin wrapper around AttentionLSTM that returns raw logits for one
    specific class — required by shap.DeepExplainer which expects a
    single-output model when explaining per class.
    """
    def __init__(self, model: AttentionLSTM, class_idx: int):
        super().__init__()
        self.model     = model
        self.class_idx = class_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)          # (B, C)
        return logits[:, self.class_idx:self.class_idx + 1]   # (B, 1)


def compute_shap_values(model: AttentionLSTM,
                         background_windows: torch.Tensor,
                         test_windows: torch.Tensor,
                         class_idx: int,
                         device: str,
                         n_background: int = 100) -> np.ndarray:
    """
    Compute SHAP GradientExplainer values for one behaviour class.
 
    SHAP values shape: (N_test, T, F) — contribution of each feature
    at each timestep to the prediction for class_idx.
 
    ── WHY THE ORIGINAL CRASHED ─────────────────────────────────────────────
    cuDNN (NVIDIA's GPU-accelerated RNN backend) has a hard restriction:
    it can only compute a backward pass when the model is in train() mode.
    In eval() mode cuDNN discards the internal state required for gradients.
    SHAP's GradientExplainer MUST compute gradients (it calls .backward()
    internally), so running it with model.eval() on a CUDA LSTM always
    produces:
        RuntimeError: cudnn RNN backward can only be called in training mode
 
    The attention heatmaps did NOT crash because get_attention_weights() runs
    under @torch.no_grad() — no backward pass, no cuDNN conflict.
 
    ── THE FIX — two targeted changes ────────────────────────────────────────
    1. model.train()  +  disable each nn.Dropout individually
       • train() satisfies cuDNN's requirement for backward support
       • Dropout must be explicitly kept in eval() mode so that stochastic
         zeroing does not pollute SHAP attribution scores with randomness.
         If dropout fires during SHAP, each attribution call sees a slightly
         different network, making the values meaningless.
 
    2. torch.backends.cudnn.flags(enabled=False) context manager
       • Falls back to PyTorch's non-cuDNN LSTM implementation ONLY for
         this call — slower but fully supports backward in any mode.
       • cuDNN is automatically re-enabled when the context exits.
       • This is the minimal-impact approach: nothing else in the pipeline
         is affected.
    ─────────────────────────────────────────────────────────────────────────
    """
    try:
        import shap
    except ImportError:
        print("[SHAP] shap not installed. Run: pip install shap")
        return None
 
    # ── Fix 1: train mode for cuDNN backward, dropout disabled ───────────────
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()   # freeze dropout — keeps attributions deterministic
 
    wrapped   = LSTMWrapper(model, class_idx).to(device)
    bg        = background_windows[:n_background].to(device)
    explainer = shap.GradientExplainer(wrapped, bg)
    test      = test_windows.to(device)
 
    # ── Fix 2: disable cuDNN for this backward pass only ─────────────────────
    with torch.backends.cudnn.flags(enabled=False):
        shap_values = explainer.shap_values(test)
 
    # ── Restore eval mode immediately after SHAP is done ─────────────────────
    model.eval()
 
    # Convert to numpy
    if torch.is_tensor(shap_values):
        shap_values = shap_values.cpu().numpy()
    else:
        shap_values = np.array(shap_values)
 
    # Shape standardisation — GradientExplainer may return (N, T, F) or (1, N, T, F)
    if shap_values.ndim == 4:
        if shap_values.shape[0] == 1:
            shap_values = shap_values[0]       # strip leading class dim → (N, T, F)
        elif shap_values.shape[-1] == 1:
            shap_values = shap_values[..., 0]  # strip trailing class dim → (N, T, F)
 
    return shap_values


def plot_shap_feature_importance(shap_values: np.ndarray,
                                  feature_names: list,
                                  class_name: str,
                                  output_path: str = "shap_feature_importance.png") -> None:
    """
    Beeswarm-style feature importance: mean |SHAP| across time and samples,
    ranked by importance. Shows which features DRIVE this behaviour class.
    """
    # Mean over timesteps and samples → (F,)
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 1))
    order = np.argsort(mean_abs_shap)[::-1]

    top_n   = min(15, len(feature_names))
    top_idx = order[:top_n]
    top_names  = [feature_names[i] for i in top_idx]
    top_values = mean_abs_shap[top_idx]

    colors = ["#d62728" if v > top_values[0] * 0.5 else
              "#ff7f0e" if v > top_values[0] * 0.2 else "#1f77b4"
              for v in top_values]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(range(top_n), top_values[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|  (higher = more important)")
    ax.set_title(f"Feature Importance via SHAP — {class_name}\n"
                 f"Top {top_n} features driving this classification")

    for bar, val in zip(bars, top_values[::-1]):
        ax.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Feature importance saved → {output_path}")


def plot_shap_over_time(shap_values: np.ndarray,
                         feature_names: list,
                         class_name: str,
                         top_n_features: int = 5,
                         output_path: str    = "shap_over_time.png") -> None:
    """
    Line plot showing how the top features' SHAP contributions evolve
    across the 60-timestep window.

    This answers: "At which MOMENT in the last second did [feature] matter?"
    E.g. brake pressure contribution may peak at t-10 (braking 170ms ago),
    confirming the model correctly identifies the trail-brake sequence.
    """
    # Mean absolute SHAP per feature per timestep → (T, F)
    mean_time_shap = np.abs(shap_values).mean(axis=0)

    # Top features by overall importance
    feat_importance = mean_time_shap.mean(axis=0)
    top_feat_idx    = np.argsort(feat_importance)[::-1][:top_n_features]

    timesteps = np.arange(-WINDOW_SIZE + 1, 1)  # -59 to 0

    fig, ax = plt.subplots(figsize=(12, 5))
    for idx in top_feat_idx:
        ax.plot(timesteps, mean_time_shap[:, idx],
                label=feature_names[idx], lw=2)

    ax.axvline(0, color="black", lw=0.8, alpha=0.4, linestyle="--", label="Now")
    ax.set_xlabel("Timestep (t-59 = 1s ago  →  t-0 = now)")
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title(f"SHAP Contribution Over Time — {class_name}\n"
                 f"(which features mattered most at each moment?)")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] SHAP over time saved → {output_path}")


def run_shap_analysis(model: AttentionLSTM,
                       train_loader,
                       test_loader,
                       device: str,
                       feature_names: list,
                       output_dir: str = "./diagnostics/shap",
                       n_background:  int = 100,
                       n_test:        int = 200) -> None:
    """
    Run SHAP analysis for all 7 behaviour classes and save all plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Background: first n_background windows from training set
    bg_windows = []
    for windows, _ in train_loader:
        bg_windows.append(windows)
        if sum(w.shape[0] for w in bg_windows) >= n_background:
            break
    bg_windows = torch.cat(bg_windows, dim=0)[:n_background]

    # Test: first n_test windows from test set
    test_windows = []
    for windows, _ in test_loader:
        test_windows.append(windows)
        if sum(w.shape[0] for w in test_windows) >= n_test:
            break
    test_windows = torch.cat(test_windows, dim=0)[:n_test]

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        print(f"[SHAP] Computing values for: {cls_name}...")
        shap_vals = compute_shap_values(
            model, bg_windows, test_windows, cls_idx, device, n_background)

        if shap_vals is None:
            continue

        safe_name = cls_name.replace(" ", "_")
        plot_shap_feature_importance(
            shap_vals, feature_names, cls_name,
            str(output_dir / f"shap_importance_{safe_name}.png"))

        plot_shap_over_time(
            shap_vals, feature_names, cls_name,
            output_path=str(output_dir / f"shap_time_{safe_name}.png"))

    print(f"[SHAP] All SHAP plots saved → {output_dir}/")


# ── Combined diagnostic runner ────────────────────────────────────────────────

def run_full_diagnostics(model: AttentionLSTM,
                          train_loader,
                          test_loader,
                          device: str,
                          labels_all: np.ndarray,
                          feature_names: list,
                          output_dir: str = "./diagnostics") -> None:
    """
    Run all diagnostic tools — attention heatmaps + SHAP — in one call.
    """
    print(f"\n[Diag] Running full diagnostics → {output_dir}/")
    plot_attention_per_class(model, test_loader, device, labels_all,
                             output_dir=f"{output_dir}/attention")
    run_shap_analysis(model, train_loader, test_loader, device,
                      feature_names, output_dir=f"{output_dir}/shap")
    print(f"[Diag] All diagnostics complete.")


# %% [markdown] cell 4
# # LSTM Module - Evaluator
# lstm_evaluator.py — Comprehensive evaluation of the trained LSTM.
#
# Produces all metrics and plots needed for Chapter 4/5 of the proposal:
#   - Per-class F1, precision, recall (macro-F1 ≥ 0.75 target per §3.7)
#   - Confusion matrices per class
#   - ROC-AUC curves per class
#   - Optimal threshold per class (maximises F1)
#   - Training curve plots (loss + F1 over epochs)
#   - Error analysis (worst predictions)

# %% cell 5
# ── Inference pass ────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model: AttentionLSTM,
                         loader,
                         device: str) -> tuple:
    """
    Run inference on a DataLoader and collect all logits, probabilities,
    predictions, and ground-truth labels.

    Returns:
        logits (N, C), probs (N, C), labels (N, C) — all numpy arrays
    """
    model.eval()
    all_logits = []
    all_labels = []

    for windows, labels in loader:
        windows = windows.to(device, non_blocking=True)
        logits  = model(windows)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())

    logits_arr = np.concatenate(all_logits, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    probs_arr  = 1 / (1 + np.exp(-logits_arr))

    return logits_arr, probs_arr, labels_arr


# ── Threshold tuning ──────────────────────────────────────────────────────────

def tune_thresholds(probs: np.ndarray,
                    labels: np.ndarray,
                    n_thresholds: int = 100) -> tuple:
    """
    Find the decision threshold that maximises F1-score for each class
    independently. Saves a per-class threshold JSON.

    Returns:
        optimal_thresholds: list of floats, one per class
        threshold_f1s:      list of best F1s at those thresholds
    """
    optimal_thresholds = []
    threshold_f1s      = []

    for cls_idx in range(probs.shape[1]):
        best_f1  = 0.0
        best_thr = 0.5
        for thr in np.linspace(0.1, 0.9, n_thresholds):
            preds = (probs[:, cls_idx] > thr).astype(int)
            f1    = f1_score(labels[:, cls_idx], preds, zero_division=0)
            if f1 > best_f1:
                best_f1  = f1
                best_thr = thr
        optimal_thresholds.append(round(best_thr, 3))
        threshold_f1s.append(round(best_f1, 4))
        print(f"  {CLASS_NAMES[cls_idx]:<30}  optimal_threshold={best_thr:.2f}  "
              f"best_F1={best_f1:.4f}")

    return optimal_thresholds, threshold_f1s


# ── Core metrics table ────────────────────────────────────────────────────────

def compute_full_metrics(probs: np.ndarray,
                          labels: np.ndarray,
                          thresholds: list = None,
                          output_path: str = "evaluation_metrics.json") -> dict:
    """
    Full per-class and aggregate metrics table.
    Uses optimal thresholds if provided, else 0.5 for all classes.
    """
    if thresholds is None:
        thresholds = [0.5] * probs.shape[1]

    preds = np.stack(
        [(probs[:, i] > thr).astype(int) for i, thr in enumerate(thresholds)],
        axis=1
    )

    results = {}

    print(f"\n{'='*70}")
    print(f"  EVALUATION METRICS")
    print(f"  {'Class':<30} {'F1':>6} {'Prec':>6} {'Rec':>6} {'AUC':>6}")
    print(f"  {'-'*55}")

    f1_scores = []
    for i, name in enumerate(CLASS_NAMES):
        f1   = f1_score(labels[:, i], preds[:, i], zero_division=0)
        prec = precision_score(labels[:, i], preds[:, i], zero_division=0)
        rec  = recall_score(labels[:, i], preds[:, i], zero_division=0)
        try:
            auc = roc_auc_score(labels[:, i], probs[:, i])
        except ValueError:
            auc = float("nan")

        f1_scores.append(f1)
        results[name] = {"f1": round(f1, 4), "precision": round(prec, 4),
                          "recall": round(rec, 4), "roc_auc": round(auc, 4),
                          "threshold": thresholds[i]}

        status = "✓" if f1 >= 0.75 else "✗"
        print(f"  {status} {name:<30} {f1:>5.3f} {prec:>6.3f} {rec:>6.3f} {auc:>6.3f}")

    macro_f1 = float(np.mean(f1_scores))
    results["macro_f1"] = round(macro_f1, 4)
    target_met = macro_f1 >= 0.75

    print(f"  {'-'*55}")
    print(f"  {'MACRO F1':<30} {macro_f1:>5.3f}")
    print(f"  Target (≥ 0.75): {'✓ MET' if target_met else '✗ NOT MET'}")
    print(f"{'='*70}\n")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Eval] Metrics saved → {output_path}")

    return results


# ── Confusion matrices ────────────────────────────────────────────────────────

def plot_confusion_matrices(preds: np.ndarray,
                              labels: np.ndarray,
                              output_path: str = "confusion_matrices.png") -> None:
    """
    Per-class binary confusion matrix grid.
    For multi-label binary classification, each class gets its own 2×2 matrix.
    """
    n_classes = labels.shape[1]
    n_cols    = 4
    n_rows    = int(np.ceil(n_classes / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i in range(n_classes):
        # need to enforce a 2x2 shape even when a class is entirely missing
        cm    = confusion_matrix(labels[:, i], preds[:, i], labels=[0, 1])
        ax    = axes[i]
        im    = ax.imshow(cm, cmap="Blues", aspect="auto")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        ax.set_title(CLASS_NAMES[i], fontsize=9)

        for row in range(2):
            for col in range(2):
                ax.text(col, row, f"{cm[row, col]:,}",
                        ha="center", va="center",
                        color="white" if cm[row, col] > cm.max() / 2 else "black",
                        fontsize=10, fontweight="bold")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Per-Class Confusion Matrices", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Confusion matrices saved → {output_path}")


# ── ROC curves ────────────────────────────────────────────────────────────────

def plot_roc_curves(probs: np.ndarray,
                    labels: np.ndarray,
                    output_path: str = "roc_curves.png") -> None:
    """
    Multi-class ROC curve grid — one panel per behaviour class.
    """
    n_classes = labels.shape[1]
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(n_classes):
        if labels[:, i].sum() == 0:
            axes[i].text(0.5, 0.5, "No positives", ha="center", va="center")
            continue
        fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
        auc = roc_auc_score(labels[:, i], probs[:, i])
        axes[i].plot(fpr, tpr, color=colors[i], lw=2, label=f"AUC={auc:.3f}")
        axes[i].plot([0, 1], [0, 1], "k--", alpha=0.3)
        axes[i].set_title(CLASS_NAMES[i], fontsize=9)
        axes[i].set_xlabel("FPR", fontsize=8)
        axes[i].set_ylabel("TPR", fontsize=8)
        axes[i].legend(fontsize=8)

    for j in range(n_classes, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("ROC Curves — LSTM Behaviour Classifiers", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[Eval] ROC curves saved → {output_path}")


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training_curves(history: dict,
                          output_path: str = "training_curves.png") -> None:
    """
    Loss and macro-F1 training curves over epochs.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train", lw=2)
    ax1.plot(epochs, history["val_loss"],   label="Val",   lw=2, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_f1"], label="Train Macro-F1", lw=2)
    ax2.plot(epochs, history["val_f1"],   label="Val Macro-F1",   lw=2, linestyle="--")
    ax2.axhline(0.75, color="red", linestyle=":", alpha=0.6, label="Target (0.75)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1-Score")
    ax2.set_title("Training & Validation Macro-F1")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Training curves saved → {output_path}")


# ── Error analysis ────────────────────────────────────────────────────────────

def error_analysis(probs: np.ndarray,
                    labels: np.ndarray,
                    windows: np.ndarray,
                    feature_names: list,
                    output_dir: str = ".",
                    n_worst: int    = 5) -> None:
    """
    Find the n_worst windows per class where the model was most confidently
    wrong (high probability on false positives or low probability on
    false negatives), and plot the raw telemetry for inspection.

    This helps you identify WHAT features the model was misled by.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        true_labels = labels[:, cls_idx]
        cls_probs   = probs[:, cls_idx]

        # False positives: predicted high probability but true label = 0
        fp_mask = (cls_probs > 0.5) & (true_labels < 0.5)
        # False negatives: predicted low probability but true label = 1
        fn_mask = (cls_probs < 0.5) & (true_labels > 0.5)

        for error_type, mask in [("FP", fp_mask), ("FN", fn_mask)]:
            if mask.sum() == 0:
                continue

            # Sort by confidence of being wrong
            indices = np.where(mask)[0]
            if error_type == "FP":
                # Most confident false alarms: highest prob
                order = np.argsort(-cls_probs[indices])
            else:
                # Missed events with highest true-class probability
                order = np.argsort(cls_probs[indices])

            worst_idxs = indices[order[:n_worst]]

            fig, axes = plt.subplots(n_worst, 1, figsize=(14, 3 * n_worst), sharex=True)
            if n_worst == 1:
                axes = [axes]

            for i, idx in enumerate(worst_idxs):
                window = windows[idx]  # (T, F)
                ax = axes[i]
                for f_idx, fname in enumerate(feature_names[:8]):  # plot first 8 features
                    ax.plot(window[:, f_idx], alpha=0.7,
                            label=fname if i == 0 else "_nolegend_")
                conf = cls_probs[idx]
                ax.set_title(f"Window {idx} | conf={conf:.3f} | true={int(true_labels[idx])}")
                ax.set_ylabel("Norm. value")

            axes[0].legend(fontsize=7, ncol=4, loc="upper right")
            axes[-1].set_xlabel("Timestep (0 = 1s ago, 59 = now)")
            fig.suptitle(f"{cls_name} — {error_type} errors (worst {n_worst})", fontsize=11)

            save_path = output_dir / f"error_{cls_name.replace(' ', '_')}_{error_type}.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
            plt.close()

    print(f"[Eval] Error analysis plots saved → {output_dir}/")


# ── Master evaluation runner ──────────────────────────────────────────────────

def run_full_evaluation(model: AttentionLSTM,
                         test_loader,
                         device: str,
                         history: dict          = None,
                         output_dir: str        = "./eval_output",
                         feature_names: list    = None) -> dict:
    """
    Run the complete evaluation suite on the held-out test set.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Eval] Running evaluation on test set...")
    logits, probs, labels = collect_predictions(model, test_loader, device)
    preds_default = (probs > 0.5).astype(int)

    # Threshold tuning
    print("[Eval] Tuning decision thresholds...")
    optimal_thresholds, _ = tune_thresholds(probs, labels)

    # Save thresholds
    thr_dict = dict(zip(LABEL_COLS, optimal_thresholds))
    with open(output_dir / "optimal_thresholds.json", "w") as f:
        json.dump(thr_dict, f, indent=2)

    preds_tuned = np.stack(
        [(probs[:, i] > t).astype(int) for i, t in enumerate(optimal_thresholds)],
        axis=1
    )

    # Metrics
    metrics = compute_full_metrics(
        probs, labels,
        thresholds=optimal_thresholds,
        output_path=str(output_dir / "evaluation_metrics.json")
    )

    # Plots
    plot_confusion_matrices(preds_tuned, labels,
                             str(output_dir / "confusion_matrices.png"))
    plot_roc_curves(probs, labels,
                    str(output_dir / "roc_curves.png"))

    if history:
        plot_training_curves(history,
                             str(output_dir / "training_curves.png"))

    # Error analysis (needs raw windows — extract from loader)
    if feature_names:
        all_windows = np.concatenate(
            [w.numpy() for w, _ in test_loader], axis=0)
        error_analysis(probs, labels, all_windows, feature_names,
                       output_dir=str(output_dir / "error_analysis"))

    return metrics

# %% [markdown] cell 6
# # LSTM Module - Inference
# lstm_inference.py — Load a trained LSTM checkpoint and run it on
# never-seen session data from scratch: raw CSV in, coaching labels out.

# %% cell 7
class LSTMInferenceEngine:
    """
    Self-contained inference engine.
    Load once, call predict() as many times as you like.

    Bundles:
      - trained model weights
      - normalizer params (fit on training data)
      - per-class optimal decision thresholds
    """

    def __init__(self,
                 checkpoint_path: str,
                 normalizer_path: str,
                 threshold_path:  str = None,
                 device:          str = None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load normalizer params
        with open(normalizer_path) as f:
            self.norm_params = json.load(f)

        # Load thresholds (fall back to 0.5 per class if file absent)
        if threshold_path and Path(threshold_path).exists():
            with open(threshold_path) as f:
                thr_dict = json.load(f)
            self.thresholds = [thr_dict.get(col, 0.5) for col in LABEL_COLS]
        else:
            self.thresholds = [0.5] * len(LABEL_COLS)

        # Load model
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg  = ckpt.get("model_config", {})
        self.model = build_model(
            input_dim   = cfg.get("input_dim",   len(FEATURE_COLS)),
            hidden_dim  = cfg.get("hidden_dim",  128),
            num_layers  = cfg.get("num_layers",  2),
            num_classes = cfg.get("num_classes", len(LABEL_COLS)),
            device      = self.device,
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"[Inference] Model loaded from {checkpoint_path}")
        print(f"[Inference] Device: {self.device} | "
              f"Thresholds: {[round(t,2) for t in self.thresholds]}")

    def predict_from_csv(self,
                          raw_csv_path: str,
                          vehicle:      str = "default",
                          stride:       int = 1) -> pd.DataFrame:
        """
        Full pipeline on a raw LFS telemetry CSV:
            adapt → normalise → window → infer → return per-window results

        Returns:
            DataFrame with columns: window_idx, timestamp_end_ms,
            + one prob column and one binary pred column per class,
            + attention_peak_timestep (which of the 60 steps got most attention)
        """
        # Adapt
        df = adapt_lfs_telemetry(raw_csv_path, vehicle=vehicle)

        # Normalise
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
        df = apply_normalizer(df, self.norm_params, feature_cols)
        df[feature_cols] = df[feature_cols].fillna(0.0)

        # Build dataset (labels absent — use zeros as placeholder)
        for col in LABEL_COLS:
            if col not in df.columns:
                df[col] = 0

        ds     = TelemetryWindowDataset(df, stride=stride)
        loader = make_dataloader(ds, batch_size=512, shuffle=False)

        all_probs  = []
        all_attn   = []

        with torch.no_grad():
            for windows, _ in loader:
                windows = windows.to(self.device)
                logits, attn = self.model(windows, return_attention=True)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_attn.append(attn.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)   # (N, C)
        all_attn  = np.concatenate(all_attn,  axis=0)   # (N, T)

        # Build results DataFrame
        results = {}
        results["window_idx"] = np.arange(len(all_probs))

        # Timestamp of the last row in each window
        if "timestamp_ms" in df.columns:
            ts_values = df["timestamp_ms"].values
            results["timestamp_end_ms"] = [
                ts_values[min(i * stride + WINDOW_SIZE - 1, len(ts_values) - 1)]
                for i in range(len(all_probs))
            ]

        for cls_idx, col in enumerate(LABEL_COLS):
            cls_name = CLASS_NAMES[cls_idx]
            results[f"prob_{cls_name.replace(' ', '_')}"] = all_probs[:, cls_idx]
            results[f"pred_{cls_name.replace(' ', '_')}"] = (
                all_probs[:, cls_idx] > self.thresholds[cls_idx]).astype(int)

        # Peak attention timestep (relative: 0 = now, -59 = 1s ago)
        peak_attn_idx = all_attn.argmax(axis=1) - (WINDOW_SIZE - 1)
        results["attention_peak_offset"] = peak_attn_idx

        return pd.DataFrame(results)

    def session_summary(self, predictions_df: pd.DataFrame) -> dict:
        """
        Aggregate per-window predictions into a session-level summary:
        total event count per class + Driving Persona label.
        """
        summary = {}
        for col in LABEL_COLS:
            cls_name = CLASS_NAMES[LABEL_COLS.index(col)]
            pred_col = f"pred_{cls_name.replace(' ', '_')}"
            if pred_col in predictions_df.columns:
                summary[cls_name] = int(predictions_df[pred_col].sum())

        # Driving Persona classification (proposal §3.3)
        brake_events    = summary.get("Brake Locked", 0)
        spin_events     = summary.get("Wheel Spin",   0)
        rough_events    = summary.get("Rough Steering", 0)
        smooth_events   = summary.get("Gentle Acceleration", 0)
        upshift_events  = summary.get("Late Upshift",  0)

        aggressive_score = brake_events + spin_events + rough_events
        smooth_score     = smooth_events
        cautious_score   = upshift_events

        if aggressive_score > smooth_score and aggressive_score > cautious_score:
            persona = "Aggressive"
        elif smooth_score > cautious_score:
            persona = "Smooth"
        else:
            persona = "Cautious"

        summary["driving_persona"] = persona

        print(f"\n[Inference] Session Summary:")
        for k, v in summary.items():
            print(f"  {k:<30}: {v}")

        return summary

# %% [markdown] cell 8
# # LSTM Module - Model
# lstm_model.py — Stacked 2-layer LSTM with temporal attention.
#
# Architecture per proposal §3.5.4:
#   - 2-layer stacked LSTM, hidden dim 128
#   - 60-timestep input window
#   - 7 binary outputs (multi-label, BCEWithLogitsLoss)
#   - Attention layer for explainability (shows WHICH timestep mattered most)
#   - NOT bidirectional — causal temporal order must be preserved for
#     meaningful feedback ("what you did 0.5s ago caused the lock-up now")

# %% cell 9
class TemporalAttention(nn.Module):
    """
    Single-head soft attention over the LSTM's hidden state sequence.
    Produces a scalar attention weight per timestep — tells you which
    moment in the 1-second window drove the prediction.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn_linear = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> tuple:
        """
        Args:
            lstm_out: (batch, seq_len, hidden_dim)
        Returns:
            context:      (batch, hidden_dim) — weighted summary
            attn_weights: (batch, seq_len)    — attention distribution
        """
        scores  = self.attn_linear(lstm_out)             # (B, T, 1)
        weights = torch.softmax(scores, dim=1)            # (B, T, 1)
        context = (lstm_out * weights).sum(dim=1)         # (B, H)
        return context, weights.squeeze(-1)               # (B, H), (B, T)


class AttentionLSTM(nn.Module):
    """
    Complete LSTM module for the bimodal driver coaching framework.

    Forward pass returns:
        logits (always):          (batch, num_classes) — pre-sigmoid scores
        attn_weights (optional):  (batch, seq_len)     — for diagnostics

    Use torch.sigmoid(logits) to get probabilities [0, 1].
    Use BCEWithLogitsLoss(logits, targets) during training — do NOT apply
    sigmoid before the loss function.
    """

    def __init__(self,
                 input_dim:   int = INPUT_DIM,
                 hidden_dim:  int = HIDDEN_DIM,
                 num_layers:  int = NUM_LAYERS,
                 num_classes: int = NUM_CLASSES,
                 dropout:     float = DROPOUT,
                 window_size: int = WINDOW_SIZE):
        super().__init__()

        self.input_dim   = input_dim
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.num_classes = num_classes
        self.window_size = window_size

        # Optional input batch norm to stabilise training across LFS/AC domains
        self.input_norm = nn.BatchNorm1d(window_size)

        # Stacked LSTM — dropout applied between layers (not after last layer)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Temporal attention
        self.attention = TemporalAttention(hidden_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=False), #predecesor of this line brings about an error in the pipeline step 7 reason bellow
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        """
        To calculate feature attributuon scores, SHAP's DeepEcplainer
        attaches custom registration hooks to the neural network layers
        to monitor gradients during a modified backward pass.
        
        Turning it from True to False, it lets PyTorch allocate clean and
        separate tensor for the layer's output.
        """

        self._init_weights()

    def _init_weights(self):
        """Orthogonal init for LSTM weights, Xavier for linear layers."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Forget gate bias = 1 — helps with long-range dependencies
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self,
                x: torch.Tensor,
                return_attention: bool = False) -> tuple:
        """
        Args:
            x:                (batch, seq_len, input_dim)
            return_attention: if True, also return attention weights

        Returns:
            logits:       (batch, num_classes)
            attn_weights: (batch, seq_len) — only if return_attention=True
        """
        # Input normalisation across the time dimension
        x = self.input_norm(x)                             # (B, T, F)

        lstm_out, _ = self.lstm(x)                         # (B, T, H)

        context, attn_weights = self.attention(lstm_out)   # (B, H), (B, T)

        logits = self.classifier(context)                  # (B, C)

        if return_attention:
            return logits, attn_weights
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper returning sigmoid probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
        return torch.sigmoid(logits)

    def count_parameters(self) -> int:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Trainable parameters: {total:,}")
        return total


def build_model(input_dim:   int   = INPUT_DIM,
                hidden_dim:  int   = HIDDEN_DIM,
                num_layers:  int   = NUM_LAYERS,
                num_classes: int   = NUM_CLASSES,
                dropout:     float = DROPOUT,
                device: str        = None) -> AttentionLSTM:
    """
    Build and print the AttentionLSTM model summary.
    Automatically selects CUDA if available and device is not specified.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AttentionLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    model.count_parameters()
    print(f"[Model] Device: {device}")
    print(f"[Model] Architecture: LSTM({input_dim}→{hidden_dim}×{num_layers}) "
          f"+ Attention → MLP → {num_classes} outputs")
    return model

# %% [markdown] cell 10
# # LSTM Data - Argumentation
# augmentation.py — Time-series augmentation techniques for the LSTM training set.
#
# Applied DURING training (not to val/test) to address class imbalance
# and improve generalisation across LFS vs AC domain differences.

# %% cell 11
class AugmentedTelemetryDataset(Dataset):
    """
    Wraps a TelemetryWindowDataset and applies on-the-fly augmentations
    to each window during training. Augmentation is stochastic — each
    epoch sees a slightly different version of the same window.

    Augmentation techniques used:
        1. Gaussian noise injection  — robustness to sensor jitter
        2. Time shift (jitter)       — robustness to irregular sampling
        3. Feature dropout           — robustness to missing channels
        4. Amplitude scaling         — data diversity without label changes
        5. Window slice              — diversity in temporal context
    """

    def __init__(self,
                 base_dataset,
                 noise_std: float          = 0.01,
                 time_shift_max: int       = 5,
                 feature_dropout_p: float  = 0.05,
                 amplitude_scale_range: tuple = (0.92, 1.08),
                 window_slice_p: float     = 0.3,
                 augment_p: float          = 0.7):
        """
        Args:
            base_dataset:         TelemetryWindowDataset instance
            noise_std:            Std of Gaussian noise added to features
            time_shift_max:       Max rows to shift the window start (±)
            feature_dropout_p:    Probability of zeroing each feature at each timestep
            amplitude_scale_range: (min, max) multiplier applied to all features
            window_slice_p:       Probability of randomly slicing a sub-window
            augment_p:            Probability of applying ANY augmentation to a sample
        """
        self.dataset             = base_dataset
        self.noise_std           = noise_std
        self.time_shift_max      = time_shift_max
        self.feature_dropout_p   = feature_dropout_p
        self.amplitude_scale_range = amplitude_scale_range
        self.window_slice_p      = window_slice_p
        self.augment_p           = augment_p

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple:
        window, label = self.dataset[idx]      # (W, F), (C,)
        window = window.numpy().copy()         # work in numpy for speed

        # Randomly skip augmentation entirely for some samples
        if np.random.random() > self.augment_p:
            return torch.from_numpy(window), label

        # 1. Gaussian noise
        if np.random.random() < 0.6:
            window += np.random.normal(0, self.noise_std, window.shape).astype(np.float32)

        # 2. Amplitude scaling
        if np.random.random() < 0.5:
            scale = np.random.uniform(*self.amplitude_scale_range)
            window *= scale

        # 3. Feature dropout (zero out random channels at random timesteps)
        if np.random.random() < 0.3:
            mask = np.random.random(window.shape) > self.feature_dropout_p
            window *= mask.astype(np.float32)

        # 4. Window slice — replace with a shorter sub-window, zero-padded
        if np.random.random() < self.window_slice_p:
            W = window.shape[0]
            slice_len = np.random.randint(int(W * 0.7), W)
            start     = np.random.randint(0, W - slice_len)
            sliced    = window[start:start + slice_len]
            padded    = np.zeros_like(window)
            padded[W - slice_len:] = sliced   # right-align: most recent data kept
            window = padded

        # Clip to keep values in valid range after augmentation
        window = np.clip(window, -0.1, 1.1).astype(np.float32)

        return torch.from_numpy(window), label

    # Attempt to solve issue in pipeline step 4
    def get_sample_weights(self) -> torch.Tensor:
        """
        Pass the sampler weights request down to the underlying
        TelemetryWindowDataset where labels actually exist
        """
        return self.dataset.get_sample_weights()


def oversample_minority_windows(windows: np.ndarray,
                                 labels: np.ndarray,
                                 target_ratio: float = 0.15) -> tuple:
    """
    Hard oversampling: duplicate windows that contain at least one positive
    label from a rare class until that class reaches at least target_ratio
    of the total dataset.

    Used as a STATIC pre-processing step before building the Dataset if
    on-the-fly sampling is insufficient.

    Args:
        windows:      (N, W, F) array
        labels:       (N, C) binary array
        target_ratio: minimum fraction of positives per class

    Returns:
        (augmented_windows, augmented_labels)
    """
    n_total = len(labels)
    extra_windows = [windows]
    extra_labels  = [labels]

    for cls_idx in range(labels.shape[1]):
        pos_mask = labels[:, cls_idx] > 0.5
        n_pos    = pos_mask.sum()
        n_target = int(n_total * target_ratio)

        if n_pos >= n_target or n_pos == 0:
            continue

        n_needed = n_target - n_pos
        pos_idxs = np.where(pos_mask)[0]

        # Repeat minority windows with replacement
        repeat_idxs = np.random.choice(pos_idxs, size=n_needed, replace=True)
        extra_windows.append(windows[repeat_idxs])
        extra_labels.append(labels[repeat_idxs])
        print(f"[Aug] Class {cls_idx}: added {n_needed:,} duplicate windows "
              f"({n_pos:,} → {n_pos + n_needed:,})")

    aug_windows = np.concatenate(extra_windows, axis=0)
    aug_labels  = np.concatenate(extra_labels,  axis=0)

    # Shuffle
    perm = np.random.permutation(len(aug_windows))
    return aug_windows[perm], aug_labels[perm]


def mixup_windows(window_a: torch.Tensor,
                   label_a:  torch.Tensor,
                   window_b: torch.Tensor,
                   label_b:  torch.Tensor,
                   alpha: float = 0.2) -> tuple:
    """
    Mixup augmentation for time-series: linearly interpolates between
    two training samples. Only applied within the same label class to
    avoid creating contradictory training signal.

    lam ~ Beta(alpha, alpha); lam closer to 0 or 1 means less mixing.
    """
    lam = np.random.beta(alpha, alpha)
    mixed_window = lam * window_a + (1 - lam) * window_b
    mixed_label  = lam * label_a  + (1 - lam) * label_b
    return mixed_window, mixed_label

# %% [markdown] cell 12
# # LSTM Data - Distributions
# distributions.py — Descriptive statistics, feature distributions,
# class imbalance diagnostics, and correlation analysis.
#
# All outputs are saved as PNG files for direct inclusion in the report.
# Run this AFTER labelling, BEFORE training.

# %% cell 13
def descriptive_stats_table(df: pd.DataFrame,
                             output_path: str = "descriptive_stats.csv") -> pd.DataFrame:
    """
    Generate a full descriptive statistics table (count, mean, std, min,
    percentiles, max) for all feature columns. Saves as CSV for the report.

    §3.6 of the proposal references statistical analysis methods — this
    table satisfies the "descriptive statistics" requirement.
    """
    feature_cols_present = [c for c in FEATURE_COLS if c in df.columns]
    stats = df[feature_cols_present].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    stats.index.name = "Feature"
    stats = stats.round(4)

    stats.to_csv(output_path)
    print(f"[Stats] Descriptive statistics saved → {output_path}")
    print(stats.to_string())
    return stats


def class_distribution_report(df: pd.DataFrame,
                                output_path: str = "class_distribution.json") -> dict:
    """
    Compute and print the class imbalance statistics for all 7 label columns.
    Returns a dict usable as pos_weight values for BCEWithLogitsLoss.

    Formula: pos_weight[i] = (N_negative_i) / (N_positive_i)
    """
    label_cols_present = [c for c in LABEL_COLS if c in df.columns]
    n_total = len(df)
    report = {}

    print(f"\n{'='*60}")
    print(f"  CLASS DISTRIBUTION REPORT  (N = {n_total:,} rows)")
    print(f"{'='*60}")
    print(f"  {'Label':<30} {'Pos':>7} {'Neg':>7} {'Pos%':>6}  {'pos_weight':>10}")
    print(f"  {'-'*60}")

    pos_weights = {}
    for col, name in zip(label_cols_present, CLASS_NAMES):
        n_pos = int(df[col].sum())
        n_neg = n_total - n_pos
        pct   = 100 * n_pos / n_total
        pw    = n_neg / max(n_pos, 1)
        pos_weights[col] = round(pw, 2)
        print(f"  {col:<30} {n_pos:>7,} {n_neg:>7,} {pct:>5.1f}%  {pw:>10.1f}x")

        report[col] = {
            "n_positive": n_pos,
            "n_negative": n_neg,
            "pct_positive": round(pct, 2),
            "pos_weight": round(pw, 2),
        }

    print(f"{'='*60}\n")

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Stats] Class distribution saved → {output_path}")
    return pos_weights


def plot_label_distribution(df: pd.DataFrame,
                              output_path: str = "label_distribution.png") -> None:
    """
    Horizontal bar chart showing positive label counts per class.
    Colour-coded: red = severe imbalance (<2%), amber = moderate, green = ok.
    """
    label_cols_present = [c for c in LABEL_COLS if c in df.columns]
    counts = [df[c].sum() for c in label_cols_present]
    pcts   = [100 * c / len(df) for c in counts]
    colors = ["#d62728" if p < 2 else "#ff7f0e" if p < 10 else "#2ca02c"
              for p in pcts]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(CLASS_NAMES[:len(label_cols_present)], counts, color=colors, alpha=0.85)
    ax.set_xlabel("Positive label count (rows)")
    ax.set_title("Label Distribution — LSTM Behaviour Classes")

    for bar, pct, count in zip(bars, pcts, counts):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                f"  {count:,}  ({pct:.1f}%)", va="center", fontsize=9)

    ax.axvline(0.02 * len(df), color="red",   linestyle="--", alpha=0.5, label="2% threshold")
    ax.axvline(0.10 * len(df), color="orange", linestyle="--", alpha=0.5, label="10% threshold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Label distribution saved → {output_path}")


def plot_feature_distributions(df: pd.DataFrame,
                                 output_path: str = "feature_distributions.png",
                                 n_cols: int = 5) -> None:
    """
    Grid of histograms for all canonical feature columns — one subplot
    per feature. Used to spot skew, outliers, and constant columns.
    """
    feature_cols_present = [c for c in FEATURE_COLS if c in df.columns]
    n = len(feature_cols_present)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols_present):
        axes[i].hist(df[col].dropna(), bins=50, alpha=0.75, color="#1f77b4", edgecolor="none")
        axes[i].set_title(col, fontsize=8)
        axes[i].tick_params(labelsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions (Canonical Schema)", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Feature distributions saved → {output_path}")


def plot_correlation_matrix(df: pd.DataFrame,
                              output_path: str = "correlation_matrix.png") -> None:
    """
    Feature correlation heatmap — helps identify redundant features
    that could be dropped to reduce LSTM input dimensionality.
    """
    feature_cols_present = [c for c in FEATURE_COLS if c in df.columns]
    corr = df[feature_cols_present].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    ax.set_title("Feature Correlation Matrix", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Correlation matrix saved → {output_path}")


def plot_time_series_sample(df: pd.DataFrame,
                              n_rows: int = 3000,
                              output_path: str = "time_series_sample.png") -> None:
    """
    Multi-panel time-series plot for the first n_rows of the session,
    showing raw telemetry alongside behaviour labels. Useful sanity check
    to visually confirm the labels were applied correctly.
    """
    sample = df.head(n_rows).copy()
    label_cols_present = [c for c in LABEL_COLS if c in df.columns]

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(4, 1, hspace=0.4)

    # Panel 1: speed, brake, throttle
    ax1 = fig.add_subplot(gs[0])
    if "speed_kmh" in sample.columns:
        ax1.plot(sample["speed_kmh"] / sample["speed_kmh"].max(), label="Speed (norm)", alpha=0.7)
    if "brake" in sample.columns:
        ax1.plot(sample["brake"], label="Brake", color="red", alpha=0.7)
    if "throttle" in sample.columns:
        ax1.plot(sample["throttle"], label="Throttle", color="green", alpha=0.7)
    ax1.set_ylabel("Value")
    ax1.set_title(f"Telemetry Sample (first {n_rows} rows)")
    ax1.legend(fontsize=7, ncol=3)

    # Panel 2: RPM & gear
    ax2 = fig.add_subplot(gs[1])
    if "engine_rpm_norm" in sample.columns:
        ax2.plot(sample["engine_rpm_norm"], label="RPM (norm)", color="purple", alpha=0.7)
    if "gear" in sample.columns:
        ax2b = ax2.twinx()
        ax2b.plot(sample["gear"], label="Gear", color="orange", alpha=0.5)
        ax2b.set_ylabel("Gear", color="orange")
    ax2.set_ylabel("RPM (norm)")
    ax2.legend(fontsize=7)

    # Panel 3: Wheel slip
    ax3 = fig.add_subplot(gs[2])
    for col in ["slip_ratio_lf", "slip_ratio_rf", "slip_ratio_lr", "slip_ratio_rr"]:
        if col in sample.columns:
            ax3.plot(sample[col], alpha=0.6, label=col.split("_")[-1].upper())
    ax3.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="Lockup threshold")
    ax3.set_ylabel("Slip Ratio")
    ax3.legend(fontsize=7, ncol=4)

    # Panel 4: Labels as stacked bands
    ax4 = fig.add_subplot(gs[3])
    label_colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    for i, (col, name) in enumerate(zip(label_cols_present, CLASS_NAMES)):
        if col in sample.columns:
            ax4.fill_between(range(len(sample)), i, i + sample[col].values,
                             alpha=0.7, color=label_colors[i % len(label_colors)], label=name)
    ax4.set_yticks(range(len(label_cols_present)))
    ax4.set_yticklabels(CLASS_NAMES[:len(label_cols_present)], fontsize=7)
    ax4.set_xlabel("Sample index")
    ax4.set_title("Applied Behaviour Labels")

    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Time-series sample saved → {output_path}")


def run_full_distribution_analysis(labelled_csv_path: str,
                                    output_dir: str = ".") -> dict:
    """
    Run all distribution analyses on a labelled CSV and save all plots/CSVs
    to output_dir. Returns the pos_weight dict for use in training.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(labelled_csv_path)
    print(f"[Analysis] Loaded {len(df):,} rows from {labelled_csv_path}")

    descriptive_stats_table(df, f"{output_dir}/descriptive_stats.csv")
    pos_weights = class_distribution_report(df, f"{output_dir}/class_distribution.json")
    plot_label_distribution(df, f"{output_dir}/label_distribution.png")
    plot_feature_distributions(df, f"{output_dir}/feature_distributions.png")
    plot_correlation_matrix(df, f"{output_dir}/correlation_matrix.png")
    plot_time_series_sample(df, output_path=f"{output_dir}/time_series_sample.png")

    return pos_weights

# %% [markdown] cell 14
# # LSTM Data - Labelling
# labelling.py — Applies the 7 threshold-based behaviour labels defined in
# proposal §1.6 to a canonical-schema DataFrame.
#
# All functions are vectorized (no Python loops) using pandas groupby + cumsum,
# making them fast enough to run overnight on large session files without issue.

# %% cell 15
# ── Core utility: vectorized consecutive-run check ────────────────────────────

def has_consecutive_true(series: pd.Series, min_count: int) -> pd.Series:
    """
    Returns a boolean Series where True means the input Series has been
    continuously True for AT LEAST min_count consecutive rows at that point.

    Uses the cumsum-groupby trick — O(n), no Python-level loops.

    Example: min_count=3, series=[F,F,T,T,T,F,T,T,F]
             output          =[F,F,F,F,T,F,F,F,F]
    """
    # Create run-group IDs: increments each time the value changes
    groups = (series != series.shift()).cumsum()

    # Within each run-group, count how many rows belong to it
    run_lengths = series.groupby(groups).transform("count")

    # Valid if: currently True AND run is long enough
    return series & (run_lengths >= min_count)


def _get_max_slip(df: pd.DataFrame) -> pd.Series:
    """Maximum slip ratio across all four wheels at each timestep."""
    slip_cols = ["slip_ratio_lf", "slip_ratio_rf", "slip_ratio_lr", "slip_ratio_rr"]
    available = [c for c in slip_cols if c in df.columns]
    if not available:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return df[available].max(axis=1)


def _get_max_wheel_spin(df: pd.DataFrame) -> pd.Series:
    """Maximum wheel spin across all four driven wheels at each timestep."""
    spin_cols = ["wheel_spin_lf", "wheel_spin_rf", "wheel_spin_lr", "wheel_spin_rr"]
    available = [c for c in spin_cols if c in df.columns]
    if not available:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return df[available].max(axis=1)


# FIX (see distribution check on raw CSV): wheel_spin_lf/rf/lr/rr are NOT
# angular velocity in rad/s and NOT a 0-1 ratio. describe() confirms they
# share the same mean/std/range as speed_kmh (mean≈102 vs speed mean≈104,
# same -40..180 span) — this is wheel SURFACE speed in km/h. Any threshold
# written as if it were a small rad/s or 0-1 value (wheel_spin_zero_thresh
# = 0.5, WHEEL_SPIN_SLIP_THRESH compared directly to the raw column) is a
# silent no-op or always-true condition against this column. Compare it to
# speed_kmh instead, the same way slip_ratio_* already does.

def _get_max_wheel_slip_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Per-wheel slip expressed as (wheel_surface_speed - car_speed) / car_speed,
    i.e. how much faster the wheel's contact patch is moving than the car
    itself — the same physical quantity slip_ratio_* already encodes, but
    derived independently from wheel_spin_* as a cross-check / fallback.
    Returns the max across all four wheels. Speed is floored at 1 km/h to
    avoid a divide-by-near-zero blowup at a dead stop.
    """
    spin_cols = ["wheel_spin_lf", "wheel_spin_rf", "wheel_spin_lr", "wheel_spin_rr"]
    available = [c for c in spin_cols if c in df.columns]
    if not available or "speed_kmh" not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index)
    speed_floor = df["speed_kmh"].clip(lower=1.0)
    ratios = df[available].sub(df["speed_kmh"], axis=0).div(speed_floor, axis=0)
    return ratios.max(axis=1)


def _get_min_wheel_speed_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Per-wheel (wheel_surface_speed / car_speed) ratio, minimum across all
    four wheels. A locked wheel has near-zero surface speed while the car
    is still moving, so this ratio collapses toward 0 regardless of what
    slip_ratio_* reports — used as the independent lockup fallback signal.
    """
    spin_cols = ["wheel_spin_lf", "wheel_spin_rf", "wheel_spin_lr", "wheel_spin_rr"]
    available = [c for c in spin_cols if c in df.columns]
    if not available or "speed_kmh" not in df.columns:
        return pd.Series(np.ones(len(df)), index=df.index)  # neutral -> never "locked"
    speed_floor = df["speed_kmh"].clip(lower=1.0)
    ratios = df[available].div(speed_floor, axis=0)
    return ratios.min(axis=1)


# ── Individual labelling functions ────────────────────────────────────────────

def label_brake_locked(df: pd.DataFrame) -> pd.Series:
    """
    The previous method of computing brake lock up was unreliable, dangours and unsafe
    thus this new throttle independent method.
    
    Race-engineer level brake lockup detection.
    Tracks individual wheel slip magnitudes and absolute wheel speed collapse
    to catch front and rear axle lockups under any braking pressure.
    """

    # <---old way
    #brake_high  = df["brake"] > BRAKE_PRESSURE_THRESH
    #slip_high   = _get_max_slip(df) >= SLIP_RATIO_THRESH
    #candidate   = brake_high & slip_high

    # <---new way
    # Ensure the driver is actually braking and the car is moving, prevents false positives while stationary in the pits
    braking_active = (df["brake"] > brake_activation_thresh) & (df["speed_kmh"] > speed_gate_kmh)

    # Extract absolute magnitudes of slip ratios, fixes the negative sign issue during deceleration
    lf_slip_lock = df["slip_ratio_lf"].abs() >= slip_lock_thresh
    rf_slip_lock = df["slip_ratio_rf"].abs() >= slip_lock_thresh
    lr_slip_lock = df["slip_ratio_lr"].abs() >= slip_lock_thresh
    rr_slip_lock = df["slip_ratio_rr"].abs() >= slip_lock_thresh

    # Wheel Spin Fallback
    # FIX: wheel_spin_xx is wheel SURFACE SPEED in km/h (confirmed against
    # speed_kmh's matching mean/std/range in describe()), not rad/s and not
    # a 0-1 value — comparing it to wheel_spin_zero_thresh=0.5 was a no-op
    # that never fired. A locked wheel's surface speed collapses toward 0
    # *relative to the car's speed*, so compare per-wheel speed against
    # car speed instead of against a fixed small constant.
    wheel_speed_ratio_lf = df["wheel_spin_lf"] / df["speed_kmh"].clip(lower=1.0)
    wheel_speed_ratio_rf = df["wheel_spin_rf"] / df["speed_kmh"].clip(lower=1.0)
    wheel_speed_ratio_lr = df["wheel_spin_lr"] / df["speed_kmh"].clip(lower=1.0)
    wheel_speed_ratio_rr = df["wheel_spin_rr"] / df["speed_kmh"].clip(lower=1.0)

    lf_spin_lock = wheel_speed_ratio_lf < wheel_locked_speed_ratio
    rf_spin_lock = wheel_speed_ratio_rf < wheel_locked_speed_ratio
    lr_spin_lock = wheel_speed_ratio_lr < wheel_locked_speed_ratio
    rr_spin_lock = wheel_speed_ratio_rr < wheel_locked_speed_ratio

    # Consolidate axle lockups
    # Front lockup matches the symptom of not able to steer at all
    front_locked = (lf_slip_lock | lf_spin_lock) | (rf_slip_lock | rf_spin_lock)
    rear_locked  = (lr_slip_lock | lr_spin_lock) | (rr_slip_lock | rr_spin_lock)
    
    any_wheel_locked = front_locked | rear_locked

    # Gate by active braking context
    candidate = braking_active & any_wheel_locked

    # Pass back into your consecutive frame window constraint (e.g., LOCKUP_MIN_SAMPLES)

    return has_consecutive_true(candidate, LOCKUP_MIN_SAMPLES).astype(int)


def label_wheel_spin(df: pd.DataFrame) -> pd.Series:
    """
    §1.6 definition: max driven wheel spin > 0.3 while throttle > 0.1
    for at least 2 consecutive samples.

    FIX: wheel_spin_xx is wheel surface speed in km/h, not a 0-1 slip
    fraction (see describe() — same scale as speed_kmh). The original
    code compared the raw ~100-scale column directly to
    WHEEL_SPIN_SLIP_THRESH=0.30, which is true almost any time the car
    is moving at all — that's why this label fired on 85.4% of rows.
    Use the wheel-speed-vs-car-speed ratio instead; slip_ratio_* (already
    a correctly-scaled dimensionless ratio) is checked too, and either
    signal firing counts as wheelspin — this makes the label robust even
    if one sensor channel is noisy.
    """
    wheel_slip_ratio = _get_max_wheel_slip_ratio(df)     # (wheel_speed - car_speed) / car_speed
    slip_ratio        = _get_max_slip(df)                # existing slip_ratio_* columns

    spin_high    = (wheel_slip_ratio > WHEEL_SPIN_SLIP_THRESH) | (slip_ratio > WHEEL_SPIN_SLIP_THRESH)
    accelerating = df["throttle"] > WHEEL_SPIN_THROTTLE_MIN
    candidate    = spin_high & accelerating
    return has_consecutive_true(candidate, WHEEL_SPIN_MIN_SAMPLES).astype(int)


def label_trail_brake(df: pd.DataFrame,
                       apex_reference: dict = None) -> pd.Series:
    """
    §1.6 definition: brake pressure > 20% after the geometric apex point.

    If apex_reference is provided (dict mapping track name to list of
    apex_dist_m values per corner), uses distance-based detection.
    Otherwise falls back to: brake > threshold while speed is rising
    (car is already past slowest point of corner).

    apex_reference format:
        {"blackwood_gp": [245.0, 612.0, 890.0, ...]}
    """
    brake_active = df["brake"] > TRAIL_BRAKE_THRESH

    if apex_reference is not None and "lap_dist_m" in df.columns:
        # Build a per-row boolean: True if we're in the post-apex zone of any corner
        post_apex_zone = pd.Series(False, index=df.index)
        for apex_dist in apex_reference:
            window_start = apex_dist
            window_end   = apex_dist + 150  # 150m past apex counts as "post apex"
            in_zone = (
                (df["lap_dist_m"] >= window_start) &
                (df["lap_dist_m"] <= window_end)
            )
            post_apex_zone = post_apex_zone | in_zone
        return (brake_active & post_apex_zone).astype(int)
    else:
        # Fallback: braking while speed_kmh is rising (past the slowest point)
        speed_rising = df["speed_kmh"].diff() > 0 if "speed_kmh" in df.columns \
                       else pd.Series(False, index=df.index)
        return (brake_active & speed_rising).astype(int)


def label_aggressive_downshift(df: pd.DataFrame,
                                vehicle_redline:        float = 8500.0,
                                rpm_danger_pct:         float = DOWNSHIFT_RPM_PCT, #<--- wired to constants cell (0.95, matches docstring)
                                slip_angle_threshold:   float = 4.0,
                                long_g_rate_threshold:  float = 0.40,
                                post_shift_window:      int   = DOWNSHIFT_POST_SHIFT_WINDOW) -> pd.Series: #<--- wired to its own constant (12 rows/200ms), NOT LATE_UPSHIFT_MIN_SAMPLES
    """
    Race-engineer definition of an aggressive downshift.
 
    An aggressive downshift is one whose CONSEQUENCES indicate mechanical
    stress or vehicle instability. It is NOT about what the RPM was
    BEFORE the shift — it is about what happens AFTER the gear engages.
 
    A race engineer watching data would flag a downshift as aggressive when
    any of the following appear in the data within ~200ms after the shift:
 
        a) Post-shift RPM overrev: the engine is pushed above 95% of
           redline by the downshift. This means the driver rev-matched
           too aggressively or downshifted too early in the braking zone,
           forcing the engine to spin faster than intended.
 
        b) Driveline shock: the rate of change of longitudinal deceleration
           spikes sharply immediately after the shift completes. This
           indicates the gearbox engaged abruptly rather than smoothly,
           sending a shock through the driveline (heard as a "clunk" and
           felt as a lurch). Measured as the absolute rate of change of
           longitudinal G exceeding a threshold.
 
        c) Rear instability: body slip angle spikes beyond a threshold
           immediately post-shift. In a rear-wheel-drive car, an aggressive
           downshift under braking can unsettle the rear axle via sudden
           engine braking torque, causing oversteer. Body slip angle
           increasing abruptly after a downshift event is the signature.
 
    ── WHY THE ORIGINAL WAS WRONG (51.7% label rate) ─────────────────────
    The original checked: RPM > 90% of redline AND a gear decrease happened
    in the last 5 rows. The problem: being at 90%+ of redline is completely
    NORMAL at the top of any gear's range. That is literally where you drive
    on a race track. With a 5-row (83ms) look-back window, almost any
    downshift near the top of a gear range was being flagged, regardless of
    whether anything problematic actually happened. The fix is to look at
    what happens AFTER the shift, not what the RPM was before it.
 
    Args:
        vehicle_redline:        Redline RPM of the current car.
        rpm_danger_pct:         Fraction of redline that post-shift RPM must
                                exceed to count as an overrev. Default 0.95.
        slip_angle_threshold:   Body slip angle (degrees) threshold for
                                instability detection post-shift. Default 4.0°.
        long_g_rate_threshold:  Rate of change of longitudinal G (G/sample)
                                threshold for driveline shock detection.
                                Default 0.40 G/sample.
        post_shift_window:      Number of samples (rows) after the gear-decrease
                                event to look for consequences. Default 12 rows
                                = 200ms at 60Hz.
    """
 
    # Step 1: Identify the downshift event row (the exact moment gear decreases)
    if "gear" not in df.columns:
        return pd.Series(0, index=df.index)
 
    gear_diff       = df["gear"].diff()
    downshift_event = gear_diff < 0   # True only at the exact row of the shift
 
    # Step 2: Create a "post-shift window" — the next N rows after the shift event.
    # This is where consequences manifest. Rolling max forward-propagates the flag.
    in_post_window = (
        downshift_event
        .rolling(post_shift_window, min_periods=1)
        .max()
        .astype(bool)
    )
 
    # Step 3a: Post-shift RPM overrev
    # The downshift drove the engine above 95% of redline.
    if "engine_rpm_norm" in df.columns:
        rpm_overrev = df["engine_rpm_norm"] > rpm_danger_pct
    elif "engine_rpm" in df.columns:
        rpm_overrev = df["engine_rpm"] > (vehicle_redline * rpm_danger_pct)
    else:
        rpm_overrev = pd.Series(False, index=df.index)
 
    # Step 3b: Driveline shock via longitudinal G rate of change
    # A smooth shift produces gradual G changes. A shock produces a spike.
    if "longitudinal_g" in df.columns:
        long_g_rate    = df["longitudinal_g"].diff().abs()
        driveline_shock = long_g_rate > long_g_rate_threshold
    else:
        driveline_shock = pd.Series(False, index=df.index)
 
    # Step 3c: Rear instability via body slip angle spike
    # If the car was stable going INTO the braking zone and the slip angle
    # jumps sharply within the post-shift window, the downshift unsettled the rear.
    if "body_slip_angle" in df.columns:
        slip_abs     = df["body_slip_angle"].abs()
        # Rate of slip angle change — a sudden jump signals instability
        slip_rate    = slip_abs.diff().abs()
        rear_unstable = (slip_abs > slip_angle_threshold) | (slip_rate > 1.5)
    else:
        rear_unstable = pd.Series(False, index=df.index)
 
    # Step 4: Combine — a downshift is aggressive if it happened recently
    # AND at least one danger sign is present in the post-shift window.
    aggressive = in_post_window & (rpm_overrev | driveline_shock | rear_unstable)
 
    # Require at least 2 consecutive rows to filter single-sample noise spikes
    return has_consecutive_true(aggressive, 2).astype(int)


def label_late_upshift(df: pd.DataFrame,
                       vehicle_power_peak_norm: float = 0.847,
                       throttle_threshold:      float = 0.70,
                       min_sustained_samples:   int   = LATE_UPSHIFT_MIN_SAMPLES) -> pd.Series: #<--- wired to constants cell (20 samples/0.33s, matches docstring)
    """
    Race-engineer definition of a late upshift.
 
    A late upshift is NOT simply "RPM above power peak." Every driver
    spends time above power peak on every lap — that is where peak power
    is delivered. What a race engineer flags is:
 
        The driver is on full (or near-full) throttle AND has allowed RPM
        to climb into the overrev zone past the power peak AND has not
        upshifted despite being there for long enough that power is being
        lost by staying in the current gear.
 
    ── WHY THE ORIGINAL WAS WRONG (100% label rate) ─────────────────────
    The original condition was: engine_rpm_norm > power_peak_norm sustained
    for 90 consecutive samples. There was NO requirement for the driver to
    be accelerating. During engine braking, trailing throttle, corner entry,
    and any coasting phase, RPM sits comfortably above power peak. Since the
    driver is always transitioning through the power-peak zone between shifts,
    almost every row qualified, giving 100%.
 
    ── WHAT A RACE ENGINEER ACTUALLY LOOKS FOR ──────────────────────────
    1. Driver is meaningfully ON the throttle (throttle > 0.70).
       Engine braking / coasting above power peak is NORMAL technique,
       not a mistake. Filtering on throttle eliminates those cases.
 
    2. RPM is above the power-peak norm.
       The driver is in the zone where staying longer costs them lap time.
 
    3. Gear is not neutral / reverse and a gear increase did not just happen
       (we don't want to double-flag the exact moment of a correct upshift).
 
    4. The condition is sustained for min_sustained_samples rows (default 20
       rows = 0.33 seconds at 60Hz). A skilled driver shifts within ~0.1–0.25
       seconds of passing the power peak. Staying there for 0.33 seconds on
       full throttle is a measurable and consistent late upshift.
 
    Args:
        vehicle_power_peak_norm: power_peak_rpm / redline_rpm.
                                 Default 0.847 ≈ 7200/8500 for a typical LFS GT car.
        throttle_threshold:      Minimum throttle input to count as accelerating.
                                 0.70 = 70% pedal travel — filters coasting / braking.
        min_sustained_samples:   How many consecutive qualifying rows before firing.
                                 20 samples = 0.33 s at 60Hz.
    """
 
    # Condition 1: RPM above power peak
    if "engine_rpm_norm" in df.columns:
        above_peak = df["engine_rpm_norm"] > vehicle_power_peak_norm
    elif "engine_rpm" in df.columns:
        above_peak = df["engine_rpm"] > (vehicle_power_peak_norm * 8500.0)
    else:
        return pd.Series(0, index=df.index)
 
    # Condition 2: Driver is meaningfully accelerating — the critical missing gate
    # that caused the 100% label rate in the original implementation.
    # Without this, engine braking at high RPM fires the label constantly.
    if "throttle" in df.columns:
        accelerating = df["throttle"] > throttle_threshold
    else:
        accelerating = pd.Series(True, index=df.index)
 
    # Condition 3: Not currently in the process of upshifting.
    # A gear increase in the last 3 rows means the driver IS shifting — do not
    # penalise them for the brief window while the shift is executing.
    if "gear" in df.columns:
        gear_increasing     = df["gear"].diff() > 0
        shift_in_progress   = gear_increasing.rolling(3, min_periods=1).max().astype(bool)
        not_shifting        = ~shift_in_progress
        # Also exclude neutral and reverse (gear <= 1) where "upshift" is meaningless
        in_valid_gear       = df["gear"] > 1
    else:
        not_shifting  = pd.Series(True, index=df.index)
        in_valid_gear = pd.Series(True, index=df.index)
 
    # Combine all gates, then apply the run-length check
    candidate = above_peak & accelerating & not_shifting & in_valid_gear
    return has_consecutive_true(candidate, min_sustained_samples).astype(int)

def label_rough_steering(df: pd.DataFrame,
                        lock_to_lock_deg: float = 540.0,
                        base_rough_thresh_deg_s: float = ROUGH_STEER_RATE_THRESH, #<--- wired to constants cell (180.0, low-speed endpoint)
                        high_speed_rough_thresh_deg_s: float = ROUGH_STEER_HIGH_SPEED_THRESH, #<--- wired to constants cell (90.0, high-speed endpoint)
                        speed_low_gate_kmh: float = 40.0,
                        speed_high_gate_kmh: float = 140.0,
                        min_duration_samples: int = 3) -> pd.Series:
    """
    Race-engineer grade rough steering and chassis instability detection.
    Converts normalized steering data to physical degrees/sec, applies a 
    dynamic, speed-sensitive threshold, and filters out high-frequency noise.
    """
    #<---old method
    #if "steer_rate" not in df.columns:
        # Derive from steering angle differences if rate not available
    #    steer_rate = df["steer"].diff().abs() * SAMPLE_RATE_HZ if "steer" in df.columns \
    #                 else pd.Series(np.zeros(len(df)), index=df.index)
    #else:
    #    steer_rate = df["steer_rate"].abs()

    #<---new method

    # Compute or extract steering rate, already in physical degrees/second
    # FIX: steer_rate (precomputed by the LFS telemetry script) is confirmed
    # via describe() to already be in degrees/second — mean≈0, std≈3.9,
    # range -185..328. That range only makes sense as deg/s (a fast wheel
    # flick can hit a few hundred deg/s); read as "normalized units/sec" it
    # would imply traversing the full steering range ~160 times a second,
    # which is physically absurd. The old code re-multiplied this already-
    # physical value by deg_per_norm_unit (270), inflating any nonzero
    # reading by 270x and pushing it past the 90-180 deg/s threshold almost
    # every time. Only the "steer" position-diff fallback (genuinely on the
    # -1..1 normalized scale) needs that conversion.
    deg_per_norm_unit = lock_to_lock_deg / 2.0   # used only by the fallback branch below
    if "steer_rate" in df.columns:
        steer_rate_deg_s = df["steer_rate"].abs()
    elif "steer" in df.columns:
        norm_steer_rate = df["steer"].diff().abs() * SAMPLE_RATE_HZ
        norm_steer_rate = norm_steer_rate.fillna(0.0)
        steer_rate_deg_s = norm_steer_rate * deg_per_norm_unit
    else:
        return pd.Series(np.zeros(len(df)), index=df.index).astype(int)

    # Dynamic Thresholding based on Vehicle Speed
    # Extract raw speed prioritizing non-normalized speed from the adapter stage
    speed = df["speed_kmh"] if "speed_kmh" in df.columns else (df["speed_kmh_norm"] * 250.0)
    
    # Linearly interpolate the threshold between low-speed handling and high-speed stability
    # At 40 km/h: threshold is 180 deg/s. At 140 km/h: threshold tightens to 90 deg/s.
    speed_fraction = ((speed - speed_low_gate_kmh) / (speed_high_gate_kmh - speed_low_gate_kmh)).clip(0.0, 1.0)
    dynamic_threshold = base_rough_thresh_deg_s - (speed_fraction * (base_rough_thresh_deg_s - high_speed_rough_thresh_deg_s))

    # Check if steering velocity violates dynamic safety envelope
    # Also ignore parking/pit maneuvers below 20 km/h entirely
    steering_violating = (steer_rate_deg_s > dynamic_threshold) & (speed > 20.0)

    # Cross-Reference with Chassis Instability
    # High steering inputs are especially "rough" if they cause sudden spikes in lateral loading
    if "lateral_g" in df.columns:
        lateral_jerk = df["lateral_g"].diff().abs() * SAMPLE_RATE_HZ
        # A jerk over 4.0 G/sec indicates violent platform snapping
        heavy_chassis_transient = (lateral_jerk > 4.0) & (speed > 20.0)
        # FIX: original was `steering_violating | (steering_violating & heavy_chassis_transient)`,
        # which is a tautology (A | (A & B) == A) — heavy_chassis_transient never
        # actually affected the label. Genuine OR: fire on either a pure steering-rate
        # overload, or a chassis-snap event, matching the docstring's stated intent.
        candidate = steering_violating | heavy_chassis_transient
    else:
        candidate = steering_violating

    # Apply persistence filter (e.g., 3 consecutive samples / 50ms)
    # Prevents sharp single-frame force feedback spikes from polluting labels
    return has_consecutive_true(candidate, min_duration_samples).astype(int)
#return (steer_rate > ROUGH_STEER_RATE_THRESH).astype(int)


def label_gentle_accel(df: pd.DataFrame,
                        apex_reference: dict = None) -> pd.Series:
    """
    §1.6 definition: throttle application RATE below 15%/sample from apex exit,
    indicating smooth and controlled power delivery. Positive label.

    Uses the same apex reference structure as trail_brake.
    Falls back to any sustained-throttle region if no apex reference provided.
    """
    # NOTE: throttle_rate (from the LFS telemetry script) is on a
    # percentage-points-per-sample scale (describe(): min=-117.7, max=100.2),
    # NOT a 0-1 fraction. GENTLE_ACCEL_RATE_THRESH is fixed at the constant
    # level (config cell) to match this scale — if throttle_rate is ever
    # re-derived from a differently-scaled column, re-check this threshold.
    if "throttle_rate" not in df.columns:
        throttle_rate = df["throttle"].diff().clip(lower=0) * 100.0 if "throttle" in df.columns \
                        else pd.Series(np.zeros(len(df)), index=df.index)
    else:
        throttle_rate = df["throttle_rate"].clip(lower=0)

    smooth_accel = throttle_rate < GENTLE_ACCEL_RATE_THRESH
    throttle_active = df["throttle"] > 0.1 if "throttle" in df.columns \
                      else pd.Series(True, index=df.index)

    candidate = smooth_accel & throttle_active

    if apex_reference is not None and "lap_dist_m" in df.columns:
        post_apex_zone = pd.Series(False, index=df.index)
        for apex_dist in apex_reference:
            in_zone = (
                (df["lap_dist_m"] >= apex_dist) &
                (df["lap_dist_m"] <= apex_dist + 200)  # 200m exit zone
            )
            post_apex_zone = post_apex_zone | in_zone
        candidate = candidate & post_apex_zone

    return has_consecutive_true(candidate, GENTLE_ACCEL_MIN_SAMPLES).astype(int)


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



# %% [markdown] cell 16
# # LSTM Data -Windowing
# windowing.py — Converts a labelled canonical DataFrame into sliding
# 60-timestep windows ready for LSTM training. Handles normalization,
# session-stratified train/val/test splitting, and DataLoader creation.

# %% cell 17
# ── Normalization ─────────────────────────────────────────────────────────────

def fit_normalizer(df: pd.DataFrame,
                   feature_cols: list,
                   save_path: str = "normalizer_params.json") -> dict:
    """
    Fit MinMax normalization on the TRAINING set only.
    Saves min/max per feature to JSON so inference can reproduce
    the exact same transform on never-seen data.

    Returns:
        dict: {"feature_name": {"min": float, "max": float}, ...}
    """
    params = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        params[col] = {"min": col_min, "max": col_max}

    if save_path:
        with open(save_path, "w") as f:
            json.dump(params, f, indent=2)
        print(f"[Norm] Normalizer params saved → {save_path}")

    return params


def apply_normalizer(df: pd.DataFrame,
                     params: dict,
                     feature_cols: list) -> pd.DataFrame:
    """
    Apply saved min-max normalization to a DataFrame using pre-fitted params.
    Clips values to [0, 1] — out-of-range values (e.g. from AC data)
    are clipped rather than allowed to explode the model input.
    """
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns or col not in params:
            continue
        col_min = params[col]["min"]
        col_max = params[col]["max"]
        rng = col_max - col_min
        if rng == 0:
            df[col] = 0.0
        else:
            df[col] = ((df[col] - col_min) / rng).clip(0, 1)
    return df


def load_normalizer(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


# ── Session-stratified split ──────────────────────────────────────────────────

def session_split(df: pd.DataFrame,
                  train_sessions: list,
                  val_sessions: list,
                  test_sessions: list,
                  session_col: str = "session_id") -> tuple:
    """
    Split a multi-session DataFrame into train/val/test by session ID.
    This prevents data leakage — no frame from the same lap appears
    in both training and evaluation sets.

    Args:
        train_sessions: list of session_id values for training
        val_sessions:   list of session_id values for validation
        test_sessions:  list of session_id values for test (held-out Nurburgring)

    Returns:
        (train_df, val_df, test_df)
    """
    if session_col not in df.columns:
        # Single-session fallback: split by row index 70/15/15
        n = len(df)
        t = int(n * 0.70)
        v = int(n * 0.85)
        print("[Split] No session_id column found — using 70/15/15 row split.")
        return df.iloc[:t], df.iloc[t:v], df.iloc[v:]

    train_df = df[df[session_col].isin(train_sessions)].reset_index(drop=True)
    val_df   = df[df[session_col].isin(val_sessions)].reset_index(drop=True)
    test_df  = df[df[session_col].isin(test_sessions)].reset_index(drop=True)

    print(f"[Split] Train: {len(train_df):,} rows | "
          f"Val: {len(val_df):,} rows | "
          f"Test: {len(test_df):,} rows")
    return train_df, val_df, test_df


def lap_split(df: pd.DataFrame,
              train_frac: float = 0.70,
              val_frac: float   = 0.15) -> tuple:
    """
    Single-session fallback: split by lap number rather than by row,
    so consecutive laps stay together and data leakage is minimised.
    """
    if "lap" not in df.columns:
        return session_split(df, [], [], [])

    laps = sorted(df["lap"].unique())
    n = len(laps)
    t_end = int(n * train_frac)
    v_end = int(n * (train_frac + val_frac))

    train_laps = laps[:t_end]
    val_laps   = laps[t_end:v_end]
    test_laps  = laps[v_end:]

    train_df = df[df["lap"].isin(train_laps)].reset_index(drop=True)
    val_df   = df[df["lap"].isin(val_laps)].reset_index(drop=True)
    test_df  = df[df["lap"].isin(test_laps)].reset_index(drop=True)

    print(f"[Split] Train laps: {train_laps} | Val laps: {val_laps} | Test laps: {test_laps}")
    print(f"        Rows → Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    return train_df, val_df, test_df


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class TelemetryWindowDataset(Dataset):
    """
    Converts a labelled, normalised canonical DataFrame into a PyTorch
    Dataset of (window, label) pairs.

    Each sample:
        window: Tensor of shape (WINDOW_SIZE, num_features) = (60, 25)
        label:  Tensor of shape (NUM_CLASSES,)               = (7,)
                Multi-label binary vector — BCEWithLogitsLoss target.

    The label assigned to each window is taken from the LAST row of that
    window (the most recent timestep), matching the LSTM's causal design.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 feature_cols: list = None,
                 label_cols: list   = None,
                 window_size: int   = WINDOW_SIZE,
                 stride: int        = TRAIN_STRIDE):

        self.feature_cols = [c for c in (feature_cols or FEATURE_COLS) if c in df.columns]
        self.label_cols   = [c for c in (label_cols   or LABEL_COLS)   if c in df.columns]
        self.window_size  = window_size

        # Pre-build all windows as numpy arrays for fast __getitem__
        features = df[self.feature_cols].values.astype(np.float32)  # (N, F)
        labels   = df[self.label_cols  ].values.astype(np.float32)  # (N, C)

        self.windows = []
        self.labels  = []

        for start in range(0, len(df) - window_size + 1, stride):
            end = start + window_size
            self.windows.append(features[start:end])              # (W, F)
            self.labels.append(labels[end - 1])                   # (C,) label at last row

        self.windows = np.stack(self.windows)   # (n_windows, W, F)
        self.labels  = np.stack(self.labels)    # (n_windows, C)

        print(f"[Dataset] {len(self.windows):,} windows | "
              f"shape: {self.windows.shape} | "
              f"labels: {self.labels.shape}")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple:
        return (
            torch.from_numpy(self.windows[idx]),
            torch.from_numpy(self.labels[idx]),
        )

    def get_sample_weights(self) -> torch.Tensor:
        """
        Compute per-sample weights for WeightedRandomSampler.
        Samples containing at least one rare-event positive label get
        higher weight, counteracting class imbalance in the DataLoader.

        Weight = max pos_weight across all active labels for that window.
        """
        pos_counts = self.labels.sum(axis=0)                     # (C,)
        neg_counts = len(self.labels) - pos_counts
        pos_weights = neg_counts / np.maximum(pos_counts, 1)     # (C,)

        # Per-sample weight = max weight across its active labels
        sample_weights = np.ones(len(self.labels))
        for i, label_vec in enumerate(self.labels):
            active = label_vec > 0.5
            if active.any():
                sample_weights[i] = pos_weights[active].max()

        return torch.tensor(sample_weights, dtype=torch.float32)


# ── DataLoader factory ────────────────────────────────────────────────────────

def make_dataloader(dataset: TelemetryWindowDataset,
                    batch_size: int   = BATCH_SIZE,
                    shuffle: bool     = False,
                    use_sampler: bool = False,
                    num_workers: int  = 0) -> DataLoader:
    """
    Wrap a TelemetryWindowDataset in a DataLoader.

    use_sampler=True enables WeightedRandomSampler — use this for the
    training DataLoader to oversample minority-class windows. Leave False
    for val/test to get an unbiased evaluation.
    """
    sampler = None
    if use_sampler:
        weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False  # mutually exclusive with sampler

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ── Full windowing pipeline ───────────────────────────────────────────────────

def build_datasets(labelled_csv_path: str,
                   normalizer_save_path: str = "normalizer_params.json",
                   train_sessions: list = None,
                   val_sessions:   list = None,
                   test_sessions:  list = None,
                   train_stride: int    = TRAIN_STRIDE,
                   eval_stride: int     = EVAL_STRIDE) -> tuple:
    """
    Full pipeline: load labelled CSV → split → normalise → build datasets.

    Returns:
        (train_dataset, val_dataset, test_dataset, normalizer_params)
    """
    df = pd.read_csv(labelled_csv_path)
    print(f"[Window] Loaded {len(df):,} rows")

    # Missing values — forward fill (handles session init lag)
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    df[feature_cols] = df[feature_cols].fillna(method="ffill").fillna(0.0)

    # Session split
    if train_sessions is not None:
        train_df, val_df, test_df = session_split(
            df, train_sessions, val_sessions, test_sessions)
    else:
        train_df, val_df, test_df = lap_split(df)

    # Fit normalizer on train only
    norm_params = fit_normalizer(train_df, feature_cols, save_path=normalizer_save_path)

    # Apply normalizer to all splits
    train_df = apply_normalizer(train_df, norm_params, feature_cols)
    val_df   = apply_normalizer(val_df,   norm_params, feature_cols)
    test_df  = apply_normalizer(test_df,  norm_params, feature_cols)

    # Build datasets
    train_ds = TelemetryWindowDataset(train_df, stride=train_stride)
    val_ds   = TelemetryWindowDataset(val_df,   stride=eval_stride)
    test_ds  = TelemetryWindowDataset(test_df,  stride=eval_stride)

    return train_ds, val_ds, test_ds, norm_params

# %% [markdown] cell 18
# # LFS Multistream Injestion
#
# High-performance pre-ingestion layer for LFS telemetry logs.
#
# Discovers, chronologically sorts, shifts lap boundaries, and merges multi-file datasets.

# %% cell 19
def load_and_compile_telemetry_folder(input_source, fallback_engine='c') -> pd.DataFrame:
    # Handle input source parsing
    if isinstance(input_source, (str, Path)):
        folder = Path(input_source)
        # Find all CSV files in the directory
        csv_files = list(folder.glob("*.csv"))
    elif isinstance(input_source, list):
        csv_files = [Path(f) for f in input_source]
    else:
        raise ValueError("input_source must be a folder path string, Path object, or list of file paths, you shmack.")

    if not csv_files:
        raise FileNotFoundError(f"No CSV telemetry files discovered for input: {input_source}. What are doing here, cuzzo!")

    # Extract numeric timestamp from filename and sort chronologically
    def extract_timestamp(path: Path) -> int:
        match = re.search(r'\d+', path.name)
        return int(match.group()) if match else 0

    sorted_files = sorted(csv_files, key=extract_timestamp)
    print(f"Found and sorted {len(sorted_files)} telemetry files chronologically")

    # Check if pyarrow is installed for high-speed multi-threaded parsing
    try:
        import pyarrow
        csv_engine = 'pyarrow'
    except ImportError:
        csv_engine = fallback_engine
    print(f"Using high-performance CSV engine: '{csv_engine}'...")

    compiled_dfs = []
    cumulative_lap_offset = 0

    # Read and vectorized-shift laps across file boundaries
    for idx, file_path in enumerate(sorted_files):
        # Read file with optimized engine configuration
        df = pd.read_csv(file_path, engine=csv_engine)
        
        if df.empty:
            continue
            
        if 'Lap' not in df.columns:
            raise KeyError(f"Critical error: 'Lap' column missing in telemetry file: {file_path.name} \nIf you are seeing this, you definetly using the wrong dataset")

        # If it's not the first file, shift its laps sequentially 
        # Formula: current_file_laps = cumulative_previous_max_laps + current_file_laps
        if idx > 0 and cumulative_lap_offset > 0:
            df['Lap'] += cumulative_lap_offset

        # Update the cumulative lap offset based on the new max lap observed
        cumulative_lap_offset = int(df['Lap'].max())
        
        compiled_dfs.append(df)
        print(f" ---- Processed: {file_path.name}\nTotal rows: {len(df):,}\nAbsolute Laps: {df['Lap'].min()} to {df['Lap'].max()}")

    # Perform a single-pass efficient memory allocation merge
    print("Merging dataset blocks into unified global stream...")
    unified_df = pd.concat(compiled_dfs, ignore_index=True)
    
    print(f"Pre-ingestion complete! Consolidated Data Shape: {unified_df.shape}")
    return unified_df

# %% [markdown] cell 20
# # LFS Adapter
# lfs_adapter.py — Maps raw LFS telemetry CSV columns to the canonical schema
# and resolves all unit discrepancies.
#
# Run this FIRST before any labelling or preprocessing.

# %% cell 21
# ── Raw LFS column → canonical column mapping ─────────────────────────────────
LFS_TO_CANONICAL = {
    "Timestamp_MS"   : "timestamp_ms",
    "Session_Time_S" : "session_time_s",
    "Lap"            : "lap",
    "Lap_Dist_M"     : "lap_dist_m",
    "Distance_M"     : "distance_m",
    "Speed_KMH"      : "speed_kmh",
    "Engine_RPM"     : "engine_rpm",
    "Gear"           : "gear",
    "Throttle"       : "throttle",
    "Brake"          : "brake",
    "Steer"          : "steer",
    "Clutch"         : "clutch",
    "Handbrake"      : "handbrake",
    "Throttle_Rate"  : "throttle_rate",
    "Brake_Rate"     : "brake_rate",
    "Steer_Rate"     : "steer_rate",
    "Slip_Ratio_LF"  : "slip_ratio_lf",
    "Slip_Ratio_RF"  : "slip_ratio_rf",
    "Slip_Ratio_LR"  : "slip_ratio_lr",
    "Slip_Ratio_RR"  : "slip_ratio_rr",
    "Body_Slip_Angle": "body_slip_angle",
    "AngVel_X"       : "angvel_x",
    "AngVel_Y"       : "angvel_y",
    "AngVel_Z"       : "angvel_z",
    "Susp_Load_LF"   : "susp_load_lf",
    "Susp_Load_RF"   : "susp_load_rf",
    "Susp_Load_LR"   : "susp_load_lr",
    "Susp_Load_RR"   : "susp_load_rr",
    "Wheel_Spin_LF"  : "wheel_spin_lf",
    "Wheel_Spin_RF"  : "wheel_spin_rf",
    "Wheel_Spin_LR"  : "wheel_spin_lr",
    "Wheel_Spin_RR"  : "wheel_spin_rr",
    "Accel_X"        : "accel_x_ms2",
    "Accel_Y"        : "accel_y_ms2",
    "Accel_Z"        : "accel_z_ms2",
    "Roll"           : "roll",
    "Pitch"          : "pitch",
    "Heading"        : "heading",
}

# Vehicle-specific constants — update per car used in LFS
# FIX: this "default" profile (redline 8500) previously disagreed with
# VEHICLE_REDLINE=7031 set in the main pipeline config cell. convert_units()
# always runs with vehicle="default" (see adapt_lfs_telemetry's default arg,
# used as-is in main()), so THIS redline_rpm is the one actually used to
# compute engine_rpm_norm — not whatever VEHICLE_REDLINE says elsewhere.
# Set redline_rpm here to your real car's redline so there's one source of
# truth, and update power_peak_rpm to the actual dyno/in-game peak-power RPM
# for that car (7200 below is a placeholder ratio-matched default, NOT a
# verified value for this specific vehicle — replace it).
VEHICLE_CONFIG = {
    "default": {
        "redline_rpm"   : 7031,   # was 8500 — now matches VEHICLE_REDLINE
        "power_peak_rpm": 5956,   # placeholder = redline * 0.847 ratio default — VERIFY against real car
        "max_speed_kmh" : 280,
        "susp_load_max" : 5000,   # Newton-metres, normalise by this
    }
}


def load_raw_lfs_csv(csv_path: str) -> pd.DataFrame:
    """Load the raw LFS telemetry CSV and do basic sanity checks."""
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
      - Accel X/Y/Z: m/s² → G  (AC gives G natively; LFS gives m/s²)
      - RPM: normalize by redline
      - Speed: normalize by max speed
      - Suspension load: normalize by max load
    """
    cfg = VEHICLE_CONFIG.get(vehicle, VEHICLE_CONFIG["default"])

    # m/s² → G
    for axis in ["x", "y", "z"]:
        col = f"accel_{axis}_ms2"
        g_col = col.replace("_ms2", "_g").replace("accel_", "")
        if col in df.columns:
            df[f"{'lateral_g' if axis == 'y' else 'longitudinal_g' if axis == 'x' else 'vertical_g'}"] = \
                df[col] / 9.81

    # Normalise RPM → fraction of redline
    if "engine_rpm" in df.columns:
        df["engine_rpm_norm"] = (df["engine_rpm"] / cfg["redline_rpm"]).clip(0, 1.2)

    # Normalise speed
    if "speed_kmh" in df.columns:
        df["speed_kmh_norm"] = (df["speed_kmh"] / cfg["max_speed_kmh"]).clip(0, 1.2)

    # Normalise suspension loads
    for corner in ["lf", "rf", "lr", "rr"]:
        raw_col = f"susp_load_{corner}"
        if raw_col in df.columns:
            df[f"{raw_col}_norm"] = (df[raw_col] / cfg["susp_load_max"]).clip(0, 2)

    return df


def drop_duplicates_and_resample(df: pd.DataFrame,
                                  target_interval_ms: int = 17) -> pd.DataFrame:
    """
    Fix the sampling-rate jitter logged in the telemetry diagnostics
    (mean 17.24ms, max spike 156ms, min gap 0ms):
      1. Drop exact-duplicate timestamps (min gap = 0ms rows)
      2. Resample to a fixed target_interval_ms grid via linear interpolation
      3. Flag any interpolated segment > 3 original samples as low-confidence

    target_interval_ms=17 ≈ 58.8Hz (matches observed actual mean).
    """
    df = df.copy()
    df = df.sort_values("timestamp_ms").reset_index(drop=True)

    # Drop duplicate timestamps
    df = df.drop_duplicates(subset="timestamp_ms", keep="first")

    # Build regular time grid
    t_start = int(df["timestamp_ms"].iloc[0])
    t_end   = int(df["timestamp_ms"].iloc[-1])
    regular_times = np.arange(t_start, t_end + 1, target_interval_ms)

    # Set timestamp as index and reindex onto regular grid, interpolate
    df = df.set_index("timestamp_ms")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Merge original + regular grid, interpolate numerics only
    df = df.reindex(df.index.union(regular_times))
    df[numeric_cols] = df[numeric_cols].interpolate(method="index")
    df = df.reindex(regular_times)
    df.index.name = "timestamp_ms"
    df = df.reset_index()

    # Flag low-confidence interpolated regions (gaps > 3 original samples = ~50ms)
    # We do this by checking where the original had large gaps before resampling
    df["interpolated"] = False  # placeholder — full implementation in labelling stage
    return df


def add_source_column(df: pd.DataFrame, source: str = "lfs") -> pd.DataFrame:
    """Tag every row with its data source for downstream traceability."""
    df["data_source"] = source
    return df


def adapt_lfs_telemetry(csv_path: Union[str, Path, pd.DataFrame],
                         vehicle: str = "default",
                         output_path: str = None) -> pd.DataFrame:
    """
    Full adapter pipeline: load → rename → convert units → resample → tag source.

    Args:
        csv_path:    Path to raw LFS telemetry CSV
        vehicle:     Vehicle key in VEHICLE_CONFIG (update per LFS car)
        output_path: If provided, save the canonical CSV here

    Returns:
        Canonical schema DataFrame, ready for labelling.
    """
    # Versatility layer, checking the kind of input we have and routing accordingly
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
        print(f"  Saved → {output_path}")

    return df


def adapt_ac_telemetry(csv_path: str,
                        vehicle: str = "default",
                        output_path: str = None) -> pd.DataFrame:
    """
    Adapter for Assetto Corsa / Telemetrick CSV output.
    AC already provides G-forces and hydraulic brake pressure natively,
    so unit conversion is lighter. Column mapping handles the name differences.

    NOTE: Fill in the AC_TO_CANONICAL mapping below once you have an AC CSV.
    """
    AC_TO_CANONICAL = {
        # Placeholder — map AC Telemetrick column names here
        # e.g. "BrakePressure" : "brake",
        #      "Speed"         : "speed_kmh",
        # ... et al.
    }
    print("[Adapter] AC adapter — column mapping not yet defined. "
          "Add AC→canonical mappings to AC_TO_CANONICAL dict.")
    df = pd.read_csv(csv_path)
    rename_map = {k: v for k, v in AC_TO_CANONICAL.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    df = add_source_column(df, source="ac")

    if output_path:
        df.to_csv(output_path, index=False)
    return df


def merge_sessions(csv_paths: list,
                   vehicle: str = "default",
                   output_path: str = None) -> pd.DataFrame:
    """
    Adapt and concatenate multiple LFS session CSVs into one combined
    canonical DataFrame, with a global session_id column added per file.
    """
    frames = []
    for i, path in enumerate(csv_paths):
        df = adapt_lfs_telemetry(path, vehicle=vehicle)
        df["session_id"] = i
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"[Adapter] Merged {len(csv_paths)} sessions → {len(combined):,} rows")

    if output_path:
        combined.to_csv(output_path, index=False)
        print(f"  Saved → {output_path}")
    return combined

# %% [markdown] cell 22
# # LSTM Module - Fine Tuning
# lstm_finetune.py — Fine-tune a pre-trained LSTM on new data.
#
# Two strategies:
#   Strategy A — Head-only (fast, ~5 epochs):
#       Freeze the LSTM backbone, only train the classifier head.
#       Use when the new data has similar feature structure but different
#       driving style distribution (e.g. adding a new driver's sessions).
#
#   Strategy B — Full fine-tune (thorough, ~15-20 epochs):
#       Unfreeze everything, very low LR (1e-5).
#       Use when the new data comes from a different simulator source
#       (e.g. LFS-trained model receiving first Assetto Corsa data).
#
# Strategy A then B in sequence is the recommended default — same pattern
# used by the CNN module's freeze-then-finetune approach.

# %% cell 23
def load_pretrained(checkpoint_path: str,
                    device: str = None) -> AttentionLSTM:
    """Load a saved best_model.pt checkpoint and return the model."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(checkpoint_path, map_location=device)
    cfg    = ckpt.get("model_config", {})

    model = build_model(
        input_dim   = cfg.get("input_dim",   len(LABEL_COLS)),
        hidden_dim  = cfg.get("hidden_dim",  128),
        num_layers  = cfg.get("num_layers",  2),
        num_classes = cfg.get("num_classes", len(LABEL_COLS)),
        device      = device,
    )
    model.load_state_dict(ckpt["model_state"])
    print(f"[Finetune] Loaded checkpoint: {checkpoint_path}")
    print(f"[Finetune] Pre-trained val F1: {ckpt.get('val_f1', 'N/A'):.4f}")
    return model


def freeze_backbone(model: AttentionLSTM) -> None:
    """Freeze LSTM + input_norm layers; leave classifier head trainable."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Finetune] Strategy A — head only | Trainable params: {trainable:,}")


def unfreeze_all(model: AttentionLSTM) -> None:
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad_(True)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Finetune] Strategy B — full fine-tune | Total params: {total:,}")


class LSTMFineTuner:
    """
    Manages the freeze → fine-tune sequence with separate early stopping
    for each phase.
    """

    def __init__(self,
                 checkpoint_path:  str,
                 train_loader,
                 val_loader,
                 pos_weights_dict: dict,
                 output_dir:       str   = "/kaggle/working/finetune",
                 device:           str   = None):

        self.device        = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model         = load_pretrained(checkpoint_path, self.device)
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.pos_weights   = pos_weights_dict
        self.output_dir    = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history       = {"phase_a": [], "phase_b": []}

    def _run_phase(self,
                   phase_name:    str,
                   learning_rate: float,
                   max_epochs:    int,
                   patience:      int) -> float:
        """Run one fine-tuning phase. Returns best val F1 achieved."""
        criterion = build_criterion(self.pos_weights, LABEL_COLS, self.device)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, verbose=True)

        best_val_f1    = 0.0
        patience_count = 0
        phase_history  = []

        print(f"\n[Finetune] === Phase {phase_name} | LR={learning_rate} | "
              f"Max epochs={max_epochs} ===")

        for epoch in range(1, max_epochs + 1):
            train_m = run_epoch(self.model, self.train_loader, criterion,
                                optimizer, self.device, training=True)
            val_m   = run_epoch(self.model, self.val_loader, criterion,
                                optimizer, self.device, training=False)
            scheduler.step(val_m["macro_f1"])

            phase_history.append({
                "epoch": epoch,
                "train_loss": train_m["loss"], "val_loss": val_m["loss"],
                "train_f1":  train_m["macro_f1"], "val_f1": val_m["macro_f1"],
            })

            print(f"  Ep {epoch:03d}  train_F1={train_m['macro_f1']:.4f}  "
                  f"val_F1={val_m['macro_f1']:.4f}")

            if val_m["macro_f1"] > best_val_f1:
                best_val_f1    = val_m["macro_f1"]
                patience_count = 0
                self._save(phase_name, best_val_f1)
                print(f"    ★ New best: {best_val_f1:.4f}")
            else:
                patience_count += 1
                if patience_count >= patience:
                    print(f"  Early stop at epoch {epoch}")
                    break

        self.history[f"phase_{phase_name.lower()}"] = phase_history
        return best_val_f1

    def run(self,
            phase_a_epochs: int = 10,
            phase_b_epochs: int = 25,
            phase_a_lr:     float = LEARNING_RATE,
            phase_b_lr:     float = FINETUNE_LR) -> dict:
        """
        Execute Strategy A (head-only) then Strategy B (full fine-tune).
        """
        # Phase A
        freeze_backbone(self.model)
        f1_a = self._run_phase("A", phase_a_lr, phase_a_epochs, patience=5)

        # Phase B — unfreeze, very low LR
        unfreeze_all(self.model)
        f1_b = self._run_phase("B", phase_b_lr, phase_b_epochs, patience=PATIENCE)

        print(f"\n[Finetune] Done. Phase A best F1: {f1_a:.4f} | "
              f"Phase B best F1: {f1_b:.4f}")

        history_path = self.output_dir / "finetune_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def _save(self, phase: str, val_f1: float):
        path = self.output_dir / f"finetune_phase{phase}_best.pt"
        torch.save({
            "model_state": self.model.state_dict(),
            "val_f1":      val_f1,
            "model_config": {
                "input_dim":   self.model.input_dim,
                "hidden_dim":  self.model.hidden_dim,
                "num_layers":  self.model.num_layers,
                "num_classes": self.model.num_classes,
            },
        }, path)

# %% [markdown] cell 24
# # LSTM Module - Best Model Download
#
# Designed to generate a download link for the `bestmodel.pt` in the cell block output

# %% cell 25
def download_bestmodel (print_files=False):
    # 1. Display all files in /kaggle/working/checkpoints
    print("Checking Checkpoint Directory Contents...")
    if CHECKPOINT_DIR.exists():
        files = list(CHECKPOINT_DIR.iterdir())
        if files:
            for f in files:
                size_mb = f.stat().st_size / (1024 * 1024)
                if print_files == True:
                    print(f"  📄 {f.name} ({size_mb:.2f} MB)")
        else:
            print("  (Directory is empty)")
            return
    else:
        print("   Path /kaggle/working/checkpoints does not exist.")
        return
    print("Found file in search in Checkpoint Directory Contents")
    print(f"Making a copy in {BESTMODEL_DIR}...")

    BESTMODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Copy best_model.pt into the new 'bestmodel' folder
    source_path = CHECKPOINT_DIR / "best_model.pt"
    dest_path   = BESTMODEL_DIR / "best_model.pt"
    
    if source_path.exists():
        shutil.copy(source_path, dest_path)
        print(f"\n✅ Copied 'best_model.pt' -> '{BESTMODEL_DIR}/best_model.pt'")
        print("You can now download it directly from the Output pane under 'bestmodel/'!")
    else:
        print(f"\n⚠️ Could not find {source_path}. Make sure training has saved a checkpoint first.")
    
    # 3. Creating a direct clickable Download link
    print("Generating model download link...")
    from IPython.display import FileLink
    print(f'Download the model below\n')
    return FileLink('bestmodel/best_model.pt')

# %% [markdown] cell 26
# # LSTM Module - Trianer
# lstm_trainer.py — Full training loop for the AttentionLSTM.
#
# Designed to run on Kaggle Notebooks (NVIDIA P100/T4 GPU).
# Saves checkpoints to /kaggle/working/ (or a local path you specify)
# and best model by val macro-F1.

# %% cell 27
# ── Loss function ─────────────────────────────────────────────────────────────

class FocalLossWithLogits(nn.Module):
    """
    Multi-label focal loss (Lin et al. 2017), built on the same per-class
    pos_weight signal build_criterion already computes.

    Standard BCE+pos_weight upweights rare-class positives in the loss, but
    every example still contributes ~equally once weighted — easy examples
    (majority class, obviously-negative rows) still dominate the gradient
    just by sheer volume. Focal loss additionally down-weights easy/confident
    examples via (1-p_t)^gamma, focusing training on hard/rare/ambiguous
    ones — often a better fit than a raw pos_weight cap for a class as
    extreme as Brake Locked (0.5% base rate), and tends to produce probability
    outputs that don't need such extreme per-class thresholds to work with.

    Toggle via USE_FOCAL_LOSS in the constants cell — compare val macro-F1
    against a plain-BCE run before committing to it; it isn't strictly better
    in every case and needs its own gamma/alpha tuning pass.
    """
    def __init__(self, pos_weight: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = torch.sigmoid(logits)
        p_t     = probs * targets + (1 - probs) * (1 - targets)
        focal_w = (1 - p_t).clamp(min=1e-6) ** self.gamma

        # per-class alpha: reuses the same normalized pos_weight signal so
        # rare classes still get extra weight on top of the focal term
        alpha_t = self.alpha * self.pos_weight * targets + (1 - self.alpha) * (1 - targets)

        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        loss = alpha_t * focal_w * bce
        return loss.mean()


def build_criterion(pos_weights_dict: dict,
                    label_cols: list,
                    device: str):
    """
    Build the training loss with class-specific positive weights.

    pos_weights_dict: output of distributions.class_distribution_report()
                      e.g. {"label_brake_locked": 47.3, "label_wheel_spin": 12.1, ...}

    pos_weight tensor shape must match the number of output classes.
    Returns FocalLossWithLogits if USE_FOCAL_LOSS else plain BCEWithLogitsLoss —
    both consumed identically by run_epoch (criterion(logits, labels)).
    """
    weights = []
    for col in label_cols:
        w = pos_weights_dict.get(col, 1.0)
        # Cap weight at 50x to prevent instability on extremely rare classes
        weights.append(min(float(w), 50.0))

    pos_weight = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"[Loss] pos_weights: {dict(zip(CLASS_NAMES, [round(w,1) for w in weights]))}")

    if USE_FOCAL_LOSS:
        print(f"[Loss] Using FocalLossWithLogits (gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA})")
        return FocalLossWithLogits(pos_weight, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def diagnose_class_separation(all_logits: np.ndarray,
                               all_labels: np.ndarray,
                               class_names: list,
                               thresholds: np.ndarray = None) -> list:
    """
    Per-class diagnostic that goes beyond the fixed threshold=0.50 F1 used by
    compute_epoch_metrics(). For each class, reports:
      - best_f1 / best_threshold: the best F1 achievable at ANY threshold,
        found by a coarse sweep (0.05 to 0.95). This is what tune_thresholds()
        finds properly at the end of training — this gives an early read on
        it during training.
      - mean_prob_pos / mean_prob_neg: mean predicted probability for rows
        that are truly positive vs truly negative for this class. The gap
        between these ("separation") tells you whether the model has learned
        ANY useful signal for this class, independent of where the decision
        threshold happens to sit.

    Why this matters: a class showing F1=0.000 at the default 0.50 threshold
    can look "dead," but for extreme-imbalance classes trained with a large
    pos_weight, the model's raw probability outputs are often systematically
    compressed toward 0 even once real separation exists — the true optimal
    threshold can sit well below 0.50 (Brake Locked needed 0.30, Gentle Accel
    needed 0.19 in one run). This diagnostic distinguishes "no signal yet,
    needs more training/capacity/weight" (small or negative separation) from
    "signal exists, just needs a different threshold" (real separation, but
    fixed-0.5 F1 is 0) — the two require completely different fixes.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    probs = 1 / (1 + np.exp(-all_logits))

    results = []
    for c, name in enumerate(class_names):
        y_true = all_labels[:, c]
        y_prob = probs[:, c]
        pos_mask = y_true > 0.5
        neg_mask = ~pos_mask

        mean_pos = float(y_prob[pos_mask].mean()) if pos_mask.any() else float("nan")
        mean_neg = float(y_prob[neg_mask].mean()) if neg_mask.any() else float("nan")

        best_f1, best_thr = 0.0, 0.5
        if pos_mask.any():
            for t in thresholds:
                preds = (y_prob > t).astype(int)
                f1 = f1_score(y_true, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_thr = f1, float(t)

        results.append({
            "name":            name,
            "mean_prob_pos":   mean_pos,
            "mean_prob_neg":   mean_neg,
            "separation":      (mean_pos - mean_neg) if pos_mask.any() else float("nan"),
            "best_f1":         best_f1,
            "best_threshold":  best_thr,
        })
    return results


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_epoch_metrics(all_logits: np.ndarray,
                           all_labels: np.ndarray,
                           threshold: float = 0.50) -> dict:
    """
    Compute per-class and macro F1-score from raw logits.

    Returns dict with:
        macro_f1: float
        per_class_f1: list of floats
    """
    probs = 1 / (1 + np.exp(-all_logits))   # sigmoid without torch
    preds = (probs > threshold).astype(int)

    per_class_f1 = f1_score(all_labels, preds, average=None,
                             zero_division=0).tolist()
    macro_f1     = float(np.mean(per_class_f1))
    return {"macro_f1": macro_f1, "per_class_f1": per_class_f1}


# ── Training epoch ────────────────────────────────────────────────────────────

def run_epoch(model: AttentionLSTM,
              loader,
              criterion: nn.BCEWithLogitsLoss,
              optimizer: torch.optim.Optimizer,
              device: str,
              training: bool = True) -> dict:
    """
    Run one full epoch (training or validation).

    Returns:
        dict with "loss" (float), "macro_f1" (float), "per_class_f1" (list)
    """
    model.train(training)
    total_loss   = 0.0
    all_logits   = []
    all_labels   = []

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for windows, labels in loader:
            windows = windows.to(device, non_blocking=True)  # (B, T, F)
            labels  = labels.to(device, non_blocking=True)   # (B, C)

            logits = model(windows)                           # (B, C)
            loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item() * len(windows)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    avg_loss   = total_loss / len(loader.dataset)

    metrics = compute_epoch_metrics(all_logits, all_labels)
    metrics["loss"]   = avg_loss
    metrics["logits"] = all_logits   # kept for diagnose_class_separation() — cheap, already in memory
    metrics["labels"] = all_labels
    return metrics


# ── Main trainer class ────────────────────────────────────────────────────────

class LSTMTrainer:
    """
    Manages the complete training lifecycle:
      - Class-weighted loss
      - ReduceLROnPlateau on val macro-F1
      - Early stopping (patience=10 epochs)
      - Checkpoint every epoch + best model save
      - Detailed per-epoch logging
    """

    def __init__(self,
                 model: AttentionLSTM,
                 train_loader,
                 val_loader,
                 pos_weights_dict: dict,
                 label_cols: list     = None,
                 learning_rate: float = LEARNING_RATE,
                 checkpoint_dir: str  = "/kaggle/working/checkpoints",
                 device: str          = None):

        self.model         = model
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.label_cols    = label_cols or LABEL_COLS
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = build_criterion(pos_weights_dict, self.label_cols, self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",          # maximise val macro-F1
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            #verbose=True, # This line raised a type error, still keeping it in the event I opt to use PyTorch version 2.1 and lower
        )

        """
        In PyTorch 2.2, verbose was deprecated and completely
        removed in version 2.2+
        """

        self.history       = {"train_loss": [], "val_loss": [],
                               "train_f1": [],  "val_f1": [],
                               "per_class_f1": []}
        self.best_val_f1   = 0.0
        self.patience_count = 0

    def train(self, max_epochs: int = MAX_EPOCHS) -> dict:
        """
        Execute the full training run with early stopping.

        Returns:
            self.history dict for plotting.
        """
        print(f"\n{'='*65}")
        print(f"  LSTM TRAINING  |  Device: {self.device}  |  "
              f"Max epochs: {max_epochs}")
        print(f"  Train batches: {len(self.train_loader):,}  |  "
              f"Val batches: {len(self.val_loader):,}")
        print(f"{'='*65}\n")

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()

            train_metrics = run_epoch(
                self.model, self.train_loader, self.criterion,
                self.optimizer, self.device, training=True)

            val_metrics = run_epoch(
                self.model, self.val_loader, self.criterion,
                self.optimizer, self.device, training=False)

            # LR scheduling on val macro-F1
            self.scheduler.step(val_metrics["macro_f1"])

            # Log history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_f1"].append(train_metrics["macro_f1"])
            self.history["val_f1"].append(val_metrics["macro_f1"])
            self.history["per_class_f1"].append(val_metrics["per_class_f1"])

            elapsed = time.time() - t0
            print(f"Epoch {epoch:03d}/{max_epochs}  "
                  f"train_loss={train_metrics['loss']:.4f}  "
                  f"val_loss={val_metrics['loss']:.4f}  "
                  f"train_F1={train_metrics['macro_f1']:.4f}  "
                  f"val_F1={val_metrics['macro_f1']:.4f}  "
                  f"({elapsed:.1f}s)")

            # Per-class F1 every 5 epochs
            if epoch % 5 == 0:
                self._print_per_class_f1(val_metrics["per_class_f1"], epoch)
                self._print_class_separation(val_metrics["logits"], val_metrics["labels"])

            # Save checkpoint every epoch
            self._save_checkpoint(epoch, val_metrics["macro_f1"])

            # Best model
            if val_metrics["macro_f1"] > self.best_val_f1:
                self.best_val_f1   = val_metrics["macro_f1"]
                self.patience_count = 0
                self._save_best_model()
                print(f"  ★ New best val macro-F1: {self.best_val_f1:.4f}")
            else:
                self.patience_count += 1
                if self.patience_count >= PATIENCE:
                    print(f"\n[Train] Early stopping at epoch {epoch} "
                          f"(no improvement for {PATIENCE} epochs)")
                    break

        print(f"\n[Train] Training complete. Best val macro-F1: {self.best_val_f1:.4f}")
        self._save_history()
        return self.history

    def _print_per_class_f1(self, per_class_f1: list, epoch: int):
        print(f"\n  Per-class F1 @ epoch {epoch} (fixed threshold=0.50):")
        for name, f1 in zip(CLASS_NAMES, per_class_f1):
            bar = "█" * int(f1 * 20)
            status = "✓" if f1 >= 0.75 else "✗"
            print(f"    {status} {name:<30} {f1:.3f}  {bar}")
        print()

    def _print_class_separation(self, logits: np.ndarray, labels: np.ndarray):
        """
        Diagnostic table: for each class, is F1=0.000 above meaning "no signal
        yet" or "signal exists, wrong threshold"? separation = mean predicted
        probability for true positives minus true negatives — near 0 (or
        negative) means the model genuinely hasn't learned to separate that
        class yet; a clear positive gap means it has, even if fixed-0.50 F1
        says 0.000. best_f1/best_thr show what tune_thresholds() would likely
        find at the end of training, computed live from THIS epoch's val set.
        """
        diag = diagnose_class_separation(logits, labels, CLASS_NAMES)
        print("  Class separation (is a 0.000 F1 above real, or just mis-thresholded?):")
        print(f"    {'':<2}{'Label':<30}{'sep(pos-neg)':>14}{'best_F1':>10}{'@thr':>8}")
        for d in diag:
            sep = d["separation"]
            flag = "  " if (sep == sep and sep > 0.05) else "??"  # NaN-safe check, flag weak/no separation
            sep_str = f"{sep:+.3f}" if sep == sep else "  n/a"
            print(f"    {flag}{d['name']:<30}{sep_str:>14}{d['best_f1']:>10.3f}{d['best_threshold']:>8.2f}")
        print()

    def _save_checkpoint(self, epoch: int, val_f1: float):
        path = self.checkpoint_dir / f"epoch_{epoch:03d}_f1={val_f1:.4f}.pt"
        torch.save({
            "epoch":           epoch,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_f1":          val_f1,
            "history":         self.history,
        }, path)

    def _save_best_model(self):
        path = self.checkpoint_dir / "best_model.pt"
        torch.save({
            "model_state":    self.model.state_dict(),
            "val_f1":         self.best_val_f1,
            "model_config": {
                "input_dim":   self.model.input_dim,
                "hidden_dim":  self.model.hidden_dim,
                "num_layers":  self.model.num_layers,
                "num_classes": self.model.num_classes,
            },
        }, path)
        print(f"  → Best model saved: {path}")

    def _save_history(self):
        path = self.checkpoint_dir / "training_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[Train] History saved → {path}")


# %% cell 28
# ---- LSTM TRAINING AT WORK ----

print(f"{'-'*136}\n")

# %% [markdown] cell 29
# # Constants
# constants.py — Single source of truth for column names, thresholds,
# and all shared constants used across the LSTM module pipeline.
#
# Matches project proposal §1.6 (behavior definitions) and §3.5.4 (LSTM spec).

# %% cell 30
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

# %% [markdown] cell 31
# # **LSTM Pipleline**
# lstm_pipeline.py — Master script for Kaggle execution.
#
# Runs the complete LSTM pipeline end-to-end:
#
# 1. Pre-injest all our training data and merging them
# 2. Adapt raw LFS CSV to canonical schema
# 3. Label all 7 behaviour classes
# 4. Distribution analysis + descriptive stats
# 5. Build windowed datasets with normalisation
# 6. Train AttentionLSTM with class-weighted loss
# 7. Evaluate: F1, confusion matrices, ROC curves
# 8. Diagnostics: attention heatmaps + SHAP
# 9. Save best model + normaliser + thresholds for inference
#
# Run on Kaggle: kernel type = Python, accelerator = GPU (P100).
# Mount your labelled CSV as a Kaggle Dataset before running.

# %% cell 32
# FIX: this was computing round(115/7031, 5) = 0.01636 — 115 is not a
# plausible peak-power RPM for any car (real values are ~5000-8000 RPM), so
# this ratio was ~50x too small. It was then passed as VEHICLE_POWER_PEAK
# below and used directly as label_late_upshift's above_peak threshold,
# making "RPM above power peak" true almost all the time (engine_rpm_norm
# mean ≈ 0.72 >> 0.016) — this is very likely why Late Upshift's positive
# rate (22.1%) and FP count were high. Compute the ratio from
# VEHICLE_CONFIG so there's one source of truth, and REPLACE power_peak_rpm
# in VEHICLE_CONFIG above with your car's real peak-power RPM before trusting this.
_power_peak_norm = VEHICLE_CONFIG["default"]["power_peak_rpm"] / VEHICLE_CONFIG["default"]["redline_rpm"]
print(f"Vehicle power peak (ratio): {round(_power_peak_norm, 5)}  "
      f"[VERIFY power_peak_rpm in VEHICLE_CONFIG is your real car's value]")

# %% cell 33
# ── Configure these paths before running ─────────────────────────────────────
RAW_CSV_PATH        = "/kaggle/input/datasets/samwelnjehia/lstm-module-telemetry-data"
APEX_JSON_PATH      = "/kaggle/input/lfs-telemetry/apex_reference.json"
TRACK_NAME          = "blackwood_gp"
VEHICLE_REDLINE     = VEHICLE_CONFIG["default"]["redline_rpm"]   # single source of truth now
# FIX: was hardcoded to 0.01636 (round(115/7031,5)) — 115 was not a real
# peak-power RPM, this silently broke the Late Upshift label (see cell 32
# note). Now derived from VEHICLE_CONFIG so it can't drift out of sync again.
VEHICLE_POWER_PEAK  = VEHICLE_CONFIG["default"]["power_peak_rpm"] / VEHICLE_CONFIG["default"]["redline_rpm"]
OUTPUT_DIR          = "/kaggle/working"
CHECKPOINT_DIR      = Path(f"{OUTPUT_DIR}/checkpoints")
BESTMODEL_DIR       = Path(f"{OUTPUT_DIR}/bestmodel")
EVAL_DIR            = f"{OUTPUT_DIR}/eval_output"
DIAG_DIR            = f"{OUTPUT_DIR}/diagnostics"

# Sessions to use for each split (by session_id or lap number)
# If your CSV is a single session, leave None — pipeline will auto-split by lap
TRAIN_SESSIONS      = None
VAL_SESSIONS        = None
TEST_SESSIONS       = None

# Training config
MAX_EPOCHS  = 512
BATCH_SIZE  = 256
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────

# %% cell 34
def main():
    print(f"\n{'='*65}")
    print(f"  BIMODAL DRIVER COACHING — LSTM MODULE")
    print(f"  Device: {DEVICE}")
    print(f"{'-'*65}\n")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ── Step 0: Pre-injestion ─────────────────────────────────────────────────────────
    print("[Step 0] Pre-injesting all telemetry data...")
    frankinstien_data = load_and_compile_telemetry_folder(RAW_CSV_PATH)

    # ── Step 1: Adapt ─────────────────────────────────────────────────────────
    canonical_path = f"{OUTPUT_DIR}/canonical_telemetry.csv"
    print("[Step 1] Adapting raw LFS CSV → canonical schema...")
    adapt_lfs_telemetry(frankinstien_data, vehicle="default",
                        output_path=canonical_path)

    # ── Step 2: Label ─────────────────────────────────────────────────────────
    labelled_path = f"{OUTPUT_DIR}/labelled_telemetry.csv"
    print("\n[Step 2] Applying behaviour labels...")
    run_labelling_pipeline(
        canonical_csv_path  = canonical_path,
        output_csv_path     = labelled_path,
        apex_json_path      = APEX_JSON_PATH if os.path.exists(APEX_JSON_PATH) else None,
        track_name          = TRACK_NAME,
        vehicle_redline     = VEHICLE_REDLINE,
        vehicle_power_peak_norm = VEHICLE_POWER_PEAK,
    )

    # ── Step 3: Distribution analysis ────────────────────────────────────────
    print("\n[Step 3] Running distribution analysis...")
    pos_weights = run_full_distribution_analysis(
        labelled_csv_path = labelled_path,
        output_dir        = f"{OUTPUT_DIR}/analysis",
    )

    # ── Step 4: Build datasets ────────────────────────────────────────────────
    print("\n[Step 4] Building windowed datasets...")
    norm_path = f"{OUTPUT_DIR}/normalizer_params.json"
    train_ds, val_ds, test_ds, norm_params = build_datasets(
        labelled_csv_path  = labelled_path,
        normalizer_save_path = norm_path,
        train_sessions     = TRAIN_SESSIONS,
        val_sessions       = VAL_SESSIONS,
        test_sessions      = TEST_SESSIONS,
    )

    # Wrap training dataset in augmentation
    aug_train_ds  = AugmentedTelemetryDataset(train_ds, augment_p=0.7)
    train_loader  = make_dataloader(aug_train_ds, batch_size=BATCH_SIZE,
                                    use_sampler=True)
    val_loader    = make_dataloader(val_ds,   batch_size=BATCH_SIZE)
    test_loader   = make_dataloader(test_ds,  batch_size=BATCH_SIZE)

    feature_cols_present = [c for c in FEATURE_COLS
                            if c in train_ds.feature_cols]

    # ── Step 5: Build + train model ───────────────────────────────────────────
    print("\n[Step 5] Building model...")
    model = build_model(
        input_dim   = len(feature_cols_present),
        device      = DEVICE,
    )

    print("\n[Step 5] Training...")
    trainer = LSTMTrainer(
        model            = model,
        train_loader     = train_loader,
        val_loader       = val_loader,
        pos_weights_dict = pos_weights,
        checkpoint_dir   = CHECKPOINT_DIR,
        device           = DEVICE,
    )
    history = trainer.train(max_epochs=MAX_EPOCHS)

    # Load best weights
    best_path = f"{CHECKPOINT_DIR}/best_model.pt"
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"\n[Step 5] Best model loaded (val F1 = {ckpt['val_f1']:.4f})")

    # ── Step 6: Evaluate ──────────────────────────────────────────────────────
    print("\n[Step 6] Running evaluation on test set...")
    metrics = run_full_evaluation(
        model          = model,
        test_loader    = test_loader,
        device         = DEVICE,
        history        = history,
        output_dir     = EVAL_DIR,
        feature_names  = feature_cols_present,
    )

    # Save thresholds alongside model
    _, probs, labels_all = collect_predictions(model, test_loader, DEVICE)
    optimal_thresholds, _ = tune_thresholds(probs, labels_all)
    thr_dict = dict(zip(LABEL_COLS, optimal_thresholds))
    thr_path = f"{OUTPUT_DIR}/optimal_thresholds.json"
    with open(thr_path, "w") as f:
        json.dump(thr_dict, f, indent=2)
    print(f"[Step 6] Thresholds saved → {thr_path}")

    # ── Step 7: Diagnostics ───────────────────────────────────────────────────
    print("\n[Step 7] Running diagnostics (attention + SHAP)...")
    run_full_diagnostics(
        model          = model,
        train_loader   = train_loader,
        test_loader    = test_loader,
        device         = DEVICE,
        labels_all     = labels_all,
        feature_names  = feature_cols_present,
        output_dir     = DIAG_DIR,
    )

    # ── Step 8: Print final summary ───────────────────────────────────────────
    macro_f1   = metrics.get("macro_f1", 0)
    target_met = macro_f1 >= 0.75
    print(f"\n{'='*65}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Test set macro-F1 : {macro_f1:.4f}")
    print(f"  Target (≥ 0.75)   : {'✓ MET' if target_met else '✗ NOT MET'}")
    print(f"  Best model        : {best_path}")
    print(f"  Normalizer        : {norm_path}")
    print(f"  Thresholds        : {thr_path}")
    print(f"  Eval output       : {EVAL_DIR}/")
    print(f"  Diagnostics       : {DIAG_DIR}/")
    print(f"{'='*65}\n")

    # ── Step 9: Best model download link ──────────────────────────────────────
    print("\n[Step 9] Creating download link...")
    download_bestmodel()

    return model, metrics, history

# %% [markdown] cell 35
# - LSTM pipeline coming alive

# %% cell 36
import pandas as pd
dataframe = pd.read_csv('/kaggle/working/labelled_telemetry.csv')
print(dataframe.shape)
comls = ['engine_rpm','throttle','throttle_rate','brake','brake_rate','speed_kmh','wheel_spin_lf','wheel_spin_rf','wheel_spin_lr','wheel_spin_rr','slip_ratio_lf','slip_ratio_rf','slip_ratio_lr','slip_ratio_rr']
dataframe[comls].describe()

# %% cell 37
#if __name__ == "__main__":
#    main()

# %% cell 38
download_bestmodel()

# %% cell 39
"""
Breake in case of new re-runs
- Avoid filling up the disk
"""

"""
import os
import shutil
from pathlib import Path

# Wipe the bloated checkpoints directory
checkpoint_dir = Path("/kaggle/working/checkpoints")
if checkpoint_dir.exists():
    shutil.rmtree(checkpoint_dir)
    print("Cleared corrupted/bloated checkpoints.")

# Re-create an empty clean directory
checkpoint_dir.mkdir(parents=True, exist_ok=True)
"""

# %% [markdown] cell 40
# # 16 · SHAP Sub-Module

# %% cell 41
#from src.models.lstm_model import AttentionLSTM, build_model
#from src.evaluation.lstm_diagnostics import run_full_diagnostics
import torch, json

# Load saved model
device   = "cuda" if torch.cuda.is_available() else "cpu"
ckpt     = torch.load("/kaggle/working/bestmodel/best_model.pt", map_location=device)
cfg      = ckpt["model_config"]
model    = build_model(
    input_dim   = cfg["input_dim"],
    hidden_dim  = cfg["hidden_dim"],
    num_layers  = cfg["num_layers"],
    num_classes = cfg["num_classes"],
    device      = device,
)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded — val F1: {ckpt['val_f1']:.4f}")

# %% [markdown] cell 42
# # 17 · Apex Reference Generator
#
# One-time-per-track utility: detects each corner's geometric apex (as the local speed minimum) from a single clean reference lap, and writes it in the format `load_apex_reference()` expects. Run this after capturing one clean lap; see the usage notes at the bottom of the next cell.

# %% cell 43
# ── Apex reference generator ────────────────────────────────────────────────
# Run this ONCE per track, after driving one clean reference lap (no traffic,
# consistent racing line, no off-track excursions). It finds each corner's
# apex as a local minimum in speed_kmh vs lap_dist_m and writes the JSON file
# load_apex_reference() expects: {"<track_name>": [apex_dist_m, ...]}.
#
# Prerequisite: run this lap's raw LFS CSV through Steps 0-1 (load_and_compile
# + adapt_lfs_telemetry) like any other session, so you have a canonical
# dataframe with lap_dist_m and speed_kmh columns. Point RAW_LAP_CSV_PATH at
# that adapted output (or reuse canonical_telemetry.csv and filter to one lap
# if the reference lap was captured inside a normal session).

def detect_apex_candidates(lap_df: pd.DataFrame,
                            min_prominence_kmh: float = 8.0,
                            min_apex_spacing_m: float = 60.0) -> list:
    """
    Detect corner apex distances from a single clean reference lap by finding
    local minima in speed_kmh vs lap_dist_m — a corner apex is approximately
    the point of minimum speed within that corner.

    lap_df:               canonical-schema rows for ONE lap only.
    min_prominence_kmh:   a dip must be at least this many km/h below the
                          lower of its two neighbouring local maxima to count
                          as a real corner — filters out small throttle-lift
                          noise and sub-dips inside a single corner/chicane.
                          Raise this if you get spurious extra apexes; lower
                          it if a genuine slow corner gets missed.
    min_apex_spacing_m:   minimum distance between two accepted apexes, so a
                          single wide corner isn't split into two candidates.
    """
    d = lap_df.sort_values("lap_dist_m").reset_index(drop=True)
    speed = d["speed_kmh"].to_numpy()
    dist  = d["lap_dist_m"].to_numpy()

    # Local minima: strictly lower than both immediate neighbours
    is_min = np.r_[False, (speed[1:-1] < speed[:-2]) & (speed[1:-1] < speed[2:]), False]
    candidate_idx = np.where(is_min)[0]

    # Prominence filter: how far above the dip do we have to climb on the
    # nearer side before hitting a higher point? Filters sensor jitter.
    accepted = []
    for idx in candidate_idx:
        left_max  = speed[:idx + 1].max()
        right_max = speed[idx:].max()
        prominence = min(left_max, right_max) - speed[idx]
        if prominence >= min_prominence_kmh:
            accepted.append((float(dist[idx]), float(speed[idx]), float(prominence)))

    # Merge apexes that are too close together — keep the lower-speed one
    accepted.sort(key=lambda x: x[0])
    merged = []
    for a in accepted:
        if merged and (a[0] - merged[-1][0]) < min_apex_spacing_m:
            if a[1] < merged[-1][1]:
                merged[-1] = a
        else:
            merged.append(a)

    return [round(a[0], 1) for a in merged]


def build_apex_reference_json(lap_df: pd.DataFrame,
                               track_name: str,
                               output_json_path: str,
                               existing_json_path: str = None,
                               min_prominence_kmh: float = 8.0,
                               min_apex_spacing_m: float = 60.0,
                               plot: bool = True) -> list:
    """
    Detect apexes for one track and write/merge them into an apex reference
    JSON file at output_json_path (format matches load_apex_reference()).

    existing_json_path: if you already have a reference file for OTHER
    tracks and want to add this track without losing them, pass its path
    here (can be the same as output_json_path to update in place).
    """
    apexes = detect_apex_candidates(lap_df, min_prominence_kmh, min_apex_spacing_m)
    print(f"[Apex] Detected {len(apexes)} corner apexes for '{track_name}':")
    print(f"       {apexes}")

    refs = {}
    if existing_json_path and Path(existing_json_path).exists():
        with open(existing_json_path) as f:
            refs = json.load(f)
    refs[track_name] = apexes

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(refs, f, indent=2)
    print(f"[Apex] Saved → {output_json_path}")

    if plot:
        d = lap_df.sort_values("lap_dist_m")
        plt.figure(figsize=(14, 4))
        plt.plot(d["lap_dist_m"], d["speed_kmh"], lw=1, color="steelblue")
        for a in apexes:
            plt.axvline(a, color="crimson", ls="--", lw=1, alpha=0.8)
        plt.xlabel("lap_dist_m")
        plt.ylabel("speed_kmh")
        plt.title(f"{track_name} — detected apexes (verify against the racing line before trusting these)")
        plt.tight_layout()
        plt.show()
        print("[Apex] SANITY CHECK: each dashed line should land at the slowest point of a "
              "real corner. Delete/adjust obvious false positives by hand in the JSON before use.")

    return apexes


# ── Usage ──────────────────────────────────────────────────────────────────
# 1. Drive one clean reference lap at the target track (no traffic, one clean
#    line through every corner — this lap's quality directly determines apex
#    accuracy).
# 2. Run that session's raw LFS CSV through Steps 0-1 as usual to get a
#    canonical dataframe, OR reuse an existing canonical_telemetry.csv and
#    filter to a single representative lap:
#
#      full_df = pd.read_csv(canonical_path)
#      one_lap = full_df[full_df["lap"] == REFERENCE_LAP_NUMBER]
#
# 3. Generate and save the reference (default path avoids Kaggle's read-only
#    /kaggle/input — write to /kaggle/working first):
#
#      apex_out_path = "/kaggle/working/apex_reference.json"
#      apexes = build_apex_reference_json(
#          lap_df            = one_lap,
#          track_name        = TRACK_NAME,          # "blackwood_gp" — must match exactly
#          output_json_path  = apex_out_path,
#          existing_json_path= apex_out_path,        # merges if the file already exists
#      )
#
# 4. Point the pipeline at it. /kaggle/working isn't attached as a dataset
#    input by default, so either:
#      a) for THIS session, just override the constant before calling main():
#           APEX_JSON_PATH = apex_out_path
#      b) for future sessions, upload apex_reference.json as a small Kaggle
#         Dataset and attach it as an input — then APEX_JSON_PATH (currently
#         "/kaggle/input/lfs-telemetry/apex_reference.json") will find it
#         automatically like any other input file.
#
# 5. Re-run Steps 1-3 (adapt → label → distribution) so label_trail_brake and
#    label_gentle_accel pick up the distance-based branch instead of the
#    noisier speed-rising fallback. Check the new class_distribution report —
#    Trail Brake's positive rate/quality should visibly change.


# %% cell 44
# BREAK incase of new track in training

"""
REFERENCE_LAP_DATAFILE = '/kaggle/input/datasets/samwelnjehia/blackwood-reference-lap/lfs_lstm_telemetry_1786178117.csv'
CANONICAL_REFERENCE_LAP = '/kaggle/working/canonical_reference_lap_telemetry.csv'
APEX_OUT_PATH = "/kaggle/working/apex_reference.json"

canonical_reference_lap = f"{OUTPUT_DIR}/canonical_reference_lap_telemetry.csv"
print("[Step 1] Adapting raw LFS reference lap CSV → canonical schema...")
reference_lap = pd.read_csv(REFERENCE_LAP_DATAFILE)
adapt_lfs_telemetry(reference_lap, vehicle="default",
                   output_path=canonical_reference_lap)

canonical_df = pd.read_csv(CANONICAL_REFERENCE_LAP)

one_lap = canonical_df[canonical_df["lap"] == 1]   # pick clean reference lap number
print(canonical_df.shape)

apexes = build_apex_reference_json(
    lap_df             = one_lap,
    track_name         = TRACK_NAME,
    output_json_path   = APEX_OUT_PATH,
    existing_json_path = APEX_OUT_PATH,
)

APEX_JSON_PATH = APEX_OUT_PATH   # override for this session, then call main()

"""
