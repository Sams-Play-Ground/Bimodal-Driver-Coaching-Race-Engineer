"""
module_metrics.py — Module-level evaluation for the CNN and LSTM models:
per-class F1/precision/recall/ROC-AUC, macro-F1 against the proposal's
0.75 target, confusion matrices, ROC curves, and confidently-wrong error
analysis, per proposal §3.7.

Ported from `final-lstm-module-codes.ipynb` ("LSTM Module - Evaluator").
The CNN module reuses these same functions (see notebooks/ cell 10 for its
original standalone copy) — behaviour is identical, only CLASS_NAMES differ.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix,
)

from src.data.constants import CLASS_NAMES, LABEL_COLS


@torch.no_grad()
def collect_predictions(model, loader, device: str) -> tuple:
    """Run inference on a DataLoader; return (logits, probs, labels) as numpy arrays."""
    model.eval()
    all_logits, all_labels = [], []

    for windows, labels in loader:
        windows = windows.to(device, non_blocking=True)
        logits = model(windows)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())

    logits_arr = np.concatenate(all_logits, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    probs_arr = 1 / (1 + np.exp(-logits_arr))
    return logits_arr, probs_arr, labels_arr


def tune_thresholds(probs: np.ndarray, labels: np.ndarray, n_thresholds: int = 100) -> tuple:
    """Find the decision threshold maximising F1 for each class independently."""
    optimal_thresholds, threshold_f1s = [], []

    for cls_idx in range(probs.shape[1]):
        best_f1, best_thr = 0.0, 0.5
        for thr in np.linspace(0.1, 0.9, n_thresholds):
            preds = (probs[:, cls_idx] > thr).astype(int)
            f1 = f1_score(labels[:, cls_idx], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        optimal_thresholds.append(round(best_thr, 3))
        threshold_f1s.append(round(best_f1, 4))
        name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
        print(f"  {name:<30}  optimal_threshold={best_thr:.2f}  best_F1={best_f1:.4f}")

    return optimal_thresholds, threshold_f1s


def compute_precision_recall_f1(y_true, y_pred, class_name: str = "") -> dict:
    """Compute precision, recall, and F1-score for a single event class."""
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "class_name": class_name,
    }


def compute_macro_f1(per_class_f1_scores) -> float:
    """Average F1 scores across all event classes to produce macro-F1."""
    values = list(per_class_f1_scores.values()) if isinstance(per_class_f1_scores, dict) else list(per_class_f1_scores)
    return float(np.mean(values))


def compute_confusion_matrix(y_true, y_pred, class_name: str = ""):
    """Generate a 2x2 confusion matrix for a single event class."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def check_against_threshold(macro_f1: float, threshold: float = 0.75) -> bool:
    """Verify whether the macro-F1 score meets the proposal's acceptance threshold."""
    return macro_f1 >= threshold


def compute_full_metrics(probs: np.ndarray, labels: np.ndarray, thresholds: list = None,
                          class_names: list = None, output_path: str = "evaluation_metrics.json") -> dict:
    """Full per-class and aggregate metrics table. Uses optimal thresholds if provided, else 0.5."""
    class_names = class_names or CLASS_NAMES
    if thresholds is None:
        thresholds = [0.5] * probs.shape[1]

    preds = np.stack([(probs[:, i] > thr).astype(int) for i, thr in enumerate(thresholds)], axis=1)

    results = {}
    print(f"\n{'='*70}\n  EVALUATION METRICS\n  {'Class':<30} {'F1':>6} {'Prec':>6} {'Rec':>6} {'AUC':>6}\n  {'-'*55}")

    f1_scores = []
    for i, name in enumerate(class_names):
        f1 = f1_score(labels[:, i], preds[:, i], zero_division=0)
        prec = precision_score(labels[:, i], preds[:, i], zero_division=0)
        rec = recall_score(labels[:, i], preds[:, i], zero_division=0)
        try:
            auc = roc_auc_score(labels[:, i], probs[:, i])
        except ValueError:
            auc = float("nan")

        f1_scores.append(f1)
        results[name] = {"f1": round(f1, 4), "precision": round(prec, 4),
                          "recall": round(rec, 4), "roc_auc": round(auc, 4), "threshold": thresholds[i]}

        status = "OK" if f1 >= 0.75 else "--"
        print(f"  [{status}] {name:<30} {f1:>5.3f} {prec:>6.3f} {rec:>6.3f} {auc:>6.3f}")

    macro_f1 = float(np.mean(f1_scores))
    results["macro_f1"] = round(macro_f1, 4)

    print(f"  {'-'*55}\n  {'MACRO F1':<30} {macro_f1:>5.3f}")
    print(f"  Target (>= 0.75): {'MET' if check_against_threshold(macro_f1) else 'NOT MET'}\n{'='*70}\n")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Eval] Metrics saved -> {output_path}")

    return results


def plot_confusion_matrices(preds: np.ndarray, labels: np.ndarray, class_names: list = None,
                             output_path: str = "confusion_matrices.png") -> None:
    """Per-class binary confusion matrix grid."""
    class_names = class_names or CLASS_NAMES
    n_classes = labels.shape[1]
    n_cols = 4
    n_rows = int(np.ceil(n_classes / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i in range(n_classes):
        cm = confusion_matrix(labels[:, i], preds[:, i], labels=[0, 1])
        ax = axes[i]
        ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"]); ax.set_yticklabels(["True 0", "True 1"])
        ax.set_title(class_names[i], fontsize=9)
        for row in range(2):
            for col in range(2):
                ax.text(col, row, f"{cm[row, col]:,}", ha="center", va="center",
                        color="white" if cm[row, col] > cm.max() / 2 else "black", fontsize=10, fontweight="bold")

    for j in range(n_classes, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Per-Class Confusion Matrices", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Confusion matrices saved -> {output_path}")


def plot_roc_curves(probs: np.ndarray, labels: np.ndarray, class_names: list = None,
                     output_path: str = "roc_curves.png") -> None:
    """Multi-class ROC curve grid — one panel per behaviour class."""
    class_names = class_names or CLASS_NAMES
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
        axes[i].plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f"AUC={auc:.3f}")
        axes[i].plot([0, 1], [0, 1], "k--", alpha=0.3)
        axes[i].set_title(class_names[i], fontsize=9)
        axes[i].set_xlabel("FPR", fontsize=8); axes[i].set_ylabel("TPR", fontsize=8)
        axes[i].legend(fontsize=8)

    for j in range(n_classes, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("ROC Curves", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[Eval] ROC curves saved -> {output_path}")


def plot_training_curves(history: dict, output_path: str = "training_curves.png") -> None:
    """Loss and macro-F1 training curves over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val", lw=2, linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Training & Validation Loss"); ax1.legend()

    ax2.plot(epochs, history["train_f1"], label="Train Macro-F1", lw=2)
    ax2.plot(epochs, history["val_f1"], label="Val Macro-F1", lw=2, linestyle="--")
    ax2.axhline(0.75, color="red", linestyle=":", alpha=0.6, label="Target (0.75)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro F1-Score"); ax2.set_title("Training & Validation Macro-F1"); ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Training curves saved -> {output_path}")


def evaluate_module_on_test_set(model, test_loader, device: str = None, history: dict = None,
                                 output_dir: str = "./eval_output", class_names: list = None) -> dict:
    """
    Run the complete evaluation suite (threshold tuning, metrics table,
    confusion matrices, ROC curves, training curves) on the held-out
    Nurburgring test set and return the metrics dict.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    class_names = class_names or CLASS_NAMES
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Eval] Running evaluation on test set...")
    logits, probs, labels = collect_predictions(model, test_loader, device)

    print("[Eval] Tuning decision thresholds...")
    optimal_thresholds, _ = tune_thresholds(probs, labels)

    thr_dict = dict(zip(LABEL_COLS[:len(optimal_thresholds)], optimal_thresholds))
    with open(output_dir / "optimal_thresholds.json", "w") as f:
        json.dump(thr_dict, f, indent=2)

    preds_tuned = np.stack([(probs[:, i] > t).astype(int) for i, t in enumerate(optimal_thresholds)], axis=1)

    metrics = compute_full_metrics(probs, labels, thresholds=optimal_thresholds, class_names=class_names,
                                    output_path=str(output_dir / "evaluation_metrics.json"))

    plot_confusion_matrices(preds_tuned, labels, class_names, str(output_dir / "confusion_matrices.png"))
    plot_roc_curves(probs, labels, class_names, str(output_dir / "roc_curves.png"))

    if history:
        plot_training_curves(history, str(output_dir / "training_curves.png"))

    return metrics
