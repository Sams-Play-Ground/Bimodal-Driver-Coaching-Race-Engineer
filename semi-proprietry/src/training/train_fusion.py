"""
train_fusion.py — Training and evaluation loop for the Late Fusion Head
(proposal §3.5.5, §3.6). Trains on fusion_training_data.csv, a tabular
dataset of pre-computed CNN + LSTM probabilities.

Ported from `final-cnn-module-codes.ipynb` ("16.4 Late Fusion Pipeline",
SECTION 3-4: Training + Evaluation).
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

from src.fusion.fusion_constants import (
    CLASS_NAMES, LABEL_COLS, INPUT_DIM, NUM_CLASSES, LR, WEIGHT_DECAY,
    MAX_EPOCHS, PATIENCE, GRAD_CLIP, CHECKPOINT_DIR, EVAL_DIR, FUSION_DATA_CSV,
)
from src.fusion.late_fusion_head import LateFusionHead, build_dataloaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_epoch(model, loader, criterion, optimizer=None, training: bool = True) -> dict:
    """One training or validation epoch. Returns loss and macro-F1."""
    model.train(training)
    total_loss = 0.0
    all_logits, all_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)

            if training and optimizer:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item() * len(x)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

    logits_arr = np.concatenate(all_logits, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0).astype(int)
    probs_arr = 1 / (1 + np.exp(-logits_arr))
    preds_arr = (probs_arr > 0.5).astype(int)

    per_cls = f1_score(labels_arr, preds_arr, average=None, zero_division=0)
    return {"loss": total_loss / len(loader.dataset), "macro_f1": float(np.mean(per_cls)), "per_class_f1": per_cls.tolist()}


def train_fusion_head(csv_path: str = FUSION_DATA_CSV, checkpoint_dir: Path = CHECKPOINT_DIR) -> LateFusionHead:
    """Full training loop for the Late Fusion Head. Returns the best model loaded from checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}\n  LATE FUSION HEAD TRAINING  |  Device: {DEVICE}\n{'='*55}\n")

    train_dl, val_dl, train_ds = build_dataloaders(csv_path)

    model = LateFusionHead().to(DEVICE)
    pw = train_ds.pos_weights().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=6)

    best_val_f1, patience_cnt = 0.0, 0
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        tr = run_epoch(model, train_dl, criterion, optimizer, training=True)
        vl = run_epoch(model, val_dl, criterion, training=False)
        scheduler.step(vl["macro_f1"])

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(vl["loss"])
        history["train_f1"].append(tr["macro_f1"])
        history["val_f1"].append(vl["macro_f1"])

        print(f"Epoch {epoch:03d}/{MAX_EPOCHS}  tr_loss={tr['loss']:.4f}  vl_loss={vl['loss']:.4f}  "
              f"tr_F1={tr['macro_f1']:.4f}  vl_F1={vl['macro_f1']:.4f}  ({time.time()-t0:.1f}s)")

        if epoch % 10 == 0:
            print("  Per-class val F1:")
            for name, f1 in zip(CLASS_NAMES, vl["per_class_f1"]):
                ok = "OK" if f1 >= 0.75 else "--"
                print(f"    [{ok}] {name:<22} {f1:.3f}")

        torch.save({
            "epoch": epoch, "model_state": model.state_dict(), "val_f1": vl["macro_f1"],
            "model_config": {"input_dim": INPUT_DIM, "num_classes": NUM_CLASSES},
        }, checkpoint_dir / f"epoch_{epoch:03d}.pt")

        if vl["macro_f1"] > best_val_f1:
            best_val_f1, patience_cnt = vl["macro_f1"], 0
            torch.save({
                "model_state": model.state_dict(), "val_f1": best_val_f1,
                "model_config": {"input_dim": INPUT_DIM, "num_classes": NUM_CLASSES},
                "class_names": CLASS_NAMES,
            }, checkpoint_dir / "best_model.pt")
            print(f"  New best val macro-F1: {best_val_f1:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n[Train] Early stopping at epoch {epoch}")
                break

    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    ckpt = torch.load(checkpoint_dir / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"\n[Train] Done. Best val macro-F1: {best_val_f1:.4f}")
    return model


@torch.no_grad()
def collect_predictions(model, loader) -> tuple:
    """Collect (probs, labels) numpy arrays from a DataLoader."""
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        probs = torch.sigmoid(model(x.to(DEVICE))).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0).astype(int)


def tune_thresholds(probs: np.ndarray, labels: np.ndarray, n: int = 100) -> list:
    """Per-class decision threshold that maximises F1 on the validation set."""
    labels = labels.astype(int)
    thresholds = []
    for i in range(probs.shape[1]):
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.1, 0.9, n):
            f1 = f1_score(labels[:, i], (probs[:, i] > t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(round(best_t, 3))
    return thresholds


def evaluate_fusion_head(model, val_dl, history: dict = None, output_dir: Path = EVAL_DIR) -> tuple:
    """Full evaluation suite: metrics table, confusion matrices, training curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probs, labels = collect_predictions(model, val_dl)
    thresholds = tune_thresholds(probs, labels)

    with open(output_dir / "optimal_thresholds.json", "w") as f:
        json.dump(dict(zip(LABEL_COLS, thresholds)), f, indent=2)

    preds = np.stack([(probs[:, i] > t).astype(int) for i, t in enumerate(thresholds)], axis=1)

    print(f"\n{'='*60}\n  FUSION HEAD EVALUATION\n  {'Class':<24} {'F1':>6} {'AUC':>6}\n  {'-'*40}")
    f1s = []
    for i, name in enumerate(CLASS_NAMES):
        f1 = f1_score(labels[:, i], preds[:, i], zero_division=0)
        try:
            auc = roc_auc_score(labels[:, i], probs[:, i])
        except ValueError:
            auc = float("nan")
        f1s.append(f1)
        ok = "OK" if f1 >= 0.75 else "--"
        print(f"  [{ok}] {name:<24} {f1:>5.3f} {auc:>6.3f}")

    macro_f1 = float(np.mean(f1s))
    print(f"  {'-'*40}\n  {'MACRO F1':<24} {macro_f1:>5.3f}")
    print(f"  Target >= 0.75: {'MET' if macro_f1 >= 0.75 else 'NOT MET'}\n{'='*60}\n")

    _plot_confusion_matrices(preds, labels, output_dir)
    if history:
        _plot_training_curves(history, output_dir)

    print(f"[Eval] Plots saved -> {output_dir}/")
    return thresholds, probs, labels


def _plot_confusion_matrices(preds, labels, output_dir: Path):
    n_cols = 5
    n_rows = int(np.ceil(NUM_CLASSES / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()
    i = 0
    for i, name in enumerate(CLASS_NAMES):
        cm = confusion_matrix(labels[:, i], preds[:, i], labels=[0, 1])
        axes[i].imshow(cm, cmap="Blues")
        axes[i].set_title(name, fontsize=8)
        for r in range(2):
            for c in range(2):
                axes[i].text(c, r, f"{cm[r, c]:,}", ha="center", va="center", fontsize=10,
                              fontweight="bold", color="white" if cm[r, c] > cm.max() / 2 else "black")
        axes[i].set_xticks([0, 1]); axes[i].set_yticks([0, 1])
        axes[i].set_xticklabels(["Pred 0", "Pred 1"], fontsize=7)
        axes[i].set_yticklabels(["True 0", "True 1"], fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Fusion Head - Confusion Matrices", fontsize=11)
    plt.tight_layout()
    plt.savefig(str(output_dir / "confusion_matrices.png"), dpi=130, bbox_inches="tight")
    plt.close()


def _plot_training_curves(history: dict, output_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(history["train_loss"]) + 1)
    ax1.plot(ep, history["train_loss"], label="Train")
    ax1.plot(ep, history["val_loss"], label="Val", linestyle="--")
    ax1.set_title("Loss"); ax1.legend()
    ax2.plot(ep, history["train_f1"], label="Train")
    ax2.plot(ep, history["val_f1"], label="Val", linestyle="--")
    ax2.axhline(0.75, color="red", linestyle=":", alpha=0.6, label="Target")
    ax2.set_title("Macro F1"); ax2.legend()
    plt.tight_layout()
    plt.savefig(str(output_dir / "training_curves.png"), dpi=130, bbox_inches="tight")
    plt.close()
