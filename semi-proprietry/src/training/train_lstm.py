"""
train_lstm.py — Training loop for the LSTM temporal module, intended to run
on Kaggle Notebooks (NVIDIA P100/T4 GPU), per proposal §3.6.

Loss is class-weighted BCEWithLogitsLoss by default; FocalLossWithLogits is
available as a toggle (USE_FOCAL_LOSS in constants.py) for the most extreme-
imbalance classes. Includes a per-epoch class-separation diagnostic that
distinguishes "no signal yet" from "signal exists, wrong threshold" — useful
for classes as rare as Brake Locked (~0.5% base rate).

Ported from `final-lstm-module-codes.ipynb` ("LSTM Module - Trainer").
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.data.constants import (
    LABEL_COLS, CLASS_NAMES, GRAD_CLIP, MAX_EPOCHS, PATIENCE, LEARNING_RATE,
    USE_FOCAL_LOSS, FOCAL_GAMMA, FOCAL_ALPHA,
)
from src.models.lstm_model import AttentionLSTM


class FocalLossWithLogits(nn.Module):
    """
    Multi-label focal loss (Lin et al. 2017), built on the same per-class
    pos_weight signal build_criterion already computes.

    Standard BCE+pos_weight upweights rare-class positives in the loss, but
    every example still contributes ~equally once weighted — easy examples
    still dominate the gradient by sheer volume. Focal loss additionally
    down-weights easy/confident examples via (1-p_t)^gamma, focusing
    training on hard/rare/ambiguous ones. Toggle via USE_FOCAL_LOSS —
    compare val macro-F1 against a plain-BCE run before committing to it.
    """

    def __init__(self, pos_weight: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_w = (1 - p_t).clamp(min=1e-6) ** self.gamma

        alpha_t = self.alpha * self.pos_weight * targets + (1 - self.alpha) * (1 - targets)

        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (alpha_t * focal_w * bce).mean()


def build_criterion(pos_weights_dict: dict, label_cols: list, device: str):
    """
    Build the training loss with class-specific positive weights.

    pos_weights_dict: e.g. {"label_brake_locked": 47.3, "label_wheel_spin": 12.1, ...}
    Returns FocalLossWithLogits if USE_FOCAL_LOSS else plain BCEWithLogitsLoss —
    both consumed identically by run_epoch (criterion(logits, labels)).
    """
    weights = [min(float(pos_weights_dict.get(col, 1.0)), 50.0) for col in label_cols]  # capped at 50x
    pos_weight = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"[Loss] pos_weights: {dict(zip(CLASS_NAMES, [round(w, 1) for w in weights]))}")

    if USE_FOCAL_LOSS:
        print(f"[Loss] Using FocalLossWithLogits (gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA})")
        return FocalLossWithLogits(pos_weight, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def diagnose_class_separation(all_logits: np.ndarray, all_labels: np.ndarray,
                               class_names: list, thresholds: np.ndarray = None) -> list:
    """
    Per-class diagnostic beyond the fixed threshold=0.50 F1. For each class:
      - best_f1 / best_threshold: best F1 at any threshold (coarse sweep).
      - separation: mean predicted prob for true positives minus true
        negatives — tells you whether the model has learned ANY useful
        signal for this class, independent of where the decision threshold
        sits. Distinguishes "no signal yet" from "signal exists, needs a
        different threshold" for extreme-imbalance classes.
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
            "name": name, "mean_prob_pos": mean_pos, "mean_prob_neg": mean_neg,
            "separation": (mean_pos - mean_neg) if pos_mask.any() else float("nan"),
            "best_f1": best_f1, "best_threshold": best_thr,
        })
    return results


def compute_epoch_metrics(all_logits: np.ndarray, all_labels: np.ndarray, threshold: float = 0.50) -> dict:
    """Compute per-class and macro F1-score from raw logits."""
    probs = 1 / (1 + np.exp(-all_logits))
    preds = (probs > threshold).astype(int)
    per_class_f1 = f1_score(all_labels, preds, average=None, zero_division=0).tolist()
    return {"macro_f1": float(np.mean(per_class_f1)), "per_class_f1": per_class_f1}


def run_epoch(model: AttentionLSTM, loader, criterion, optimizer, device: str, training: bool = True) -> dict:
    """Run one full epoch (training or validation)."""
    model.train(training)
    total_loss = 0.0
    all_logits, all_labels = [], []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for windows, labels in loader:
            windows = windows.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(windows)
            loss = criterion(logits, labels)

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

    metrics = compute_epoch_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(loader.dataset)
    metrics["logits"] = all_logits
    metrics["labels"] = all_labels
    return metrics


class LSTMTrainer:
    """
    Handles the full LSTM training loop over telemetry sliding windows:
      - Class-weighted (or focal) loss
      - ReduceLROnPlateau on val macro-F1
      - Early stopping
      - Checkpoint every epoch + best model save
      - Per-epoch class-separation diagnostics
    """

    def __init__(self, model: AttentionLSTM, train_loader, val_loader, pos_weights_dict: dict,
                 label_cols: list = None, learning_rate: float = LEARNING_RATE,
                 checkpoint_dir: str = "/kaggle/working/checkpoints", device: str = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.label_cols = label_cols or LABEL_COLS
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = build_criterion(pos_weights_dict, self.label_cols, self.device)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6,
        )

        self.history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": [], "per_class_f1": []}
        self.best_val_f1 = 0.0
        self.patience_count = 0

    def run_epoch(self, data_loader, training: bool = True) -> dict:
        """Run a single epoch and return loss/metric summary."""
        return run_epoch(self.model, data_loader, self.criterion, self.optimizer, self.device, training=training)

    def train(self, max_epochs: int = MAX_EPOCHS) -> dict:
        """Execute the full training run with early stopping."""
        print(f"\n{'='*65}\n  LSTM TRAINING  |  Device: {self.device}  |  Max epochs: {max_epochs}")
        print(f"  Train batches: {len(self.train_loader):,}  |  Val batches: {len(self.val_loader):,}\n{'='*65}\n")

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()

            train_metrics = self.run_epoch(self.train_loader, training=True)
            val_metrics = self.run_epoch(self.val_loader, training=False)

            self.scheduler.step(val_metrics["macro_f1"])

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_f1"].append(train_metrics["macro_f1"])
            self.history["val_f1"].append(val_metrics["macro_f1"])
            self.history["per_class_f1"].append(val_metrics["per_class_f1"])

            elapsed = time.time() - t0
            print(f"Epoch {epoch:03d}/{max_epochs}  train_loss={train_metrics['loss']:.4f}  "
                  f"val_loss={val_metrics['loss']:.4f}  train_F1={train_metrics['macro_f1']:.4f}  "
                  f"val_F1={val_metrics['macro_f1']:.4f}  ({elapsed:.1f}s)")

            if epoch % 5 == 0:
                self._print_per_class_f1(val_metrics["per_class_f1"], epoch)
                self._print_class_separation(val_metrics["logits"], val_metrics["labels"])

            self.save_checkpoint(epoch, self.checkpoint_dir, val_metrics["macro_f1"])

            if val_metrics["macro_f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["macro_f1"]
                self.patience_count = 0
                self._save_best_model()
                print(f"  New best val macro-F1: {self.best_val_f1:.4f}")
            else:
                self.patience_count += 1
                if self.patience_count >= PATIENCE:
                    print(f"\n[Train] Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                    break

        print(f"\n[Train] Training complete. Best val macro-F1: {self.best_val_f1:.4f}")
        self._save_history()
        return self.history

    def _print_per_class_f1(self, per_class_f1: list, epoch: int):
        print(f"\n  Per-class F1 @ epoch {epoch} (fixed threshold=0.50):")
        for name, f1 in zip(CLASS_NAMES, per_class_f1):
            status = "OK" if f1 >= 0.75 else "--"
            print(f"    [{status}] {name:<30} {f1:.3f}")
        print()

    def _print_class_separation(self, logits: np.ndarray, labels: np.ndarray):
        """
        Is a 0.000 F1 above real (model hasn't learned this class yet), or
        just mis-thresholded (real separation exists at a different cutoff)?
        """
        diag = diagnose_class_separation(logits, labels, CLASS_NAMES)
        print("  Class separation:")
        print(f"    {'':<2}{'Label':<30}{'sep(pos-neg)':>14}{'best_F1':>10}{'@thr':>8}")
        for d in diag:
            sep = d["separation"]
            flag = "  " if (sep == sep and sep > 0.05) else "??"
            sep_str = f"{sep:+.3f}" if sep == sep else "  n/a"
            print(f"    {flag}{d['name']:<30}{sep_str:>14}{d['best_f1']:>10.3f}{d['best_threshold']:>8.2f}")
        print()

    def save_checkpoint(self, epoch: int, checkpoint_dir, val_f1: float = None) -> str:
        """Save model weights to the checkpoints directory after each epoch."""
        path = Path(checkpoint_dir) / f"epoch_{epoch:03d}_f1={val_f1:.4f}.pt"
        torch.save({
            "epoch": epoch, "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(), "val_f1": val_f1, "history": self.history,
        }, path)
        return str(path)

    def _save_best_model(self):
        path = self.checkpoint_dir / "best_model.pt"
        torch.save({
            "model_state": self.model.state_dict(), "val_f1": self.best_val_f1,
            "model_config": {
                "input_dim": self.model.input_dim, "hidden_dim": self.model.hidden_dim,
                "num_layers": self.model.num_layers, "num_classes": self.model.num_classes,
            },
        }, path)
        print(f"  -> Best model saved: {path}")

    def _save_history(self):
        path = self.checkpoint_dir / "training_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[Train] History saved -> {path}")
