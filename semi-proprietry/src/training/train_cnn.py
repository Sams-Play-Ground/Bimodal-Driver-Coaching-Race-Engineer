"""
train_cnn.py — Training loop for the CNN spatial module, intended to run on
Kaggle Notebooks (NVIDIA P100 GPU), per proposal §3.6.

Two-phase schedule: backbone frozen for the first FREEZE_EPOCHS epochs
(head-only warmup), then unfrozen for end-to-end fine-tuning at a much lower
learning rate. Loss is BCEWithLogitsLoss with per-class pos_weight from
label distribution analysis, fixing an earlier F1=0.00-on-minority-classes
issue from unweighted training.

Ported from `final-cnn-module-codes.ipynb` ("9 - Training loop").
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.data.cnn_constants import (
    CLASS_NAMES, CHECKPOINT_DIR, EPOCHS, PATIENCE, LR_HEAD, LR_FINETUNE,
    FREEZE_EPOCHS, GRAD_CLIP, WEIGHT_DECAY,
)


def build_criterion(pos_weights: dict, device: str = None) -> nn.BCEWithLogitsLoss:
    """
    Build BCEWithLogitsLoss with per-class positive weights (from
    analyse_label_distribution). Higher weight = rare positive class is
    penalised more heavily for misses, forcing the model to actually learn
    to detect it rather than defaulting to the safe majority class.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = [pos_weights.get(name, 1.0) for name in CLASS_NAMES]
    pw = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"[Loss] BCEWithLogitsLoss pos_weight: {dict(zip(CLASS_NAMES, [round(w, 1) for w in weights]))}")
    return nn.BCEWithLogitsLoss(pos_weight=pw)


def compute_metrics(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute per-class and macro F1 from raw logits."""
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > threshold).astype(int)
    per_cls = f1_score(labels, preds, average=None, zero_division=0).tolist()
    return {"macro_f1": float(np.mean(per_cls)), "per_class_f1": per_cls}


def run_epoch(model, loader, criterion, optimizer=None, training: bool = True, device: str = None) -> dict:
    """Run one full epoch. Returns loss and F1 metrics."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train(training)
    total_loss = 0.0
    all_logits, all_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, labels)

            if training and optimizer:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item() * len(imgs)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


class CNNTrainer:
    """
    Handles the full CNN training loop: backbone freezing for the first
    FREEZE_EPOCHS epochs, then end-to-end fine-tuning at LR_FINETUNE.
    Mirrors LSTMTrainer in structure and logging conventions.
    """

    def __init__(self, model, train_dl, val_dl, pos_weights, checkpoint_dir=CHECKPOINT_DIR):
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.criterion = build_criterion(pos_weights)
        self.history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": [], "per_class_f1": []}
        self.best_val_f1 = 0.0
        self.patience_cnt = 0

        # Phase 1: freeze backbone, train head only
        self.model.freeze_backbone()
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD, weight_decay=WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=8)
        self._phase = 1

    def _maybe_unfreeze(self, epoch: int):
        if epoch == FREEZE_EPOCHS and self._phase == 1:
            self.model.unfreeze_all()
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=8)
            self._phase = 2

    def run_epoch(self, data_loader, training: bool = True) -> dict:
        """Run a single epoch and return loss/metric summary."""
        return run_epoch(self.model, data_loader, self.criterion,
                          self.optimizer if training else None, training=training)

    def train(self, max_epochs: int = EPOCHS) -> dict:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n{'='*65}\n  CNN TRAINING  |  Device: {device}  |  Max epochs: {max_epochs}")
        print(f"  Train batches: {len(self.train_dl):,}  |  Val batches: {len(self.val_dl):,}")
        print(f"  Early stopping: patience={PATIENCE} on val macro-F1\n{'='*65}\n")

        for epoch in range(1, max_epochs + 1):
            self._maybe_unfreeze(epoch)
            t0 = time.time()

            tr = self.run_epoch(self.train_dl, training=True)
            vl = self.run_epoch(self.val_dl, training=False)

            self.scheduler.step(vl["macro_f1"])

            self.history["train_loss"].append(tr["loss"])
            self.history["val_loss"].append(vl["loss"])
            self.history["train_f1"].append(tr["macro_f1"])
            self.history["val_f1"].append(vl["macro_f1"])
            self.history["per_class_f1"].append(vl["per_class_f1"])

            elapsed = time.time() - t0
            print(f"Epoch {epoch:03d}/{max_epochs}  tr_loss={tr['loss']:.4f}  vl_loss={vl['loss']:.4f}  "
                  f"tr_F1={tr['macro_f1']:.4f}  vl_F1={vl['macro_f1']:.4f}  ({elapsed:.1f}s)  [Phase {self._phase}]")

            if epoch % 5 == 0:
                print(f"  Per-class val F1 @ epoch {epoch}:")
                for name, f1 in zip(CLASS_NAMES, vl["per_class_f1"]):
                    status = "OK" if f1 >= 0.75 else "--"
                    print(f"    [{status}] {name:<15} {f1:.3f}")

            self.save_checkpoint(epoch, self.checkpoint_dir, vl["macro_f1"])

            if vl["macro_f1"] > self.best_val_f1:
                self.best_val_f1 = vl["macro_f1"]
                self.patience_cnt = 0
                self._save_best()
                print(f"  New best val macro-F1: {self.best_val_f1:.4f}")
            else:
                self.patience_cnt += 1
                if self.patience_cnt >= PATIENCE:
                    print(f"\n[Train] Early stopping at epoch {epoch}")
                    break

        print(f"\n[Train] Complete. Best val macro-F1: {self.best_val_f1:.4f}")
        self._save_history()
        return self.history

    def save_checkpoint(self, epoch: int, checkpoint_dir, val_f1: float = None) -> str:
        """Save model weights to the checkpoints directory after each epoch."""
        path = Path(checkpoint_dir) / f"epoch_{epoch:03d}_f1={val_f1:.4f}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_f1": val_f1,
            "model_config": {"backbone": self.model.backbone_name, "num_classes": self.model.num_classes},
        }, path)
        return str(path)

    def _save_best(self):
        path = self.checkpoint_dir / "best_model.pt"
        torch.save({
            "model_state": self.model.state_dict(),
            "val_f1": self.best_val_f1,
            "model_config": {"backbone": self.model.backbone_name, "num_classes": self.model.num_classes},
            "class_names": CLASS_NAMES,
            "notes": "CNN spatial module — EfficientNet-B0, 3-output, class-weighted loss",
        }, path)
        print(f"  -> Best model saved: {path}")

    def _save_history(self):
        with open(self.checkpoint_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)


def load_dataset_from_kaggle(dataset_path: str):
    """Load the mounted Kaggle Dataset containing preprocessed frames + labels (see src.data.cnn_dataset)."""
    from src.data.cnn_dataset import build_dataloaders
    return build_dataloaders(labels_path=Path(dataset_path) / "labels_cnn_3output.csv",
                              frames_dir=Path(dataset_path) / "frames_fullres")
