"""
late_fusion_head.py — Learned MLP that fuses 3 CNN spatial probabilities +
7 LSTM temporal probabilities into 10 coaching-relevant event classes,
per proposal §3.5.5.

A learned head (rather than simple averaging) can capture that
"apex miss (CNN) co-occurring with late braking (LSTM)" is a more
significant compound event than either alone, and learns which stream to
trust more per class.

Ported from `final-cnn-module-codes.ipynb` ("16.4 Late Fusion Pipeline",
SECTION 1-2: Dataset + Model).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.fusion.fusion_constants import INPUT_COLS, LABEL_COLS, INPUT_DIM, NUM_CLASSES, BATCH_SIZE, MODALITY_DROPOUT_P


class FusionDataset(Dataset):
    """
    Wraps fusion_training_data.csv (produced by fusion_data_generator —
    see notebooks/ for the original data-assembly cell).

    Each sample:
        x: (10,) float32 — concatenated CNN + LSTM sigmoid probabilities
        y: (10,) float32 — multi-label ground truth (BCEWithLogitsLoss target)

    Modality Dropout (proposal §3.5.5):
        During training, randomly zero the entire CNN stream (first 3
        values) OR the entire LSTM stream (last 7 values) with probability
        MODALITY_DROPOUT_P each. Forces the fusion head to remain functional
        when one stream is absent — e.g. user uploads only telemetry, no
        video, or vice versa.
    """

    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.x = df[INPUT_COLS].values.astype(np.float32)
        self.y = df[LABEL_COLS].values.astype(np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple:
        x = self.x[idx].copy()
        y = self.y[idx]

        if self.augment:
            r = np.random.random()
            if r < MODALITY_DROPOUT_P:
                x[:3] = 0.0    # drop CNN stream
            elif r < MODALITY_DROPOUT_P * 2:
                x[3:] = 0.0    # drop LSTM stream

        return torch.from_numpy(x), torch.from_numpy(y)

    def pos_weights(self) -> torch.Tensor:
        """Compute BCEWithLogitsLoss pos_weight from label distribution, capped at 50x."""
        n = len(self.y)
        n_pos = self.y.sum(axis=0)
        n_neg = n - n_pos
        pw = np.minimum(n_neg / np.maximum(n_pos, 1), 50.0)
        return torch.tensor(pw, dtype=torch.float32)


def build_dataloaders(csv_path: str, val_frac: float = 0.15, batch_size: int = BATCH_SIZE) -> tuple:
    """
    Load fusion CSV, 85/15 random split, return train/val DataLoaders.
    Fusion data is tabular — row-level split is fine here since there is no
    frame-level data leakage risk (probabilities are already aggregated).
    """
    df = pd.read_csv(csv_path)
    n = len(df)
    idx = np.random.permutation(n)
    cut = int(n * (1 - val_frac))

    train_df = df.iloc[idx[:cut]].reset_index(drop=True)
    val_df = df.iloc[idx[cut:]].reset_index(drop=True)

    train_ds = FusionDataset(train_df, augment=True)
    val_ds = FusionDataset(val_df, augment=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"[Data] Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    return train_dl, val_dl, train_ds


class LateFusionHead(nn.Module):
    """
    Two-layer MLP over the concatenated CNN+LSTM probability vector.

    Architecture:
        10-dim input -> Linear(64) -> ReLU -> Dropout(0.3)
                     -> Linear(32) -> ReLU
                     -> Linear(10) -> raw logits
    """

    def __init__(self, input_dim: int = INPUT_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),  # raw logits — sigmoid applied by loss / at inference
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 10) -> logits: (batch, 10)"""
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid probabilities for inference."""
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self(x))
