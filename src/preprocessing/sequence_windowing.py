"""
sequence_windowing.py — Constructs the 60-timestep sliding windows used as
LSTM input, per proposal §3.5.4. One window equals one second of driving
at the ~59Hz post-resample rate.

Ported from `final-lstm-module-codes.ipynb` ("LSTM Data - Windowing").
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.data.constants import FEATURE_COLS, LABEL_COLS, WINDOW_SIZE, TRAIN_STRIDE, EVAL_STRIDE, BATCH_SIZE
from src.preprocessing.telemetry_normalization import (
    fit_normalizer, apply_normalizer, forward_fill_missing_values,
    session_split, lap_split,
)


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

    def __init__(self, df: pd.DataFrame, feature_cols: list = None, label_cols: list = None,
                 window_size: int = WINDOW_SIZE, stride: int = TRAIN_STRIDE):
        self.feature_cols = [c for c in (feature_cols or FEATURE_COLS) if c in df.columns]
        self.label_cols = [c for c in (label_cols or LABEL_COLS) if c in df.columns]
        self.window_size = window_size

        features = df[self.feature_cols].values.astype(np.float32)  # (N, F)
        labels = df[self.label_cols].values.astype(np.float32)      # (N, C)

        self.windows, self.labels = [], []
        for start in range(0, len(df) - window_size + 1, stride):
            end = start + window_size
            self.windows.append(features[start:end])
            self.labels.append(labels[end - 1])

        self.windows = np.stack(self.windows)   # (n_windows, W, F)
        self.labels = np.stack(self.labels)     # (n_windows, C)

        print(f"[Dataset] {len(self.windows):,} windows | shape: {self.windows.shape} | labels: {self.labels.shape}")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple:
        return torch.from_numpy(self.windows[idx]), torch.from_numpy(self.labels[idx])

    def get_sample_weights(self) -> torch.Tensor:
        """
        Per-sample weight for WeightedRandomSampler. Windows containing at
        least one rare-event positive label get higher weight, counteracting
        class imbalance in the DataLoader. Weight = max pos_weight across
        the window's active labels.
        """
        pos_counts = self.labels.sum(axis=0)
        neg_counts = len(self.labels) - pos_counts
        pos_weights = neg_counts / np.maximum(pos_counts, 1)

        sample_weights = np.ones(len(self.labels))
        for i, label_vec in enumerate(self.labels):
            active = label_vec > 0.5
            if active.any():
                sample_weights[i] = pos_weights[active].max()

        return torch.tensor(sample_weights, dtype=torch.float32)


def make_dataloader(dataset: TelemetryWindowDataset, batch_size: int = BATCH_SIZE,
                     shuffle: bool = False, use_sampler: bool = False, num_workers: int = 0) -> DataLoader:
    """
    Wrap a TelemetryWindowDataset in a DataLoader. use_sampler=True enables
    WeightedRandomSampler to oversample minority-class windows — use this
    for the training loader only; leave False for val/test to keep an
    unbiased evaluation.
    """
    sampler = None
    if use_sampler:
        weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )


def build_datasets(labelled_csv_path: str, normalizer_save_path: str = "normalizer_params.json",
                    train_sessions: list = None, val_sessions: list = None, test_sessions: list = None,
                    train_stride: int = TRAIN_STRIDE, eval_stride: int = EVAL_STRIDE) -> tuple:
    """
    Full pipeline: load labelled CSV -> split -> normalise -> build datasets.

    Returns:
        (train_dataset, val_dataset, test_dataset, normalizer_params)
    """
    df = pd.read_csv(labelled_csv_path)
    print(f"[Window] Loaded {len(df):,} rows")

    df = forward_fill_missing_values(df)
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    if train_sessions is not None:
        train_df, val_df, test_df = session_split(df, train_sessions, val_sessions, test_sessions)
    else:
        train_df, val_df, test_df = lap_split(df)

    norm_params = fit_normalizer(train_df, feature_cols, save_path=normalizer_save_path)

    train_df = apply_normalizer(train_df, norm_params, feature_cols)
    val_df = apply_normalizer(val_df, norm_params, feature_cols)
    test_df = apply_normalizer(test_df, norm_params, feature_cols)

    train_ds = TelemetryWindowDataset(train_df, stride=train_stride)
    val_ds = TelemetryWindowDataset(val_df, stride=eval_stride)
    test_ds = TelemetryWindowDataset(test_df, stride=eval_stride)

    return train_ds, val_ds, test_ds, norm_params


def build_sliding_windows(df: pd.DataFrame, window_size: int = WINDOW_SIZE, stride: int = 1) -> list:
    """Thin functional wrapper: slice a normalized DataFrame into raw overlapping windows (no Dataset object)."""
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    features = df[feature_cols].values.astype(np.float32)
    return [features[start:start + window_size] for start in range(0, len(df) - window_size + 1, stride)]
