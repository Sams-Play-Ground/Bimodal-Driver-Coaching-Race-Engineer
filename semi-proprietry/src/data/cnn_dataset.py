"""
cnn_dataset.py — PyTorch Dataset for the CNN spatial classification module,
per proposal §3.5.2 / §3.5.3.

Applies the SAME letterbox transform used at inference time (see
src.preprocessing.frame_extraction.letterbox in the public repo) so training
and inference never see a spatial mismatch.

Ported from `final-cnn-module-codes.ipynb` ("7 - Dataset & augmentation").
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from src.data.cnn_constants import (
    MODEL_INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD, LABELS_PATH, FRAMES_DIR, BATCH_SIZE,
)
from src.preprocessing.frame_extraction import letterbox  # public repo module


class DrivingDataset(Dataset):
    """
    Preprocessing pipeline (applied identically to training, val, and inference):
        1. cv2.imread (BGR) -> cvtColor -> RGB numpy array
        2. letterbox(frame, 640, 288) -> aspect-ratio-preserving resize + black pad
        3. transforms.ToTensor() -> (3, 288, 640) float32
        4. Resize(MODEL_INPUT_SIZE) -> (3, 224, 224) square input for EfficientNet
        5. Normalize(ImageNet mean/std)

    Augmentations applied ONLY when augment=True (training set):
        random rotation, colour jitter, random erasing.

    Labels: torch.float32 tensor of shape (3,) = [off_track, apex_miss, sliding]
    """

    NORMALIZE = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __init__(self, df: pd.DataFrame, frames_dir, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.frames_dir = Path(frames_dir)
        self.augment = augment

        self._train_tf = transforms.Compose([
            transforms.Resize(MODEL_INPUT_SIZE),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(degrees=4),
            transforms.ToTensor(),
            self.NORMALIZE,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.085)),
        ])
        self._val_tf = transforms.Compose([
            transforms.Resize(MODEL_INPUT_SIZE),
            transforms.ToTensor(),
            self.NORMALIZE,
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        row = self.df.iloc[idx]
        img = cv2.imread(str(self.frames_dir / row["frame"]))
        if img is None:
            raise FileNotFoundError(f"Frame not found: {self.frames_dir / row['frame']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = letterbox(img, target_w=640, target_h=288)
        img = Image.fromarray(img)

        tensor = self._train_tf(img) if self.augment else self._val_tf(img)
        labels = torch.tensor([row["off_track"], row["apex_miss"], row["sliding"]], dtype=torch.float32)
        return tensor, labels

    def get_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler — up-weights rare positive labels."""
        cols = ["off_track", "apex_miss", "sliding"]
        pos_counts = self.df[cols].sum().values
        neg_counts = len(self.df) - pos_counts
        weights_per_class = np.sqrt(neg_counts / np.maximum(pos_counts, 1))

        sample_weights = np.ones(len(self.df))
        for i, row in self.df.iterrows():
            label_vec = row[cols].values.astype(bool)
            if label_vec.any():
                sample_weights[i] = weights_per_class[label_vec].max()
        return torch.tensor(sample_weights, dtype=torch.float32)


def build_dataloaders(labels_path: Path = LABELS_PATH, frames_dir=FRAMES_DIR,
                       batch_size: int = BATCH_SIZE, val_frac: float = 0.2,
                       use_sampler: bool = True) -> tuple:
    """
    Load labels, split 80/20 by frame group (group_id prevents augmented
    copies of the same frame leaking across train/val), and return
    (train_dl, val_dl, train_ds, val_ds).
    """
    df = pd.read_csv(labels_path)

    def get_group(name):
        parts = Path(name).stem.split("_")
        return parts[-1] if len(parts) > 1 else name

    df["group_id"] = df["frame"].apply(get_group)
    groups = df["group_id"].unique()
    np.random.seed(42)
    np.random.shuffle(groups)
    split = int(len(groups) * (1 - val_frac))
    train_grp, val_grp = set(groups[:split]), set(groups[split:])

    train_df = df[df["group_id"].isin(train_grp)].reset_index(drop=True)
    val_df = df[df["group_id"].isin(val_grp)].reset_index(drop=True)

    train_ds = DrivingDataset(train_df, frames_dir, augment=True)
    val_ds = DrivingDataset(val_df, frames_dir, augment=False)

    sampler, shuffle = None, True
    if use_sampler:
        w = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        shuffle = False

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                           sampler=sampler, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                         num_workers=2, pin_memory=True)

    print(f"[Data] Train: {len(train_ds):,} frames | Val: {len(val_ds):,} frames")
    print(f"[Data] Train batches: {len(train_dl):,} | Val batches: {len(val_dl):,}")
    print(f"[Data] WeightedRandomSampler: {use_sampler}")
    return train_dl, val_dl, train_ds, val_ds
