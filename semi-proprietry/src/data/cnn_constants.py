"""
cnn_constants.py — Single source of truth for the CNN module's architecture,
training, and labelling constants, per proposal §3.5.3 / §3.6.

Mirrors the LSTM module's src/data/constants.py — change once, works
everywhere. Kaggle-specific I/O paths are left as documented defaults;
override them for local runs.

Ported from `final-cnn-module-codes.ipynb` ("2 - Constants").
"""

from pathlib import Path

# ── Model architecture ───────────────────────────────────────────────────────
BACKBONE = "efficientnet_b0"   # B0 ~ 5.3M params, runs on consumer hardware
NUM_CLASSES = 3                 # [off_track, apex_miss, sliding]
CLASS_NAMES = ["Off-Track", "Apex Miss", "Sliding"]
MODEL_INPUT_SIZE = 224          # EfficientNet canonical square input

# ── Video / frame extraction ─────────────────────────────────────────────────
FRAME_EXTRACT_FPS = 1           # 1 frame per second for labelling
INFERENCE_EVERY_NTH = 1         # process every frame during inference

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 35
PATIENCE = 16                   # early stopping patience
LR_HEAD = 1e-3                  # frozen-backbone phase
LR_FINETUNE = 5e-6              # full fine-tune phase
FREEZE_EPOCHS = 5               # backbone frozen for first n epochs
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-4
DROPOUT = 0.45

# ── Labelling (interactive tool key map) ─────────────────────────────────────
KEY_MAP = {
    "s": (0, 0, 0),   # Safe
    "o": (1, 0, 0),   # Off-track
    "a": (0, 1, 0),   # Apex miss
    "l": (0, 0, 1),   # Sliding
    "b": (1, 1, 0),   # Off + Apex
    "p": (1, 0, 1),   # Off + Sliding
    "d": (0, 1, 1),   # Apex + Sliding
    "g": (1, 1, 1),   # All three
}

# ── Inference decision thresholds (tuned per class after training) ───────────
DEFAULT_THRESHOLDS = {
    "Off-Track": 0.35,   # lower threshold — err toward catching dangerous events
    "Apex Miss": 0.47,
    "Sliding": 0.45,
}

# ── Normalisation (ImageNet stats — used by EfficientNet pretrained weights) ─
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ── Default Kaggle paths — override for local runs ───────────────────────────
VIDEO_DIR = Path("/kaggle/input/datasets/samwelnjehia/cnn-module-data")
FRAMES_DIR = Path("/kaggle/working/frames_fullres")
LABELS_PATH = Path("/kaggle/working/labels_cnn_3output.csv")
CHECKPOINT_DIR = Path("/kaggle/working/checkpoints")
EVAL_DIR = Path("/kaggle/working/eval_output")
DIAG_DIR = Path("/kaggle/working/diagnostics")
BESTMODEL_DIR = Path("/kaggle/working/bestmodel")
