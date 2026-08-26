"""
fusion_constants.py — Column layout and hyperparameters for the Late Fusion
Head, which combines 3 CNN spatial probabilities with 7 LSTM temporal
probabilities into 10 coaching-relevant event classes (proposal §3.5.5).

Ported from `final-cnn-module-codes.ipynb` ("16.4 Late Fusion Pipeline").
"""

from pathlib import Path

CNN_COLS = ["cnn_off_track", "cnn_apex_miss", "cnn_sliding"]
LSTM_COLS = [
    "lstm_brake_locked", "lstm_wheel_spin", "lstm_trail_brake",
    "lstm_aggressive_downshift", "lstm_late_upshift",
    "lstm_rough_steering", "lstm_gentle_accel",
]
INPUT_COLS = CNN_COLS + LSTM_COLS  # 10-dim concatenated vector

LABEL_COLS = [
    "label_off_track", "label_apex_miss", "label_sliding",
    "lstm_brake_locked", "lstm_wheel_spin", "lstm_trail_brake",
    "lstm_aggressive_downshift", "lstm_late_upshift",
    "lstm_rough_steering", "lstm_gentle_accel",
]

CLASS_NAMES = [
    "Off-Track", "Apex Miss", "Sliding",
    "Brake Locked", "Wheel Spin", "Trail Brake",
    "Aggressive Downshift", "Late Upshift",
    "Rough Steering", "Gentle Accel",
]

INPUT_DIM = len(INPUT_COLS)     # 10
NUM_CLASSES = len(LABEL_COLS)   # 10

BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 112
PATIENCE = 16
GRAD_CLIP = 1.0
MODALITY_DROPOUT_P = 0.20  # probability of zeroing one entire input stream during training

FUSION_DATA_CSV = "/kaggle/working/fusion_training_data.csv"
CHECKPOINT_DIR = Path("/kaggle/working/fusion/checkpoints")
EVAL_DIR = Path("/kaggle/working/fusion/eval")
DIAG_DIR = Path("/kaggle/working/fusion/diagnostics")
TEMPLATES_PATH = "/kaggle/working/feedback_templates.json"
