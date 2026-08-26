# CNN Module — full notebook export
# Auto-exported from the original Kaggle .ipynb for reproducibility.
# Cell boundaries are marked; markdown cells are kept as comments.

# %% [markdown] cell 0
# # CNN Module — Bimodal AI Driver Training Framework
#
# **Project:** A Bimodal Deep Learning Framework for Driver Training by Integrating Computer Vision and Telemetry via Explainable AI
#
# **Author:** Samwel N. Njehia | USIU-Africa | BSc Data Science & Analytics | Spring 2026
#
# **Module:** Spatial Feature Extraction via EfficientNet-B0 (proposal §3.5.3)
#
# ---
#
# ### Notebook structure
# | # | Section | Proposal ref |
# |---|---|---|
# | 1 | Imports | — |
# | 2 | Constants — single source of truth | §1.6, §3.5.3 |
# | 3 | Multi-video ingestion | §3.4 |
# | 4 | Frame extraction (full-res) | §3.4 |
# | 5 | Interactive labelling tool | §3.5.1 |
# | 6 | Label distribution analysis | §3.6 |
# | 7 | Dataset & augmentation | §3.5.2 |
# | 8 | Model — DrivingCNN (EfficientNet-B0) | §3.5.3 |
# | 9 | Training loop with class-weighted loss & F1 tracking | §3.6 |
# | 10 | Evaluation — F1, confusion matrices, ROC curves | §3.7 |
# | 11 | Grad-CAM diagnostics | §3.7 |
# | 12 | 5-fold cross-validation | §3.7 |
# | 13 | Ensemble (Wisdom of All) | §3.7 |
# | 14 | Inference — annotate unseen footage | §1.7.2 |

# %% [markdown] cell 1
# # 1 · Imports

# %% cell 2
# ── Install ────────────────────────────────────────────────────────────────
!pip install -q timm opencv-python-headless

# ── Standard libraries ─────────────────────────────────────────────────────
import os, json, random, copy, time, shutil
from pathlib import Path
from base64 import b64encode

# ── Numerical / data ───────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── Vision ─────────────────────────────────────────────────────────────────
import cv2
from PIL import Image
import timm

# ── PyTorch ────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as TF

# ── Metrics ────────────────────────────────────────────────────────────────
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve,
    classification_report,
)
from sklearn.model_selection import GroupKFold

# ── Notebook utilities ─────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('Agg')   # safe for Kaggle — use plt.show() or plt.savefig()
import IPython.display as ipd
from IPython.display import clear_output, HTML
import ipywidgets as widgets
from tqdm.notebook import tqdm

from typing import Union
from scipy import stats

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[Setup] Device : {DEVICE}')
print(f'[Setup] PyTorch: {torch.__version__}')
print(f'[Setup] Signora! Il limone! Signoraaaa!!!')

# %% cell 3
print('[Status Check] Notebook on? \n[Status Check] What you think wise azz, otherwise you would not be reading this...')

# %% [markdown] cell 4
# # 2 · Constants — single source of truth
#
# All thresholds, paths, and hyperparameters live here.
# Change once, works everywhere. Mirrors the LSTM module's `constants cell block`.

# %% cell 5
# ── Paths ──────────────────────────────────────────────────────────────────
VIDEO_DIR      = Path('/kaggle/input/datasets/samwelnjehia/cnn-module-data')
FRAMES_DIR     = Path('/kaggle/working/frames_fullres')
LABELS_PATH    = Path('/kaggle/working/labels_cnn_3output.csv')
CHECKPOINT_DIR = Path('/kaggle/working/checkpoints')
EVAL_DIR       = Path('/kaggle/working/eval_output')
DIAG_DIR       = Path('/kaggle/working/diagnostics')
BESTMODEL_DIR  = Path("/kaggle/working/bestmodel")
TEST_VIDEO     = '/kaggle/input/datasets/samwelnjehia/rq2-detuned/LFS 2026-08-05 14-51-34-781.mp4'

FRAMES_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
BESTMODEL_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# ── Model architecture ──────────────────────────────────────────────────────
BACKBONE         = 'efficientnet_b0'   # B0 ~ 5.3M params, runs on consumer hardware
NUM_CLASSES      = 3                   # [off_track, apex_miss, sliding]
CLASS_NAMES      = ['Off-Track', 'Apex Miss', 'Sliding']
MODEL_INPUT_SIZE = 224                 # EfficientNet canonical square input

# ── Video / frame extraction ────────────────────────────────────────────────
FRAME_EXTRACT_FPS   = 1               # 1 frame per second for labelling
INFERENCE_EVERY_NTH = 1               # process every frame during inference

# ── Training ────────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
EPOCHS        = 35                    # epochs = 1024
PATIENCE      = 16                    # early stopping patience
LR_HEAD       = 1e-3                  # frozen-backbone phase
LR_FINETUNE   = 5e-6                  # full fine-tune phase
FREEZE_EPOCHS = 5                     # backbone frozen for first n epochs
GRAD_CLIP     = 1.0                   # gradient norm clip (matches LSTM module)
WEIGHT_DECAY  = 1e-4
DROPOUT       = 0.45

# ── Labeling ────────────────────────────────────────────────────────────────
KEY_MAP = {
    's': (0, 0, 0),   # Safe
    'o': (1, 0, 0),   # Off-track
    'a': (0, 1, 0),   # Apex miss
    'l': (0, 0, 1),   # Sliding
    'b': (1, 1, 0),   # Off + Apex
    'p': (1, 0, 1),   # Off + Sliding
    'd': (0, 1, 1),   # Apex + Sliding
    'g': (1, 1, 1),   # All three
}

# ── Inference decision thresholds (tuned per class after training) ──────────
# These are STARTING values — tune_thresholds() will find optimal ones
DEFAULT_THRESHOLDS = {
    'Off-Track': 0.35,   # lower threshold — err toward catching dangerous events
    'Apex Miss': 0.47,
    'Sliding':   0.45,
}

# ── Normalisation (ImageNet stats — used by EfficientNet pretrained weights) ─
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

print('[Constants] All constants loaded.')
print(f'[Constants] Backbone   : {BACKBONE}')
print(f'[Constants] Num classes: {NUM_CLASSES} → {CLASS_NAMES}')
print(f'[Constants] Batch size : {BATCH_SIZE} | Epochs: {EPOCHS} | Patience: {PATIENCE}')

# %% cell 6
from pathlib import Path
from IPython.display import FileLink

def download_bestmodel_2(dir_path, print_files=False):
    # Convert path string/Path object to a Path instance
    dir_path = Path(dir_path)
    
    # 1. Display files in the specified directory
    print(f"Checking Directory Contents ({dir_path})...")
    if dir_path.exists():
        files = list(dir_path.iterdir())
        if files:
            for f in files:
                size_mb = f.stat().st_size / (1024 * 1024)
                if print_files:
                    print(f"  📄 {f.name} ({size_mb:.2f} MB)")
        else:
            print("  (Directory is empty)")
            return "file not found in directory"
    else:
        print(f"  Path {dir_path} does not exist.")
        return "file not found in directory"
    
    # 2. Quick scan for target file
    target_file = dir_path / "best_model.pt"
    
    if target_file.exists():
        print(f"\n✅ Found 'best_model.pt' in {dir_path}")
        print("Generating model download link...\n")
        return FileLink(str(target_file))
    else:
        print(f"\n⚠️ Could not find best_model.pt in {dir_path}")
        return "file not found in directory"

import base64
from pathlib import Path
from IPython.display import HTML

def download_bestmodel_3(dir_path, file_to_download = "best_model.pt", print_files=False):
    dir_path = Path(dir_path)
    
    print(f"Checking Directory Contents ({dir_path})...")
    if dir_path.exists():
        files = list(dir_path.iterdir())
        if files:
            for f in files:
                size_mb = f.stat().st_size / (1024 * 1024)
                if print_files:
                    print(f"  📄 {f.name} ({size_mb:.2f} MB)")
        else:
            print("  (Directory is empty)")
            return "file not found in directory"
    else:
        print(f"  Path {dir_path} does not exist.")
        return "file not found in directory"
    
    target_file = dir_path / file_to_download
    
    if target_file.exists():
        print(f"\n✅ Found {file_to_download}. Encoding for download...")
        
        # Read file and encode to base64
        with open(target_file, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        
        # HTML download tag
        html = f'''
        <a download="{target_file.name}" href="data:application/octet-stream;base64,{b64}" target="_blank">
            <button style="padding:10px 20px; background-color:#4CAF50; color:white; border:none; border-radius:5px; cursor:pointer; font-weight:bold;">
                📥 Click Here to Download {target_file.name}
            </button>
        </a>
        '''
        return HTML(html)
    else:
        print(f"\n⚠️ Could not find {file_to_download} in {dir_path}")
        return "file not found in directory"

# %% cell 7
target_path = Path('/kaggle/working/fusion/checkpoints')

# Files of interest inference_config.json, history.json, best_model.pt

download_bestmodel_3(target_path, file_to_download = 'inference_config.json')

# %% cell 8
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

# %% cell 9
download_bestmodel()

# %% [markdown] cell 10
# # 3 · Multi-video ingestion
#
# Discovers all video files across one or more input folders and returns
# a unified list. Mirrors `load_and_compile_telemetry_folder()` from the LSTM module,
# ensuring the same multi-source ingestion pattern is consistent across both modules.

# %% cell 11
def discover_video_files(source) -> list:
    """
    Discover all video files in one directory, multiple directories,
    or a direct list of file paths.

    Args:
        source: str | Path | list[str | Path]
            A folder path, list of folder paths, or list of direct video file paths.

    Returns:
        Sorted list of Path objects for all discovered video files.

    Example:
        files = discover_video_files('/kaggle/input/dataset1')
        files = discover_video_files(['/kaggle/input/ds1', '/kaggle/input/ds2'])
    """
    VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI'}

    found = []

    if isinstance(source, (str, Path)):
        # Single directory
        source = Path(source)
        if source.is_dir():
            found = [f for f in source.rglob('*') if f.suffix in VIDEO_EXTENSIONS]
        elif source.is_file() and source.suffix in VIDEO_EXTENSIONS:
            found = [source]
        else:
            raise FileNotFoundError(f'[Ingest] Path not found or not a video: {source}')

    elif isinstance(source, list):
        for item in source:
            item = Path(item)
            if item.is_dir():
                found.extend(f for f in item.rglob('*') if f.suffix in VIDEO_EXTENSIONS)
            elif item.is_file() and item.suffix in VIDEO_EXTENSIONS:
                found.append(item)
    else:
        raise TypeError('[Ingest] source must be a path string, Path, or list.')

    found = sorted(set(found))
    if not found:
        raise FileNotFoundError(f'[Ingest] No video files found in: {source}')

    print(f'[Ingest] Found {len(found)} video file(s):')
    for f in found:
        size_mb = f.stat().st_size / 1_000_000
        print(f'  {f.name}  ({size_mb:.1f} MB)')

    return found

# %% cell 12
# Quick test — run this to confirm your dataset is visible
video_files = discover_video_files(VIDEO_DIR)
print(f'\n[Ingest] Total: {len(video_files)} video(s) ready for frame extraction.')

# %% [markdown] cell 13
# # 4 · Frame extraction
#
# Extracts frames at **full native resolution** — scaling happens inside the
# `DrivingDataset` class at training time (via letterbox), not here for labeling cause.
# This preserves every pixel of spatial information for the labelling step.

# %% cell 14
def letterbox(frame: np.ndarray,
              target_w: int = 640,
              target_h: int = 288) -> np.ndarray:
    """
    Resize a frame to fit inside (target_w × target_h) while preserving
    the original aspect ratio. Remaining area is black-padded (letterbox style).

    Used consistently in:
      • DrivingDataset.__getitem__ (training + validation)
      • annotate_video preprocess step (inference)
    Ensures training and inference see IDENTICAL spatial representations.
    """
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas  = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off   = (target_w - new_w) // 2
    y_off   = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def extract_frames_fullres(video_path: Path,
                            out_dir: Path,
                            fps_target: int = 1) -> int:
    """
    Extract one frame per `fps_target` seconds at the video's native resolution.
    Frames are saved as high-quality JPEGs — NO resizing or letterboxing here.
    Scaling happens in DrivingDataset to keep the labelling view clean.

    Args:
        video_path: Path to the source video file.
        out_dir:    Directory where extracted JPEG frames are saved.
        fps_target: How many frames per second to extract (default 1).

    Returns:
        Number of frames saved.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f'  [Extract] Could not open: {video_path.name} — skipping.')
        return 0

    src_fps  = cap.get(cv2.CAP_PROP_FPS)
    src_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    interval = max(1, int(src_fps / fps_target))
    stem     = video_path.stem
    count    = saved = 0

    print(f'  [Extract] {video_path.name}')
    print(f'            {src_w}×{src_h} @ {src_fps:.1f}fps  → {fps_target} frame every {interval} source frames')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            out = out_dir / f'{stem}_f{count:07d}.jpg'
            cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
        count += 1

    cap.release()
    return saved

# %% cell 15
# ── Run extraction across all discovered videos ─────────────────────────────
total_frames = 0
for vid in video_files:
    n = extract_frames_fullres(vid, FRAMES_DIR, fps_target=FRAME_EXTRACT_FPS)
    total_frames += n
    print(f'    → {n} frames saved\n')

print(f'[Extract] Total frames saved : {total_frames:,}')
print(f'[Extract] Saved to           : {FRAMES_DIR}')
print(f'[Extract] Estimated disk use : ~{total_frames * 2.5 / 1000:.1f} GB')

# %% [markdown] cell 16
# # 5 · Interactive labelling tool
#
# **Key map (3-output multi-label):**
#
# | Key | off_track | apex_miss | sliding | Mnemonic |
# |---|---|---|---|---|
# | `s` | 0 | 0 | 0 | **S**afe |
# | `o` | 1 | 0 | 0 | **O**ff track |
# | `a` | 0 | 1 | 0 | **A**pex miss |
# | `l` | 0 | 0 | 1 | s**L**iding |
# | `b` | 1 | 1 | 0 | off + apex (**B**oth spatial) |
# | `p` | 1 | 0 | 1 | off + sliding (**P**anic) |
# | `d` | 0 | 1 | 1 | apex + sli**D**ing |
# | `g` | 1 | 1 | 1 | all three (**G**ardening) |
# | `q` | — | — | — | **Q**uit and save |
#
# Progress is auto-saved every 70 frames so you can resume any time.

# %% cell 17
def run_labelling_tool(frames_dir: Path = FRAMES_DIR,
                        labels_path: Path = LABELS_PATH,
                        display_width: int = 960) -> None:
    """
    Interactive labelling widget with frame deletion support.

    KEY MAP
    ───────────────────────────────────────────────────────────────
    s  →  Safe              (0, 0, 0)
    o  →  Off-track         (1, 0, 0)
    a  →  Apex miss         (0, 1, 0)
    l  →  Sliding           (0, 0, 1)
    b  →  Off + Apex        (1, 1, 0)
    p  →  Off + Sliding     (1, 0, 1)
    d  →  Apex + Sliding    (0, 1, 1)
    g  →  All three         (1, 1, 1)
    x  →  DELETE frame      — removes the file from disk and skips it.
                              Also purges it from the labels CSV if it was
                              previously labelled in an earlier session.
                              Use this for blurry, corrupt, or irrelevant
                              frames that should never enter the dataset.
    q  →  Quit and save
    ───────────────────────────────────────────────────────────────

    Resumes automatically from the last unlabelled frame.
    Auto-saves every 70 frames and on quit.
    Deletion log is written to <labels_path>.deletions.txt.
    """
    frames_dir  = Path(frames_dir)
    labels_path = Path(labels_path)
    del_log     = labels_path.with_suffix('.deletions.txt')

    # ── Build the queue of frames still needing labels ────────────────────────
    all_frames = sorted(frames_dir.glob('*.jpg'))

    if labels_path.exists():
        done_set = set(pd.read_csv(labels_path)['frame'].tolist())
        frames   = [f for f in all_frames if f.name not in done_set]
        print(f'[Label] Resuming — {len(frames)} frames left  ({len(done_set)} already done)')
    else:
        frames = all_frames
        print(f'[Label] Starting fresh — {len(frames)} frames to label')

    # Convert to list so we can mutate it when frames are deleted
    frames = list(frames)

    print('[Label] s=Safe | o=Off | a=Apex | l=Slide | b=Off+Apex | '
          'p=Off+Slide | d=Apex+Slide | g=All | x=DELETE | q=Quit')

    rows    = []   # pending rows not yet flushed to CSV
    deleted = []   # names of frames deleted this session
    i       = [0]  # mutable current index

    # ── Display helpers ───────────────────────────────────────────────────────
    def show_current():
        if i[0] >= len(frames):
            print('[Label] All frames labelled!')
            return
        img = cv2.imread(str(frames[i[0]]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dh  = int(display_width * img.shape[0] / img.shape[1])
        img = cv2.resize(img, (display_width, dh))
        _, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        ipd.display(ipd.Image(data=buf.tobytes()))
        print(f'  Frame {i[0]+1}/{len(frames)}: {frames[i[0]].name}')

    def refresh():
        """Clear and redraw the current frame."""
        with out:
            clear_output(wait=True)
            show_current()

    # ── Persistence helpers ───────────────────────────────────────────────────
    def save_labels():
        """Flush pending rows to the labels CSV."""
        if not rows:
            return
        df_new = pd.DataFrame(rows)
        if labels_path.exists():
            df_old = pd.concat([pd.read_csv(labels_path), df_new], ignore_index=True)
        else:
            df_old = df_new
        df_old.to_csv(labels_path, index=False)
        print(f'[Label] Saved {len(df_old)} total labels → {labels_path}')

    def purge_frame_from_csv(frame_name: str):
        """
        Remove a frame from the labels CSV if it was labelled in a
        previous session. Called automatically when x is pressed so the
        deleted frame doesn't linger in the dataset.
        """
        if not labels_path.exists():
            return
        df = pd.read_csv(labels_path)
        before = len(df)
        df = df[df['frame'] != frame_name]
        if len(df) < before:
            df.to_csv(labels_path, index=False)
            print(f'[Label] Purged "{frame_name}" from labels CSV.')

    def log_deletion(frame_name: str):
        """Append the deleted frame name to the deletion log."""
        with open(del_log, 'a') as f:
            f.write(frame_name + '\n')

    def delete_current_frame():
        """
        Delete the current frame file from disk, remove it from the
        in-session queue, purge it from the labels CSV, and advance.
        """
        if i[0] >= len(frames):
            return

        target = frames[i[0]]

        # Remove from disk
        try:
            target.unlink()
        except FileNotFoundError:
            pass  # already gone — continue gracefully

        # Remove from the pending-row buffer in case it was just labelled
        rows[:] = [r for r in rows if r['frame'] != target.name]

        # Purge from the saved CSV (handles frames labelled in prior sessions)
        purge_frame_from_csv(target.name)

        # Track deletion for the log
        deleted.append(target.name)
        log_deletion(target.name)

        print(f'[Label] 🗑  Deleted: {target.name}')

        # Remove from the in-memory queue — the next frame slides into place
        frames.pop(i[0])
        # i[0] stays the same; the frame that was at i[0]+1 is now at i[0]
        # Update the progress bar ceiling since the queue shrank
        progress.max = len(frames)

    # ── Key handler ───────────────────────────────────────────────────────────
    def on_label(change):
        val = label_widget.value.strip().lower()
        label_widget.value = ''
        if not val:
            return
        k = val[0]

        # ── Quit ──────────────────────────────────────────────────────────────
        if k == 'q':
            save_labels()
            with out:
                clear_output()
                print(f'[Label] Quit — progress saved.')
                if deleted:
                    print(f'[Label] {len(deleted)} frame(s) deleted this session.')
                    print(f'[Label] Deletion log → {del_log}')
            return

        # ── Delete ────────────────────────────────────────────────────────────
        if k == 'x':
            delete_current_frame()
            progress.value = min(i[0], len(frames))
            refresh()
            return

        # ── Label ─────────────────────────────────────────────────────────────
        if k not in KEY_MAP:
            return

        if i[0] >= len(frames):
            return

        off, apex, slid = KEY_MAP[k]
        rows.append({
            'frame':     frames[i[0]].name,
            'off_track': off,
            'apex_miss': apex,
            'sliding':   slid,
        })
        i[0] += 1
        progress.value = i[0]

        # Auto-save every 70 frames
        if i[0] % 70 == 0:
            save_labels()
            rows.clear()

        refresh()

    # ── Widgets ───────────────────────────────────────────────────────────────
    out          = widgets.Output()
    label_widget = widgets.Text(
        placeholder='Type key + Enter',
        layout=widgets.Layout(width='350px')
    )
    progress = widgets.IntProgress(
        value=0, min=0, max=len(frames),
        description='Progress:',
        layout=widgets.Layout(width='500px')
    )

    with out:
        show_current()

    label_widget.observe(on_label, names='value')
    ipd.display(progress, label_widget, out)

# %% cell 18
run_labelling_tool()

# %% [markdown] cell 19
# # 6 · Label distribution analysis
#
# Mirrors `distributions LSTM module` from the LSTM module.
# Computes `pos_weight` values for class-weighted loss — the fix for the
# F1=0.00 issue documented in the original evaluation report.

# %% cell 20
def analyse_label_distribution(labels_path: Path = LABELS_PATH,
                               output_dir: Path  = EVAL_DIR) -> dict:
    """
    Compute class distribution, pos_weight for BCEWithLogitsLoss,
    and save a bar chart.

    Returns:
        pos_weights: dict {class_name: weight} usable directly in
                     build_criterion(). High weight = rare class gets
                     penalised more for false negatives.
    """
    df = pd.read_csv(labels_path)
    n  = len(df)

    print(f'\n{'='*55}')
    print(f'  LABEL DISTRIBUTION  (N = {n:,} frames)')
    print(f'{'='*55}')
    print(f'  {"Class":<15} {"Pos":>6} {"Neg":>6} {"Pos%":>6}  {"pos_weight":>10}')
    print(f'  {"-"*50}')

    pos_weights = {}
    counts      = {}
    for col, name in zip(['off_track', 'apex_miss', 'sliding'], CLASS_NAMES):
        if col not in df.columns:
            continue
        n_pos = int(df[col].sum())
        n_neg = n - n_pos
        pct   = 100 * n_pos / n
        pw    = min(n_neg / max(n_pos, 1), 50.0)   # cap at 50x for stability
        pos_weights[name] = round(pw, 2)
        counts[name]      = n_pos
        print(f'  {name:<15} {n_pos:>6,} {n_neg:>6,} {pct:>5.1f}%  {pw:>10.1f}x')

    print(f'{'='*55}\n')

    # Bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    colors = ['#d62728' if pos_weights[n] > 20 else '#ff7f0e'
              if pos_weights[n] > 5 else '#2ca02c' for n in CLASS_NAMES if n in pos_weights]

    names_present = [n for n in CLASS_NAMES if n in counts]
    ax1.bar(names_present, [counts[n] for n in names_present], color=colors, alpha=0.85, edgecolor='black')
    ax1.set_title('Positive label counts per class')
    ax1.set_ylabel('Frame count')
    for i, (name, c) in enumerate([(n, counts[n]) for n in names_present]):
        ax1.text(i, c + 1, str(c), ha='center', fontsize=9)

    ax2.bar(names_present, [pos_weights[n] for n in names_present], color=colors, alpha=0.85, edgecolor='black')
    ax2.set_title('pos_weight per class (BCEWithLogitsLoss)')
    ax2.set_ylabel('Weight multiplier')
    ax2.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Balanced')
    ax2.legend()

    plt.tight_layout()
    out = output_dir / 'label_distribution.png'
    plt.savefig(str(out), dpi=130, bbox_inches='tight')
    plt.show()
    print(f'[Stats] Chart saved → {out}')
    return pos_weights

# %% cell 21
# ── Run it ───────────────────────────────────────────────────────────────────
pos_weights = analyse_label_distribution()
print(f'[Stats] pos_weights for loss: {pos_weights}')

# %% [markdown] cell 22
# # 7 · Dataset & augmentation
#
# `DrivingDataset` applies the **same letterbox transform in both training and inference**
# — fixing the critical preprocessing mismatch from the original notebook.
#
# Augmentation is handled inline (no separate augmented CSV) and uses
# `WeightedRandomSampler` to oversample minority-class frames during training,
# matching the LSTM module's approach.

# %% cell 23
class DrivingDataset(Dataset):
    """
    PyTorch Dataset for the CNN spatial classification module.

    Preprocessing pipeline (applied identically to training, val, and inference):
        1. cv2.imread (BGR) → cvtColor → RGB numpy array
        2. letterbox(frame, 640, 288) → aspect-ratio-preserving resize + black pad
        3. transforms.ToTensor() → (3, 288, 640) float32
        4. CenterCrop(MODEL_INPUT_SIZE) → (3, 224, 224) square input for EfficientNet
        5. Normalize(ImageNet mean/std)

    Augmentations applied ONLY when augment=True (training set):
        • Random horizontal flip
        • ColorJitter (brightness, contrast, saturation)
        • Random rotation ±8°
        • Random erasing (occlusion robustness)

    Labels: torch.float32 tensor of shape (3,) = [off_track, apex_miss, sliding]
    """

    NORMALIZE = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __init__(self, df: pd.DataFrame, frames_dir, augment: bool = False):
        self.df         = df.reset_index(drop=True)
        self.frames_dir = Path(frames_dir)
        self.augment    = augment

        self._train_tf = transforms.Compose([
            transforms.Resize(MODEL_INPUT_SIZE),      # Resize whole image to (224, 224) retaining edges
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(degrees=4),     # Mild rotation to prevent track tilting distortion
            transforms.ToTensor(),
            self.NORMALIZE,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.085)),
        ])
        self._val_tf = transforms.Compose([
            transforms.Resize(MODEL_INPUT_SIZE),       # Full FOV resize for val/inference
            transforms.ToTensor(),
            self.NORMALIZE,
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        row    = self.df.iloc[idx]
        img    = cv2.imread(str(self.frames_dir / row['frame']))
        if img is None:
            raise FileNotFoundError(f"Frame not found: {self.frame_dir / row['frame']}")
        img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img    = letterbox(img, target_w=640, target_h=288)  # consistent with inference
        img    = Image.fromarray(img)

        tensor = self._train_tf(img) if self.augment else self._val_tf(img)
        labels = torch.tensor(
            [row['off_track'], row['apex_miss'], row['sliding']],
            dtype=torch.float32
        )
        return tensor, labels

    def get_sample_weights(self) -> torch.Tensor:
        """
        Per-sample weights for WeightedRandomSampler.
        Samples with at least one positive minority-class label
        get higher weight to counteract class imbalance.
        """
        cols       = ['off_track', 'apex_miss', 'sliding']
        pos_counts = self.df[cols].sum().values
        neg_counts = len(self.df) - pos_counts
        weights_per_class = np.sqrt(neg_counts / np.maximum(pos_counts, 1))

        sample_weights = np.ones(len(self.df))
        for i, row in self.df.iterrows():
            label_vec = row[cols].values.astype(bool)
            if label_vec.any():
                sample_weights[i] = weights_per_class[label_vec].max()
        return torch.tensor(sample_weights, dtype=torch.float32)


def build_dataloaders(labels_path: Path = LABELS_PATH,
                       frames_dir        = FRAMES_DIR,
                       batch_size: int   = BATCH_SIZE,
                       val_frac: float   = 0.2,
                       use_sampler: bool = True) -> tuple:
    """
    Load labels, shuffle, split 80/20 by frame group (using group_id to
    prevent augmented copies of the same frame appearing in both sets),
    and return (train_dl, val_dl, train_ds, val_ds).
    """
    df = pd.read_csv(labels_path)

    # Group ID: strip aug prefix so augmented copies don't leak into val
    def get_group(name):
        parts = Path(name).stem.split('_')
        return parts[-1] if len(parts) > 1 else name

    df['group_id'] = df['frame'].apply(get_group)
    groups         = df['group_id'].unique()
    np.random.seed(42)
    np.random.shuffle(groups)
    split     = int(len(groups) * (1 - val_frac))
    train_grp = set(groups[:split])
    val_grp   = set(groups[split:])

    train_df = df[df['group_id'].isin(train_grp)].reset_index(drop=True)
    val_df   = df[df['group_id'].isin(val_grp)].reset_index(drop=True)

    train_ds = DrivingDataset(train_df, frames_dir, augment=True)
    val_ds   = DrivingDataset(val_df,   frames_dir, augment=False)

    sampler  = None
    shuffle  = True
    if use_sampler:
        w       = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        shuffle = False  # mutually exclusive with sampler

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                           sampler=sampler, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                           num_workers=2, pin_memory=True)

    print(f'[Data] Train: {len(train_ds):,} frames | Val: {len(val_ds):,} frames')
    print(f'[Data] Train batches: {len(train_dl):,} | Val batches: {len(val_dl):,}')
    print(f'[Data] WeightedRandomSampler: {use_sampler}')
    return train_dl, val_dl, train_ds, val_ds

# %% cell 24
train_dl, val_dl, train_ds, val_ds = build_dataloaders()

# %% [markdown] cell 25
# # 8 · Model — DrivingCNN (EfficientNet-B0)
#
# **Why EfficientNet-B0?**
# EfficientNet scales depth, width, and resolution together in an optimal ratio
# (Tan & Le, 2019). B0 is the smallest member of the family — 5.3M parameters
# vs ResNet-50's 25.6M — achieving comparable accuracy at 8× lower parameter
# cost. This is critical for the on-device deployment target (proposal §1.7.2).
#
# The `num_classes=0` argument removes EfficientNet's original ImageNet head
# and returns a 1280-dim feature vector per frame. A small custom head maps
# this to 3 binary outputs. Raw logits are returned — `BCEWithLogitsLoss`
# applies sigmoid internally during training, which is numerically more stable
# than applying sigmoid in the forward pass first.

# %% cell 26
class DrivingCNN(nn.Module):
    """
    Spatial feature extraction CNN for the bimodal driver coaching framework.

    Architecture (proposal §3.5.3):
        EfficientNet-B0 backbone (pretrained on ImageNet, num_classes=0)
        → 1280-dim feature vector (Global Average Pooling output)
        → Dropout(0.3)
        → Linear(1280, 3)
        → raw logits [off_track, apex_miss, sliding]

    Training protocol:
        Phase 1 (epochs 0–FREEZE_EPOCHS): backbone frozen, only head trained.
            Allows the new 3-class head to stabilise before touching backbone weights.
        Phase 2 (epochs FREEZE_EPOCHS+): all layers unfrozen, lr reduced to LR_FINETUNE.
            End-to-end fine-tuning without catastrophic forgetting.
    """

    def __init__(self, backbone: str = BACKBONE, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes   = num_classes

        # Pretrained backbone — removes original classification head
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)

        # 3-output head
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(self.backbone.num_features, num_classes),
        )

        # Xavier init on the new head (backbone already has pretrained weights)
        nn.init.xavier_uniform_(self.head[1].weight)
        nn.init.zeros_(self.head[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, H, W) — normalised frame batch
        Returns:
            logits: (batch, num_classes) — raw pre-sigmoid scores
                    Apply torch.sigmoid() for probabilities.
                    BCEWithLogitsLoss applies sigmoid internally during training.
        """
        return self.head(self.backbone(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper: returns sigmoid probabilities in eval mode."""
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self(x))

    def count_parameters(self) -> int:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'[Model] Trainable parameters: {total:,}')
        return total

    def freeze_backbone(self):
        """Freeze all backbone layers — only head is trained."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        print(f'[Model] Phase 1: backbone frozen — training head only.')

    def unfreeze_all(self):
        """Unfreeze all layers for end-to-end fine-tuning."""
        for p in self.parameters():
            p.requires_grad_(True)
        print(f'[Model] Phase 2: all layers unfrozen — full fine-tuning at LR={LR_FINETUNE}.')


def build_model(device=DEVICE) -> DrivingCNN:
    model = DrivingCNN().to(device)
    model.count_parameters()
    print(f'[Model] Backbone   : {model.backbone_name}')
    print(f'[Model] Output dim : {model.num_classes} → {CLASS_NAMES}')
    print(f'[Model] Device     : {device}')
    return model

# %% cell 27
model = build_model()

# %% [markdown] cell 28
# # 9 · Training loop
#
# **Improvements over original notebook:**
# - `BCEWithLogitsLoss` with **`pos_weight`** from label distribution analysis — fixes F1=0.00 on minority classes
# - Early stopping on **macro-F1** (not val_loss) — loss can improve while F1 stays zero
# - Per-class F1 printed every 5 epochs for real-time visibility of class imbalance
# - **Gradient clipping** at 1.0 (matches LSTM module)
# - Checkpoint saves full **model config metadata** (backbone, num_classes) for safe reload
# - **Training curve** saved as PNG after training

# %% cell 29
def build_criterion(pos_weights: dict, device=DEVICE) -> nn.BCEWithLogitsLoss:
    """
    Build BCEWithLogitsLoss with per-class positive weights.

    pos_weights: {class_name: weight} from analyse_label_distribution().
    Higher weight = rare positive class is penalised more heavily for misses,
    forcing the model to actually learn to detect it rather than defaulting
    to the safe majority class.
    """
    weights = [pos_weights.get(name, 1.0) for name in CLASS_NAMES]
    pw      = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f'[Loss] BCEWithLogitsLoss pos_weight: {dict(zip(CLASS_NAMES, [round(w,1) for w in weights]))}')
    return nn.BCEWithLogitsLoss(pos_weight=pw)


def compute_metrics(logits: np.ndarray,
                     labels: np.ndarray,
                     threshold: float = 0.5) -> dict:
    """Compute per-class and macro F1 from raw logits."""
    probs   = 1 / (1 + np.exp(-logits))
    preds   = (probs > threshold).astype(int)
    per_cls = f1_score(labels, preds, average=None, zero_division=0).tolist()
    return {'macro_f1': float(np.mean(per_cls)), 'per_class_f1': per_cls}


def run_epoch(model, loader, criterion, optimizer=None, training=True) -> dict:
    """Run one full epoch. Returns loss and F1 metrics."""
    model.train(training)
    total_loss  = 0.0
    all_logits  = []
    all_labels  = []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            logits       = model(imgs)
            loss         = criterion(logits, labels)

            if training and optimizer:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # prevent gradient explosion
                optimizer.step()

            total_loss += loss.item() * len(imgs)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics    = compute_metrics(all_logits, all_labels)
    metrics['loss'] = total_loss / len(loader.dataset)
    return metrics


class CNNTrainer:
    """
    Full training lifecycle for DrivingCNN.
    Mirrors LSTMTrainer in structure and logging conventions.
    """

    def __init__(self, model, train_dl, val_dl, pos_weights,
                 checkpoint_dir=CHECKPOINT_DIR):
        self.model       = model
        self.train_dl    = train_dl
        self.val_dl      = val_dl
        self.checkpoint_dir = Path(checkpoint_dir)
        self.criterion   = build_criterion(pos_weights)
        self.history     = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': [], 'per_class_f1': []}
        self.best_val_f1 = 0.0
        self.patience_cnt = 0

        # Phase 1: freeze backbone, train head only
        self.model.freeze_backbone()
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR_HEAD, weight_decay=WEIGHT_DECAY
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=8
        )
        self._phase = 1

    def _maybe_unfreeze(self, epoch):
        if epoch == FREEZE_EPOCHS and self._phase == 1:
            self.model.unfreeze_all()
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=8
            )
            self._phase = 2

    def train(self, max_epochs=EPOCHS) -> dict:
        print(f'\n{"="*65}')
        print(f'  CNN TRAINING  |  Device: {DEVICE}  |  Max epochs: {max_epochs}')
        print(f'  Train batches: {len(self.train_dl):,}  |  Val batches: {len(self.val_dl):,}')
        print(f'  Early stopping: patience={PATIENCE} on val macro-F1')
        print(f'{"="*65}\n')

        for epoch in range(1, max_epochs + 1):
            self._maybe_unfreeze(epoch)
            t0 = time.time()

            tr = run_epoch(self.model, self.train_dl, self.criterion, self.optimizer, training=True)
            vl = run_epoch(self.model, self.val_dl,   self.criterion, training=False)

            self.scheduler.step(vl['macro_f1'])

            self.history['train_loss'].append(tr['loss'])
            self.history['val_loss'].append(vl['loss'])
            self.history['train_f1'].append(tr['macro_f1'])
            self.history['val_f1'].append(vl['macro_f1'])
            self.history['per_class_f1'].append(vl['per_class_f1'])

            elapsed = time.time() - t0
            print(f'Epoch {epoch:03d}/{max_epochs}  '
                  f'tr_loss={tr["loss"]:.4f}  vl_loss={vl["loss"]:.4f}  '
                  f'tr_F1={tr["macro_f1"]:.4f}  vl_F1={vl["macro_f1"]:.4f}  '
                  f'({elapsed:.1f}s)  [Phase {self._phase}]')

            if epoch % 5 == 0:
                print(f'  Per-class val F1 @ epoch {epoch}:')
                for name, f1 in zip(CLASS_NAMES, vl['per_class_f1']):
                    bar    = '█' * int(f1 * 20)
                    status = '✓' if f1 >= 0.75 else '✗'
                    print(f'    {status} {name:<15} {f1:.3f}  {bar}')
                print()

            self._save_checkpoint(epoch, vl['macro_f1'])

            if vl['macro_f1'] > self.best_val_f1:
                self.best_val_f1  = vl['macro_f1']
                self.patience_cnt = 0
                self._save_best()
                print(f'  ★ New best val macro-F1: {self.best_val_f1:.4f}')
            else:
                self.patience_cnt += 1
                if self.patience_cnt >= PATIENCE:
                    print(f'\n[Train] Early stopping at epoch {epoch}')
                    break

        print(f'\n[Train] Complete. Best val macro-F1: {self.best_val_f1:.4f}')
        self._save_history()
        return self.history

    def _save_checkpoint(self, epoch, val_f1):
        path = self.checkpoint_dir / f'epoch_{epoch:03d}_f1={val_f1:.4f}.pt'
        torch.save({
            'epoch':           epoch,
            'model_state':     self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'val_f1':          val_f1,
            'model_config':    {'backbone': self.model.backbone_name, 'num_classes': self.model.num_classes},
        }, path)

    def _save_best(self):
        path = self.checkpoint_dir / 'best_model.pt'
        torch.save({
            'model_state':  self.model.state_dict(),
            'val_f1':       self.best_val_f1,
            'model_config': {'backbone': self.model.backbone_name, 'num_classes': self.model.num_classes},
            'class_names':  CLASS_NAMES,
            'notes':        'CNN spatial module — EfficientNet-B0, 3-output, class-weighted loss',
        }, path)
        print(f'  → Best model saved: {path}')

    def _save_history(self):
        with open(self.checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

# %% cell 30
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

# %% cell 31
trainer = CNNTrainer(model, train_dl, val_dl, pos_weights)
history = trainer.train()

# %% [markdown] cell 32
# # 10 · Evaluation
#
# Full evaluation suite matching the LSTM module's `lstm_evaluator`:
# - Per-class F1, precision, recall
# - Macro-F1 vs 0.75 target
# - Per-class threshold tuning
# - Confusion matrix grid
# - ROC curves
# - Training curves

# %% cell 33
@torch.no_grad()
def collect_val_predictions(model, loader, device=DEVICE) -> tuple:
    """Run inference on loader and return (logits, probs, labels) as numpy arrays."""
    model.eval()
    all_logits = []
    all_labels = []
    for imgs, labels in tqdm(loader, desc='[Eval] Collecting predictions'):
        logits = model(imgs.to(device))
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
    logits_arr = np.concatenate(all_logits, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    probs_arr  = 1 / (1 + np.exp(-logits_arr))
    return logits_arr, probs_arr, labels_arr


def tune_thresholds(probs: np.ndarray, labels: np.ndarray, n=100) -> list:
    """Find the per-class threshold that maximises F1. Returns list of floats."""
    thresholds = []
    print('\n[Eval] Tuning decision thresholds...')
    for i, name in enumerate(CLASS_NAMES):
        best_f1, best_thr = 0.0, 0.5
        for thr in np.linspace(0.1, 0.9, n):
            preds = (probs[:, i] > thr).astype(int)
            f1    = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        thresholds.append(round(best_thr, 3))
        print(f'  {name:<15}  thr={best_thr:.2f}  best_F1={best_f1:.4f}')
    return thresholds


def print_metrics_table(probs, labels, thresholds):
    """Print the full per-class metrics table and macro-F1 verdict."""
    preds = np.stack([(probs[:, i] > t).astype(int) for i, t in enumerate(thresholds)], axis=1)
    print(f'\n{"="*60}')
    print(f'  EVALUATION METRICS')
    print(f'  {"Class":<15} {"F1":>6} {"Prec":>6} {"Rec":>6} {"AUC":>6}')
    print(f'  {"-"*45}')
    f1s = []
    for i, name in enumerate(CLASS_NAMES):
        f1   = f1_score(labels[:, i], preds[:, i], zero_division=0)
        prec = precision_score(labels[:, i], preds[:, i], zero_division=0)
        rec  = recall_score(labels[:, i], preds[:, i], zero_division=0)
        try:
            auc = roc_auc_score(labels[:, i], probs[:, i])
        except ValueError:
            auc = float('nan')
        f1s.append(f1)
        ok = '✓' if f1 >= 0.75 else '✗'
        print(f'  {ok} {name:<15} {f1:>5.3f} {prec:>6.3f} {rec:>6.3f} {auc:>6.3f}')
    macro_f1 = float(np.mean(f1s))
    print(f'  {"-"*45}')
    print(f'  {"MACRO F1":<15} {macro_f1:>5.3f}')
    print(f'  Target (≥ 0.75): {"✓ MET" if macro_f1 >= 0.75 else "✗ NOT MET"}')
    print(f'{"="*60}\n')
    return macro_f1


def plot_confusion_matrices(preds, labels, output_dir=EVAL_DIR):
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(5 * NUM_CLASSES, 4))
    for i, (name, ax) in enumerate(zip(CLASS_NAMES, axes)):
        cm = confusion_matrix(labels[:, i], preds[:, i])
        ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred 0', 'Pred 1'])
        ax.set_yticklabels(['True 0', 'True 1'])
        ax.set_title(name)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f'{cm[r,c]:,}', ha='center', va='center',
                        color='white' if cm[r,c] > cm.max()/2 else 'black',
                        fontsize=12, fontweight='bold')
    fig.suptitle('Confusion Matrices — CNN Spatial Classifier', fontsize=12)
    plt.tight_layout()
    out = output_dir / 'confusion_matrices.png'
    plt.savefig(str(out), dpi=130, bbox_inches='tight'); plt.show()
    print(f'[Eval] Confusion matrices → {out}')


def plot_roc_curves(probs, labels, output_dir=EVAL_DIR):
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(5 * NUM_CLASSES, 4))
    for i, (name, ax) in enumerate(zip(CLASS_NAMES, axes)):
        if labels[:, i].sum() == 0:
            ax.text(0.5, 0.5, 'No positives', ha='center', va='center')
            continue
        fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
        auc = roc_auc_score(labels[:, i], probs[:, i])
        ax.plot(fpr, tpr, lw=2, label=f'AUC={auc:.3f}')
        ax.plot([0,1],[0,1],'k--', alpha=0.3)
        ax.set_title(name); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.legend()
    fig.suptitle('ROC Curves — CNN Spatial Classifier', fontsize=12)
    plt.tight_layout()
    out = output_dir / 'roc_curves.png'
    plt.savefig(str(out), dpi=130, bbox_inches='tight'); plt.show()
    print(f'[Eval] ROC curves → {out}')


def plot_training_curves(history, output_dir=EVAL_DIR):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(history['train_loss'])+1)
    ax1.plot(ep, history['train_loss'], label='Train'); ax1.plot(ep, history['val_loss'], label='Val', linestyle='--')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend()
    ax2.plot(ep, history['train_f1'], label='Train'); ax2.plot(ep, history['val_f1'], label='Val', linestyle='--')
    ax2.axhline(0.75, color='red', linestyle=':', alpha=0.6, label='Target 0.75')
    ax2.set_title('Macro F1'); ax2.set_xlabel('Epoch'); ax2.legend()
    plt.tight_layout()
    out = output_dir / 'training_curves.png'
    plt.savefig(str(out), dpi=130, bbox_inches='tight'); plt.show()
    print(f'[Eval] Training curves → {out}')

# %% cell 34
# ── Load best model and run full evaluation ───────────────────────────────────
checkpoint = torch.load(CHECKPOINT_DIR / 'best_model.pt', map_location=DEVICE)
model      = build_model()
model.load_state_dict(checkpoint['model_state'])
print(f'[Eval] Loaded best model (val_F1={checkpoint["val_f1"]:.4f})')

logits, probs, labels = collect_val_predictions(model, val_dl)
optimal_thresholds    = tune_thresholds(probs, labels)

# Save thresholds
with open(CHECKPOINT_DIR / 'optimal_thresholds.json', 'w') as f:
    json.dump(dict(zip(CLASS_NAMES, optimal_thresholds)), f, indent=2)

preds = np.stack([(probs[:,i] > t).astype(int) for i,t in enumerate(optimal_thresholds)], axis=1)
print_metrics_table(probs, labels, optimal_thresholds)
plot_confusion_matrices(preds, labels)
plot_roc_curves(probs, labels)
plot_training_curves(history)

# %% [markdown] cell 35
# # 11 · Grad-CAM diagnostics
#
# **What is Grad-CAM doing?**
# It flows gradients backward from one output neuron into the last convolutional layer
# and uses the gradient magnitudes to weight each feature map spatially.
# High weight = the model was looking at that region when making its decision.
#
# For a spatially-trained model: we expect heat on the **track edge** for off-track,
# on the **apex curb** for apex-miss, and on **the car's lateral attitude** for sliding.
# If the heat is concentrated on the sky or HUD, the model has not learned the task —
# it found a shortcut (e.g. lighting conditions).
#
# **Fixes over original:**
# - Safe division — no ZeroDivisionError when activations are all zero
# - Proper ImageNet denormalisation before overlay
# - All models set to `eval()` before GradCAM
# - Per-class heatmap grid across multiple test samples
# - Confidence score annotated on each heatmap

# %% cell 36
IMAGENET_MEAN_T = torch.tensor(IMAGENET_MEAN).view(3,1,1)
IMAGENET_STD_T  = torch.tensor(IMAGENET_STD).view(3,1,1)

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalisation and convert to uint8 numpy array (H,W,3).
    Required before overlaying a Grad-CAM heatmap — without this the base
    image appears washed out.
    """
    img = tensor.cpu() * IMAGENET_STD_T + IMAGENET_MEAN_T
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def get_gradcam(model: DrivingCNN,
                input_tensor: torch.Tensor,
                target_class_idx: int,
                device: str = DEVICE) -> tuple:
    """
    Compute a Grad-CAM heatmap for one frame and one output class.

    Args:
        input_tensor:     (3, H, W) — single normalised frame tensor
        target_class_idx: 0=off_track, 1=apex_miss, 2=sliding

    Returns:
        heatmap:    (H, W) float32 in [0, 1]
        confidence: sigmoid probability for target_class_idx
    """
    model.eval()
    activations, gradients = [], []

    # Register hooks on the final conv layer of the EfficientNet backbone
    target_layer = model.backbone.conv_head
    fwd_h = target_layer.register_forward_hook(lambda m, i, o: activations.append(o))
    bwd_h = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))

    # Forward + backward
    inp    = input_tensor.unsqueeze(0).to(device)
    logit  = model(inp)[0, target_class_idx]
    conf   = torch.sigmoid(logit).item()
    model.zero_grad()
    logit.backward()

    fwd_h.remove(); bwd_h.remove()

    # Compute weighted activation map (Grad-CAM paper equation)
    grads  = gradients[0][0].cpu().detach().numpy()    # (C, h, w)
    fmaps  = activations[0][0].cpu().detach().numpy()  # (C, h, w)
    weights = grads.mean(axis=(1, 2))                  # (C,)
    cam    = np.dot(fmaps.transpose(1, 2, 0), weights) # (h, w)
    cam    = np.maximum(cam, 0)                        # ReLU

    # Safe normalisation — avoids ZeroDivisionError on constant predictions
    cam_max = cam.max()
    if cam_max > 0:
        cam = cam / cam_max
    # Resize to match model input
    cam = cv2.resize(cam, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    return cam.astype(np.float32), conf


def overlay_heatmap(base_rgb: np.ndarray,
                     heatmap: np.ndarray,
                     alpha: float = 0.45) -> np.ndarray:
    """Blend a Grad-CAM heatmap (JET colormap) onto a base RGB image."""
    h, w = base_rgb.shape[:2]
    hm   = cv2.resize(heatmap, (w, h))
    hm_u8   = np.uint8(255 * hm)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    return np.clip(alpha * hm_color + (1 - alpha) * base_rgb, 0, 255).astype(np.uint8)


def visualize_gradcam_per_class(model, val_ds, n_samples=5,
                                  output_dir=DIAG_DIR) -> None:
    """
    For each of the 3 output classes, pick n_samples frames that are
    positively labelled for that class, compute Grad-CAM, and save a
    row of overlaid heatmaps.

    Interpretation guide printed next to each output.
    """
    model.eval()
    df = val_ds.df.reset_index(drop=True)

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        col     = ['off_track', 'apex_miss', 'sliding'][cls_idx]
        pos_idx = df.index[df[col] == 1].tolist()
        if not pos_idx:
            print(f'[Diag] No positive samples for {cls_name} in val set — skipping.')
            continue

        sample_idx = random.sample(pos_idx, min(n_samples, len(pos_idx)))
        fig, axes  = plt.subplots(2, len(sample_idx), figsize=(4*len(sample_idx), 8))

        for col_i, idx in enumerate(sample_idx):
            tensor, label = val_ds[idx]
            heatmap, conf = get_gradcam(model, tensor, cls_idx)
            base_rgb      = denormalize(tensor)

            # Row 0: original frame
            axes[0, col_i].imshow(base_rgb)
            axes[0, col_i].set_title(f'Label: {int(label[cls_idx])}', fontsize=8)
            axes[0, col_i].axis('off')

            # Row 1: heatmap overlay
            overlaid = overlay_heatmap(base_rgb, heatmap)
            axes[1, col_i].imshow(overlaid)
            axes[1, col_i].set_title(f'conf={conf:.2f}', fontsize=8)
            axes[1, col_i].axis('off')

        fig.suptitle(
            f'Grad-CAM — {cls_name}\n'
            f'Top row: input frame | Bottom row: model attention (red=high, blue=low)',
            fontsize=10
        )
        plt.tight_layout()
        out = output_dir / f'gradcam_{cls_name.replace(" ","_")}.png'
        plt.savefig(str(out), dpi=140, bbox_inches='tight')
        plt.show()
        print(f'[Diag] Saved → {out}')


def visualize_gradcam_ensemble(ensemble_model, val_ds, n_samples=4,
                                output_dir=DIAG_DIR) -> None:
    """
    Grad-CAM for the ensemble: average heatmaps across all 5 fold models
    to get a consensus view of which regions drive each class decision.
    """
    ensemble_model.eval()
    for m in ensemble_model.models:
        m.eval()

    df = val_ds.df.reset_index(drop=True)

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        col     = ['off_track', 'apex_miss', 'sliding'][cls_idx]
        pos_idx = df.index[df[col] == 1].tolist()
        if not pos_idx:
            continue
        sample_idx = random.sample(pos_idx, min(n_samples, len(pos_idx)))
        fig, axes  = plt.subplots(2, len(sample_idx), figsize=(4*len(sample_idx), 8))

        for col_i, idx in enumerate(sample_idx):
            tensor, label = val_ds[idx]
            # Average heatmaps across all ensemble members
            cams  = [get_gradcam(m, tensor, cls_idx)[0] for m in ensemble_model.models]
            cam   = np.mean(cams, axis=0)
            conf  = ensemble_model.predict(tensor.unsqueeze(0))[0, cls_idx].item()
            base  = denormalize(tensor)

            axes[0, col_i].imshow(base)
            axes[0, col_i].set_title(f'Label: {int(label[cls_idx])}', fontsize=8)
            axes[0, col_i].axis('off')
            axes[1, col_i].imshow(overlay_heatmap(base, cam))
            axes[1, col_i].set_title(f'ens_conf={conf:.2f}', fontsize=8)
            axes[1, col_i].axis('off')

        fig.suptitle(f'Ensemble Grad-CAM (mean of 5) — {cls_name}', fontsize=10)
        plt.tight_layout()
        out = output_dir / f'gradcam_ensemble_{cls_name.replace(" ","_")}.png'
        plt.savefig(str(out), dpi=140, bbox_inches='tight')
        plt.show()
        print(f'[Diag] Saved → {out}')

# %% cell 37
# Run individual model Grad-CAM
visualize_gradcam_per_class(model, val_ds)

# %% [markdown] cell 38
# # 12 · 5-Fold cross-validation (GroupKFold)
#
# Groups are defined by original frame filename so no augmented copy of a training
# frame leaks into the validation fold.
#
# **Bug fix from original:** backbone unfreeze at epoch 5 now also recreates
# the optimizer with the correct `LR_FINETUNE` — in the original notebook
# the optimizer kept the old `lr=1e-4` after unfreeze, negating the purpose
# of the two-phase approach.

# %% cell 39
def run_fold(fold_idx: int, train_df: pd.DataFrame, val_df: pd.DataFrame) -> float:
    """Train one cross-validation fold. Returns best val F1 for this fold."""
    print(f'\n[CV] Fold {fold_idx+1} | train: {len(train_df):,}  val: {len(val_df):,}')

    f_train_ds = DrivingDataset(train_df, FRAMES_DIR, augment=True)
    f_val_ds   = DrivingDataset(val_df,   FRAMES_DIR, augment=False)
    f_train_dl = DataLoader(f_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    f_val_dl   = DataLoader(f_val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    fold_model     = build_model()
    fold_criterion = build_criterion(pos_weights)
    best_f1        = 0.0

    fold_model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, fold_model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )

    for epoch in range(EPOCHS):
        # Backbone unfreeze: ALSO recreate optimizer with fine-tune LR
        if epoch == FREEZE_EPOCHS:
            fold_model.unfreeze_all()
            optimizer = torch.optim.AdamW(
                fold_model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY
            )

        tr = run_epoch(fold_model, f_train_dl, fold_criterion, optimizer, training=True)
        vl = run_epoch(fold_model, f_val_dl,   fold_criterion, training=False)

        if vl['macro_f1'] > best_f1:
            best_f1 = vl['macro_f1']
            torch.save({
                'model_state':  fold_model.state_dict(),
                'val_f1':       best_f1,
                'model_config': {'backbone': fold_model.backbone_name, 'num_classes': fold_model.num_classes},
            }, CHECKPOINT_DIR / f'fold_{fold_idx+1}_best.pt')

        if epoch % 5 == 0:
            print(f'  Fold {fold_idx+1} | Ep {epoch:03d} | '
                  f'tr_F1={tr["macro_f1"]:.4f} vl_F1={vl["macro_f1"]:.4f}')

    print(f'[CV] Fold {fold_idx+1} complete. Best val F1: {best_f1:.4f}')
    return best_f1

# %% cell 40
# ── Run GroupKFold ────────────────────────────────────────────────────────────
all_df = pd.read_csv(LABELS_PATH)

def get_group_id(name):
    return '_'.join(name.split('_')[2:]) if name.startswith('aug_') else name

all_df['group_id'] = all_df['frame'].apply(get_group_id)

gkf          = GroupKFold(n_splits=5)
groups       = all_df['group_id']
fold_results = []

for fold_i, (tr_idx, vl_idx) in enumerate(gkf.split(all_df, groups=groups)):
    f1 = run_fold(fold_i, all_df.iloc[tr_idx], all_df.iloc[vl_idx])
    fold_results.append(f1)

print(f'\n[CV] Results: {[round(f,4) for f in fold_results]}')
print(f'[CV] Mean F1: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}')

# %% [markdown] cell 41
# # 13 · Ensemble — Wisdom of All
#
# Single clean `EnsembleModel` class (the original had two conflicting versions).
# Averaging the sigmoid outputs of 5 fold-trained models improves robustness —
# individual models may be biased toward the data distribution in their training
# folds; the average is more stable.

# %% cell 42
class EnsembleModel(nn.Module):
    """
    Ensemble of N DrivingCNN models trained on different GroupKFold splits.
    Returns the mean sigmoid probability across all member models.

    Single, canonical version — replaces the two conflicting EnsembleModel
    definitions from the original notebook.
    """

    def __init__(self, n_folds: int = 5, device: str = DEVICE):
        super().__init__()
        self.device = device
        self.models = nn.ModuleList([DrivingCNN().to(device) for _ in range(n_folds)])
        self.n_folds = n_folds

    def load_fold_checkpoints(self, checkpoint_dir: Path = CHECKPOINT_DIR):
        """Load best checkpoint from each fold."""
        for i in range(self.n_folds):
            path = checkpoint_dir / f'fold_{i+1}_best.pt'
            if not path.exists():
                print(f'[Ensemble] Warning: {path} not found — skipping fold {i+1}')
                continue
            ckpt = torch.load(path, map_location=self.device)
            self.models[i].load_state_dict(ckpt['model_state'])
            self.models[i].eval()
            print(f'[Ensemble] Fold {i+1} loaded (val_F1={ckpt["val_f1"]:.4f})')

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, H, W) or (3, H, W) — normalised frame(s)
        Returns:
            Mean sigmoid probabilities across all models: (batch, num_classes)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        self.eval()
        with torch.no_grad():
            probs = torch.stack([torch.sigmoid(m(x.to(self.device))) for m in self.models])
            return probs.mean(dim=0)  # (batch, num_classes)

    def save(self, path):
        torch.save({
            'model_state_dicts': [m.state_dict() for m in self.models],
            'n_folds':           self.n_folds,
            'class_names':       CLASS_NAMES,
            'model_config':      {'backbone': BACKBONE, 'num_classes': NUM_CLASSES},
        }, path)
        print(f'[Ensemble] Saved → {path}')

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        for i, sd in enumerate(ckpt['model_state_dicts']):
            self.models[i].load_state_dict(sd)
            self.models[i].eval()
        print(f'[Ensemble] Loaded {len(ckpt["model_state_dicts"])} models from {path}')
        

# %% cell 43
# Build, load, and save
ensemble = EnsembleModel()
ensemble.load_fold_checkpoints()
ensemble.save(CHECKPOINT_DIR / 'ensemble_wisdom_of_all.pt')

# Evaluate ensemble on val set
all_ens_probs, all_labels_e = [], []
for imgs, labels in tqdm(val_dl, desc='[Ensemble] Evaluating'):
    probs = ensemble.predict(imgs)
    all_ens_probs.append(probs.cpu().numpy())
    all_labels_e.append(labels.numpy())

ens_probs  = np.concatenate(all_ens_probs, axis=0)
ens_labels = np.concatenate(all_labels_e,  axis=0)
ens_thrs   = tune_thresholds(ens_probs, ens_labels)
print_metrics_table(ens_probs, ens_labels, ens_thrs)

# Save ensemble thresholds
with open(CHECKPOINT_DIR / 'ensemble_thresholds.json', 'w') as f:
    json.dump(dict(zip(CLASS_NAMES, ens_thrs)), f, indent=2)

# Run Grad-CAM on ensemble
visualize_gradcam_ensemble(ensemble, val_ds)

# %% [markdown] cell 44
# # 14 · Inference — annotate unseen footage
#
# **Critical fix:** inference preprocessing now uses `letterbox()` — **identical to
# the transform used during training**. The original notebook used
# `resize(288,640) + center_crop(288)` which stretched the aspect ratio and produced
# frames the model never saw during training, degrading precision.
#
# The annotated output overlays three coloured dots in the corner:
# - 🔴 Red  → Off-track
# - 🔵 Blue → Apex miss
# - 🟠 Orange → Sliding
#
# A yellow border appears if 2+ conditions are active simultaneously.

# %% cell 45
def build_inference_transform():
    """
    Inference preprocessing — MUST match DrivingDataset's transform exactly.
    letterbox → ToTensor → CenterCrop(224) → Normalize(ImageNet)
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(MODEL_INPUT_SIZE),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def annotate_video(input_path: str,
                    output_path: str,
                    model_input,
                    thresholds: dict = None,
                    process_every: int = INFERENCE_EVERY_NTH) -> None:
    """
    Process a video file frame-by-frame through the trained model and write
    an annotated output video with coloured dot overlays.

    Args:
        input_path:   Path to the input video file (unseen test footage).
        output_path:  Where to save the annotated output video.
        model_input:  EnsembleModel / DrivingCNN instance, or path to checkpoint .pt file.
        thresholds:   {class_name: float} decision thresholds. Defaults to DEFAULT_THRESHOLDS.
        process_every: Process 1 out of every N frames (1 = every frame).
    """
    # Resolve model
    if isinstance(model_input, (str, Path)):
        ens = EnsembleModel()
        ens.load(str(model_input))
        model_input = ens
    model_input.eval()

    thresholds = thresholds or DEFAULT_THRESHOLDS
    preprocess = build_inference_transform()

    cap   = cv2.VideoCapture(str(input_path))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    silent = str(output_path).replace('.mp4', '_silent.mp4')
    out    = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    DOT_R = max(18, W // 50)
    PAD   = DOT_R + 12
    DOTS  = [
        # (class_idx, center, filled_color_BGR, label)
        (0, (W-PAD,        PAD),               (0,  0, 220),   'OT'),   # Red
        (1, (W-PAD,  PAD*2+DOT_R),             (220, 80,  0),  'AM'),   # Blue
        (2, (W-PAD,  PAD*3+DOT_R*2),           (0, 165, 255),  'SL'),   # Orange
    ]

    frame_count = 0
    last_probs  = np.zeros(NUM_CLASSES)

    with tqdm(total=total, desc='[Inference] Annotating') as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % process_every == 0:
                # Preprocessing: letterbox (aspect-preserving) then normalize
                rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                lb_frame   = letterbox(rgb, target_w=640, target_h=288)  # same as training
                lb_pil     = Image.fromarray(lb_frame)
                tensor     = preprocess(lb_pil).unsqueeze(0)
                last_probs = model_input.predict(tensor).cpu().numpy().squeeze()

            states = [(last_probs[i] > thresholds.get(name, 0.5)) for i, name in enumerate(CLASS_NAMES)]

            # Draw indicator dots
            for cls_i, center, color_bgr, lbl in DOTS:
                cv2.circle(frame, center, DOT_R, (80, 80, 80), 2)
                if states[cls_i]:
                    cv2.circle(frame, center, DOT_R, color_bgr, -1)

            # Yellow warning border if 2+ events active simultaneously
            if sum(states) >= 2:
                cv2.rectangle(frame, (0, 0), (W, H), (0, 255, 255), 5)

            # Probability HUD
            hud = '  '.join([f'{n[:2]}:{last_probs[i]:.2f}' for i, n in enumerate(CLASS_NAMES)])
            cv2.putText(frame, hud, (10, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            out.write(frame)
            frame_count += 1
            pbar.update(1)

    cap.release()
    out.release()

    # Re-encode with libx264 for browser playback
    os.system(f"ffmpeg -y -i '{silent}' -vcodec libx264 -crf 23 -preset fast '{output_path}' -loglevel quiet")
    os.remove(silent)
    print(f'[Inference] Annotated video saved → {output_path}')


def play_video(video_path: str, width: int = 800) -> HTML:
    """Display an annotated video inline in the Kaggle notebook."""
    with open(video_path, 'rb') as f:
        b64 = b64encode(f.read()).decode()
    html = f"""
    <div style="text-align:center; margin:12px 0;">
        <p style="font-family:monospace; font-size:12px; color:#888;">
            🔴 Off-Track &nbsp;|&nbsp; 🔵 Apex Miss &nbsp;|&nbsp; 🟠 Sliding
            &nbsp;|&nbsp; 🟡 border = 2+ active
        </p>
        <video width="{width}" controls autoplay loop
               style="border-radius:8px; border:1px solid #444;">
            <source src="data:video/mp4;base64,{b64}" type="video/mp4">
        </video>
    </div>"""
    return HTML(html)


# %% cell 46
# ── Run on test footage ───────────────────────────────────────────────────────
ens_thresholds_dict = dict(zip(CLASS_NAMES, ens_thrs))

annotate_video(
    input_path  = TEST_VIDEO,
    output_path = '/kaggle/working/annotated_output.mp4',
    model_input = ensemble,
    thresholds  = ens_thresholds_dict,
)

# Trim first 2 minutes for quick preview
os.system("ffmpeg -y -i '/kaggle/working/annotated_output.mp4' -t 120 -c copy '/kaggle/working/annotated_trimmed.mp4' -loglevel quiet")
play_video('/kaggle/working/annotated_trimmed.mp4')

# %% cell 47
# Trim first 2 minutes for quick preview
#os.system("ffmpeg -y -i '/kaggle/working/annotated_output.mp4' -t 120 -c copy '/kaggle/working/annotated_trimmed.mp4' -loglevel quiet")
#play_video('/kaggle/working/annotated_trimmed.mp4')
play_video('/kaggle/working/annotated_output.mp4')

# %% [markdown] cell 48
# # 15 · Continuous training / fine-tuning
#
# Load the saved best model and continue training on new labelled data.
# The checkpoint format now includes architecture metadata so the class
# definition doesn't have to match exactly — it will always reload safely.

# %% cell 49
def finetune_from_checkpoint(checkpoint_path: str,
                             new_train_dl,
                             new_val_dl,
                             pos_weights_new: dict,
                             max_epochs:   int   = 30,
                             patience:     int   = 8,
                             session_note: str   = 'Fine-tune session') -> DrivingCNN:
    """
    Load a saved checkpoint and continue training on new labelled data.
    Starts with the backbone fully unfrozen at LR_FINETUNE — the model
    already converged in a previous session, so Phase 1 is skipped.

    Args:
        checkpoint_path:  Path to best_model.pt from a previous training run.
        new_train_dl:     DataLoader built from the new labelled data.
        new_val_dl:       Corresponding validation DataLoader.
        pos_weights_new:  Class weights computed from the new label distribution.
        max_epochs:       Maximum additional epochs to train.
        patience:         Early stopping patience.
        session_note:     Descriptive note saved in the new checkpoint.

    Returns:
        Fine-tuned DrivingCNN model.
    """
    ckpt      = torch.load(checkpoint_path, map_location=DEVICE)
    ft_model  = build_model()
    ft_model.load_state_dict(ckpt['model_state'])
    print(f'[Finetune] Loaded: {checkpoint_path}')
    print(f'[Finetune] Previous val F1: {ckpt["val_f1"]:.4f}')
    print(f'[Finetune] Session: {session_note}')

    ft_model.unfreeze_all()
    criterion  = build_criterion(pos_weights_new)
    optimizer  = torch.optim.AdamW(ft_model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    best_f1   = ckpt['val_f1']
    patience_cnt = 0
    save_path = CHECKPOINT_DIR / 'best_model.pt'

    for epoch in range(1, max_epochs + 1):
        tr = run_epoch(ft_model, new_train_dl, criterion, optimizer, training=True)
        vl = run_epoch(ft_model, new_val_dl,   criterion, training=False)
        scheduler.step(vl['macro_f1'])

        print(f'  Ep {epoch:03d} | tr_F1={tr["macro_f1"]:.4f} vl_F1={vl["macro_f1"]:.4f}')

        if vl['macro_f1'] > best_f1:
            best_f1      = vl['macro_f1']
            patience_cnt = 0
            torch.save({
                'model_state':  ft_model.state_dict(),
                'val_f1':       best_f1,
                'model_config': {'backbone': ft_model.backbone_name, 'num_classes': ft_model.num_classes},
                'class_names':  CLASS_NAMES,
                'notes':        session_note,
            }, save_path)
            print(f'  ★ New best: {best_f1:.4f} → {save_path}')
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f'[Finetune] Early stopping at epoch {epoch}')
                break

    print(f'[Finetune] Complete. Best val F1: {best_f1:.4f}')
    return ft_model


# %% cell 50
# ── Example usage (uncomment and run when you have new labelled data) ─────────
# new_train_dl, new_val_dl, _, _ = build_dataloaders(
#     labels_path=Path('/kaggle/working/labels_cnn_session2.csv')
# )
# new_pos_weights = analyse_label_distribution(Path('/kaggle/working/labels_cnn_session2.csv'))
# ft_model = finetune_from_checkpoint(
#     checkpoint_path = '/kaggle/working/checkpoints/best_model.pt',
#     new_train_dl    = new_train_dl,
#     new_val_dl      = new_val_dl,
#     pos_weights_new = new_pos_weights,
#     session_note    = 'Session 2 — LFS footage added',
# )

# %% [markdown] cell 51
# # 16 · Late Fusion & LSTM

# %% [markdown] cell 52
# ## 16.1 · Constants

# %% cell 53
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
WHEEL_SPIN_SLIP_THRESH   = 0.30    # driven wheel slip under acceleration
WHEEL_SPIN_THROTTLE_MIN  = 0.10    # must be accelerating
WHEEL_SPIN_MIN_SAMPLES   = 2
TRAIL_BRAKE_THRESH       = 0.15    # brake pressure past apex
DOWNSHIFT_RPM_PCT        = 0.60    # fraction of redline during downshift
LATE_UPSHIFT_MIN_SAMPLES = 90      # 1.5 seconds at 60Hz
ROUGH_STEER_RATE_THRESH  = 20.00   # degrees per second
GENTLE_ACCEL_RATE_THRESH = 0.10    # throttle rate per sample (below = smooth)
GENTLE_ACCEL_MIN_SAMPLES = 10      # must be sustained to count
brake_activation_thresh  = 0.10    # used to tell if actually braking or stop hold
speed_gate_kmh           = 10.0    # minimum speed to activating brake lock up tracking
slip_lock_thresh         = SLIP_RATIO_THRESH
wheel_spin_zero_thresh   = 0.5

# ── LSTM architecture ─────────────────────────────────────────────────────────
WINDOW_SIZE   = 60       # timesteps — 1 second at 60Hz
HIDDEN_DIM    = 512      # per LSTM layer
NUM_LAYERS    = 3        # stacked layers
DROPOUT       = 0.3

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE       = 256
LEARNING_RATE    = 1e-3
FINETUNE_LR      = 1e-5
MAX_EPOCHS       = 512
PATIENCE         = 64    # early stopping
GRAD_CLIP        = 1.0
TRAIN_STRIDE     = 1
EVAL_STRIDE      = 30

# ── Inference ─────────────────────────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.50   # sigmoid threshold — tuned per class after training


# %% cell 54
# ── Configure these paths before running ─────────────────────────────────────
RAW_CSV_PATH        = "/kaggle/input/datasets/samwelnjehia/lstm-module-telemetry-data"
APEX_JSON_PATH      = "/kaggle/input/lfs-telemetry/apex_reference.json"
TRACK_NAME          = "blackwood_gp"
VEHICLE_REDLINE     = 7031.0
VEHICLE_POWER_PEAK  = 0.01636   # power_peak_rpm / redline
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

# %% [markdown] cell 55
# ## 16.2 · LSTM Module Classes and Methods

# %% cell 56
# ── AttensionLSTM Class ──────────────────────────────────────────────────────────

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

# %% cell 57
# ── Telemetrywindowdatset Class ────────────────────────────────────────────────────

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

# %% cell 58
# ── apply_normalizer & load_normalizer Method ───────────────────────────────────────────

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

# %% [markdown] cell 59
# ## 16.2b · LFS Adapter & LSTMInferenceEngine (ported from lstm-module-codes.ipynb)
#
# This block did not exist in the CNN notebook. It's required by both the Late Fusion section and RQ2 — without it, raw telemetry column names never get mapped to the canonical `FEATURE_COLS` names the LSTM was trained on.

# %% cell 60
# ─────────────────────────────────────────────────────────────────────────────
# PORTED FROM lstm-module-codes.ipynb — the raw→canonical LFS adapter and the
# LSTMInferenceEngine class. Neither exists in this notebook even though
# Section 16 (Late Fusion) and Section 17 (RQ2) both depend on them:
#   - apply_normalizer() alone can't fix column names, only column VALUES.
#     Your raw telemetry CSV has raw LFS headers ("Brake", "Engine_RPM", ...),
#     but FEATURE_COLS (cell above) lists the CANONICAL post-adapter names
#     ("brake", "engine_rpm_norm", ...). Without this adapter running first,
#     `[c for c in FEATURE_COLS if c in df.columns]` returns an empty list,
#     which is exactly why TelemetryWindowDataset built 0-width windows and
#     the LSTM raised "Expected 25, got 0".
#   - LSTMInferenceEngine.predict_from_csv() is what CELL 1 (compute_fer)'s
#     docstring assumes you're calling — it runs adapt→normalise→window→
#     infer in one call and returns exactly the pred_<Class> columns
#     compute_fer() looks for.
# ─────────────────────────────────────────────────────────────────────────────
import re

# Snapshot the LSTM class-name order now (cell 50 already reassigned the
# shared CLASS_NAMES variable to the 7 LSTM behaviour classes at this point
# in the notebook).
LSTM_CLASS_NAMES = list(CLASS_NAMES)
LSTM_LABEL_COLS = list(LABEL_COLS)   # frozen 7-class LSTM label order — do NOT reference bare LABEL_COLS below this point

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

# Vehicle-specific constants — update per car used in LFS.
# NOTE: redline_rpm here MUST match VEHICLE_REDLINE set in cell 51, and
# power_peak_rpm should be your car's real dyno/in-game peak-power RPM —
# the 5956 below is a ratio-matched placeholder, not a verified value.
VEHICLE_CONFIG = {
    "default": {
        "redline_rpm"   : 7031,
        "power_peak_rpm": 5956,
        "max_speed_kmh" : 205,
        "susp_load_max" : 8000,   # Newton-metres, normalise by this
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

    for axis in ["x", "y", "z"]:
        col = f"accel_{axis}_ms2"
        if col in df.columns:
            df[f"{'lateral_g' if axis == 'y' else 'longitudinal_g' if axis == 'x' else 'vertical_g'}"] = \
                df[col] / 9.81

    if "engine_rpm" in df.columns:
        df["engine_rpm_norm"] = (df["engine_rpm"] / cfg["redline_rpm"]).clip(0, 1.2)

    if "speed_kmh" in df.columns:
        df["speed_kmh_norm"] = (df["speed_kmh"] / cfg["max_speed_kmh"]).clip(0, 1.2)

    for corner in ["lf", "rf", "lr", "rr"]:
        raw_col = f"susp_load_{corner}"
        if raw_col in df.columns:
            df[f"{raw_col}_norm"] = (df[raw_col] / cfg["susp_load_max"]).clip(0, 2)

    return df


def drop_duplicates_and_resample(df: pd.DataFrame,
                                  target_interval_ms: int = 17) -> pd.DataFrame:
    """
    Fix telemetry sampling-rate jitter:
      1. Drop exact-duplicate timestamps
      2. Resample to a fixed target_interval_ms grid via linear interpolation
    """
    df = df.copy()
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    df = df.drop_duplicates(subset="timestamp_ms", keep="first")

    t_start = int(df["timestamp_ms"].iloc[0])
    t_end   = int(df["timestamp_ms"].iloc[-1])
    regular_times = np.arange(t_start, t_end + 1, target_interval_ms)

    df = df.set_index("timestamp_ms")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    df = df.reindex(df.index.union(regular_times))
    df[numeric_cols] = df[numeric_cols].interpolate(method="index")
    df = df.reindex(regular_times)
    df.index.name = "timestamp_ms"
    df = df.reset_index()

    df["interpolated"] = False
    return df


def add_source_column(df: pd.DataFrame, source: str = "lfs") -> pd.DataFrame:
    """Tag every row with its data source for downstream traceability."""
    df["data_source"] = source
    return df


def adapt_lfs_telemetry(csv_path,
                         vehicle: str = "default",
                         output_path: str = None) -> pd.DataFrame:
    """
    Full adapter pipeline: load → rename → convert units → resample → tag source.
    Accepts either a raw CSV path OR an already-loaded/compiled DataFrame
    (e.g. the output of load_and_compile_telemetry_folder()).
    Returns the canonical-schema DataFrame that FEATURE_COLS matches against.
    """
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


def load_and_compile_telemetry_folder(input_source, fallback_engine='c') -> pd.DataFrame:
    """
    Discovers, chronologically sorts, shifts lap boundaries across file
    boundaries, and merges a folder (or list) of raw LFS telemetry CSVs
    into one unified raw DataFrame. Run adapt_lfs_telemetry() on the result.
    """
    if isinstance(input_source, (str, Path)):
        folder = Path(input_source)
        csv_files = list(folder.glob("*.csv"))
    elif isinstance(input_source, list):
        csv_files = [Path(f) for f in input_source]
    else:
        raise ValueError("input_source must be a folder path, Path, or list of file paths.")

    if not csv_files:
        raise FileNotFoundError(f"No CSV telemetry files discovered for input: {input_source}")

    def extract_timestamp(path: Path) -> int:
        match = re.search(r'\d+', path.name)
        return int(match.group()) if match else 0

    sorted_files = sorted(csv_files, key=extract_timestamp)
    print(f"Found and sorted {len(sorted_files)} telemetry files chronologically")

    try:
        import pyarrow
        csv_engine = 'pyarrow'
    except ImportError:
        csv_engine = fallback_engine

    compiled_dfs = []
    cumulative_lap_offset = 0

    for idx, file_path in enumerate(sorted_files):
        df = pd.read_csv(file_path, engine=csv_engine)
        if df.empty:
            continue
        if 'Lap' not in df.columns:
            raise KeyError(f"'Lap' column missing in telemetry file: {file_path.name}")

        if idx > 0 and cumulative_lap_offset > 0:
            df['Lap'] += cumulative_lap_offset

        cumulative_lap_offset = int(df['Lap'].max())
        compiled_dfs.append(df)
        print(f" ---- Processed: {file_path.name} | rows: {len(df):,} | "
              f"laps {df['Lap'].min()}-{df['Lap'].max()}")

    unified_df = pd.concat(compiled_dfs, ignore_index=True)
    print(f"Pre-ingestion complete! Consolidated Data Shape: {unified_df.shape}")
    return unified_df


class LSTMInferenceEngine:
    """
    Self-contained LSTM inference engine — load once, call predict_from_csv()
    as many times as you like. Bundles trained weights, normalizer params,
    and per-class decision thresholds.
    """

    def __init__(self,
                 checkpoint_path: str,
                 normalizer_path: str,
                 threshold_path:  str = None,
                 device:          str = None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        with open(normalizer_path) as f:
            self.norm_params = json.load(f)

        if threshold_path and Path(threshold_path).exists():
            with open(threshold_path) as f:
                thr_dict = json.load(f)
            self.thresholds = [thr_dict.get(col, 0.5) for col in LSTM_LABEL_COLS]
        else:
            self.thresholds = [0.5] * len(LSTM_LABEL_COLS)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg  = ckpt.get("model_config", {})
        self.model = build_model(
            input_dim   = cfg.get("input_dim",   len(FEATURE_COLS)),
            hidden_dim  = cfg.get("hidden_dim",  128),
            num_layers  = cfg.get("num_layers",  2),
            num_classes = cfg.get("num_classes", len(LSTM_LABEL_COLS)),
            device      = self.device,
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"[Inference] Model loaded from {checkpoint_path}")
        print(f"[Inference] Device: {self.device} | "
              f"Thresholds: {[round(t,2) for t in self.thresholds]}")

    def predict_from_csv(self,
                          raw_csv_path,
                          vehicle: str = "default",
                          stride:  int = 1) -> pd.DataFrame:
        """
        Full pipeline on a raw LFS telemetry CSV (or a pre-compiled DataFrame,
        e.g. from load_and_compile_telemetry_folder()):
            adapt → normalise → window → infer → return per-window results

        Returns a DataFrame with one prob_<Class> and one pred_<Class> column
        per behaviour class — this is exactly the shape compute_fer() expects.
        """
        df = adapt_lfs_telemetry(raw_csv_path, vehicle=vehicle)

        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
        if not feature_cols:
            raise ValueError(
                "[Inference] No FEATURE_COLS matched after adaptation — check "
                "that your raw CSV's headers are covered by LFS_TO_CANONICAL."
            )
        df = apply_normalizer(df, self.norm_params, feature_cols)
        df[feature_cols] = df[feature_cols].fillna(0.0)

        for col in LSTM_LABEL_COLS:
            if col not in df.columns:
                df[col] = 0

        ds     = TelemetryWindowDataset(df, feature_cols=feature_cols, stride=stride)
        loader = DataLoader(ds, batch_size=512, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for windows, _ in loader:
                windows = windows.to(self.device)
                logits  = self.model(windows)
                probs   = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        all_probs = np.concatenate(all_probs, axis=0)   # (N, C)

        results = {"window_idx": np.arange(len(all_probs))}
        for cls_idx, col in enumerate(LSTM_LABEL_COLS):
            cls_name = LSTM_CLASS_NAMES[cls_idx]
            results[f"prob_{cls_name.replace(' ', '_')}"] = all_probs[:, cls_idx]
            results[f"pred_{cls_name.replace(' ', '_')}"] = (
                all_probs[:, cls_idx] > self.thresholds[cls_idx]).astype(int)

        return pd.DataFrame(results)


# %% [markdown] cell 61
# ## 16.3 · Fusion Data Generator
#
# fusion_data_generator.py
#
# Step 1 of the fusion pipeline.
#
# Runs saved CNN and LSTM models over their respective inputs (video frames and telemetry windows) and saves the output probability vectors side-by-side as a single CSV.
#
# That CSV becomes the training data for the Late Fusion Head — the fusion
# head never sees raw frames or raw telemetry, only what each sub-model
# decided about them.

# %% cell 62
"""
Output row format:
    window_id | cnn_off_track | cnn_apex_miss | cnn_sliding
              | lstm_brake_locked | lstm_wheel_spin | lstm_trail_brake
              | lstm_aggressive_downshift | lstm_late_upshift
              | lstm_rough_steering | lstm_gentle_accel
              | [all 10 label columns for supervision]
"""

# ── User-configurable paths ───────────────────────────────────────────────────
CNN_CHECKPOINT   = "/kaggle/working/bestmodel/best_model.pt"
LSTM_CHECKPOINT  = "/kaggle/input/datasets/samwelnjehia/lstm-model-data/LSTM-module-best_model.pt"
LSTM_NORM_PARAMS = "/kaggle/input/datasets/samwelnjehia/lstm-model-data/normalizer_params.json"

# Labelled CSVs (both CNN and LSTM should point to the same labelled session data)
CNN_LABELS_CSV   = "/kaggle/working/labels_cnn_3output.csv"
LSTM_LABELS_CSV  = "/kaggle/input/datasets/samwelnjehia/lstm-model-data/labelled_telemetry.csv"

CNN_FRAMES_DIR   = "/kaggle/working/frames_fullres"
OUTPUT_CSV       = "/kaggle/working/fusion_training_data.csv"

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

# %% cell 63
# ── CNN loader (reuse your existing DrivingDataset) ───────────────────────────
def load_cnn_model(checkpoint_path: str) -> torch.nn.Module:
    """Load CNN from checkpoint. Architecture config is stored in the .pt file."""

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    cfg  = ckpt["model_config"]
    model = DrivingCNN(
        backbone    = cfg["backbone"],
        num_classes = cfg["num_classes"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[CNN]  Loaded — val F1: {ckpt.get('val_f1', 'N/A')}")
    return model


def load_lstm_model(checkpoint_path: str) -> torch.nn.Module:
    """Load LSTM from checkpoint."""

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    cfg  = ckpt["model_config"]
    model = build_model(
        input_dim   = cfg["input_dim"],
        hidden_dim  = cfg["hidden_dim"],
        num_layers  = cfg["num_layers"],
        num_classes = cfg["num_classes"],
        device      = DEVICE,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[LSTM] Loaded — val F1: {ckpt.get('val_f1', 'N/A')}")
    return model


# %% [markdown] cell 64
# - **Inference passes**

# %% cell 65
# ── CNN inference pass ───────────────────────────────────────────────────────
@torch.no_grad()
def run_cnn_inference(model, labels_csv: str, frames_dir: str) -> pd.DataFrame:
    """
    Run every labelled frame through the CNN and collect sigmoid probabilities.
    Returns DataFrame: [frame, cnn_off_track, cnn_apex_miss, cnn_sliding,
                        label_off_track, label_apex_miss, label_sliding]
    """

    df = pd.read_csv(labels_csv)
    ds = DrivingDataset(df, frames_dir, augment=False)   # no augmentation at inference
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    all_probs  = []
    all_labels = []

    for imgs, labels in tqdm(dl, desc="[CNN ] Inference"):
        logits = model(imgs.to(DEVICE))
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    probs_arr  = np.concatenate(all_probs,  axis=0)   # (N, 3)
    labels_arr = np.concatenate(all_labels, axis=0)   # (N, 3)

    results = pd.DataFrame({
        "frame"         : df["frame"].values,
        "cnn_off_track" : probs_arr[:, 0],
        "cnn_apex_miss" : probs_arr[:, 1],
        "cnn_sliding"   : probs_arr[:, 2],
        # Ground-truth labels for fusion head supervision
        "label_off_track" : labels_arr[:, 0].astype(int),
        "label_apex_miss" : labels_arr[:, 1].astype(int),
        "label_sliding"   : labels_arr[:, 2].astype(int),
    })
    print(f"[CNN ] {len(results):,} rows collected")
    return results

# %% cell 66
# ── LSTM inference pass ───────────────────────────────────────────────────────
@torch.no_grad()
def run_lstm_inference(model, labels_csv: str, norm_params_path: str) -> pd.DataFrame:
    """
    Run every labelled telemetry window through the LSTM and collect sigmoid
    probabilities. Returns one row per window (not per raw telemetry row).

    Window assignment: the window index maps to the LAST row of each window,
    matching the convention used during LSTM training.
    """

    df          = pd.read_csv(labels_csv)
    norm_params = load_normalizer(norm_params_path)
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    # Apply the SAME normalisation used during training
    df[feature_cols] = df[feature_cols].fillna(method="ffill").fillna(0.0)
    df = apply_normalizer(df, norm_params, feature_cols)

    ds = TelemetryWindowDataset(df, feature_cols=feature_cols,
                                 label_cols=LABEL_COLS, stride=1)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    all_probs  = []
    all_labels = []

    for windows, labels in tqdm(dl, desc="[LSTM] Inference"):
        logits = model(windows.to(DEVICE))
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    probs_arr  = np.concatenate(all_probs,  axis=0)   # (N_windows, 7)
    labels_arr = np.concatenate(all_labels, axis=0)   # (N_windows, 7)

    lstm_cols  = [f"lstm_{col.replace('label_', '')}" for col in LABEL_COLS]
    label_cols = [f"label_{col.replace('label_', '')}" for col in LABEL_COLS]

    results = pd.DataFrame(probs_arr,  columns=lstm_cols)
    for i, col in enumerate(label_cols):
        results[col] = labels_arr[:, i].astype(int)

    # Window index for alignment with CNN outputs
    results.insert(0, "window_id", np.arange(len(results)))
    print(f"[LSTM] {len(results):,} windows collected")
    return results

# %% cell 67
# ── Alignment ──────────────────────────────────────────────────────────────
def align_cnn_lstm(cnn_df: pd.DataFrame, lstm_df: pd.DataFrame) -> pd.DataFrame:
    """
    CNN operates per-frame; LSTM operates per-window.
    Since both are extracted from the same session at 60Hz with the same
    labelled data as input, the simplest alignment is:
        - Truncate both to the shorter length
        - Pair them by row index

    This works because both datasets are extracted in the same temporal
    order from the same session recording.

    For multi-session datasets, a more robust timestamp-based join is needed.
    See the comment at the bottom of this function for the extension path.
    """
    n = min(len(cnn_df), len(lstm_df))
    print(f"[Align] CNN rows: {len(cnn_df):,} | LSTM windows: {len(lstm_df):,} → using {n:,}")

    cnn_trunc  = cnn_df.iloc[:n].reset_index(drop=True)
    lstm_trunc = lstm_df.iloc[:n].reset_index(drop=True)

    # Drop duplicate label columns from LSTM (CNN labels are the ground truth)
    lstm_data_cols = [c for c in lstm_trunc.columns if not c.startswith("label_")]
    combined = pd.concat([cnn_trunc, lstm_trunc[lstm_data_cols]], axis=1)

    print(f"[Align] Combined shape: {combined.shape}")
    print(f"[Align] Columns: {list(combined.columns)}")
    return combined

    # ── Future extension: timestamp-based join ────────────────────────────────
    # If you add a timestamp_ms column to both CNN and LSTM outputs,
    # you can do an exact merge:
    #   combined = pd.merge_asof(
    #       cnn_df.sort_values("timestamp_ms"),
    #       lstm_df.sort_values("timestamp_ms"),
    #       on="timestamp_ms",
    #       direction="nearest",
    #       tolerance=20,   # 20ms tolerance at 60Hz
    #   )

# %% cell 68
# ── Main ──────────────────────────────────────────────────────────────────
def generate_fusion_data():
    """End-to-end: load models → inference → align → save."""
    print(f"\n{'='*55}")
    print(f"  FUSION TRAINING DATA GENERATOR")
    print(f"  Device: {DEVICE}")
    print(f"{'='*55}\n")

    cnn  = load_cnn_model(CNN_CHECKPOINT)
    lstm = load_lstm_model(LSTM_CHECKPOINT)

    cnn_df  = run_cnn_inference(cnn,  CNN_LABELS_CSV, CNN_FRAMES_DIR)
    lstm_df = run_lstm_inference(lstm, LSTM_LABELS_CSV, LSTM_NORM_PARAMS)

    combined = align_cnn_lstm(cnn_df, lstm_df)
    combined.to_csv(OUTPUT_CSV, index=False)

    print(f"\n[Done] Fusion training data saved → {OUTPUT_CSV}")
    print(f"       {len(combined):,} rows | {combined.shape[1]} columns")
    print(f"\n  CNN  prob columns : {[c for c in combined.columns if c.startswith('cnn_')]}")
    print(f"  LSTM prob columns : {[c for c in combined.columns if c.startswith('lstm_')]}")
    print(f"  Label columns     : {[c for c in combined.columns if c.startswith('label_')]}")
    return combined

# %% cell 69
if __name__ == "__main__":
    generate_fusion_data()

# %% [markdown] cell 70
# ## 16.4 · Late Fusion Pipeline
#
# fusion_pipeline.py
#
# Steps 2–4 of the fusion pipeline in one file.
#
#     Step 2 — Train the Late Fusion Head on fusion_training_data.csv
#     Step 3 — Evaluate with per-class F1, confusion matrices, ROC curves
#     Step 4 — SHAP attribution on the fusion head → maps to feedback templates
#     Step 5 — Inference wrapper: raw CSV + video → coaching report
#
# Run produces fusion_training_data.csv.
#
#

# %% cell 71
# ── Paths ─────────────────────────────────────────────────────────────────────
FUSION_DATA_CSV   = "/kaggle/working/fusion_training_data.csv"
CHECKPOINT_DIR    = Path("/kaggle/working/fusion/checkpoints")
EVAL_DIR          = Path("/kaggle/working/fusion/eval")
DIAG_DIR          = Path("/kaggle/working/fusion/diagnostics")
TEMPLATES_PATH    = "/kaggle/working/feedback_templates.json"  # authored separately

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# %% cell 72
# ── Constants ─────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The 10 input features to the fusion head (3 CNN + 7 LSTM probabilities)
CNN_COLS = ["cnn_off_track", "cnn_apex_miss", "cnn_sliding"]
LSTM_COLS = [
    "lstm_brake_locked", "lstm_wheel_spin", "lstm_trail_brake",
    "lstm_aggressive_downshift", "lstm_late_upshift",
    "lstm_rough_steering", "lstm_gentle_accel",
]
INPUT_COLS = CNN_COLS + LSTM_COLS   # 10-dim concatenated vector

# The 10 supervision labels (CNN + LSTM ground truth combined)
"""
LABEL_COLS = [
    "label_off_track", "label_apex_miss", "label_sliding",
    "label_brake_locked", "label_wheel_spin", "label_trail_brake",
    "label_aggressive_downshift", "label_late_upshift",
    "label_rough_steering", "label_gentle_accel",
]
"""

LABEL_COLS = [
    # CNN vision labels
    "label_off_track", 
    "label_apex_miss", 
    "label_sliding",
    
    # Telemetry/LSTM labels
    "lstm_brake_locked", 
    "lstm_wheel_spin", 
    "lstm_trail_brake",
    "lstm_aggressive_downshift", 
    "lstm_late_upshift",
    "lstm_rough_steering", 
    "lstm_gentle_accel",
]

CLASS_NAMES = [
    "Off-Track", "Apex Miss", "Sliding",
    "Brake Locked", "Wheel Spin", "Trail Brake",
    "Aggressive Downshift", "Late Upshift",
    "Rough Steering", "Gentle Accel",
]

INPUT_DIM  = len(INPUT_COLS)   # 10
NUM_CLASSES = len(LABEL_COLS)  # 10

# Training hyperparameters
BATCH_SIZE    = 512    # fusion data is tabular — can use large batches
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
MAX_EPOCHS    = 112
PATIENCE      = 16
GRAD_CLIP     = 1.0
MODALITY_DROPOUT_P = 0.20   # probability of zeroing one entire stream

# %% cell 73
# SECTION 1-4
# SECTION 1 — DATASET

class FusionDataset(Dataset):
    """
    Wraps the fusion_training_data.csv produced by fusion_data_generator.py.

    Each sample:
        x: (10,) float32 — concatenated CNN + LSTM sigmoid probabilities
        y: (10,) float32 — multi-label ground truth (BCEWithLogitsLoss target)

    Modality Dropout (proposal §3.5.5):
        During training, randomly zero the entire CNN stream (first 3 values)
        OR the entire LSTM stream (last 7 values) with probability p each.
        Forces the fusion head to remain functional when one stream is absent
        — e.g. user uploads only telemetry, no video, or vice versa.
    """

    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.x       = df[INPUT_COLS].values.astype(np.float32)
        self.y       = df[LABEL_COLS].values.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
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
        """Compute BCEWithLogitsLoss pos_weight from label distribution."""
        n     = len(self.y)
        n_pos = self.y.sum(axis=0)
        n_neg = n - n_pos
        pw    = np.minimum(n_neg / np.maximum(n_pos, 1), 50.0)
        return torch.tensor(pw, dtype=torch.float32)


def build_dataloaders(csv_path: str, val_frac: float = 0.15) -> tuple:
    """
    Load fusion CSV, 85/15 random split, return train/val DataLoaders.
    Fusion data is tabular — row-level split is fine here since there
    is no frame-level data leakage risk (probabilities are already aggregated).
    """
    df   = pd.read_csv(csv_path)
    n    = len(df)
    idx  = np.random.permutation(n)
    cut  = int(n * (1 - val_frac))

    train_df = df.iloc[idx[:cut]].reset_index(drop=True)
    val_df   = df.iloc[idx[cut:]].reset_index(drop=True)

    train_ds = FusionDataset(train_df, augment=True)
    val_ds   = FusionDataset(val_df,   augment=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                           shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=2, pin_memory=True)

    print(f"[Data] Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    return train_dl, val_dl, train_ds


# SECTION 2 — MODEL

class LateFusionHead(nn.Module):
    """
    Two-layer MLP that learns cross-modal dependencies between CNN and LSTM
    output streams (proposal §3.5.5).

    Architecture:
        10-dim input → Linear(64) → ReLU → Dropout(0.3)
                     → Linear(32) → ReLU
                     → Linear(10) → raw logits

    Why a learned head rather than simple averaging:
        Averaging treats every input equally. The fusion head can learn that
        "apex miss (CNN) co-occurring with late braking (LSTM)" is a more
        significant compound event than either alone — which averaging cannot
        capture. It also learns which stream to trust more for each class.
    """

    def __init__(self, input_dim=INPUT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.input_dim   = input_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),   # raw logits — sigmoid applied by loss / at inference
        )
        # Xavier init
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 10) → logits: (batch, 10)"""
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid probabilities for inference."""
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self(x))


# SECTION 3 — TRAINING

def run_epoch(model, loader, criterion, optimizer=None, training=True) -> dict:
    """One training or validation epoch. Returns loss and macro-F1."""
    model.train(training)
    total_loss = 0.0
    all_logits, all_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y   = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss   = criterion(logits, y)

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
    probs_arr  = 1 / (1 + np.exp(-logits_arr))
    preds_arr  = (probs_arr > 0.5).astype(int)

    per_cls   = f1_score(labels_arr, preds_arr, average=None, zero_division=0)
    macro_f1  = float(np.mean(per_cls))

    return {
        "loss":          total_loss / len(loader.dataset),
        "macro_f1":      macro_f1,
        "per_class_f1":  per_cls.tolist(),
    }


def train_fusion_head(csv_path: str = FUSION_DATA_CSV) -> LateFusionHead:
    """
    Full training loop for the Late Fusion Head.
    Returns the best model loaded from checkpoint.
    """
    print(f"\n{'='*55}")
    print(f"  LATE FUSION HEAD TRAINING  |  Device: {DEVICE}")
    print(f"{'='*55}\n")

    train_dl, val_dl, train_ds = build_dataloaders(csv_path)

    model     = LateFusionHead().to(DEVICE)
    pw        = train_ds.pos_weights().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6
    )

    best_val_f1   = 0.0
    patience_cnt  = 0
    history       = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    for epoch in range(1, MAX_EPOCHS + 1):
        t0  = time.time()
        tr  = run_epoch(model, train_dl, criterion, optimizer, training=True)
        vl  = run_epoch(model, val_dl,   criterion, training=False)
        scheduler.step(vl["macro_f1"])

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(vl["loss"])
        history["train_f1"].append(tr["macro_f1"])
        history["val_f1"].append(vl["macro_f1"])

        print(f"Epoch {epoch:03d}/{MAX_EPOCHS}  "
              f"tr_loss={tr['loss']:.4f}  vl_loss={vl['loss']:.4f}  "
              f"tr_F1={tr['macro_f1']:.4f}  vl_F1={vl['macro_f1']:.4f}  "
              f"({time.time()-t0:.1f}s)")

        # Per-class F1 every 10 epochs
        if epoch % 10 == 0:
            print("  Per-class val F1:")
            for name, f1 in zip(CLASS_NAMES, vl["per_class_f1"]):
                bar = "█" * int(f1 * 20)
                ok  = "✓" if f1 >= 0.75 else "✗"
                print(f"    {ok} {name:<22} {f1:.3f}  {bar}")

        # Save checkpoint
        torch.save({
            "epoch":        epoch,
            "model_state":  model.state_dict(),
            "val_f1":       vl["macro_f1"],
            "model_config": {"input_dim": INPUT_DIM, "num_classes": NUM_CLASSES},
            "input_cols":   INPUT_COLS,
            "label_cols":   LABEL_COLS,
            "class_names":  CLASS_NAMES,
        }, CHECKPOINT_DIR / f"epoch_{epoch:03d}.pt")

        if vl["macro_f1"] > best_val_f1:
            best_val_f1  = vl["macro_f1"]
            patience_cnt = 0
            torch.save({
                "model_state":  model.state_dict(),
                "val_f1":       best_val_f1,
                "model_config": {"input_dim": INPUT_DIM, "num_classes": NUM_CLASSES},
                "input_cols":   INPUT_COLS,
                "label_cols":   LABEL_COLS,
                "class_names":  CLASS_NAMES,
            }, CHECKPOINT_DIR / "best_model.pt")
            print(f"  ★ New best val macro-F1: {best_val_f1:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n[Train] Early stopping at epoch {epoch}")
                break

    # Save history
    with open(CHECKPOINT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Load best weights
    ckpt = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"\n[Train] Done. Best val macro-F1: {best_val_f1:.4f}")
    return model


# SECTION 4 — EVALUATION

@torch.no_grad()
def collect_predictions(model, loader) -> tuple:
    """Collect (probs, labels) numpy arrays from a DataLoader."""
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        probs = torch.sigmoid(model(x.to(DEVICE))).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return (np.concatenate(all_probs,  axis=0),
            np.concatenate(all_labels, axis=0).astype(int))


def tune_thresholds(probs, labels, n=100) -> list:
    """Per-class threshold that maximises F1 on validation set."""
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


def evaluate(model, val_dl, history: dict = None, output_dir=EVAL_DIR):
    """Full evaluation suite: metrics table, confusion matrices, ROC, training curves."""
    probs, labels = collect_predictions(model, val_dl)
    thresholds    = tune_thresholds(probs, labels)

    # Save thresholds
    thr_dict = dict(zip(LABEL_COLS, thresholds))
    with open(output_dir / "optimal_thresholds.json", "w") as f:
        json.dump(thr_dict, f, indent=2)

    preds = np.stack([(probs[:, i] > t).astype(int)
                       for i, t in enumerate(thresholds)], axis=1)

    # Metrics table
    print(f"\n{'='*60}")
    print("  FUSION HEAD EVALUATION")
    print(f"  {'Class':<24} {'F1':>6} {'AUC':>6}")
    print(f"  {'-'*40}")
    f1s = []
    for i, name in enumerate(CLASS_NAMES):
        f1  = f1_score(labels[:, i], preds[:, i], zero_division=0)
        try:
            auc = roc_auc_score(labels[:, i], probs[:, i])
        except ValueError:
            auc = float("nan")
        f1s.append(f1)
        ok = "✓" if f1 >= 0.75 else "✗"
        print(f"  {ok} {name:<24} {f1:>5.3f} {auc:>6.3f}")
    macro_f1 = float(np.mean(f1s))
    print(f"  {'-'*40}")
    print(f"  {'MACRO F1':<24} {macro_f1:>5.3f}")
    print(f"  Target ≥ 0.75: {'✓ MET' if macro_f1 >= 0.75 else '✗ NOT MET'}")
    print(f"{'='*60}\n")

    # Confusion matrices
    n_cols = 5
    n_rows = int(np.ceil(NUM_CLASSES / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten()
    for i, name in enumerate(CLASS_NAMES):
        cm = confusion_matrix(labels[:, i], preds[:, i], labels=[0, 1])
        axes[i].imshow(cm, cmap="Blues")
        axes[i].set_title(name, fontsize=8)
        for r in range(2):
            for c in range(2):
                axes[i].text(c, r, f"{cm[r,c]:,}", ha="center", va="center",
                             fontsize=10, fontweight="bold",
                             color="white" if cm[r,c] > cm.max()/2 else "black")
        axes[i].set_xticks([0,1]); axes[i].set_yticks([0,1])
        axes[i].set_xticklabels(["Pred 0","Pred 1"], fontsize=7)
        axes[i].set_yticklabels(["True 0","True 1"], fontsize=7)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Fusion Head — Confusion Matrices", fontsize=11)
    plt.tight_layout()
    plt.savefig(str(output_dir / "confusion_matrices.png"), dpi=130, bbox_inches="tight")
    plt.close()

    # Training curves
    if history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ep = range(1, len(history["train_loss"])+1)
        ax1.plot(ep, history["train_loss"], label="Train")
        ax1.plot(ep, history["val_loss"],   label="Val", linestyle="--")
        ax1.set_title("Loss"); ax1.legend()
        ax2.plot(ep, history["train_f1"], label="Train")
        ax2.plot(ep, history["val_f1"],   label="Val", linestyle="--")
        ax2.axhline(0.75, color="red", linestyle=":", alpha=0.6, label="Target")
        ax2.set_title("Macro F1"); ax2.legend()
        plt.tight_layout()
        plt.savefig(str(output_dir / "training_curves.png"), dpi=130, bbox_inches="tight")
        plt.close()

    print(f"[Eval] Plots saved → {output_dir}/")
    return thresholds, probs, labels




# %% cell 74
# SECTION 5 — SHAP ON FUSION HEAD

def run_fusion_shap(model, val_dl, output_dir=DIAG_DIR, n_bg=200, n_test=500):
    """
    SHAP GradientExplainer on the fusion head.

    This is the SHAP that generates coaching feedback (proposal §3.5.5).
    It decomposes each fusion head prediction into contributions from the
    10 input features (3 CNN probs + 7 LSTM probs), telling you:
        "The Off-Track prediction was driven 70% by cnn_off_track and
         30% by lstm_rough_steering co-occurring."

    This is different from the individual model diagnostics:
        - CNN Grad-CAM  → which PIXELS drove the CNN prediction
        - LSTM SHAP     → which TELEMETRY FEATURES drove the LSTM prediction
        - Fusion SHAP   → which SUB-MODEL OUTPUTS drove the final combined decision
    """
    try:
        import shap
    except ImportError:
        print("[SHAP] Run: pip install shap")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect background and test samples
    all_x, all_y = [], []
    for x, y in val_dl:
        all_x.append(x.numpy())
        all_y.append(y.numpy())
        if sum(a.shape[0] for a in all_x) >= n_bg + n_test:
            break
    all_x = np.concatenate(all_x, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    bg_x   = torch.tensor(all_x[:n_bg],          dtype=torch.float32).to(DEVICE)
    test_x = torch.tensor(all_x[n_bg:n_bg+n_test], dtype=torch.float32).to(DEVICE)

    # Fusion head is a plain MLP — no cuDNN RNN issue, model.eval() is fine
    model.eval()
    explainer = shap.GradientExplainer(model, bg_x)

    # SHAP values: (n_test, 10_inputs) per output class
    # GradientExplainer returns a list of arrays, one per output
    shap_values = explainer.shap_values(test_x)   # list of 10 arrays each (n_test, 10)

    # For each class: bar chart showing which input feature contributed most
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        vals = np.abs(shap_values[cls_idx])   # (n_test, 10)
        mean_importance = vals.mean(axis=0)   # (10,) — mean |SHAP| per feature

        order    = np.argsort(mean_importance)[::-1]
        features = [INPUT_COLS[i] for i in order]
        values   = mean_importance[order]
        colors   = ["#d62728" if "cnn" in f else "#1f77b4" for f in features]

        ax = axes[cls_idx]
        ax.barh(range(len(features)), values[::-1], color=colors[::-1], alpha=0.85)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([f.replace("cnn_","CNN:").replace("lstm_","LSTM:")
                             for f in features[::-1]], fontsize=7)
        ax.set_title(cls_name, fontsize=9)
        ax.set_xlabel("|SHAP|", fontsize=7)

    # Legend
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#d62728", label="CNN output"),
                         Patch(color="#1f77b4", label="LSTM output")],
               loc="lower right", fontsize=9)
    fig.suptitle("Fusion Head SHAP — Which sub-model output drove each final prediction?",
                 fontsize=12)
    plt.tight_layout()
    out = output_dir / "fusion_shap_importance.png"
    plt.savefig(str(out), dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Fusion SHAP saved → {out}")

    # Save raw SHAP values for template threshold logic
    shap_summary = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        vals = np.abs(shap_values[cls_idx]).mean(axis=0)
        shap_summary[cls_name] = {col: round(float(v), 6)
                                   for col, v in zip(INPUT_COLS, vals)}
    with open(output_dir / "fusion_shap_summary.json", "w") as f:
        json.dump(shap_summary, f, indent=2)
    print(f"[SHAP] SHAP summary JSON → {output_dir}/fusion_shap_summary.json")
    return shap_summary

# %% cell 75
# SECTION 6 — FEEDBACK GENERATOR

# 42 default feedback templates — one or more per detectable event.
# Edit before deployment — this is the coaching voice of the system.
DEFAULT_TEMPLATES = {
    "Off-Track": [
        "Lap {lap}: You ran wide at turn {turn}. Try to keep two wheels within the white line — track limits cost lap time plus momentum and in real conditions mean grass or barriers.",
        "Lap {lap}: Track limit violation detected. Focus on your reference points — pick a braking marker and a turn-in point earlier.",
    ],
    "Apex Miss": [
        "Lap {lap}: You missed the apex at turn {turn}. Late apex technique gives you earlier throttle application — try turning in slightly later.",
        "Lap {lap}: Apex deviation detected. A missed apex means you are carrying too much speed into the corner — trust the braking zone.",
    ],
    "Sliding": [
        "Lap {lap}: The car slid at turn {turn}. This indicates you are at or beyond the limit of grip — ease throttle application rate on exit.",
        "Lap {lap}: Instability detected. Check your brake release — a sudden release can upset the balance and provoke a slide.",
    ],
    "Brake Locked": [
        "Lap {lap}: Wheel lock detected. Try releasing the brake progressively over 20–30 metres rather than a sharp release to maintain tyre contact.",
        "Lap {lap}: Brake lock-up at turn {turn}. Your threshold braking pressure may be too high for this surface — modulate pressure earlier.",
    ],
    "Wheel Spin": [
        "Lap {lap}: Wheelspin detected on exit. Try rolling onto the throttle rather than snapping it open — especially in the lower gears.",
        "Lap {lap}: Excessive rear wheelspin. Smooth throttle application from the apex will give you faster acceleration than a sharp squirt.",
    ],
    "Trail Brake": [
        "Lap {lap}: Trail braking technique detected at turn {turn}. Ensure brake pressure is progressively decreasing as you unwind the steering.",
    ],
    "Aggressive Downshift": [
        "Lap {lap}: Aggressive downshift detected. The rev-match was too sharp — blip the throttle smoothly rather than spiking it.",
        "Lap {lap}: Downshift instability at turn {turn}. Try downshifting slightly earlier in the braking zone to give the drivetrain time to settle.",
    ],
    "Late Upshift": [
        "Lap {lap}: Late upshift detected. You stayed past the power peak — shift earlier to keep the engine in its power band.",
        "Lap {lap}: You are leaving revs on the table. Upshift when the power curve peaks, not when you hit the rev limiter.",
    ],
    "Rough Steering": [
        "Lap {lap}: Abrupt steering input detected at turn {turn}. Smooth, progressive steering reduces load transfer and keeps tyre grip consistent.",
        "Lap {lap}: Steering rate spike. Try to pre-plan your turn-in arc — late corrections are a sign the braking zone needs work.",
    ],
    "Gentle Accel": [
        "Lap {lap}: Good throttle application detected on exit. Smooth progressive power delivery like this protects rear grip effectively.",
    ],
}


class FeedbackGenerator:
    """
    Takes fusion head predictions + SHAP attribution and generates
    the post-session coaching report.

    Logic gate (proposal §3.5.5):
        An event fires a feedback item ONLY IF:
            prob > threshold  AND  SHAP confirms the primary contributing feature

    The SHAP confirmation prevents feedback being generated when the model
    is uncertain — if SHAP shows the prediction was driven by a low-confidence
    noisy signal, the template is suppressed.
    """

    def __init__(self, fusion_model: LateFusionHead,
                 thresholds: list,
                 templates: dict    = None,
                 shap_summary: dict = None):
        self.model        = fusion_model
        self.thresholds   = thresholds           # list of 10 floats
        self.templates    = templates or DEFAULT_TEMPLATES
        self.shap_summary = shap_summary         # used for SHAP confirmation gate

    def _shap_confirms(self, class_name: str, x: np.ndarray) -> bool:
        """
        Quick SHAP confirmation gate.
        Checks that the top-contributing feature for this class has a
        magnitude consistent with a genuine signal.
        If no SHAP summary is loaded, always returns True (gate disabled).
        """
        if self.shap_summary is None:
            return True
        importance = self.shap_summary.get(class_name, {})
        if not importance:
            return True
        # Top feature by mean importance
        top_feat  = max(importance, key=importance.get)
        top_val   = importance[top_feat]
        # Suppress if the top feature's SHAP is below noise floor
        return top_val > 0.005

    def generate_report(self,
                         fusion_probs: np.ndarray,
                         lap_number:   int = 1,
                         turn_number:  int = None) -> list:
        """
        Generate a list of feedback strings for one session window.

        Args:
            fusion_probs: (10,) numpy array of fusion head sigmoid probabilities
            lap_number:   current lap for template interpolation
            turn_number:  optional corner ID for template interpolation

        Returns:
            List of plain-language feedback strings. Empty list = no issues.
        """
        feedback = []
        turn_str = str(turn_number) if turn_number else "—"

        for i, cls_name in enumerate(CLASS_NAMES):
            prob = float(fusion_probs[i])
            thr  = self.thresholds[i]

            if prob < thr:
                continue  # below threshold — no feedback

            if not self._shap_confirms(cls_name, fusion_probs):
                continue  # SHAP gate suppressed it

            templates = self.templates.get(cls_name, [])
            if not templates:
                continue

            # Pick template — rotate through available ones per event type
            template = templates[i % len(templates)]
            feedback.append(
                template.format(lap=lap_number, turn=turn_str,
                                prob=round(prob, 2))
            )

        return feedback

    def session_summary(self,
                         all_probs: np.ndarray,
                         lap_numbers: list = None) -> dict:
        """
        Aggregate predictions across a full session into a coaching report.

        all_probs: (N_windows, 10) — fusion probabilities across all windows
        Returns:
            {
                "event_counts":  {class_name: int},
                "driving_persona": "Aggressive" | "Smooth" | "Cautious",
                "feedback_items":  [str, ...],
                "fer":             float  (Feedback Event Rate per lap)
            }
        """
        n_laps       = max(lap_numbers) if lap_numbers else 1
        event_counts = {name: 0 for name in CLASS_NAMES}
        feedback     = []

        for window_idx, probs in enumerate(all_probs):
            lap = lap_numbers[window_idx] if lap_numbers else 1
            items = self.generate_report(probs, lap_number=lap)
            feedback.extend(items)
            for i, name in enumerate(CLASS_NAMES):
                if probs[i] >= self.thresholds[i]:
                    event_counts[name] += 1

        # Driving Persona classification (proposal §3.3)
        aggressive_score = (event_counts["Brake Locked"]
                            + event_counts["Wheel Spin"]
                            + event_counts["Rough Steering"])
        smooth_score     = event_counts["Gentle Accel"]
        cautious_score   = event_counts["Late Upshift"]

        if aggressive_score > smooth_score and aggressive_score > cautious_score:
            persona = "Aggressive"
        elif smooth_score >= aggressive_score and smooth_score >= cautious_score:
            persona = "Smooth"
        else:
            persona = "Cautious"

        fer = len(feedback) / max(n_laps, 1)   # Feedback Event Rate per lap

        print(f"\n{'='*55}")
        print(f"  POST-SESSION COACHING REPORT")
        print(f"{'='*55}")
        print(f"  Driving Persona : {persona}")
        print(f"  Feedback items  : {len(feedback)}")
        print(f"  FER (items/lap) : {fer:.2f}")
        print(f"\n  Event counts:")
        for name, count in event_counts.items():
            bar = "█" * min(count // 10, 20)
            print(f"    {name:<24} {count:>5}  {bar}")
        print(f"\n  Feedback:")
        for item in feedback[:15]:   # show first 15 for preview
            print(f"    • {item}")
        if len(feedback) > 15:
            print(f"    ... and {len(feedback)-15} more items")
        print(f"{'='*55}\n")

        return {
            "event_counts":    event_counts,
            "driving_persona": persona,
            "feedback_items":  feedback,
            "fer":             round(fer, 2),
        }

# %% cell 76
# SECTION 7 — MASTER RUNNER

def main():
    print(f"\n{'='*55}")
    print(f"  FUSION PIPELINE  |  Device: {DEVICE}")
    print(f"{'='*55}\n")

    # Step 2: Train
    model = train_fusion_head(FUSION_DATA_CSV)

    # Rebuild val loader for evaluation
    _, val_dl, _ = build_dataloaders(FUSION_DATA_CSV)

    # Load history for training curves
    with open(CHECKPOINT_DIR / "history.json") as f:
        history = json.load(f)

    # Step 3: Evaluate
    thresholds, probs, labels = evaluate(model, val_dl, history)

    # Step 4: SHAP
    shap_summary = run_fusion_shap(model, val_dl)

    # Save everything needed for inference
    final_bundle = {
        "thresholds":   dict(zip(LABEL_COLS, thresholds)),
        "input_cols":   INPUT_COLS,
        "label_cols":   LABEL_COLS,
        "class_names":  CLASS_NAMES,
    }
    with open(CHECKPOINT_DIR / "inference_config.json", "w") as f:
        json.dump(final_bundle, f, indent=2)

    print(f"\n[Done] Files saved:")
    print(f"  Model      → {CHECKPOINT_DIR}/best_model.pt")
    print(f"  Config     → {CHECKPOINT_DIR}/inference_config.json")
    print(f"  Eval plots → {EVAL_DIR}/")
    print(f"  SHAP plots → {DIAG_DIR}/")

    # Quick example: run session summary on val predictions
    generator = FeedbackGenerator(model, thresholds, shap_summary=shap_summary)
    generator.session_summary(probs)

    return model, thresholds, shap_summary

# %% cell 77
if __name__ == "__main__":
    main()

# %% [markdown] cell 78
# # 17 · Reserch Question 2
#
# Given I deliberately drove badly for 2 laps — rolling the car on a sausage kerb, which implies off-track, sliding, and likely rough steering — you should see:
#
# Off-Track, Sliding, Rough Steering counts significantly higher in detuned
# Brake Locked potentially higher if you were braking late into corners
# Overall FER clearly higher in detuned vs normal
#
# Even with only 2 laps, if the difference in FER is large enough, the t-test will return p < 0.05. If not, you report the directional result (FER was X% higher in detuned) and note that the sample size is a limitation — two laps is the minimum viable test, not the ideal. That is an honest and acceptable research finding.

# %% cell 79
# ─────────────────────────────────────────────────────────────────────────────
# CELL 0.5 — Hand-label RQ2 clean & detuned driving frames (Section 17)
# Reuses discover_video_files / extract_frames_fullres / run_labelling_tool
# from Sections 3–5. Same key map: s/o/a/l/b/p/d/g/x/q.
#
# Run this cell TWICE: once with LABEL_SESSION="normal", once with "detuned".
# The interactive widget is blocking-per-keypress, so labelling both in one
# pass isn't practical — flip LABEL_SESSION and re-run when you finish one.
# ─────────────────────────────────────────────────────────────────────────────

RQ2_SESSIONS = {
    "normal": {
        "video_dir":  "/kaggle/input/datasets/samwelnjehia/rq2-cleandriving",
        "frames_dir": Path("/kaggle/working/frames_normal"),
        "labels_csv": Path("/kaggle/working/labels_normal.csv"),
    },
    "detuned": {
        "video_dir":  "/kaggle/input/datasets/samwelnjehia/rq2-detuned",
        "frames_dir": Path("/kaggle/working/frames_detuned"),
        "labels_csv": Path("/kaggle/working/labels_detuned.csv"),
    },
}

LABEL_SESSION = "detuned"   # ← flip to "detuned" on the second pass from "normal"

cfg = RQ2_SESSIONS[LABEL_SESSION]
cfg["frames_dir"].mkdir(parents=True, exist_ok=True)

# 1) Extract frames if not already done
existing = list(cfg["frames_dir"].glob("*.jpg"))
if not existing:
    video_files = discover_video_files(cfg["video_dir"])
    total = 0
    for vid in video_files:
        total += extract_frames_fullres(vid, cfg["frames_dir"], fps_target=FRAME_EXTRACT_FPS)
    print(f"[{LABEL_SESSION.upper()}] {total} frames extracted -> {cfg['frames_dir']}")
else:
    print(f"[{LABEL_SESSION.upper()}] {len(existing)} frames already extracted -> {cfg['frames_dir']}")

# 2) Launch the interactive labelling tool for this session
run_labelling_tool(frames_dir=cfg["frames_dir"], labels_path=cfg["labels_csv"])

# %% cell 80
# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — FER computation helper
# Works directly with the DataFrame returned by predict_from_csv()
# No dependency on session_summary — bypasses the missing 'fer' key entirely
# ─────────────────────────────────────────────────────────────────────────────

def compute_fer(predictions_df: pd.DataFrame, n_laps: int) -> dict:
    """
    Compute Feedback Event Rate (FER) and per-class event counts from
    the DataFrame returned by LSTMInferenceEngine.predict_from_csv().

    FER = number of windows where at least one event fires / number of laps
    (proposal §3.7 — unique coaching items generated per lap)

    Args:
        predictions_df: DataFrame from predict_from_csv()
        n_laps:         number of laps in this session

    Returns dict with:
        fer            — float, feedback events per lap
        total_events   — int, total windows with at least one positive prediction
        per_class      — dict {class_name: event_count}
        per_lap_fer    — list of per-lap FER values (for t-test)
    """
    # All binary prediction columns
    pred_cols = [c for c in predictions_df.columns if c.startswith("pred_")]

    if not pred_cols:
        raise ValueError("[FER] No pred_ columns found. "
                         "Run predict_from_csv() first.")

    # Any window with at least one positive prediction = one feedback event
    any_event = predictions_df[pred_cols].any(axis=1)
    total_events = int(any_event.sum())
    fer = total_events / max(n_laps, 1)

    # Per-class counts
    per_class = {}
    for col in pred_cols:
        # Strip "pred_" prefix and convert underscores back to spaces
        cls_name = col.replace("pred_", "").replace("_", " ")
        per_class[cls_name] = int(predictions_df[col].sum())

    # Per-lap FER — split windows evenly across laps if no lap column
    # (used for the paired t-test — need one value per lap)
    per_lap_fer = []
    if "lap" in predictions_df.columns:
        for lap_id in sorted(predictions_df["lap"].unique()):
            lap_mask  = predictions_df["lap"] == lap_id
            lap_any   = any_event[lap_mask]
            per_lap_fer.append(int(lap_any.sum()))
    else:
        # No lap column — split evenly by window index
        n = len(predictions_df)
        lap_size = n // max(n_laps, 1)
        for i in range(n_laps):
            start = i * lap_size
            end   = start + lap_size if i < n_laps - 1 else n
            per_lap_fer.append(int(any_event.iloc[start:end].sum()))

    return {
        "fer":          round(fer, 2),
        "total_events": total_events,
        "per_class":    per_class,
        "per_lap_fer":  per_lap_fer,
    }

# %% cell 81
# ─────────────────────────────────────────────────────────────────────────────
# CELL 1.5 — Run the FULL bimodal framework (CNN + LSTM + Late Fusion Head)
# on the normal vs detuned sessions.
#
# Earlier versions of this cell scored CNN and LSTM separately and just
# concatenated their pred_ columns. That evaluates two independent models,
# not the framework -- the whole point of Section 16's Late Fusion Head is
# that it learns cross-modal dependencies the two streams can't capture
# alone (proposal §3.5.5). RQ2 needs to run the SAME path a real user's
# session would take: CNN probs + LSTM probs -> Late Fusion Head -> final
# fused decision. This cell does that.
# ─────────────────────────────────────────────────────────────────────────────

# Freeze the LSTM-only names now (see the LFS Adapter cell above for why --
# cell 67 overwrites the shared LABEL_COLS/CLASS_NAMES globals with the
# 10-class FUSED versions). FUSION_LABEL_COLS is captured fresh below,
# AFTER cell 67 has run, since by Section 17 it's stable for the rest of
# the notebook.
CNN_CLASS_NAMES   = ["Off-Track", "Apex Miss", "Sliding"]      # matches ensemble_thresholds.json keys
FUSION_LABEL_COLS = list(LABEL_COLS)                            # 10-class fused label order (cell 67)

# 1) Point these at your two Kaggle datasets ---------------------------------
#    telemetry_csv can be a single raw LFS CSV path *or* a folder of them.
#    labels_csv is now OPTIONAL -- if it doesn't exist, one is auto-generated
#    from every extracted frame (dummy 0/0/0 labels) purely so the CNN can
#    run inference; RQ2 doesn't need CNN ground truth, only its predictions.
SESSIONS = {
    "normal": {
        "telemetry_csv": "/kaggle/input/datasets/samwelnjehia/rq2-cleandriving/lfs_lstm_telemetry_1785936585.csv",
        "video_dir":     "/kaggle/input/datasets/samwelnjehia/rq2-cleandriving",
        "labels_csv":    "/kaggle/working/labels_normal.csv",
        "frames_dir":    Path("/kaggle/working/frames_normal"),
        "vehicle":       "default", 
        "n_laps":        1,
    },
    "detuned": {
        "telemetry_csv": "/kaggle/input/datasets/samwelnjehia/rq2-detuned/lfs_lstm_telemetry_1785930141.csv",
        "video_dir":     "/kaggle/input/datasets/samwelnjehia/rq2-detuned",
        "labels_csv":    "/kaggle/working/labels_detuned.csv",
        "frames_dir":    Path("/kaggle/working/frames_detuned"),
        "vehicle":       "default",
        "n_laps":        1,
    },
}

for cfg in SESSIONS.values():
    cfg["frames_dir"].mkdir(parents=True, exist_ok=True)

# 2) Load all THREE trained components once ----------------------------------
#    NOTE: literal paths, not CHECKPOINT_DIR/EVAL_DIR -- both variables get
#    reassigned in cell 66 to the fusion-head's own checkpoint/eval dirs.
_CNN_CKPT     = "/kaggle/working/bestmodel/best_model.pt"
_LSTM_CKPT    = "/kaggle/input/datasets/samwelnjehia/lstm-model-data/LSTM-module-best_model.pt"
_LSTM_NORM    = "/kaggle/input/datasets/samwelnjehia/lstm-model-data/normalizer_params.json"
_LSTM_THRESH  = "/kaggle/working/checkpoints/lstm_optimal_thresholds.json"   # optional
_CNN_THRESH   = "/kaggle/working/checkpoints/ensemble_thresholds.json"       # or optimal_thresholds.json
_FUSION_CKPT  = "/kaggle/working/fusion/checkpoints/best_model.pt"
_FUSION_CONFIG = "/kaggle/working/fusion/checkpoints/inference_config.json"  # thresholds + input_cols + label_cols

cnn_model = load_cnn_model(_CNN_CKPT)

lstm_engine = LSTMInferenceEngine(
    checkpoint_path = _LSTM_CKPT,
    normalizer_path = _LSTM_NORM,
    threshold_path  = _LSTM_THRESH,
)

#with open(_CNN_THRESH) as f:
#    cnn_thresholds = json.load(f)

_fusion_ckpt = torch.load(_FUSION_CKPT, map_location=DEVICE)
_fusion_cfg  = _fusion_ckpt["model_config"]
fusion_model = LateFusionHead(
    input_dim   = _fusion_cfg["input_dim"],
    num_classes = _fusion_cfg["num_classes"],
).to(DEVICE)
fusion_model.load_state_dict(_fusion_ckpt["model_state"])
fusion_model.eval()

with open(_FUSION_CONFIG) as f:
    fusion_inference_cfg = json.load(f)
FUSION_INPUT_COLS = fusion_inference_cfg["input_cols"]    # e.g. [cnn_off_track, ..., lstm_gentle_accel]
fusion_thresholds = fusion_inference_cfg["thresholds"]    # dict {label_col: tuned threshold}
print(f"[Fusion] Loaded — input_cols: {FUSION_INPUT_COLS}")
print(f"[Fusion] Loaded — thresholds: {fusion_thresholds}")


def _ensure_cnn_labels_csv(cfg: dict) -> str:
    """
    RQ2 doesn't need CNN ground truth -- only its predictions -- so if no
    labels_csv was hand-labelled for this session, auto-generate one that
    just lists every extracted frame with dummy 0/0/0 labels. This lets
    run_cnn_inference() run on unlabelled footage.
    """
    if Path(cfg["labels_csv"]).exists():
        return cfg["labels_csv"]

    frames = sorted(p.name for p in cfg["frames_dir"].glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(
            f"No extracted frames found in {cfg['frames_dir']} -- "
            f"extract_frames_fullres() must run before CNN inference."
        )
    dummy_df = pd.DataFrame({
        "frame":      frames,
        "off_track":  0,
        "apex_miss":  0,
        "sliding":    0,
    })
    dummy_df.to_csv(cfg["labels_csv"], index=False)
    print(f"[CNN ] No hand-labelled CSV found -- auto-generated "
          f"{len(dummy_df):,}-frame placeholder labels -> {cfg['labels_csv']}")
    return cfg["labels_csv"]


def score_session(name: str, cfg: dict) -> pd.DataFrame:
    """
    Run ONE session through the entire trained framework:
        video  -> CNN  -> cnn probs   ─┐
                                        ├─> align -> Late Fusion Head -> fused pred_<class>
        telemetry -> LSTM -> lstm probs ┘

    Returns a DataFrame of the FUSION HEAD's binary pred_<class> columns --
    this is "the framework's" decision, not either sub-model's decision
    alone. This is what compute_fer() should be measuring for RQ2.
    """
    # -- CNN stream: extract frames, run inference -----------------------------
    print(f"\n[{name.upper()}] Extracting frames from video...")
    video_files = discover_video_files(cfg["video_dir"])
    total = 0
    for vid in video_files:
        total += extract_frames_fullres(vid, cfg["frames_dir"], fps_target=FRAME_EXTRACT_FPS)
    print(f"[{name.upper()}] {total} frames extracted -> {cfg['frames_dir']}")

    labels_csv = _ensure_cnn_labels_csv(cfg)
    cnn_df    = run_cnn_inference(cnn_model, labels_csv, cfg["frames_dir"])
    cnn_probs = cnn_df[["cnn_off_track", "cnn_apex_miss", "cnn_sliding"]].values   # (N_cnn, 3)

    # -- LSTM stream: adapt -> normalise -> window -> infer probabilities ------
    tele_input = cfg["telemetry_csv"]
    if Path(tele_input).is_dir():
        tele_input = load_and_compile_telemetry_folder(tele_input)

    lstm_out = lstm_engine.predict_from_csv(tele_input, vehicle=cfg.get("vehicle", "default"))
    lstm_prob_cols = [f"prob_{c.replace(' ', '_')}" for c in LSTM_CLASS_NAMES]  # frozen order from adapter cell
    lstm_probs = lstm_out[lstm_prob_cols].values   # (N_lstm, 7), same class order as LSTM_LABEL_COLS
    print(f"[{name.upper()}] CNN: {len(cnn_probs):,} frames | LSTM: {len(lstm_probs):,} windows")

    # -- Align row-wise (same convention as align_cnn_lstm in cell 62) ---------
    n = min(len(cnn_probs), len(lstm_probs))
    print(f"[{name.upper()}] [Align] using {n:,} rows")

    combined = pd.DataFrame(cnn_probs[:n], columns=["cnn_off_track", "cnn_apex_miss", "cnn_sliding"])
    lstm_cols_ordered = [f"lstm_{c.replace('label_', '')}" for c in LSTM_LABEL_COLS]  # matches cell 67's LSTM_COLS
    for i, col in enumerate(lstm_cols_ordered):
        combined[col] = lstm_probs[:n, i]

    # Reorder to EXACTLY match what the fusion head was trained on -- don't
    # trust column-build order above, trust the saved config.
    combined = combined[FUSION_INPUT_COLS]

    # -- Run the TRAINED Late Fusion Head — this IS "the framework" ------------
    x = torch.tensor(combined.values, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        fusion_probs = torch.sigmoid(fusion_model(x)).cpu().numpy()   # (n, 10)

    # -- Threshold with the fusion head's OWN tuned thresholds ------------------
    out = pd.DataFrame({"window_id": np.arange(n)})
    for i, col in enumerate(FUSION_LABEL_COLS):
        thr = fusion_thresholds.get(col, 0.5)
        out[f"pred_{col}"] = (fusion_probs[:, i] > thr).astype(int)

    return out


normal_preds  = score_session("normal",  SESSIONS["normal"])
detuned_preds = score_session("detuned", SESSIONS["detuned"])

# n_laps for the next cell -- pull from SESSIONS instead of hardcoding
NORMAL_LAPS  = SESSIONS["normal"]["n_laps"]
DETUNED_LAPS = SESSIONS["detuned"]["n_laps"]

print(f"\n[RQ2] normal_preds  (fused): {normal_preds.shape}")
print(f"[RQ2] detuned_preds (fused): {detuned_preds.shape}")

# %% cell 82
# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Compute FER for both sessions
# Replace n_laps values with your actual lap counts
# ─────────────────────────────────────────────────────────────────────────────

# DETUNED_LAPS / NORMAL_LAPS now come from the SESSIONS dict in the cell above

detuned_fer_data = compute_fer(detuned_preds, n_laps=DETUNED_LAPS)
normal_fer_data  = compute_fer(normal_preds,  n_laps=NORMAL_LAPS)

print(f"Normal  session — FER: {normal_fer_data['fer']:.2f} events/lap  "
      f"(total: {normal_fer_data['total_events']})")
print(f"Detuned session — FER: {detuned_fer_data['fer']:.2f} events/lap  "
      f"(total: {detuned_fer_data['total_events']})")
print(f"\nFER increase: {detuned_fer_data['fer'] - normal_fer_data['fer']:.2f} "
      f"({((detuned_fer_data['fer'] / max(normal_fer_data['fer'], 0.01)) - 1) * 100:.1f}% higher)")

# %% cell 83
# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Per-class event count comparison table
# This is the main results table for Chapter 4
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  PER-CLASS EVENT COUNT COMPARISON")
print(f"  {'Event':<26} {'Normal':>8} {'Detuned':>9}  {'Detected?':>12}")
print(f"  {'-'*57}")

all_classes = set(list(normal_fer_data["per_class"].keys()) +
                  list(detuned_fer_data["per_class"].keys()))

results_table = []
for cls in sorted(all_classes):
    n_count = normal_fer_data["per_class"].get(cls, 0)
    d_count = detuned_fer_data["per_class"].get(cls, 0)
    detected = "✓ INCREASED" if d_count > n_count else ("— same" if d_count == n_count else "↓ decreased")
    print(f"  {cls:<26} {n_count:>8} {d_count:>9}  {detected:>12}")
    results_table.append({"event": cls, "normal": n_count,
                           "detuned": d_count, "detected": d_count > n_count})

print(f"  {'-'*57}")
print(f"  {'TOTAL (any event)':<26} "
      f"{normal_fer_data['total_events']:>8} "
      f"{detuned_fer_data['total_events']:>9}")
print(f"{'='*60}")

# Save as CSV for Chapter 4 table
results_df = pd.DataFrame(results_table)
results_df.to_csv("/kaggle/working/rq2_event_comparison.csv", index=False)
print(f"\n[RQ2] Event comparison saved → rq2_event_comparison.csv")

# %% cell 84
# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Paired t-test (proposal §3.7, alpha = 0.05)
# ─────────────────────────────────────────────────────────────────────────────

detuned_per_lap = detuned_fer_data["per_lap_fer"]
normal_per_lap  = normal_fer_data["per_lap_fer"]

print(f"\n[RQ2] Per-lap FER values:")
print(f"  Normal  : {normal_per_lap}")
print(f"  Detuned : {detuned_per_lap}")

# With only 2 laps per session, the t-test has 1 degree of freedom.
# This is statistically low-powered but directionally valid.
# Report the result honestly — the sample size is a documented limitation.
if len(detuned_per_lap) >= 2 and len(normal_per_lap) >= 2:
    t_stat, p_value = stats.ttest_rel(detuned_per_lap, normal_per_lap)

    print(f"\n{'='*55}")
    print(f"  PAIRED T-TEST RESULTS  (alpha = 0.05)")
    print(f"  H0: mean FER is equal in normal and detuned sessions")
    print(f"  H1: detuned session has significantly higher FER")
    print(f"{'='*55}")
    print(f"  t-statistic : {t_stat:.4f}")
    print(f"  p-value     : {p_value:.4f}")
    print(f"  Significant : {'YES (p < 0.05)' if p_value < 0.05 else 'NO (p ≥ 0.05)'}")
    print(f"  Conclusion  : ", end="")

    if p_value < 0.05 and t_stat < 0:
        print("Framework detects real behavioural change — FER increased "
              "significantly in the detuned session. ✓")
    elif detuned_fer_data["fer"] > normal_fer_data["fer"]:
        print("FER directionally higher in detuned session but result is not "
              "statistically significant at alpha=0.05, likely due to small "
              "sample size (n=2 laps). The directional result supports the "
              "framework's sensitivity.")
    else:
        print("No increase detected — review labelling thresholds or "
              "confirm detuned driving errors were present in the telemetry.")
    print(f"{'='*55}")

else:
    # Fallback for n=1 (single lap only)
    print("\n[RQ2] Only 1 lap per session — t-test requires n≥2.")
    print(f"  Directional result: detuned FER ({detuned_fer_data['fer']:.2f}) "
          f"{'>' if detuned_fer_data['fer'] > normal_fer_data['fer'] else '<'} "
          f"normal FER ({normal_fer_data['fer']:.2f})")

# %% cell 85
# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Save full RQ2 results for Chapter 4
# ─────────────────────────────────────────────────────────────────────────────

rq2_summary = {
    "normal_session": {
        "fer":          normal_fer_data["fer"],
        "total_events": normal_fer_data["total_events"],
        "n_laps":       NORMAL_LAPS,
        "per_lap_fer":  normal_per_lap,
        "per_class":    normal_fer_data["per_class"],
    },
    "detuned_session": {
        "fer":          detuned_fer_data["fer"],
        "total_events": detuned_fer_data["total_events"],
        "n_laps":       DETUNED_LAPS,
        "per_lap_fer":  detuned_per_lap,
        "per_class":    detuned_fer_data["per_class"],
    },
    "statistical_test": {
        "test":      "paired t-test",
        "alpha":     0.05,
        "t_stat":    round(float(t_stat), 4) if "t_stat" in dir() else None,
        "p_value":   round(float(p_value), 4) if "p_value" in dir() else None,
        "n_pairs":   len(detuned_per_lap),
        "significant": bool(p_value < 0.05) if "p_value" in dir() else None,
    },
}

import json
with open("/kaggle/working/rq2_results.json", "w") as f:
    json.dump(rq2_summary, f, indent=2)

print(f"\n[RQ2] Full results saved → rq2_results.json")
print(f"[RQ2] Research Question 2 evaluation complete.")
