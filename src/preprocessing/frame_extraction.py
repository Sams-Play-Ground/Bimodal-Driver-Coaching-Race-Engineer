"""
frame_extraction.py — Discovers session videos and extracts frames at full
native resolution for the CNN spatial module, per proposal §3.5.2.

Frames are saved untouched (no resize/letterbox) so the interactive
labelling tool shows a clean, full-quality view. The letterbox transform
that the CNN actually trains on is applied later, identically at both
training and inference time, in DrivingDataset / the inference pipeline.

Ported from `final-cnn-module-codes.ipynb` ("3 - Multi-video ingestion",
"4 - Frame extraction").
"""

from pathlib import Path

import cv2
import numpy as np


def discover_video_files(source) -> list:
    """
    Discover all video files in one directory, multiple directories, or a
    direct list of file paths.

    Args:
        source: str | Path | list[str | Path] — folder path(s) or direct
                video file path(s).

    Returns:
        Sorted list of Path objects for all discovered video files.
    """
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".MP4", ".MOV", ".AVI"}
    found = []

    if isinstance(source, (str, Path)):
        source = Path(source)
        if source.is_dir():
            found = [f for f in source.rglob("*") if f.suffix in VIDEO_EXTENSIONS]
        elif source.is_file() and source.suffix in VIDEO_EXTENSIONS:
            found = [source]
        else:
            raise FileNotFoundError(f"[Ingest] Path not found or not a video: {source}")
    elif isinstance(source, list):
        for item in source:
            item = Path(item)
            if item.is_dir():
                found.extend(f for f in item.rglob("*") if f.suffix in VIDEO_EXTENSIONS)
            elif item.is_file() and item.suffix in VIDEO_EXTENSIONS:
                found.append(item)
    else:
        raise TypeError("[Ingest] source must be a path string, Path, or list.")

    found = sorted(set(found))
    if not found:
        raise FileNotFoundError(f"[Ingest] No video files found in: {source}")

    print(f"[Ingest] Found {len(found)} video file(s):")
    for f in found:
        print(f"  {f.name}  ({f.stat().st_size / 1_000_000:.1f} MB)")

    return found


def letterbox(frame: np.ndarray, target_w: int = 640, target_h: int = 288) -> np.ndarray:
    """
    Resize a frame to fit inside (target_w x target_h) while preserving the
    original aspect ratio; remaining area is black-padded. Used identically
    for both training (DrivingDataset) and inference so the CNN never sees
    a spatial mismatch between the two.
    """
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def extract_frames_fullres(video_path: Path, out_dir: Path, fps_target: int = 1) -> int:
    """
    Extract one frame per `fps_target` seconds at the video's native
    resolution, saved as high-quality JPEGs (no resizing/letterboxing here).

    Returns:
        Number of frames saved.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [Extract] Could not open: {video_path.name} — skipping.")
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    interval = max(1, int(src_fps / fps_target))
    stem = video_path.stem
    count = saved = 0

    print(f"  [Extract] {video_path.name}")
    print(f"            {src_w}x{src_h} @ {src_fps:.1f}fps -> {fps_target} frame every {interval} source frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            out = out_dir / f"{stem}_f{count:07d}.jpg"
            cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
        count += 1

    cap.release()
    return saved


def batch_extract(video_files: list, out_dir: Path, fps_target: int = 1) -> int:
    """Run extract_frames_fullres across every discovered video, returning the total frame count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total_frames = 0
    for vid in video_files:
        n = extract_frames_fullres(vid, out_dir, fps_target=fps_target)
        total_frames += n
        print(f"    -> {n} frames saved")
    print(f"[Extract] Total: {total_frames} frame(s) across {len(video_files)} video(s)")
    return total_frames
