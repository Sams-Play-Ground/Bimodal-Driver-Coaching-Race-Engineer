"""
Video annotation — ported 1:1 from cnn-module-codes.ipynb Section 14
(annotate_video). Every processed frame is run through the real CNN
(VisionEngine.predict_frame), and the three indicator dots light up
based on that frame's actual classification — not telemetry, not a
placeholder.

  Off-Track -> red dot (top)
  Apex Miss -> blue dot (middle)
  Sliding   -> orange dot (bottom)
  Yellow border when 2+ are active on the same frame.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable

from core.cnn_engine import VisionEngine, CLASS_NAMES, DEFAULT_THRESHOLDS, NUM_CLASSES

INFERENCE_EVERY_NTH = 1  # 1 = classify every frame. Raise to trade accuracy for render speed.


class VideoProcessor:
    """Renders the annotated replay: real CNN classification per frame,
    drawn as the three indicator dots from the notebook's dashboard."""

    def process_and_annotate_video(
        self,
        video_input_path: Path,
        output_path: Path,
        vision_engine: VisionEngine,
        thresholds: Optional[dict] = None,
        process_every: int = INFERENCE_EVERY_NTH,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Path:
        if not vision_engine.available:
            raise RuntimeError(
                f"CNN is not available ({vision_engine.status_reason}) — cannot annotate footage "
                "without vision classification."
            )

        thresholds = thresholds or DEFAULT_THRESHOLDS

        cap = cv2.VideoCapture(str(video_input_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        dot_r = max(18, width // 50)
        pad = dot_r + 12
        dots = [
            # (class_idx, center, filled_color_BGR, label)
            (0, (width - pad, pad), (0, 0, 220), "OT"),               # Off-Track: red
            (1, (width - pad, pad * 2 + dot_r), (220, 80, 0), "AM"),  # Apex Miss: blue
            (2, (width - pad, pad * 3 + dot_r * 2), (0, 165, 255), "SL"),  # Sliding: orange
        ]

        frame_idx = 0
        last_probs = np.zeros(NUM_CLASSES, dtype=np.float32)
        frames_classified = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % process_every == 0:
                last_probs = vision_engine.predict_frame(frame)  # <-- real CNN forward pass, every Nth frame
                frames_classified += 1

            states = [last_probs[i] > thresholds.get(name, 0.5) for i, name in enumerate(CLASS_NAMES)]

            for cls_i, center, color_bgr, _lbl in dots:
                cv2.circle(frame, center, dot_r, (80, 80, 80), 2)
                if states[cls_i]:
                    cv2.circle(frame, center, dot_r, color_bgr, -1)

            if sum(states) >= 2:
                cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 255), 5)

            hud = "  ".join(f"{n[:2]}:{last_probs[i]:.2f}" for i, n in enumerate(CLASS_NAMES))
            cv2.putText(frame, hud, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            out.write(frame)
            frame_idx += 1
            if progress_callback and frame_idx % 5 == 0:
                progress_callback(min(99, int(100 * frame_idx / total_frames)))

        cap.release()
        out.release()
        if progress_callback:
            progress_callback(100)

        if frames_classified == 0:
            raise RuntimeError("No frames were classified by the CNN — footage may be empty or unreadable.")
        return output_path
