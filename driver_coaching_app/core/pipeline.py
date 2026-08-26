import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Optional, Callable, Dict, Any

from core.data_manager import DataManager
from core.cnn_engine import VisionEngine, CLASS_NAMES as CNN_CLASS_NAMES
from core.lstm_engine import TelemetryEngine
from core.fusion_engine import BimodalFusionEngine, FUSED_LABEL_COLS, FUSED_CLASS_NAMES
from core.feedback_generator import FeedbackGenerator


class DrivingPipeline:
    """Runs the full 'Process data' step, mirroring score_session() from
    Section 17 of the notebook: video -> CNN probs, telemetry -> LSTM
    probs, both -> trained Late Fusion Head -> fused pred_<class> columns.
    progress_callback(percent, message) is invoked as work proceeds.
    """

    def __init__(self):
        self.vision_engine = VisionEngine(model_path=DataManager.CNN_CHECKPOINT)
        self.telemetry_engine = TelemetryEngine(
            weights_path=DataManager.LSTM_CHECKPOINT,
            normalizer_path=DataManager.LSTM_NORMALIZER,
            threshold_path=DataManager.LSTM_THRESHOLDS,
        )
        self.fusion_engine = BimodalFusionEngine(
            weights_path=DataManager.FUSION_CHECKPOINT,
            config_path=DataManager.FUSION_CONFIG,
        )

    def run(
        self,
        footage_path: Optional[Path],
        telemetry_path: Optional[Path],
        vehicle: str = "default",
        frame_stride_sec: float = 1.0,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:

        def report(pct, msg):
            if progress_callback:
                progress_callback(pct, msg)

        report(1, self._checkpoint_status_message())

        report(10, "Extracting frames from footage...")
        cnn_probs = self._run_vision(footage_path, frame_stride_sec, report)

        report(45, "Running telemetry model (LSTM)...")
        lstm_probs = self._run_telemetry(telemetry_path, vehicle, report)
        raw_telemetry_df = self._load_raw_telemetry(telemetry_path)

        report(70, f"Fusing through the trained fusion head "
                   f"({len(cnn_probs)} CNN frames, {len(lstm_probs)} LSTM windows)...")
        fused_df = self.fusion_engine.evaluate(cnn_probs, lstm_probs)

        report(85, "Generating coaching feedback...")
        feedback_result = self._generate_feedback(fused_df)

        report(90, "Saving session report...")
        if fused_df.empty:
            saved_path = None
        else:
            saved_path = DataManager.save_fusion_output(fused_df, "session_fusion_coaching_report.csv")

        report(100, "Done.")
        return {
            "results_df": fused_df,
            "report_path": saved_path,
            "cnn_frame_count": len(cnn_probs),
            "lstm_window_count": len(lstm_probs),
            "vision_used": len(cnn_probs) > 0,
            "telemetry_used": len(lstm_probs) > 0,
            "telemetry_df": raw_telemetry_df,  # used for top-speed stat on the report page only
            "feedback_items": feedback_result["feedback_items"],
            "driving_persona": feedback_result["driving_persona"],
            "fer": feedback_result["fer"],
            "summary": self._summarize(fused_df, raw_telemetry_df),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _checkpoint_status_message() -> str:
        status = DataManager.checkpoint_status()
        parts = [f"{label}: {'loaded' if info['found'] else 'MISSING'}" for label, info in status.items()]
        return "Checkpoints - " + " | ".join(parts)

    def _run_vision(self, footage_path: Optional[Path], frame_stride_sec: float,
                     report: Callable[[int, str], None]) -> np.ndarray:
        if not footage_path or not Path(footage_path).exists():
            return np.zeros((0, 3), dtype=np.float32)
        if not self.vision_engine.available:
            report(12, f"Vision skipped - {self.vision_engine.status_reason} (telemetry-only fusion if available).")
            return np.zeros((0, 3), dtype=np.float32)

        cap = cv2.VideoCapture(str(footage_path))
        if not cap.isOpened():
            report(12, "Could not open footage file — skipping vision.")
            return np.zeros((0, 3), dtype=np.float32)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = max(1, int(fps * frame_stride_sec))

        probs, count = [], 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % interval == 0:
                probs.append(self.vision_engine.predict_frame(frame))
            count += 1
        cap.release()
        result = np.array(probs, dtype=np.float32) if probs else np.zeros((0, 3), dtype=np.float32)
        report(40, f"CNN classified {len(result)} frames.")
        return result

    def _generate_feedback(self, fused_df: pd.DataFrame) -> Dict[str, Any]:
        """Turns the fusion head's per-window probabilities into coaching
        text via FeedbackGenerator (ported from notebook Section 16.4).
        Returns an empty result if the fusion head never ran."""
        empty = {"feedback_items": [], "driving_persona": None, "fer": 0.0}
        if fused_df.empty:
            return empty

        prob_cols = [f"prob_{c}" for c in FUSED_LABEL_COLS]
        if not all(c in fused_df.columns for c in prob_cols):
            return empty

        all_probs = fused_df[prob_cols].values
        thresholds = FeedbackGenerator.thresholds_from_fusion_engine(self.fusion_engine)
        generator = FeedbackGenerator(thresholds=thresholds)
        return generator.session_summary(all_probs)

    @staticmethod
    def _load_raw_telemetry(telemetry_path: Optional[Path]) -> Optional[pd.DataFrame]:
        """Raw (pre-adapter) telemetry, kept around for the annotated-replay
        HUD in video_processor.py — that overlay wants the original LFS
        column names (Speed_KMH/Brake/Steer), not the LSTM engine's
        canonicalised copy."""
        if not telemetry_path or not Path(telemetry_path).exists():
            return None
        try:
            return pd.read_csv(telemetry_path)
        except Exception:
            return None

    def _run_telemetry(self, telemetry_path: Optional[Path], vehicle: str,
                        report: Callable[[int, str], None]) -> np.ndarray:
        if not telemetry_path or not Path(telemetry_path).exists():
            return np.zeros((0, 7), dtype=np.float32)
        if not self.telemetry_engine.available:
            report(46, "LSTM checkpoint/normalizer not loaded — skipping telemetry (vision-only fusion if available).")
            return np.zeros((0, 7), dtype=np.float32)
        try:
            result = self.telemetry_engine.predict_from_csv(telemetry_path, vehicle=vehicle)
            report(65, f"LSTM classified {len(result)} telemetry windows.")
            return result
        except Exception as exc:
            # Surfaced, not swallowed — a real schema/shape error should be
            # visible in the progress log, not indistinguishable from "no
            # telemetry was given".
            report(65, f"LSTM inference failed on this telemetry file: {exc}")
            return np.zeros((0, 7), dtype=np.float32)

    @staticmethod
    def _summarize(results_df: pd.DataFrame, raw_telemetry_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        top_speed = None
        if raw_telemetry_df is not None:
            for col in ("Speed_KMH", "Speed_kmh"):
                if col in raw_telemetry_df.columns:
                    top_speed = float(raw_telemetry_df[col].max())
                    break

        if results_df.empty:
            return {"total_windows": 0, "flagged_windows": 0, "event_counts": {},
                    "top_event": None, "top_speed_kmh": top_speed}

        pred_cols = [c for c in results_df.columns if c.startswith("pred_")]
        event_counts = {c.replace("pred_", ""): int(results_df[c].sum()) for c in pred_cols}
        any_event = results_df[pred_cols].any(axis=1)
        return {
            "total_windows": len(results_df),
            "flagged_windows": int(any_event.sum()),
            "event_counts": event_counts,
            "top_event": max(event_counts, key=event_counts.get) if any(event_counts.values()) else None,
            "top_speed_kmh": top_speed,
        }
