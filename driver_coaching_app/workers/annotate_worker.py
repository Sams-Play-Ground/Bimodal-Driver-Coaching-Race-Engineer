from PyQt6.QtCore import QThread, pyqtSignal
from pathlib import Path

from core.cnn_engine import VisionEngine
from core.video_processor import VideoProcessor
from core.data_manager import DataManager


class AnnotateWorker(QThread):
    """Renders the annotated replay video off the UI thread. The overlay
    is CNN-only (3 indicator dots per notebook Section 14) — telemetry
    does not factor into this visual, so it isn't required here."""

    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(str)  # path to annotated mp4
    failed = pyqtSignal(str)

    def __init__(self, footage_path: Path, parent=None):
        super().__init__(parent)
        self.footage_path = footage_path

    def run(self):
        try:
            vision_engine = VisionEngine(model_path=DataManager.CNN_CHECKPOINT)
            processor = VideoProcessor()
            output_path = DataManager.OUTPUTS_DIR / "annotated_replay.mp4"
            processor.process_and_annotate_video(
                video_input_path=self.footage_path,
                output_path=output_path,
                vision_engine=vision_engine,
                progress_callback=lambda pct: self.progress.emit(pct),
            )
            self.finished_ok.emit(str(output_path))
        except Exception as exc:
            self.failed.emit(str(exc))
