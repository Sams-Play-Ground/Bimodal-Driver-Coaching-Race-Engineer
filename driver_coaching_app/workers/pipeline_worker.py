from PyQt6.QtCore import QThread, pyqtSignal
from pathlib import Path
from typing import Optional

from core.pipeline import DrivingPipeline


class PipelineWorker(QThread):
    """Runs DrivingPipeline.run() off the UI thread. Powers both the
    'Process data' step on the Main Page and the 'Deploy AI framework'
    step on the Hub page."""

    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, footage_path: Optional[Path], telemetry_path: Optional[Path], parent=None):
        super().__init__(parent)
        self.footage_path = footage_path
        self.telemetry_path = telemetry_path

    def run(self):
        try:
            pipeline = DrivingPipeline()
            result = pipeline.run(
                footage_path=self.footage_path,
                telemetry_path=self.telemetry_path,
                progress_callback=lambda pct, msg: self.progress.emit(pct, msg),
            )
            self.finished_ok.emit(result)
        except Exception as exc:  # surface any error to the UI instead of crashing
            self.failed.emit(str(exc))
