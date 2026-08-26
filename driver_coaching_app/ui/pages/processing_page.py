from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QHBoxLayout, QTextEdit
from PyQt6.QtCore import Qt
from ui.widgets import Card, SectionTitle, make_button
from ui.state import AppState
from workers.pipeline_worker import PipelineWorker


class ProcessingPage(QWidget):
    """'Frame work is processing data...' screen from the wireframe:
    a progress bar plus a scrolling preview of the current pipeline state."""

    def __init__(self, state: AppState, navigate, parent=None):
        super().__init__(parent)
        self.state = state
        self.navigate = navigate
        self.worker = None

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)

        self.heading = SectionTitle("Frame work is processing data...")
        root.addWidget(self.heading)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        root.addWidget(self.progress_bar)

        preview_card = Card()
        preview_card.add(QLabel("Preview of current state..."))
        self.preview_box = QTextEdit()
        self.preview_box.setReadOnly(True)
        self.preview_box.setFixedHeight(180)
        preview_card.add(self.preview_box)
        root.addWidget(preview_card)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.continue_btn = make_button("Continue \u2192", "GreenButton")
        self.continue_btn.setEnabled(False)
        self.continue_btn.clicked.connect(lambda: self.navigate("report"))
        btn_row.addWidget(self.continue_btn)
        root.addLayout(btn_row)

        root.addStretch()

    def on_navigate(self, **kwargs):
        self._run()

    def _run(self):
        self.progress_bar.setValue(0)
        self.preview_box.clear()
        self.continue_btn.setEnabled(False)

        self.worker = PipelineWorker(self.state.footage_path, self.state.telemetry_path)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.preview_box.append(f"[{pct}%] {msg}")

    def _on_finished(self, result: dict):
        self.state.pipeline_result = result
        self.progress_bar.setValue(100)
        self.preview_box.append("AI framework deployment complete.")
        self.continue_btn.setEnabled(True)

    def _on_failed(self, error_msg: str):
        self.preview_box.append(f"Error: {error_msg}")
