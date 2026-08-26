from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QListWidget, QSizePolicy
)
from PyQt6.QtCore import Qt

from ui.widgets import Card, SectionTitle, Subtitle, make_button
from ui.state import AppState
from core.data_manager import DataManager
from core.vehicle_manager import VehicleManager
from workers.pipeline_worker import PipelineWorker


class MainPage(QWidget):
    """The wireframe's central hub: ingest footage + telemetry, review car
    spec, kick off processing, and watch the processing status panel."""

    def __init__(self, state: AppState, navigate, parent=None):
        super().__init__(parent)
        self.state = state
        self.navigate = navigate
        self.worker = None

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)

        root.addWidget(SectionTitle("Main Page"))
        root.addWidget(Subtitle("Bring in this session's footage and telemetry, confirm your car spec, then process."))

        # -- Ingestion row -------------------------------------------------
        ingest_row = QHBoxLayout()
        ingest_row.setSpacing(16)

        self.footage_label, self.footage_clear_btn = self._build_ingest_card(
            "footage_card", "Enter Driving footage", "Choose a .mp4 / .avi file",
            self._pick_footage, self._clear_footage,
        )
        self.telemetry_label, self.telemetry_clear_btn = self._build_ingest_card(
            "telemetry_card", "Enter Telemetry data", "Choose a .csv file",
            self._pick_telemetry, self._clear_telemetry,
        )
        ingest_row.addWidget(self.footage_card)
        ingest_row.addWidget(self.telemetry_card)
        root.addLayout(ingest_row)
        self._ingest_row = ingest_row

        # -- Car spec + Process data row ------------------------------------
        action_row = QHBoxLayout()
        action_row.setSpacing(16)

        self.car_spec_btn = make_button("Car spec", "OrangeButton")
        self.car_spec_btn.clicked.connect(lambda: self.navigate("car_spec"))
        action_row.addWidget(self.car_spec_btn)
        action_row.addStretch()

        self.process_btn = make_button("Process data", "OrangeButton")
        self.process_btn.clicked.connect(self._start_processing)
        action_row.addWidget(self.process_btn)
        root.addLayout(action_row)

        # -- Processing status card -----------------------------------------
        status_card = Card()
        status_card.add(SectionTitle("Processing Status", size=14))
        self.status_list = QListWidget()
        self.status_list.setFixedHeight(160)
        status_card.add(self.status_list)
        root.addWidget(status_card)

        # -- Continue arrow (only enabled once processing has completed) ----
        continue_row = QHBoxLayout()
        continue_row.addStretch()
        self.continue_btn = make_button("Continue \u2192", "GreenButton")
        self.continue_btn.setEnabled(False)
        self.continue_btn.clicked.connect(lambda: self.navigate("hub"))
        continue_row.addWidget(self.continue_btn)
        root.addLayout(continue_row)

        root.addStretch()
        self.refresh_car_spec_label()
        self._refresh_clear_buttons()

    # ------------------------------------------------------------------
    def _build_ingest_card(self, attr_prefix, title, placeholder, on_browse, on_clear):
        card = Card()
        card.add(SectionTitle(title, size=14))
        label = QLabel(placeholder)
        label.setWordWrap(True)
        label.setStyleSheet("color: #5B5B5B;")
        card.add(label)

        btn_row = QHBoxLayout()
        browse_btn = make_button("Browse...", "GreenButton")
        browse_btn.clicked.connect(lambda: on_browse(label))
        btn_row.addWidget(browse_btn)

        clear_btn = make_button("Clear", "OrangeButton")
        clear_btn.clicked.connect(on_clear)
        btn_row.addWidget(clear_btn)
        card.layout().addLayout(btn_row)

        setattr(self, f"{attr_prefix}", card)
        return label, clear_btn

    def _pick_footage(self, label: QLabel):
        path, _ = QFileDialog.getOpenFileName(self, "Select driving footage", "", "Video files (*.mp4 *.avi *.mov)")
        if path:
            dest = DataManager.copy_local_file(path, "raw_uploads")
            self.state.footage_path = dest
            label.setText(f"\u2713 {dest.name}")
            self._refresh_clear_buttons()

    def _pick_telemetry(self, label: QLabel):
        path, _ = QFileDialog.getOpenFileName(self, "Select telemetry data", "", "CSV files (*.csv)")
        if path:
            dest = DataManager.copy_local_file(path, "raw_uploads")
            self.state.telemetry_path = dest
            label.setText(f"\u2713 {dest.name}")
            self._refresh_clear_buttons()

    def _clear_footage(self):
        DataManager.delete_uploaded_file(self.state.footage_path)
        self.state.footage_path = None
        self.footage_label.setText("Choose a .mp4 / .avi file")
        self._refresh_clear_buttons()

    def _clear_telemetry(self):
        DataManager.delete_uploaded_file(self.state.telemetry_path)
        self.state.telemetry_path = None
        self.telemetry_label.setText("Choose a .csv file")
        self._refresh_clear_buttons()

    def _refresh_clear_buttons(self):
        self.footage_clear_btn.setEnabled(self.state.footage_path is not None)
        self.telemetry_clear_btn.setEnabled(self.state.telemetry_path is not None)

    def refresh_car_spec_label(self):
        vehicle = VehicleManager.get_active_vehicle()
        if vehicle:
            self.car_spec_btn.setText(f"Car spec ({vehicle.get('car_name', 'Unnamed')})")
        else:
            self.car_spec_btn.setText("Car spec (add a vehicle)")

    # ------------------------------------------------------------------
    def _start_processing(self):
        self.status_list.clear()
        self.continue_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self._log("Starting ingestion pipeline...")

        self.worker = PipelineWorker(self.state.footage_path, self.state.telemetry_path)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, pct, msg):
        self._log(f"[{pct}%] {msg}")

    def _on_finished(self, result: dict):
        self.state.pipeline_result = result
        self._log("Processing complete. Ready to review diagnostics.")
        self.process_btn.setEnabled(True)
        self.continue_btn.setEnabled(True)

    def _on_failed(self, error_msg: str):
        self._log(f"Error: {error_msg}")
        self.process_btn.setEnabled(True)

    def _log(self, text: str):
        self.status_list.addItem(text)
        self.status_list.scrollToBottom()

    def on_shown(self):
        self.refresh_car_spec_label()
