from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QFileDialog, QMessageBox
from ui.widgets import Card, SectionTitle, make_button
from ui.state import AppState
from core.vehicle_manager import VehicleManager
from core.report_generator import ReportGenerator
import shutil


class ReportPage(QWidget):
    """'The driving coach will see you right away!' screen: shows the
    Driving Report Preview window plus Save report pdf / Watch annotated
    video actions."""

    def __init__(self, state: AppState, navigate, parent=None):
        super().__init__(parent)
        self.state = state
        self.navigate = navigate

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)

        top_row = QHBoxLayout()
        top_row.addWidget(SectionTitle("The driving coach will see you right away!", size=16))
        top_row.addStretch()
        save_btn = make_button("Save report pdf", "GhostButton")
        save_btn.clicked.connect(self._save_pdf)
        top_row.addWidget(save_btn)
        root.addLayout(top_row)

        preview_card = Card()
        preview_card.add(QLabel("Driving Report Preview window"))
        self.preview_box = QTextEdit()
        self.preview_box.setReadOnly(True)
        self.preview_box.setMinimumHeight(260)
        preview_card.add(self.preview_box)
        root.addWidget(preview_card)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        watch_btn = make_button("Watch annotated video", "GreenButton")
        watch_btn.clicked.connect(lambda: self.navigate("annotating"))
        btn_row.addWidget(watch_btn)
        root.addLayout(btn_row)

        root.addStretch()

    def on_navigate(self, **kwargs):
        self._render_preview()

    def _render_preview(self):
        result = self.state.pipeline_result
        vehicle = VehicleManager.get_active_vehicle() or {}
        self.preview_box.clear()

        if not result:
            self.preview_box.append("No session data yet - run 'Process data' or 'Deploy AI framework' first.")
            return

        summary = result.get("summary", {})
        feedback_items = result.get("feedback_items", [])
        persona = result.get("driving_persona")
        fer = result.get("fer")

        lines = [
            f"Vehicle: {vehicle.get('car_name', 'N/A')} (ID: {vehicle.get('car_id', 'N/A')})",
            f"Top RPM {vehicle.get('top_rpm', 'N/A')} / Top Speed {vehicle.get('top_speed', 'N/A')} km/h / "
            f"Max Engine RPM {vehicle.get('max_engine_rpm', 'N/A')} / "
            f"Max Suspension Load {vehicle.get('max_suspension_load', 'N/A')} / "
            f"Peak Power RPM {vehicle.get('peak_power_rpm', 'N/A')}",
            "",
        ]
        if persona:
            lines.append(f"Driving persona: {persona}   |   Feedback Event Rate: {fer} items/lap")
        lines += [
            f"Analysis windows: {summary.get('total_windows', 0)}",
            f"Flagged windows: {summary.get('flagged_windows', 0)}",
        ]
        top_speed = summary.get("top_speed_kmh")
        if top_speed is not None:
            lines.append(f"Top speed recorded this session: {top_speed:.1f} km/h")
        lines.append(f"Most frequent finding: {summary.get('top_event') or 'Smooth Driving'}")
        lines.append("")
        lines.append("Detected Event Breakdown:")
        for event, count in summary.get("event_counts", {}).items():
            lines.append(f"  - {event}: {count}")

        lines.append("")
        lines.append("Coaching Feedback:")
        if feedback_items:
            for item in feedback_items:
                lines.append(f"  \u2022 {item}")
        else:
            lines.append("  No issues flagged above threshold this session — clean driving.")

        self.preview_box.append("\n".join(lines))

    def _save_pdf(self):
        result = self.state.pipeline_result
        if not result:
            QMessageBox.information(self, "No report yet", "Process a session before saving a report.")
            return

        vehicle = VehicleManager.get_active_vehicle() or {}
        pdf_path = ReportGenerator.build_pdf(
            summary=result.get("summary", {}),
            vehicle=vehicle,
            feedback_items=result.get("feedback_items", []),
            driving_persona=result.get("driving_persona"),
            fer=result.get("fer"),
        )
        self.state.report_pdf_path = pdf_path

        dest, _ = QFileDialog.getSaveFileName(self, "Save report", pdf_path.name, "PDF files (*.pdf);;Text files (*.txt)")
        if dest:
            shutil.copy2(pdf_path, dest)
            QMessageBox.information(self, "Saved", f"Report saved to:\n{dest}")
