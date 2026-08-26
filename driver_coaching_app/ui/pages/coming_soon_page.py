from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PyQt6.QtCore import Qt
from ui.widgets import SectionTitle, make_button
from ui.state import AppState


class ComingSoonPage(QWidget):
    """Generic placeholder used for the two not-yet-built diagnostic
    screens in the wireframe (Telemetry data plots / Footage metadata info)."""

    def __init__(self, state: AppState, navigate, parent=None):
        super().__init__(parent)
        self.state = state
        self.navigate = navigate

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)

        self.title_label = SectionTitle("Feature")
        root.addWidget(self.title_label)

        root.addStretch()
        msg = QLabel("This feature is coming soon")
        msg.setStyleSheet("font-size: 22px; color: #5B5B5B;")
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(msg)
        root.addStretch()

        back_row = QHBoxLayout()
        back_row.addStretch()
        back_btn = make_button("\u2190 Back to Hub", "GhostButton")
        back_btn.clicked.connect(lambda: self.navigate("hub"))
        back_row.addWidget(back_btn)
        back_row.addStretch()
        root.addLayout(back_row)

    def on_navigate(self, title: str = "Feature", **kwargs):
        self.title_label.setText(title)
