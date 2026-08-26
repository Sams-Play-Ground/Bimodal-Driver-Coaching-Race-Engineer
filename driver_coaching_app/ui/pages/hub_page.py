from PyQt6.QtWidgets import QWidget, QVBoxLayout
from ui.widgets import Card, SectionTitle, Subtitle, HubRow
from ui.theme import COLORS
from ui.state import AppState


class HubPage(QWidget):
    """The traffic-light screen from the wireframe: three rows, each with
    a colored status dot and a labeled action."""

    def __init__(self, state: AppState, navigate, parent=None):
        super().__init__(parent)
        self.state = state
        self.navigate = navigate

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)

        root.addWidget(SectionTitle("Diagnostics Hub"))
        root.addWidget(Subtitle("Pick a diagnostic stream to review, or deploy the AI framework for full coaching."))

        card = Card()

        row_red = HubRow(COLORS["red"], "Telemetry data plots")
        row_red.clicked.connect(lambda: self.navigate("coming_soon", title="Telemetry data plots"))
        card.add(row_red)

        row_orange = HubRow(COLORS["orange"], "Footage metadata info")
        row_orange.clicked.connect(lambda: self.navigate("coming_soon", title="Footage metadata info"))
        card.add(row_orange)

        row_green = HubRow(COLORS["green"], "Deploy AI framework", enabled_style="GreenButton")
        row_green.clicked.connect(lambda: self.navigate("processing"))
        card.add(row_green)

        root.addWidget(card)
        root.addStretch()
