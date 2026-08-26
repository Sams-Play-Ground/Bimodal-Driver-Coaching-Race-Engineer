from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QPushButton
from ui.widgets import Card, SectionTitle, Subtitle
from ui.state import AppState
from core.vehicle_manager import VehicleManager

COLUMNS = [
    ("car_name", "Name"),
    ("car_id", "Car ID"),
    ("top_rpm", "Top RPM"),
    ("top_speed", "Top Speed"),
    ("max_engine_rpm", "Max Engine RPM"),
    ("max_suspension_load", "Max Susp. Load"),
    ("peak_power_rpm", "Peak Power RPM"),
]


class TrackRecordPage(QWidget):
    """'FER Track record' - shows every vehicle profile saved so far, and
    lets you switch which one is active (used by Car Spec / Main Page /
    reports going forward)."""

    def __init__(self, state: AppState, navigate, parent=None):
        super().__init__(parent)
        self.state = state
        self.navigate = navigate

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)

        root.addWidget(SectionTitle("FER Track record"))
        root.addWidget(Subtitle("Vehicle profiles saved via Add new vehicle. Click 'Set Active' to switch cars."))

        card = Card()
        self.table = QTableWidget(0, len(COLUMNS) + 1)
        self.table.setHorizontalHeaderLabels([label for _, label in COLUMNS] + ["Active"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        card.add(self.table)
        root.addWidget(card)

        root.addStretch()

    def on_navigate(self, **kwargs):
        self._refresh()

    def on_shown(self):
        self._refresh()

    def _refresh(self):
        vehicles = VehicleManager.list_vehicles()
        active = VehicleManager.get_active_vehicle()
        active_id = active.get("car_id") if active else None

        self.table.setRowCount(len(vehicles))
        for row, v in enumerate(vehicles):
            for col, (key, _) in enumerate(COLUMNS):
                self.table.setItem(row, col, QTableWidgetItem(str(v.get(key, ""))))

            action_col = len(COLUMNS)
            if v.get("car_id") == active_id:
                item = QTableWidgetItem("\u2713 Active")
                self.table.setItem(row, action_col, item)
            else:
                btn = QPushButton("Set Active")
                btn.setCursor(self.table.cursor())
                btn.clicked.connect(lambda checked=False, car_id=v.get("car_id"): self._set_active(car_id))
                self.table.setCellWidget(row, action_col, btn)

    def _set_active(self, car_id: str):
        VehicleManager.set_active_vehicle(car_id)
        self._refresh()
