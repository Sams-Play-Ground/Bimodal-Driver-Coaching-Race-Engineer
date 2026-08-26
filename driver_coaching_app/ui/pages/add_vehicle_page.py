from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QDoubleSpinBox, QTextEdit, QMessageBox
)
from ui.widgets import Card, SectionTitle, Subtitle, make_button
from ui.state import AppState
from core.vehicle_manager import VehicleManager


class AddVehiclePage(QWidget):
    """'Add new vehicle' - the fourth hamburger-menu destination. Creates
    a brand new vehicle profile with a unique car ID; identifiers (name/ID)
    are set once here, spec values can later be tweaked from Car Spec."""

    def __init__(self, state: AppState, navigate, parent=None):
        super().__init__(parent)
        self.state = state
        self.navigate = navigate

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)

        root.addWidget(SectionTitle("Add new vehicle"))
        root.addWidget(Subtitle("Register a new car profile so it can be tracked and selected across sessions."))

        # -- Identity ----------------------------------------------------
        id_card = Card()
        id_card.add(SectionTitle("Identity", size=14))

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Car name"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g. Track Car #2")
        name_row.addWidget(self.name_input)
        id_card.layout().addLayout(name_row)

        id_row = QHBoxLayout()
        id_row.addWidget(QLabel("Car ID"))
        self.car_id_input = QLineEdit()
        self.car_id_input.setPlaceholderText("Number plate or chassis number (must be unique)")
        id_row.addWidget(self.car_id_input)
        id_card.layout().addLayout(id_row)

        root.addWidget(id_card)

        # -- Spec ------------------------------------------------------
        spec_card = Card()
        spec_card.add(SectionTitle("Spec", size=14))

        self.top_rpm_input = self._add_spin_row(spec_card, "Top RPM", 0, 20000, 0)
        self.top_speed_input = self._add_spin_row(spec_card, "Top Speed", 0, 500, 0, suffix=" km/h")
        self.max_engine_rpm_input = self._add_spin_row(spec_card, "Max Engine RPM", 0, 20000, 0)
        self.max_suspension_load_input = self._add_spin_row(spec_card, "Max Suspension Load", 0, 5000, 0, suffix=" kg")
        self.peak_power_rpm_input = self._add_spin_row(spec_card, "Peak Power RPM", 0, 20000, 0)

        root.addWidget(spec_card)

        # -- Notes -------------------------------------------------------
        notes_card = Card()
        notes_card.add(QLabel("Notes"))
        self.notes_input = QTextEdit()
        self.notes_input.setFixedHeight(70)
        notes_card.add(self.notes_input)
        root.addWidget(notes_card)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        save_btn = make_button("Save vehicle", "GreenButton")
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(save_btn)
        root.addLayout(btn_row)

        root.addStretch()

    @staticmethod
    def _add_spin_row(card: Card, label: str, lo: float, hi: float, default: float, suffix: str = "") -> QDoubleSpinBox:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(0)
        spin.setValue(default)
        if suffix:
            spin.setSuffix(suffix)
        row.addWidget(spin)
        card.layout().addLayout(row)
        return spin

    def _save(self):
        car_name = self.name_input.text().strip()
        car_id = self.car_id_input.text().strip()

        if not car_name or not car_id:
            QMessageBox.warning(self, "Missing info", "Both car name and car ID are required.")
            return

        try:
            record = VehicleManager.add_vehicle(
                car_name=car_name,
                car_id=car_id,
                top_rpm=self.top_rpm_input.value(),
                top_speed=self.top_speed_input.value(),
                max_engine_rpm=self.max_engine_rpm_input.value(),
                max_suspension_load=self.max_suspension_load_input.value(),
                peak_power_rpm=self.peak_power_rpm_input.value(),
                notes=self.notes_input.toPlainText().strip(),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Car ID already exists", str(exc))
            return

        self.state.active_vehicle = record

        self.name_input.clear()
        self.car_id_input.clear()
        for spin in (self.top_rpm_input, self.top_speed_input, self.max_engine_rpm_input,
                     self.max_suspension_load_input, self.peak_power_rpm_input):
            spin.setValue(0)
        self.notes_input.clear()

        QMessageBox.information(self, "Saved", f"'{car_name}' added and set as the active vehicle.")
        self.navigate("track_record")
