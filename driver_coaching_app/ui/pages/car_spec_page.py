from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox
from ui.widgets import Card, SectionTitle, Subtitle, make_button
from ui.state import AppState
from core.vehicle_manager import VehicleManager


class CarSpecPage(QWidget):
    """Quick-edit screen for the *active* vehicle's spec values. Identity
    (car name / car ID) is set once on Add new vehicle and shown here
    read-only; this page edits the numeric spec fields only. If no
    vehicle has been added yet, it points the user at Add new vehicle
    instead of showing an empty form."""

    def __init__(self, state: AppState, navigate, parent=None):
        super().__init__(parent)
        self.state = state
        self.navigate = navigate
        self.active_car_id = None

        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(24, 20, 24, 24)
        self.root.setSpacing(16)

        self.root.addWidget(SectionTitle("Car Spec"))
        self.subtitle = Subtitle("Set the reference limits used to contextualize this vehicle's telemetry.")
        self.root.addWidget(self.subtitle)

        # -- Identity (read-only) ------------------------------------
        self.identity_card = Card()
        self.name_label = QLabel()
        self.id_label = QLabel()
        self.id_label.setStyleSheet("color: #5B5B5B;")
        self.identity_card.add(self.name_label)
        self.identity_card.add(self.id_label)
        self.root.addWidget(self.identity_card)

        # -- Spec (editable) -------------------------------------------
        self.spec_card = Card()
        self.top_rpm_input = self._add_spin_row(self.spec_card, "Top RPM", 0, 20000)
        self.top_speed_input = self._add_spin_row(self.spec_card, "Top Speed", 0, 500, suffix=" km/h")
        self.max_engine_rpm_input = self._add_spin_row(self.spec_card, "Max Engine RPM", 0, 20000)
        self.max_suspension_load_input = self._add_spin_row(self.spec_card, "Max Suspension Load", 0, 5000, suffix=" kg")
        self.peak_power_rpm_input = self._add_spin_row(self.spec_card, "Peak Power RPM", 0, 20000)
        self.root.addWidget(self.spec_card)

        self.btn_row = QHBoxLayout()
        self.btn_row.addStretch()
        self.save_btn = make_button("Save", "GreenButton")
        self.save_btn.clicked.connect(self._save)
        self.btn_row.addWidget(self.save_btn)
        self.root.addLayout(self.btn_row)

        # -- Empty state (shown instead of the form if no vehicle exists) --
        self.empty_label = QLabel("No vehicle added yet. Add one first to set its spec.")
        self.empty_label.setStyleSheet("color: #5B5B5B; font-size: 15px;")
        self.root.addWidget(self.empty_label)

        self.add_vehicle_btn = make_button("Add new vehicle \u2192", "OrangeButton")
        self.add_vehicle_btn.clicked.connect(lambda: self.navigate("add_vehicle"))
        self.root.addWidget(self.add_vehicle_btn)

        self.root.addStretch()

    @staticmethod
    def _add_spin_row(card: Card, label: str, lo: float, hi: float, suffix: str = "") -> QDoubleSpinBox:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(0)
        if suffix:
            spin.setSuffix(suffix)
        row.addWidget(spin)
        card.layout().addLayout(row)
        return spin

    def on_navigate(self, **kwargs):
        self._load()

    def on_shown(self):
        self._load()

    def _load(self):
        vehicle = VehicleManager.get_active_vehicle()
        has_vehicle = vehicle is not None

        self.identity_card.setVisible(has_vehicle)
        self.spec_card.setVisible(has_vehicle)
        self.save_btn.setVisible(has_vehicle)
        self.empty_label.setVisible(not has_vehicle)
        self.add_vehicle_btn.setVisible(not has_vehicle)

        if not has_vehicle:
            self.active_car_id = None
            return

        self.active_car_id = vehicle["car_id"]
        self.name_label.setText(f"<b>{vehicle.get('car_name', '')}</b>")
        self.id_label.setText(f"Car ID: {vehicle.get('car_id', '')}")

        self.top_rpm_input.setValue(float(vehicle.get("top_rpm", 0) or 0))
        self.top_speed_input.setValue(float(vehicle.get("top_speed", 0) or 0))
        self.max_engine_rpm_input.setValue(float(vehicle.get("max_engine_rpm", 0) or 0))
        self.max_suspension_load_input.setValue(float(vehicle.get("max_suspension_load", 0) or 0))
        self.peak_power_rpm_input.setValue(float(vehicle.get("peak_power_rpm", 0) or 0))

    def _save(self):
        if not self.active_car_id:
            return
        VehicleManager.update_vehicle(
            self.active_car_id,
            top_rpm=self.top_rpm_input.value(),
            top_speed=self.top_speed_input.value(),
            max_engine_rpm=self.max_engine_rpm_input.value(),
            max_suspension_load=self.max_suspension_load_input.value(),
            peak_power_rpm=self.peak_power_rpm_input.value(),
        )
        self.navigate("main")
