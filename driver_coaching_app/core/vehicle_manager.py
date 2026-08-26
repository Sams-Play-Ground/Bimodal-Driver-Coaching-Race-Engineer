from typing import List, Dict, Optional
from core.data_manager import DataManager

VEHICLES_FILE = "vehicles.json"
ACTIVE_VEHICLE_FILE = "active_vehicle.json"

# Every vehicle record has exactly these fields. car_name/car_id are
# identifiers (car_id must be unique - plate or chassis number); the rest
# are numeric spec values used elsewhere in the pipeline/report.
SPEC_FIELDS = [
    "top_rpm",
    "top_speed",
    "max_engine_rpm",
    "max_suspension_load",
    "peak_power_rpm",
]


class VehicleManager:
    """Persists vehicle/car-spec profiles to storage/config/vehicles.json,
    and tracks which one is currently active (the vehicle new sessions and
    reports are attributed to) in storage/config/active_vehicle.json.
    """

    # -- Reading -----------------------------------------------------
    @staticmethod
    def list_vehicles() -> List[Dict]:
        data = DataManager.load_config(VEHICLES_FILE)
        return data.get("vehicles", []) if data else []

    @staticmethod
    def get_vehicle(car_id: str) -> Optional[Dict]:
        for v in VehicleManager.list_vehicles():
            if v.get("car_id") == car_id:
                return v
        return None

    @staticmethod
    def get_active_vehicle() -> Optional[Dict]:
        cfg = DataManager.load_config(ACTIVE_VEHICLE_FILE)
        active_id = cfg.get("active_car_id") if cfg else None
        if active_id:
            vehicle = VehicleManager.get_vehicle(active_id)
            if vehicle:
                return vehicle
        # Fall back to the most recently added vehicle if no active
        # pointer is set yet (e.g. fresh install).
        vehicles = VehicleManager.list_vehicles()
        return vehicles[-1] if vehicles else None

    # -- Writing -------------------------------------------------------
    @staticmethod
    def add_vehicle(
        car_name: str,
        car_id: str,
        top_rpm: float = 0,
        top_speed: float = 0,
        max_engine_rpm: float = 0,
        max_suspension_load: float = 0,
        peak_power_rpm: float = 0,
        notes: str = "",
    ) -> Dict:
        """Creates a new vehicle record. car_id must be unique - raises
        ValueError if a vehicle with that ID already exists (use
        update_vehicle to edit an existing one instead)."""
        vehicles = VehicleManager.list_vehicles()
        if any(v.get("car_id") == car_id for v in vehicles):
            raise ValueError(f"A vehicle with car ID '{car_id}' already exists.")

        record = {
            "car_name": car_name,
            "car_id": car_id,
            "top_rpm": top_rpm,
            "top_speed": top_speed,
            "max_engine_rpm": max_engine_rpm,
            "max_suspension_load": max_suspension_load,
            "peak_power_rpm": peak_power_rpm,
            "notes": notes,
        }
        vehicles.append(record)
        DataManager.save_config(VEHICLES_FILE, {"vehicles": vehicles})
        VehicleManager.set_active_vehicle(car_id)
        return record

    @staticmethod
    def update_vehicle(car_id: str, **fields) -> Dict:
        """Edits an existing vehicle's fields (e.g. spec values from the
        Car Spec quick-edit screen). car_name/car_id are intentionally
        editable here too, but changing car_id will break the active
        pointer unless you also call set_active_vehicle with the new id."""
        vehicles = VehicleManager.list_vehicles()
        for v in vehicles:
            if v.get("car_id") == car_id:
                v.update(fields)
                DataManager.save_config(VEHICLES_FILE, {"vehicles": vehicles})
                return v
        raise ValueError(f"No vehicle found with car ID '{car_id}'.")

    @staticmethod
    def set_active_vehicle(car_id: str):
        DataManager.save_config(ACTIVE_VEHICLE_FILE, {"active_car_id": car_id})
