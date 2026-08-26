"""
Lightweight smoke tests for the core (non-UI) modules. Run with:
    pytest tests/
These don't require a display, so they're safe to run in CI.
"""
import numpy as np
from core.fusion_engine import BimodalFusionEngine, FUSED_LABEL_COLS
from core.lstm_engine import TelemetryEngine
from core.data_manager import DataManager
from core.vehicle_manager import VehicleManager


def test_data_manager_storage_init():
    DataManager.initialize_storage()
    assert DataManager.MODELS_DIR.exists()
    assert DataManager.CONFIG_DIR.exists()


def test_vehicle_manager_roundtrip():
    VehicleManager.add_vehicle(car_name="Test Car", car_id="TEST-001", top_rpm=8500, top_speed=210)
    vehicles = VehicleManager.list_vehicles()
    assert any(v["car_id"] == "TEST-001" for v in vehicles)
    active = VehicleManager.get_active_vehicle()
    assert active["car_id"] == "TEST-001"


def test_fusion_engine_vision_only():
    """Fusion must still run with only one modality present (no fallback,
    no collapsed row count) — this is the modality-dropout behaviour the
    fusion head was trained for."""
    engine = BimodalFusionEngine(weights_path=DataManager.FUSION_CHECKPOINT,
                                  config_path=DataManager.FUSION_CONFIG)
    cnn_probs = np.random.rand(10, 3).astype(np.float32)
    lstm_probs = np.zeros((0, 7), dtype=np.float32)
    df = engine.evaluate(cnn_probs, lstm_probs)
    if engine.available:
        assert len(df) == 10
        assert all(f"prob_{c}" in df.columns for c in FUSED_LABEL_COLS)


def test_fusion_engine_telemetry_only():
    engine = BimodalFusionEngine(weights_path=DataManager.FUSION_CHECKPOINT,
                                  config_path=DataManager.FUSION_CONFIG)
    cnn_probs = np.zeros((0, 3), dtype=np.float32)
    lstm_probs = np.random.rand(8, 7).astype(np.float32)
    df = engine.evaluate(cnn_probs, lstm_probs)
    if engine.available:
        assert len(df) == 8


def test_fusion_engine_both_missing_returns_empty():
    engine = BimodalFusionEngine(weights_path=DataManager.FUSION_CHECKPOINT,
                                  config_path=DataManager.FUSION_CONFIG)
    df = engine.evaluate(np.zeros((0, 3)), np.zeros((0, 7)))
    assert df.empty


def test_telemetry_engine_reports_availability():
    engine = TelemetryEngine(
        weights_path=DataManager.LSTM_CHECKPOINT,
        normalizer_path=DataManager.LSTM_NORMALIZER,
        threshold_path=DataManager.LSTM_THRESHOLDS,
    )
    assert isinstance(engine.available, bool)
