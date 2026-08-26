import json
import shutil
from pathlib import Path
from typing import Optional
import pandas as pd

# Base storage layout -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
MODELS_DIR = STORAGE_DIR / "models"
CONFIG_DIR = STORAGE_DIR / "config"
UPLOADS_DIR = STORAGE_DIR / "raw_uploads"
OUTPUTS_DIR = STORAGE_DIR / "outputs"


class DataManager:
    """Central helper for every read/write the app does to local disk."""

    MODELS_DIR = MODELS_DIR
    CONFIG_DIR = CONFIG_DIR
    UPLOADS_DIR = UPLOADS_DIR
    OUTPUTS_DIR = OUTPUTS_DIR

    # Expected checkpoint filenames inside storage/models/. These match
    # your actual exported artifact names from the notebook.
    CNN_CHECKPOINT = MODELS_DIR / "CNN-module-best_model.pt"
    LSTM_CHECKPOINT = MODELS_DIR / "LSTM-module-best_model.pt"
    FUSION_CHECKPOINT = MODELS_DIR / "late_fusion_head.pt"

    # Supporting artifacts the engines need alongside the weights.
    # Export these from the notebook — see the deployment checklist in README.md.
    LSTM_NORMALIZER = CONFIG_DIR / "normalizer_params.json"
    LSTM_THRESHOLDS = CONFIG_DIR / "lstm_optimal_thresholds.json"
    FUSION_CONFIG = CONFIG_DIR / "inference_config.json"

    @staticmethod
    def initialize_storage():
        """Ensure all persistent directories exist. Call once at startup."""
        for path in [MODELS_DIR, CONFIG_DIR, UPLOADS_DIR, OUTPUTS_DIR]:
            path.mkdir(parents=True, exist_ok=True)

    # -- File ingestion ------------------------------------------------
    @staticmethod
    def copy_local_file(source_path: str, target_subfolder: str = "raw_uploads") -> Path:
        """Copies a file picked from the native file dialog into local storage."""
        source = Path(source_path)
        target_dir = STORAGE_DIR / target_subfolder
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / source.name
        if source.resolve() != dest.resolve():
            shutil.copy2(source, dest)
        return dest

    @staticmethod
    def delete_uploaded_file(path: Optional[Path]) -> bool:
        """Deletes a previously-ingested file from storage/raw_uploads/.
        Used by the Main Page's 'Clear' buttons. Only deletes files that
        actually live under UPLOADS_DIR — never anything else on disk,
        even if a path was passed in from elsewhere."""
        if not path:
            return False
        path = Path(path)
        try:
            path.resolve().relative_to(UPLOADS_DIR.resolve())
        except ValueError:
            return False  # not inside raw_uploads/ — refuse to touch it
        if path.exists():
            path.unlink()
            return True
        return False

    # -- JSON config -----------------------------------------------------
    @staticmethod
    def load_config(filename: str) -> dict:
        path = CONFIG_DIR / filename
        if not path.exists():
            return {}
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def save_config(filename: str, data: dict):
        path = CONFIG_DIR / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    # -- Fusion / session outputs -----------------------------------------
    @staticmethod
    def save_fusion_output(df: pd.DataFrame, filename: str = "latest_fusion_report.csv") -> Path:
        path = OUTPUTS_DIR / filename
        df.to_csv(path, index=False)
        return path

    @staticmethod
    def list_sessions() -> list:
        """Returns saved session report CSVs, most recent first."""
        if not OUTPUTS_DIR.exists():
            return []
        files = sorted(OUTPUTS_DIR.glob("session_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        return files

    @staticmethod
    def checkpoint_status() -> dict:
        """Reports which trained checkpoints AND their supporting config
        files are actually present on disk, for display on the Settings
        page. A checkpoint without its supporting file (e.g. LSTM weights
        without the normalizer) still shows found=False, since the engine
        can't run without both."""
        checkpoints = {
            "CNN (vision)": [DataManager.CNN_CHECKPOINT],
            "LSTM (telemetry)": [DataManager.LSTM_CHECKPOINT, DataManager.LSTM_NORMALIZER],
            "Fusion head": [DataManager.FUSION_CHECKPOINT, DataManager.FUSION_CONFIG],
        }
        status = {}
        for label, paths in checkpoints.items():
            all_found = all(p.exists() for p in paths)
            main = paths[0]
            status[label] = {
                "path": main,
                "found": all_found,
                "size_mb": round(main.stat().st_size / (1024 * 1024), 2) if main.exists() else None,
            }
        return status
