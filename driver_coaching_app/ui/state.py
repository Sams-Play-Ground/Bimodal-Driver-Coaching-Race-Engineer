from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class AppState:
    """Everything pages need to share: file paths picked on the Main Page,
    the active vehicle, and results produced by the pipeline / video
    annotation steps. One instance lives on the MainWindow and is passed
    into every page.
    """
    footage_path: Optional[Path] = None
    telemetry_path: Optional[Path] = None
    pipeline_result: Optional[Dict[str, Any]] = None
    annotated_video_path: Optional[Path] = None
    report_pdf_path: Optional[Path] = None
    active_vehicle: Optional[Dict[str, Any]] = None
