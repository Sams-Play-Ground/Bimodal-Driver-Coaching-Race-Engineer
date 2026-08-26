"""
Coordinates TelemetryRecorder and VideoCapture so that a single session
captures both streams in sync and saves them to the removable storage medium
under a consistent session_id.
"""

from src.collection.telemetry_recorder import TelemetryRecorder
from src.collection.video_capture import VideoCapture
from src.common.data_types import SessionMetadata


class SessionManager:
    """
    High-level orchestrator for a single data collection session,
    per proposal §3.4 (Data Collection Methods).
    """

    def __init__(self, telemetry_recorder: TelemetryRecorder, video_capture: VideoCapture):
        pass

    def begin_session(self, metadata: SessionMetadata) -> str:
        """
        Start both telemetry and video capture simultaneously.

        Returns:
            The generated session_id for this recording.
        """
        pass

    def end_session(self) -> dict:
        """
        Stop both recordings and return their resulting file paths.

        Returns:
            Dictionary with keys 'telemetry_csv_path' and 'video_path'.
        """
        pass

    def save_session_manifest(self, metadata: SessionMetadata, file_paths: dict) -> str:
        """Write a JSON manifest describing this session to the storage medium."""
        pass
