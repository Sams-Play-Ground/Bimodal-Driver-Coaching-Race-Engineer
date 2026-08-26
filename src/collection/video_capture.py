"""
Handles capture and retrieval of onboard video footage recorded via
Assetto Corsa's built-in replay/session recording system.

Per proposal §3.4: cockpit-view onboard camera, 1280x720 @ 60fps.
"""

from src.common.data_types import SessionMetadata


class VideoCapture:
    """
    Wraps onboard video capture for a single recording session.
    """

    def __init__(self, output_directory: str, resolution: tuple, frame_rate_fps: int):
        pass

    def start_recording(self, metadata: SessionMetadata) -> None:
        """Begin onboard video capture for the current session."""
        pass

    def stop_recording(self) -> str:
        """
        Stop the current video recording and finalize the output file.

        Returns:
            Path to the finalized raw video file.
        """
        pass

    def verify_camera_view(self) -> bool:
        """Confirm the active camera view matches the required cockpit/onboard angle."""
        pass

    def get_raw_video_path(self) -> str:
        """Return the file path of the most recently recorded video."""
        pass


def estimate_file_size_gb(duration_minutes: float, resolution: tuple, frame_rate_fps: int) -> float:
    """Estimate output file size in GB for a given recording duration and spec."""
    pass
