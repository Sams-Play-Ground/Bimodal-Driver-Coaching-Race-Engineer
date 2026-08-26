"""
Upload view: lets the driver select their session video file and
telemetry CSV, then triggers processing.
"""

import tkinter as tk


class UploadView(tk.Frame):
    """
    Frame containing file-picker widgets for video and telemetry input,
    plus a "Process Session" button.
    """

    def __init__(self, parent, on_submit_callback):
        super().__init__(parent)
        pass

    def build_widgets(self):
        """Construct file picker buttons, labels, and the submit button."""
        pass

    def select_video_file(self):
        """Open a file dialog for selecting the session video file."""
        pass

    def select_telemetry_file(self):
        """Open a file dialog for selecting the telemetry CSV file."""
        pass

    def on_submit(self):
        """Validate selected files and invoke the processing callback."""
        pass

    def show_processing_indicator(self):
        """Display a progress indicator while the pipeline runs."""
        pass
