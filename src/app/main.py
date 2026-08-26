"""
Entry point for the locally installable Tkinter desktop application,
per proposal §1.7.2 / §3.3: a simple upload-and-process interface that
takes session video + telemetry CSV and returns a coaching report.
"""

import tkinter as tk


class DriverCoachApp(tk.Tk):
    """
    Root application window. Hosts the upload view and results view,
    switching between them via the AppController.
    """

    def __init__(self):
        super().__init__()
        pass

    def configure_window(self):
        """Set window title, dimensions, and basic styling."""
        pass

    def launch_upload_view(self):
        """Display the initial upload screen."""
        pass

    def launch_results_view(self, feedback_items: list):
        """Display the coaching report once processing is complete."""
        pass


def main():
    """Application entry point — instantiate and run the Tkinter mainloop."""
    pass


if __name__ == "__main__":
    main()
