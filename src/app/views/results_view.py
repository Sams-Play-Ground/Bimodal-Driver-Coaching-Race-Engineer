"""
Results view: displays the final post-session coaching report and the
assigned Driving Persona for the session.
"""

import tkinter as tk


class ResultsView(tk.Frame):
    """
    Frame displaying the list of FeedbackItem messages and the session's
    classified Driving Persona.
    """

    def __init__(self, parent, feedback_items: list, persona):
        super().__init__(parent)
        pass

    def build_widgets(self):
        """Construct the scrollable feedback list and persona label."""
        pass

    def render_feedback_list(self, feedback_items: list):
        """Populate the view with each feedback message in order."""
        pass

    def render_persona_badge(self, persona):
        """Display the classified Driving Persona prominently."""
        pass

    def export_report_to_file(self, output_path: str):
        """Save the displayed coaching report to a text or PDF file."""
        pass
