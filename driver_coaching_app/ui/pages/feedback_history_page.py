import pandas as pd
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QHBoxLayout
from ui.widgets import Card, SectionTitle, Subtitle, make_button
from ui.state import AppState
from core.data_manager import DataManager
from core.fusion_engine import BimodalFusionEngine, FUSED_LABEL_COLS
from core.feedback_generator import FeedbackGenerator


class FeedbackHistoryPage(QWidget):
    """Lists every saved session report so the driver can revisit past
    feedback (one of the four hamburger-menu destinations)."""

    def __init__(self, state: AppState, navigate, parent=None):
        super().__init__(parent)
        self.state = state
        self.navigate = navigate

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)

        root.addWidget(SectionTitle("Feedback History"))
        root.addWidget(Subtitle("Every saved session report lives here."))

        card = Card()
        self.list_widget = QListWidget()
        card.add(self.list_widget)
        root.addWidget(card)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        open_btn = make_button("Open selected report", "GreenButton")
        open_btn.clicked.connect(self._open_selected)
        btn_row.addWidget(open_btn)
        root.addLayout(btn_row)

        root.addStretch()

    def on_navigate(self, **kwargs):
        self._refresh()

    def _refresh(self):
        self.list_widget.clear()
        sessions = DataManager.list_sessions()
        if not sessions:
            self.list_widget.addItem("No sessions saved yet - run 'Process data' from the Main Page.")
            return
        for path in sessions:
            item = QListWidgetItem(path.name)
            item.setData(1000, str(path))
            self.list_widget.addItem(item)

    def _open_selected(self):
        item = self.list_widget.currentItem()
        if not item or not item.data(1000):
            return
        path = item.data(1000)
        df = pd.read_csv(path)
        # Saved fusion reports use one pred_<label>/prob_<label> pair per
        # class (see core/fusion_engine.py) rather than a single
        # Detected_Event column.
        pred_cols = [c for c in df.columns if c.startswith("pred_")]
        event_counts = {c.replace("pred_", ""): int(df[c].sum()) for c in pred_cols}
        any_event = df[pred_cols].any(axis=1) if pred_cols else pd.Series([], dtype=bool)
        summary = {
            "total_windows": len(df),
            "flagged_windows": int(any_event.sum()) if pred_cols else 0,
            "event_counts": event_counts,
            "top_event": max(event_counts, key=event_counts.get) if any(event_counts.values()) else None,
            "top_speed_kmh": None,
        }

        # Regenerate the coaching text from the saved probabilities so
        # revisiting a past session still shows feedback, not just counts.
        feedback_result = {"feedback_items": [], "driving_persona": None, "fer": 0.0}
        prob_cols = [f"prob_{c}" for c in FUSED_LABEL_COLS]
        if all(c in df.columns for c in prob_cols):
            thresholds = FeedbackGenerator.thresholds_from_fusion_engine(BimodalFusionEngine(
                weights_path=DataManager.FUSION_CHECKPOINT, config_path=DataManager.FUSION_CONFIG,
            ))
            feedback_result = FeedbackGenerator(thresholds=thresholds).session_summary(df[prob_cols].values)

        self.state.pipeline_result = {
            "results_df": df, "summary": summary, "telemetry_df": None,
            "feedback_items": feedback_result["feedback_items"],
            "driving_persona": feedback_result["driving_persona"],
            "fer": feedback_result["fer"],
        }
        self.navigate("report")
