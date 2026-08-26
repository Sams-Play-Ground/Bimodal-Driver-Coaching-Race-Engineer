from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget, QScrollArea
from PyQt6.QtCore import Qt

from ui.widgets import TopBar, Sidebar
from ui.state import AppState

from ui.pages.main_page import MainPage
from ui.pages.car_spec_page import CarSpecPage
from ui.pages.hub_page import HubPage
from ui.pages.coming_soon_page import ComingSoonPage
from ui.pages.processing_page import ProcessingPage
from ui.pages.report_page import ReportPage
from ui.pages.annotating_page import AnnotatingPage
from ui.pages.video_player_page import VideoPlayerPage
from ui.pages.feedback_history_page import FeedbackHistoryPage
from ui.pages.track_record_page import TrackRecordPage
from ui.pages.add_vehicle_page import AddVehiclePage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BI Modal Driving Feedback Race Engineer")
        self.resize(1100, 760)

        self.state = AppState()

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # -- Top bar ---------------------------------------------------
        self.top_bar = TopBar()
        self.top_bar.menu_clicked.connect(self._toggle_sidebar)
        self.top_bar.back_clicked.connect(self.go_back)
        outer.addWidget(self.top_bar)

        # -- Body: sidebar drawer + page stack --------------------------
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        outer.addLayout(body)

        self.sidebar = Sidebar()
        self.sidebar.navigate.connect(self.navigate_to)
        self.sidebar.setVisible(False)
        body.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        body.addWidget(self.stack)

        # -- Build every page, keyed for navigation ----------------------
        self.pages = {}
        self._scroll_areas = {}
        self._register("main", MainPage(self.state, self.navigate_to))
        self._register("car_spec", CarSpecPage(self.state, self.navigate_to))
        self._register("hub", HubPage(self.state, self.navigate_to))
        self._register("coming_soon", ComingSoonPage(self.state, self.navigate_to))
        self._register("processing", ProcessingPage(self.state, self.navigate_to))
        self._register("report", ReportPage(self.state, self.navigate_to))
        self._register("annotating", AnnotatingPage(self.state, self.navigate_to))
        self._register("video_player", VideoPlayerPage(self.state, self.navigate_to))
        self._register("feedback_history", FeedbackHistoryPage(self.state, self.navigate_to))
        self._register("track_record", TrackRecordPage(self.state, self.navigate_to))
        self._register("add_vehicle", AddVehiclePage(self.state, self.navigate_to))

        # -- Navigation history (for the Back button) --------------------
        self._history = []
        self._current_key = None

        self.navigate_to("main")

    def _register(self, key: str, widget: QWidget):
        """Wraps every page in a QScrollArea so content taller/wider than
        the window becomes scrollable instead of getting clipped or
        forcing the window to grow. self.pages[key] stays the actual page
        (on_navigate/on_shown are called on it); the stack holds the
        scroll wrapper."""
        self.pages[key] = widget
        scroll = QScrollArea()
        scroll.setObjectName("PageScrollArea")
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.stack.addWidget(scroll)
        self._scroll_areas[key] = scroll

    def _toggle_sidebar(self):
        self.sidebar.setVisible(not self.sidebar.isVisible())

    def navigate_to(self, key: str, **kwargs):
        """Forward navigation — pushes the current page onto history so
        Back can return to it. Sidebar items and in-page 'Continue'/'Next'
        buttons should always call this (not go_back)."""
        self._go(key, push_history=True, **kwargs)

    def go_back(self):
        """Pops the last page off history and returns to it, without
        pushing a new history entry (so Back/Forward doesn't loop)."""
        if not self._history:
            return
        prev_key = self._history.pop()
        self._go(prev_key, push_history=False)

    def _go(self, key: str, push_history: bool, **kwargs):
        page = self.pages.get(key)
        if page is None:
            return

        if push_history and self._current_key is not None and self._current_key != key:
            self._history.append(self._current_key)

        self._current_key = key
        self.stack.setCurrentWidget(self._scroll_areas[key])
        self.sidebar.setVisible(False)
        # Back is hidden on Main (the app's home) — every other page shows it.
        self.top_bar.set_back_visible(key != "main")

        if hasattr(page, "on_navigate"):
            page.on_navigate(**kwargs)
        elif hasattr(page, "on_shown"):
            page.on_shown()
