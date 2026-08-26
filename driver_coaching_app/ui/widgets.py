from PyQt6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from ui.theme import COLORS


class Card(QFrame):
    """White rounded box with a black outline - the wireframe's basic
    container unit (used for every panel/section in the app)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setProperty("class", "Card")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(20, 18, 20, 18)
        self._layout.setSpacing(10)

    def layout(self):
        return self._layout

    def add(self, widget):
        self._layout.addWidget(widget)
        return widget


class SectionTitle(QLabel):
    def __init__(self, text, size=20, parent=None):
        super().__init__(text, parent)
        font = QFont()
        font.setPointSize(size)
        font.setBold(True)
        self.setFont(font)


class Subtitle(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 13px;")
        self.setWordWrap(True)


class StatusDot(QLabel):
    """Colored circle used on the traffic-light hub page."""

    SIZE = 26

    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setStyleSheet(
            f"background-color: {color}; border-radius: {self.SIZE // 2}px; border: none;"
        )


class HubRow(QWidget):
    """One row on the Deploy hub page: colored dot + labeled pill button."""

    clicked = pyqtSignal()

    def __init__(self, color: str, label: str, enabled_style: str = "GhostButton", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.addWidget(StatusDot(color))
        layout.addSpacing(14)

        self.button = QPushButton(label)
        self.button.setObjectName(enabled_style)
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button.clicked.connect(self.clicked.emit)
        layout.addWidget(self.button)
        layout.addStretch()


class TopBar(QWidget):
    """Header bar with the hamburger icon + app title, matching every
    screen in the wireframe. A Back button appears to the right of the
    hamburger on every page except Main — MainWindow toggles it."""

    menu_clicked = pyqtSignal()
    back_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TopBar")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)

        self.menu_btn = QPushButton("\u2630")  # hamburger glyph
        self.menu_btn.setObjectName("HamburgerButton")
        self.menu_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.menu_btn.clicked.connect(self.menu_clicked.emit)
        layout.addWidget(self.menu_btn)

        self.back_btn = QPushButton("\u2190 Back")
        self.back_btn.setObjectName("BackButton")
        self.back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.back_btn.clicked.connect(self.back_clicked.emit)
        self.back_btn.setVisible(False)
        layout.addWidget(self.back_btn)

        title = QLabel("BI Modal Driving Feedback Race Engineer")
        title.setObjectName("AppTitle")
        layout.addWidget(title)
        layout.addStretch()

    def set_back_visible(self, visible: bool):
        self.back_btn.setVisible(visible)


class Sidebar(QFrame):
    """Slide-out navigation menu: Main Page / Feedback History /
    FER Track record / Add new vehicle - exactly the four items in the
    wireframe's hamburger menu."""

    navigate = pyqtSignal(str)

    ITEMS = [
        ("Main Page", "main"),
        ("Feedback History", "feedback_history"),
        ("FER Track record", "track_record"),
        ("Add new vehicle", "add_vehicle"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(230)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 24, 12, 12)
        layout.setSpacing(4)

        self._buttons = {}
        for label, key in self.ITEMS:
            btn = QPushButton(label)
            btn.setObjectName("SidebarNavButton")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, k=key: self.navigate.emit(k))
            layout.addWidget(btn)
            self._buttons[key] = btn

        layout.addStretch()


def make_button(text: str, style: str = "GhostButton") -> QPushButton:
    btn = QPushButton(text)
    btn.setObjectName(style)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    return btn
