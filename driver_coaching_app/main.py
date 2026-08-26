import sys
from PyQt6.QtWidgets import QApplication

from core.data_manager import DataManager
from ui.theme import APP_STYLESHEET
from ui.main_window import MainWindow


def main():
    DataManager.initialize_storage()

    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
