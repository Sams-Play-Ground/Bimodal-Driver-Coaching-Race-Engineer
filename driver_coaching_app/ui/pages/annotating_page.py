from pathlib import Path
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QHBoxLayout
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QMovie
from ui.widgets import Card, SectionTitle, make_button
from ui.state import AppState
from workers.annotate_worker import AnnotateWorker

try:
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
    from PyQt6.QtMultimediaWidgets import QVideoWidget
    QTMULTIMEDIA_AVAILABLE = True
except ImportError:
    QTMULTIMEDIA_AVAILABLE = False

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
# Drop your looping clip at one of these filenames inside assets/ — checked
# in this order. mp4/mov/webm play back muted via QMediaPlayer; gif via QMovie.
VIDEO_CANDIDATES = ["waiting_loop.mp4", "waiting_loop.mov", "waiting_loop.webm"]
GIF_CANDIDATE = "waiting_loop.gif"


class AnnotatingPage(QWidget):
    """'Annotating video, will take a while...' screen: progress bar plus
    a looping clip/message to keep the user entertained during the wait.
    """

    def __init__(self, state: AppState, navigate, parent=None):
        super().__init__(parent)
        self.state = state
        self.navigate = navigate
        self.worker = None
        self._movie = None
        self._player = None
        self._audio_output = None

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)

        root.addWidget(SectionTitle("Annotating video, will take a while..."))

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        root.addWidget(self.progress_bar)

        entertain_card = Card()
        self.entertain_label = QLabel("Loop of video attached within the app to keep you entertained \U0001F3CE\ufe0f")
        self.entertain_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.entertain_label.setMinimumHeight(220)
        entertain_card.add(self.entertain_label)
        root.addWidget(entertain_card)
        self._entertain_card = entertain_card

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.continue_btn = make_button("Watch replay \u2192", "GreenButton")
        self.continue_btn.setEnabled(False)
        self.continue_btn.clicked.connect(lambda: self.navigate("video_player"))
        btn_row.addWidget(self.continue_btn)
        root.addLayout(btn_row)

        root.addStretch()
        self._load_entertainment_loop()

    def _load_entertainment_loop(self):
        """Priority: a real video file (mp4/mov/webm, muted + looped via
        QMediaPlayer) -> a .gif (via QMovie) -> the text placeholder.
        Drop your file at driver_coaching_app/assets/waiting_loop.<ext>."""
        for name in VIDEO_CANDIDATES:
            path = ASSETS_DIR / name
            if path.exists() and QTMULTIMEDIA_AVAILABLE:
                self._play_video_loop(path)
                return
            if path.exists() and not QTMULTIMEDIA_AVAILABLE:
                self.entertain_label.setText(
                    "Found a waiting-loop video, but PyQt6 QtMultimedia isn't available in this "
                    "environment - showing the text placeholder instead."
                )
                return

        gif_path = ASSETS_DIR / GIF_CANDIDATE
        if gif_path.exists():
            self._movie = QMovie(str(gif_path))
            self.entertain_label.setMovie(self._movie)
            self._movie.start()

    def _play_video_loop(self, path: Path):
        video_widget = QVideoWidget()
        video_widget.setMinimumHeight(220)

        self._player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._audio_output.setMuted(True)  # silent by design — a background loop, not a clip with sound
        self._player.setAudioOutput(self._audio_output)
        self._player.setVideoOutput(video_widget)
        self._player.setSource(QUrl.fromLocalFile(str(path)))
        self._player.setLoops(QMediaPlayer.Loops.Infinite)

        # Swap the placeholder label out for the actual video widget.
        self._entertain_card.layout().replaceWidget(self.entertain_label, video_widget)
        self.entertain_label.hide()

        self._player.play()

    def on_navigate(self, **kwargs):
        self._run()

    def _run(self):
        self.progress_bar.setValue(0)
        self.continue_btn.setEnabled(False)

        if not self.state.footage_path:
            # No footage was submitted this session — nothing to annotate.
            self.progress_bar.setValue(100)
            self.continue_btn.setEnabled(True)
            return

        self.worker = AnnotateWorker(self.state.footage_path)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_finished(self, output_path: str):
        self.state.annotated_video_path = Path(output_path)
        self.progress_bar.setValue(100)
        self.continue_btn.setEnabled(True)

    def _on_failed(self, error_msg: str):
        self.entertain_label.setText(f"Annotation failed: {error_msg}")
        self.continue_btn.setEnabled(True)
