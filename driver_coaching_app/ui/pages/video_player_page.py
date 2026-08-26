from pathlib import Path
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget

from ui.widgets import SectionTitle, make_button
from ui.state import AppState


class VideoPlayerPage(QWidget):
    """'Full replay of annotated video playback' screen: pause/play,
    +5 sec and -5 sec transport controls, matching the wireframe exactly."""

    def __init__(self, state: AppState, navigate, parent=None):
        super().__init__(parent)
        self.state = state
        self.navigate = navigate

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)

        root.addWidget(SectionTitle("Annotated Replay"))

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(360)
        root.addWidget(self.video_widget)

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.video_widget)

        controls = QHBoxLayout()
        controls.addStretch()

        back5_btn = make_button("-5 sec", "GhostButton")
        back5_btn.clicked.connect(lambda: self._seek(-5000))
        controls.addWidget(back5_btn)

        self.play_pause_btn = make_button("pause/play", "GhostButton")
        self.play_pause_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_pause_btn)

        fwd5_btn = make_button("+5 sec", "GhostButton")
        fwd5_btn.clicked.connect(lambda: self._seek(5000))
        controls.addWidget(fwd5_btn)

        controls.addStretch()
        root.addLayout(controls)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #5B5B5B;")
        root.addWidget(self.status_label)

        root.addStretch()

    def on_navigate(self, **kwargs):
        path = self.state.annotated_video_path
        if not path or not Path(path).exists():
            self.status_label.setText("No annotated video available yet - run 'Watch annotated video' first.")
            return

        self.status_label.setText(f"Playing: {Path(path).name}")
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.player.play()

    def _toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _seek(self, delta_ms: int):
        new_pos = max(0, self.player.position() + delta_ms)
        self.player.setPosition(new_pos)
