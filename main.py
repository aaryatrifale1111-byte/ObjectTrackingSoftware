
import sys

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox
)

from PySide6.QtCore import Qt


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "YOLO + SAM2 Object Detection and Tracking"
        )

        self.resize(1100, 750)

        self.setup_ui()

    def setup_ui(self):

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Application heading
        heading = QLabel(
            "OBJECT DETECTION AND TRACKING SYSTEM"
        )

        heading.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        heading.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            padding: 12px;
        """)

        main_layout.addWidget(heading)

        # Live camera preview
        self.camera_preview = QLabel(
            "CAMERA INPUT WILL APPEAR HERE"
        )

        self.camera_preview.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.camera_preview.setMinimumSize(800, 500)

        self.camera_preview.setStyleSheet("""
            background-color: #171717;
            color: white;
            border: 2px solid #555555;
            font-size: 18px;
        """)

        main_layout.addWidget(
            self.camera_preview,
            stretch=1
        )

        # Camera controls
        camera_group = QGroupBox("Camera Controls")

        camera_layout = QHBoxLayout(camera_group)

        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")

        self.stop_button.setEnabled(False)

        camera_layout.addWidget(self.start_button)
        camera_layout.addWidget(self.stop_button)

        main_layout.addWidget(camera_group)

        # Status information
        info_layout = QHBoxLayout()

        self.status_label = QLabel(
            "Status: Not connected"
        )

        self.fps_label = QLabel(
            "FPS: 0"
        )

        info_layout.addWidget(self.status_label)
        info_layout.addStretch()
        info_layout.addWidget(self.fps_label)

        main_layout.addLayout(info_layout)


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
