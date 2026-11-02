
import sys
import numpy as np

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
)

from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

from camera.camera_worker import CameraWorker


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLO + SAM2 Object Detection and Tracking")
        self.resize(1100, 750)

        self.gentl_file = None
        self.camera_worker = None

        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Camera preview
        self.preview = QLabel("Camera preview will appear here")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(800, 500)
        self.preview.setStyleSheet(
            "background-color: #202020; color: white;"
        )

        main_layout.addWidget(self.preview)

        # File selection
        self.file_label = QLabel("GenTL producer: Not selected")
        main_layout.addWidget(self.file_label)

        self.select_button = QPushButton("Select GenTL Producer")
        self.select_button.clicked.connect(self.select_gentl_file)
        main_layout.addWidget(self.select_button)

        # Camera controls
        button_layout = QHBoxLayout()

        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera)

        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        main_layout.addLayout(button_layout)

        # Status and FPS
        self.status_label = QLabel("Status: Not connected")
        self.fps_label = QLabel("FPS: 0.0")

        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.fps_label)

    def select_gentl_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select GenTL Producer File",
            "",
            "GenTL files (*.cti *.dll);;All files (*)"
        )

        if file_path:
            self.gentl_file = file_path
            self.file_label.setText(
                f"GenTL producer: {file_path}"
            )

    def start_camera(self):

        if not self.gentl_file:
            QMessageBox.warning(
                self,
                "GenTL File Required",
                "Please select the GenTL producer file first."
            )
            return

        if self.camera_worker and self.camera_worker.isRunning():
            return

        self.camera_worker = CameraWorker(self.gentl_file)

        self.camera_worker.frame_ready.connect(self.display_frame)
        self.camera_worker.status_changed.connect(self.update_status)
        self.camera_worker.error_occurred.connect(self.show_camera_error)
        self.camera_worker.fps_updated.connect(self.update_fps)

        self.camera_worker.finished.connect(self.camera_finished)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.camera_worker.start()

    def stop_camera(self):

        if self.camera_worker and self.camera_worker.isRunning():
            self.status_label.setText("Status: Stopping camera...")
            self.stop_button.setEnabled(False)
            self.camera_worker.stop()

    def display_frame(self, frame):

        if frame is None:
            return

        frame = np.ascontiguousarray(frame)

        if frame.ndim == 2:
            height, width = frame.shape
            image = QImage(
                frame.data,
                width,
                height,
                frame.strides[0],
                QImage.Format.Format_Grayscale8
            )

        elif frame.ndim == 3 and frame.shape[2] == 3:
            height, width, _ = frame.shape
            image = QImage(
                frame.data,
                width,
                height,
                frame.strides[0],
                QImage.Format.Format_BGR888
            )

        elif frame.ndim == 3 and frame.shape[2] == 4:
            height, width, _ = frame.shape
            image = QImage(
                frame.data,
                width,
                height,
                frame.strides[0],
                QImage.Format.Format_RGBA8888
            )

        else:
            return

        pixmap = QPixmap.fromImage(image.copy())

        self.preview.setPixmap(
            pixmap.scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def show_camera_error(self, error):
        self.status_label.setText("Status: Camera error")
        QMessageBox.critical(
            self,
            "Camera Error",
            str(error)
        )

    def camera_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.fps_label.setText("FPS: 0.0")

    def closeEvent(self, event):
        if self.camera_worker and self.camera_worker.isRunning():
            self.camera_worker.stop()

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
