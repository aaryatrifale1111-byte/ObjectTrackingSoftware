import os
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
    QGridLayout,
    QMessageBox,
    QComboBox,
    QCheckBox,
    QSpinBox,
    QGroupBox,
    QSizePolicy,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QTimer

from app_paths import YOLO_WEIGHTS, SAM2_CHECKPOINT, SAM2_CONFIG
from camera.frame_buffer import FrameBuffer
from camera.camera_worker import CameraWorker
from processing.inference_worker import InferenceWorker


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLO + SAM2 Object Detection and Tracking")
        self.resize(1200, 850)

        self.gentl_file = None
        self.video_file = None

        self.frame_buffer = None
        self.camera_worker = None
        self.inference_worker = None
        self._session = {}
        self._stopping = False
        self._error_active = False

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # ---- Source -------------------------------------------------
        layout.addWidget(QLabel("Select Camera Source:"))

        self.source_selector = QComboBox()
        self.source_selector.addItem("Laptop Webcam", "webcam")
        self.source_selector.addItem("Video file (test)", "video")
        self.source_selector.addItem("CMV-12K Industrial Camera (GenTL)", "gentl")
        self.source_selector.currentIndexChanged.connect(self.refresh_controls)
        layout.addWidget(self.source_selector)

        gentl_row = QHBoxLayout()
        self.select_gentl_button = QPushButton("Select GenTL Producer")
        self.select_gentl_button.clicked.connect(self.select_gentl_file)
        self.gentl_label = QLabel("GenTL producer: Not selected")
        gentl_row.addWidget(self.select_gentl_button)
        gentl_row.addWidget(self.gentl_label, 1)
        layout.addLayout(gentl_row)

        video_row = QHBoxLayout()
        self.select_video_button = QPushButton("Select Video File")
        self.select_video_button.clicked.connect(self.select_video_file)
        self.video_label = QLabel("Video file: Not selected")
        video_row.addWidget(self.select_video_button)
        video_row.addWidget(self.video_label, 1)
        layout.addLayout(video_row)

        # ---- Processing options ------------------------------------
        options = QHBoxLayout()

        options.addWidget(QLabel("Device:"))
        self.device_selector = QComboBox()
        self.device_selector.addItem("Auto", "auto")
        self.device_selector.addItem("CPU", "cpu")
        self.device_selector.addItem("GPU (CUDA)", "gpu")
        options.addWidget(self.device_selector)

        options.addWidget(QLabel("YOLO every N frames:"))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 30)
        self.interval_spin.setValue(5)
        options.addWidget(self.interval_spin)

        self.tracking_checkbox = QCheckBox("YOLO + SAM2 tracking")
        self.tracking_checkbox.setChecked(True)
        options.addWidget(self.tracking_checkbox)

        options.addStretch(1)
        layout.addLayout(options)

        self.cuda_label = QLabel("PyTorch / CUDA: checking...")
        layout.addWidget(self.cuda_label)

        # ---- Preview -----------------------------------------------
        self.preview = QLabel("Live view will appear here")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(800, 450)
        self.preview.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )
        self.preview.setStyleSheet("background-color: #202020; color: white;")
        layout.addWidget(self.preview, 1)

        # ---- Buttons -----------------------------------------------
        buttons = QHBoxLayout()

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_processing)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_processing)

        self.reset_button = QPushButton("Reset Tracking")
        self.reset_button.clicked.connect(self.reset_tracking)

        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.reset_button)
        layout.addLayout(buttons)

        # ---- Status and statistics ---------------------------------
        self.status_label = QLabel("Status: Not connected")
        layout.addWidget(self.status_label)

        box = QGroupBox("Live statistics")
        grid = QGridLayout(box)

        self.stat_labels = {}
        fields = [
            ("capture_fps", "Camera FPS"),
            ("processed_fps", "Processed FPS"),
            ("latency", "Frame latency (ms)"),
            ("dropped", "Dropped frames"),
            ("yolo", "YOLO (ms)"),
            ("sam", "SAM2 (ms)"),
            ("total", "Pipeline total (ms)"),
            ("device", "Device"),
            ("objects", "Active objects"),
        ]
        for i, (key, title) in enumerate(fields):
            row, col = divmod(i, 3)
            grid.addWidget(QLabel(title + ":"), row, col * 2)
            value = QLabel("-")
            value.setStyleSheet("font-weight: bold;")
            grid.addWidget(value, row, col * 2 + 1)
            self.stat_labels[key] = value

        layout.addWidget(box)

        self.refresh_controls()
        QTimer.singleShot(300, self.update_cuda_info)

    # ------------------------------------------------------------------
    # Source selection
    # ------------------------------------------------------------------
    def select_gentl_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select GenTL Producer",
            "",
            "GenTL files (*.cti *.dll);;All files (*)",
        )
        if path:
            self.gentl_file = path
            self.gentl_label.setText(f"GenTL producer: {path}")

    def select_video_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)",
        )
        if path:
            self.video_file = path
            self.video_label.setText(f"Video file: {path}")

    def update_cuda_info(self):
        try:
            import torch

            if torch.cuda.is_available():
                text = f"PyTorch {torch.__version__} | CUDA available: {torch.cuda.get_device_name(0)}"
            elif torch.version.cuda is None:
                text = f"PyTorch {torch.__version__} | CPU-only build (no CUDA support installed)"
            else:
                text = (
                    f"PyTorch {torch.__version__} | CUDA build, but no usable GPU "
                    "(check the NVIDIA driver)"
                )
        except Exception as error:
            text = f"PyTorch not available: {error}"

        self.cuda_label.setText(text)

    def is_running(self):
        return any(
            worker is not None and worker.isRunning()
            for worker in (self.camera_worker, self.inference_worker)
        )

    def refresh_controls(self):
        running = self.is_running()
        source = self.source_selector.currentData()

        self.source_selector.setEnabled(not running)
        self.select_gentl_button.setEnabled(not running and source == "gentl")
        self.select_video_button.setEnabled(not running and source == "video")
        self.device_selector.setEnabled(not running)
        self.interval_spin.setEnabled(not running)
        self.tracking_checkbox.setEnabled(not running)

        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running and not self._stopping)
        self.reset_button.setEnabled(running)

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------
    def start_processing(self):
        if self.is_running():
            return

        source = self.source_selector.currentData()
        tracking = self.tracking_checkbox.isChecked()

        if source == "gentl" and not self.gentl_file:
            QMessageBox.warning(
                self, "GenTL File Required",
                "Please select the GenTL producer file first.",
            )
            return

        if source == "video" and not self.video_file:
            QMessageBox.warning(
                self, "Video File Required", "Please select a video file first."
            )
            return

        if tracking:
            for path in (YOLO_WEIGHTS, SAM2_CHECKPOINT):
                if not os.path.exists(path):
                    QMessageBox.critical(
                        self, "Model file missing", f"Not found:\n{path}"
                    )
                    return

        self._stopping = False
        self._error_active = False
        self._session = {"source": source}

        self.frame_buffer = FrameBuffer()
        self.camera_worker = None

        self.inference_worker = InferenceWorker(
            frame_buffer=self.frame_buffer,
            tracking_enabled=tracking,
            device_mode=self.device_selector.currentData(),
            yolo_interval=self.interval_spin.value(),
            yolo_weights=YOLO_WEIGHTS,
            sam2_checkpoint=SAM2_CHECKPOINT,
            sam2_config=SAM2_CONFIG,
        )
        self.inference_worker.status_changed.connect(self.update_status)
        self.inference_worker.error_occurred.connect(self.show_error)
        self.inference_worker.models_ready.connect(self.on_models_ready)
        self.inference_worker.result_ready.connect(self.display_result)
        self.inference_worker.finished.connect(self.on_worker_finished)

        for label in self.stat_labels.values():
            label.setText("-")

        self.preview.setText(
            "Loading models..." if tracking else "Starting..."
        )
        self.inference_worker.start()
        self.refresh_controls()

    def on_models_ready(self, device_name):
        """Models are loaded: only now start the camera, so no stale frames pile up."""
        if self._stopping:
            return

        self.stat_labels["device"].setText(device_name)
        self.preview.setText("Connecting to camera...")

        self.camera_worker = CameraWorker(
            camera_type=self._session["source"],
            frame_buffer=self.frame_buffer,
            gentl_file=self.gentl_file,
            video_path=self.video_file,
        )
        self.camera_worker.status_changed.connect(self.update_status)
        self.camera_worker.error_occurred.connect(self.show_error)
        self.camera_worker.fps_updated.connect(self.update_capture_fps)
        self.camera_worker.finished.connect(self.on_worker_finished)
        self.camera_worker.start()
        self.refresh_controls()

    def stop_processing(self):
        if not self.is_running():
            return

        self._stopping = True
        self.status_label.setText("Status: Stopping...")

        for worker in (self.camera_worker, self.inference_worker):
            if worker is not None and worker.isRunning():
                worker.request_stop()

        self.refresh_controls()

    def reset_tracking(self):
        if self.inference_worker is not None and self.inference_worker.isRunning():
            self.inference_worker.request_reset()

    def on_worker_finished(self):
        if self.is_running():
            return

        self._stopping = False
        self.preview.setPixmap(QPixmap())
        self.preview.setText("Live view will appear here")
        self.stat_labels["capture_fps"].setText("-")
        self.refresh_controls()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def display_result(self, frame, stats):
        frame = np.ascontiguousarray(frame)

        if frame.ndim != 3 or frame.shape[2] != 3:
            return

        height, width, _ = frame.shape
        image = QImage(
            frame.data, width, height, frame.strides[0],
            QImage.Format.Format_BGR888,
        )
        pixmap = QPixmap.fromImage(image.copy())

        self.preview.setPixmap(
            pixmap.scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )

        labels = self.stat_labels
        labels["processed_fps"].setText(f"{stats['processed_fps']:.1f}")
        labels["latency"].setText(f"{stats['latency_ms']:.0f}")
        labels["dropped"].setText(str(stats["dropped"]))
        labels["yolo"].setText(f"{stats['yolo_ms']:.0f}")
        labels["sam"].setText(f"{stats['sam_ms']:.0f}")
        labels["total"].setText(f"{stats['total_ms']:.0f}")
        labels["device"].setText(stats["device"])

        ids = stats["active_ids"]
        text = f"{len(ids)}"
        if ids:
            text += "  (IDs " + ", ".join(str(i) for i in ids) + ")"
        if stats["lost_ids"]:
            text += "  lost: " + ", ".join(str(i) for i in stats["lost_ids"])
        labels["objects"].setText(text)

    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    def update_capture_fps(self, fps):
        self.stat_labels["capture_fps"].setText(f"{fps:.1f}")

    def show_error(self, error):
        if self._error_active:
            return
        self._error_active = True

        self.status_label.setText("Status: Error")
        self.stop_processing()

        QMessageBox.critical(self, "Error", str(error))

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        for worker in (self.camera_worker, self.inference_worker):
            if worker is not None and worker.isRunning():
                worker.request_stop()

        for worker in (self.camera_worker, self.inference_worker):
            if worker is not None:
                worker.wait(15000)

        event.accept()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()  # needed for PyInstaller on Windows

    if "--selftest" in sys.argv:
        # Headless check used to verify a packaged build: loads the models and
        # runs a few frames, then exits. Optional argument: a video file.
        import selftest

        index = sys.argv.index("--selftest")
        video = sys.argv[index + 1] if len(sys.argv) > index + 1 else None
        sys.exit(selftest.run(video))

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())