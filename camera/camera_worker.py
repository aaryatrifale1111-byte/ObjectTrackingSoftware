import time

from PySide6.QtCore import QThread, Signal


class CameraWorker(QThread):
    """
    Capture thread. It only reads frames and publishes them to the shared
    FrameBuffer; it never waits for YOLO or SAM2, so camera acquisition cannot
    be slowed down by inference.

    Camera sources: "webcam", "gentl", "video" (saved file played in real time).
    """

    status_changed = Signal(str)
    error_occurred = Signal(str)
    fps_updated = Signal(float)      # capture FPS

    def __init__(self, camera_type, frame_buffer, gentl_file=None, video_path=None):
        super().__init__()

        self.camera_type = camera_type
        self.frame_buffer = frame_buffer
        self.gentl_file = gentl_file
        self.video_path = video_path

        self.running = False
        self.camera = None

    def _create_camera(self):
        # Imported lazily so a missing optional dependency (e.g. harvesters)
        # only matters when that source is actually selected.
        if self.camera_type == "webcam":
            from camera.webcam_camera import WebcamCamera
            return WebcamCamera()

        if self.camera_type == "video":
            if not self.video_path:
                raise RuntimeError("No video file selected.")
            from camera.video_camera import VideoFileCamera
            return VideoFileCamera(self.video_path)

        if self.camera_type == "gentl":
            if not self.gentl_file:
                raise RuntimeError("No GenTL producer file selected.")
            from camera.gentl_camera import GenTLCamera
            return GenTLCamera(self.gentl_file)

        raise RuntimeError("Unknown camera source.")

    def run(self):
        self.running = True

        try:
            self.camera = self._create_camera()

            self.status_changed.emit("Connecting to camera...")
            self.camera.connect()
            self.camera.start()
            self.status_changed.emit("Camera connected. Receiving frames...")

            frame_count = 0
            fps_start = time.perf_counter()

            while self.running:
                frame = self.camera.get_frame()

                if frame is None:
                    time.sleep(0.001)
                    continue

                self.frame_buffer.put(frame)

                frame_count += 1
                elapsed = time.perf_counter() - fps_start

                if elapsed >= 1.0:
                    self.fps_updated.emit(frame_count / elapsed)
                    frame_count = 0
                    fps_start = time.perf_counter()

        except Exception as error:
            self.error_occurred.emit(str(error))

        finally:
            self.running = False

            if self.camera is not None:
                try:
                    self.camera.stop()
                except Exception:
                    pass

            self.status_changed.emit("Camera stopped.")

    def request_stop(self):
        """Non-blocking: ask the capture loop to end."""
        self.running = False

    def stop(self):
        """Blocking variant, kept for compatibility."""
        self.running = False
        self.wait(3000)