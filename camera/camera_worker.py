
import time

from PySide6.QtCore import QThread, Signal

from camera.gentl_camera import GenTLCamera
from camera.webcam_camera import WebcamCamera


class CameraWorker(QThread):

    frame_ready = Signal(object)
    status_changed = Signal(str)
    error_occurred = Signal(str)
    fps_updated = Signal(float)

    def __init__(self, camera_type, gentl_file=None):
        super().__init__()

        self.camera_type = camera_type
        self.gentl_file = gentl_file
        self.running = False
        self.camera = None

    def run(self):

        self.running = True

        try:
            # Select camera source
            if self.camera_type == "webcam":
                self.camera = WebcamCamera()

            elif self.camera_type == "gentl":
                if not self.gentl_file:
                    raise RuntimeError("No GenTL producer file selected.")

                self.camera = GenTLCamera(self.gentl_file)

            else:
                raise RuntimeError("Unknown camera source.")

            self.status_changed.emit("Connecting to camera...")

            self.camera.connect()
            self.camera.start()

            self.status_changed.emit("Camera connected. Receiving frames...")

            frame_count = 0
            fps_start = time.perf_counter()

            while self.running:

                frame = self.camera.get_frame()

                if frame is None:
                    continue

                self.frame_ready.emit(frame)

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

    def stop(self):
        self.running = False
        self.wait(3000)
