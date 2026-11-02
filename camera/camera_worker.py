
import time

from PySide6.QtCore import QThread, Signal

from camera.gentl_camera import GenTLCamera


class CameraWorker(QThread):
    frame_ready = Signal(object)
    status_changed = Signal(str)
    error_occurred = Signal(str)
    fps_updated = Signal(float)

    def __init__(self, gentl_file):
        super().__init__()
        self.gentl_file = gentl_file
        self.running = False
        self.camera = None

    def run(self):
        self.running = True
        self.camera = GenTLCamera(self.gentl_file)

        try:
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
                    fps = frame_count / elapsed
                    self.fps_updated.emit(fps)

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
