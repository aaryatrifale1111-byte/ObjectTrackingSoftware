import time

import cv2


class VideoFileCamera:
    """
    Plays a saved video as if it were a live camera: frames are released at
    the file's own frame rate, so a slow consumer really does fall behind and
    has to drop frames. Same interface as WebcamCamera / GenTLCamera:
    connect(), start(), get_frame(), stop().
    """

    def __init__(self, path, loop=True):
        self.path = path
        self.loop = loop
        self.cap = None
        self.period = 1.0 / 30.0
        self._next_time = None

    def connect(self):
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self.path}")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 1:
            self.period = 1.0 / fps

    def start(self):
        self._next_time = time.perf_counter()

    def get_frame(self):
        if self.cap is None:
            return None

        delay = self._next_time - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

        ok, frame = self.cap.read()

        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()

        if not ok:
            raise RuntimeError("End of video reached.")

        self._next_time += self.period
        # If we fell behind (e.g. a debugger pause), do not burst to catch up.
        self._next_time = max(self._next_time, time.perf_counter() - self.period)

        return frame

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None