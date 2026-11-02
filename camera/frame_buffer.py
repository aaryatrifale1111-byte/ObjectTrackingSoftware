
import threading
import time


class FrameBuffer:

    def __init__(self):

        self.lock = threading.Lock()

        self.latest_frame = None
        self.frame_number = 0
        self.timestamp = None

    def put(self, frame):

        with self.lock:

            self.latest_frame = frame.copy()

            self.frame_number += 1

            self.timestamp = time.perf_counter()

    def get(self):

        with self.lock:

            if self.latest_frame is None:
                return None

            return (
                self.latest_frame.copy(),
                self.frame_number,
                self.timestamp
            )

    def clear(self):

        with self.lock:

            self.latest_frame = None
            self.frame_number = 0
            self.timestamp = None
