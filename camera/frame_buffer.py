import threading
import time


class FrameBuffer:
    """
    Single-slot "latest frame" buffer.

    The camera thread calls put() as fast as frames arrive; every new frame
    overwrites the previous one. The inference thread calls wait_newer() and
    always receives the NEWEST frame, so stale frames are dropped instead of
    queued. There is no queue, so nothing can accumulate.

    Frame contract:
      * put() copies the frame once (set copy_on_put=False only if the
        producer guarantees it never reuses or mutates the array).
      * get()/wait_newer() return the stored array WITHOUT copying.
        Consumers must treat it as read-only (draw on a copy).
    """

    def __init__(self, copy_on_put=True):
        self._cond = threading.Condition()
        self._copy_on_put = copy_on_put

        self.latest_frame = None
        self.frame_number = 0       # sequence number of the latest frame
        self.timestamp = None       # perf_counter() time the frame was captured
        self._closed = False

    def put(self, frame, timestamp=None):
        """Store a new frame. Returns its sequence number."""
        if self._copy_on_put:
            frame = frame.copy()

        if timestamp is None:
            timestamp = time.perf_counter()

        with self._cond:
            self.latest_frame = frame
            self.frame_number += 1
            self.timestamp = timestamp
            self._cond.notify_all()
            return self.frame_number

    def get(self):
        """Non-blocking. Returns (frame, frame_number, timestamp) or None."""
        with self._cond:
            if self.latest_frame is None:
                return None
            return self.latest_frame, self.frame_number, self.timestamp

    def wait_newer(self, last_seq, timeout=None):
        """
        Block until a frame newer than last_seq exists, then return
        (frame, frame_number, timestamp).

        Returns None on timeout, or if the buffer was closed and there is no
        newer frame. frame_number - last_seq - 1 is the number of frames
        dropped since the last one the caller processed.
        """
        deadline = None if timeout is None else time.perf_counter() + timeout

        with self._cond:
            while self.frame_number <= last_seq:
                if self._closed:
                    return None

                remaining = None
                if deadline is not None:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        return None

                self._cond.wait(remaining)

            return self.latest_frame, self.frame_number, self.timestamp

    def close(self):
        """Wake any waiting consumer; no more frames will arrive."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def clear(self):
        with self._cond:
            self.latest_frame = None
            self.frame_number = 0
            self.timestamp = None
            self._closed = False
            self._cond.notify_all()