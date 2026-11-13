import cv2
import numpy as np


class FramePreparer:
    """
    Turns whatever the camera delivers into the 8-bit, 3-channel BGR image the
    pipeline expects.

    Order matters for speed: downscale FIRST (cheap on the raw mono image),
    then reduce bit depth and expand to 3 channels on the small image. A 12 MP
    frame is never converted at full size.

    16-bit input: the bit depth of the data is guessed ONCE from the first
    frame (max > 4095 -> 16-bit, > 1023 -> 12-bit, > 255 -> 10-bit) unless
    `bit_depth` is given. This is a placeholder until the real camera's pixel
    format is known (see gentl_diagnose.py).
    """

    def __init__(self, max_side=1280, bit_depth=None):
        self.max_side = max_side
        self.bit_depth = bit_depth

    def prepare(self, frame):
        if frame.ndim == 3 and frame.shape[2] == 1:
            frame = frame[:, :, 0]

        h, w = frame.shape[:2]
        longest = max(h, w)
        if self.max_side and longest > self.max_side:
            scale = self.max_side / longest
            frame = cv2.resize(
                frame,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

        frame = self._to_uint8(frame)

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.shape[2] != 3:
            raise ValueError(f"Unsupported frame shape: {frame.shape}")

        return np.ascontiguousarray(frame)

    def _to_uint8(self, frame):
        if frame.dtype == np.uint8:
            return frame

        if frame.dtype == np.uint16:
            if self.bit_depth is None:
                peak = int(frame.max())
                if peak > 4095:
                    self.bit_depth = 16
                elif peak > 1023:
                    self.bit_depth = 12
                elif peak > 255:
                    self.bit_depth = 10
                else:
                    self.bit_depth = 8
            shift = max(0, self.bit_depth - 8)
            return (frame >> shift).clip(0, 255).astype(np.uint8)

        # float or other integer types: stretch to 0..255
        frame = frame.astype(np.float32)
        low, high = float(frame.min()), float(frame.max())
        if high <= low:
            return np.zeros(frame.shape, np.uint8)
        return ((frame - low) * (255.0 / (high - low))).astype(np.uint8)