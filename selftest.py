"""
Headless self-test, used to verify a packaged build on a clean PC:

    SpacecraftTracker.exe --selftest
    SpacecraftTracker.exe --selftest C:\\path\\video.mp4

It loads YOLO + SAM2 from the bundled files, then processes a few frames.
With a video (or the bundled test video) it expects an object to get an ID.
Prints SELFTEST: PASS / FAIL and returns the exit code.
"""

import os
import time
import traceback

import numpy as np


def run(video_path=None, frames=6):
    from app_paths import (
        YOLO_WEIGHTS, SAM2_CHECKPOINT, SAM2_CONFIG, resource_path,
    )

    if video_path is None:
        bundled = resource_path(os.path.join("test_video", "video1.mp4"))
        if os.path.exists(bundled):
            video_path = bundled

    try:
        import cv2
        from processing.live_pipeline import LiveTrackingPipeline

        print("SELFTEST: loading models...")
        t0 = time.perf_counter()
        pipe = LiveTrackingPipeline(
            YOLO_WEIGHTS, SAM2_CHECKPOINT, SAM2_CONFIG,
            device_mode="auto", yolo_interval=1,
        )
        pipe.load_models()
        print(
            f"SELFTEST: models loaded in {time.perf_counter() - t0:.1f} s on "
            f"{pipe.device_manager.device_name}, SAM2 precision {pipe.tracker.precision}"
        )

        if not video_path:
            pipe.process_frame(np.zeros((480, 640, 3), np.uint8))
            print("SELFTEST: no video available; ran one blank frame")
            print("SELFTEST: PASS (models load and run; tracking not exercised)")
            return 0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        result = None
        for i in range(frames):
            ok, frame = cap.read()
            if not ok:
                break
            result = pipe.process_frame(frame)
            print(
                f"SELFTEST: frame {i} | {result['total_ms']:.0f} ms | "
                f"active IDs {result['active_ids']} | "
                f"mask px { {k: int(m.sum()) for k, m in result['masks'].items()} }"
            )

        if result is None or not result["active_ids"]:
            print("SELFTEST: FAIL (no object was tracked; need a video with a spacecraft)")
            return 1

        print("SELFTEST: PASS")
        return 0

    except Exception:
        traceback.print_exc()
        print("SELFTEST: FAIL")
        return 1
