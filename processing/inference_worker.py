import time
import traceback

from PySide6.QtCore import QThread, Signal

from camera.frame_utils import FramePreparer


class InferenceWorker(QThread):
    """
    Always processes the NEWEST frame in the FrameBuffer. If inference is
    slower than the camera, intermediate frames are skipped (counted as
    dropped), so the displayed frame never falls further behind.

    Models are loaded here, in the worker thread, so the GUI stays responsive.
    With tracking_enabled=False no model is loaded and frames are just shown.
    """

    status_changed = Signal(str)
    error_occurred = Signal(str)
    models_ready = Signal(str)          # device name
    result_ready = Signal(object, dict)  # annotated BGR image, stats

    def __init__(
        self,
        frame_buffer,
        tracking_enabled,
        device_mode,
        yolo_interval,
        yolo_weights,
        sam2_checkpoint,
        sam2_config,
        infer_max_side=1280,
    ):
        super().__init__()

        self.frame_buffer = frame_buffer
        self.tracking_enabled = tracking_enabled
        self.device_mode = device_mode
        self.yolo_interval = yolo_interval
        self.yolo_weights = yolo_weights
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self.infer_max_side = infer_max_side

        self._stop_requested = False
        self._reset = False

    def request_stop(self):
        self._stop_requested = True

    def request_reset(self):
        """Clear all tracked objects and restart IDs at 1 (done in the worker thread)."""
        self._reset = True

    # ------------------------------------------------------------------
    def run(self):
        pipeline = None
        draw_result = None

        try:
            if self.tracking_enabled:
                self.status_changed.emit("Loading YOLO + SAM2 models...")

                from processing.live_pipeline import (
                    LiveTrackingPipeline,
                    draw_result,
                )

                pipeline = LiveTrackingPipeline(
                    yolo_weights=self.yolo_weights,
                    sam2_checkpoint=self.sam2_checkpoint,
                    sam2_config=self.sam2_config,
                    device_mode=self.device_mode,
                    yolo_interval=self.yolo_interval,
                    infer_max_side=self.infer_max_side,
                )
                pipeline.load_models()
                device_name = pipeline.device_manager.device_name
            else:
                device_name = "tracking off"

            if self._stop_requested:
                return

            self.models_ready.emit(device_name)
            self._loop(pipeline, draw_result, device_name)

        except Exception as error:
            traceback.print_exc()
            self.error_occurred.emit(f"{type(error).__name__}: {error}")

        finally:
            self.status_changed.emit("Processing stopped.")

    def _loop(self, pipeline, draw_result, device_name):
        preparer = FramePreparer(max_side=self.infer_max_side)

        last_seq = 0
        dropped_total = 0
        window_frames = 0
        window_start = time.perf_counter()
        processed_fps = 0.0

        while not self._stop_requested:
            item = self.frame_buffer.wait_newer(last_seq, timeout=0.2)
            if item is None:
                continue

            frame, seq, captured_at = item
            if last_seq:
                dropped_total += seq - last_seq - 1
            last_seq = seq

            if self._reset and pipeline is not None:
                pipeline.reset()
                self._reset = False

            prepared = preparer.prepare(frame)

            if pipeline is not None:
                result = pipeline.process_frame(prepared, capture_ts=captured_at)
                latency_ms = (time.perf_counter() - captured_at) * 1000.0
                image = draw_result(result, [
                    f"{device_name}   objects: {len(result['active_ids'])}",
                    f"latency {latency_ms:.0f} ms",
                ])
                stats = {
                    "yolo_ms": result["yolo_ms"],
                    "sam_ms": result["sam_ms"],
                    "total_ms": result["total_ms"],
                    "active_ids": result["active_ids"],
                    "lost_ids": result["lost_ids"],
                }
            else:
                image = prepared
                stats = {
                    "yolo_ms": 0.0, "sam_ms": 0.0, "total_ms": 0.0,
                    "active_ids": [], "lost_ids": [],
                }

            window_frames += 1
            elapsed = time.perf_counter() - window_start
            if elapsed >= 1.0:
                processed_fps = window_frames / elapsed
                window_frames = 0
                window_start = time.perf_counter()

            # End-to-end age of this frame when it leaves the worker
            # (capture -> prepared -> YOLO/SAM2 -> drawn).
            stats["latency_ms"] = (time.perf_counter() - captured_at) * 1000.0
            stats["processed_fps"] = processed_fps
            stats["dropped"] = dropped_total
            stats["device"] = device_name

            self.result_ready.emit(image, stats)