"""
Live YOLOv8 + official Meta SAM2.1 pipeline.

Per frame:
  1. SAM2 tracks every existing object (persistent memory, no window replay).
  2. YOLO runs every `yolo_interval` frames, and also immediately when an
     object's mask is empty or a new-object candidate is pending.
  3. Detections are matched to existing objects by mask pixels; unmatched
     detections must persist for NEW_PERSIST consecutive YOLO runs before
     they get a new, never-recycled ID and are seeded into SAM2.

process_frame() takes one BGR frame, so it can be driven from any thread.
"""

import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from processing.device_manager import DeviceManager
from processing.id_manager import IdManager, mask_to_box
from processing.sam2_stream import SAM2StreamTracker

_PALETTE = [
    (66, 135, 245), (60, 200, 90), (245, 170, 50),
    (200, 80, 220), (60, 210, 230), (240, 90, 90),
]


class LiveTrackingPipeline:

    def __init__(
        self,
        yolo_weights,
        sam2_checkpoint,
        sam2_config,
        device_mode="auto",
        yolo_interval=5,
        detect_conf=0.10,
        yolo_imgsz=640,
        new_conf=0.25,
        new_persist=3,
        match_iou=0.30,
        chain_iou=0.30,
        max_objects=6,
        exclude_zone=(0.75, 0.88, 1.0, 1.0),
        infer_max_side=1280,
        vos_optimized=False,
        precision="auto",
    ):
        self.device_manager = DeviceManager(device_mode)
        self.device = self.device_manager.get_device()

        self.yolo_weights = yolo_weights
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self.vos_optimized = vos_optimized
        self.precision = precision

        self.yolo_interval = max(1, int(yolo_interval))
        self.detect_conf = detect_conf
        self.yolo_imgsz = yolo_imgsz
        self.exclude_zone = exclude_zone
        self.infer_max_side = infer_max_side

        self.ids = IdManager(
            match_iou=match_iou,
            chain_iou=chain_iou,
            new_conf=new_conf,
            new_persist=new_persist,
            max_objects=max_objects,
        )

        self.yolo = None
        self.tracker = None
        self.frame_count = 0

        print("Selected device:", self.device_manager.device_name)

    # ------------------------------------------------------------------
    def load_models(self):
        print("Loading YOLOv8...")
        self.yolo = YOLO(self.yolo_weights)
        print("Loading official SAM2.1...")
        self.tracker = SAM2StreamTracker(
            self.sam2_checkpoint,
            self.sam2_config,
            self.device,
            vos_optimized=self.vos_optimized,
            precision=self.precision,
        )
        print(
            "Models loaded on", self.device_manager.device_name,
            "| SAM2 precision:", self.tracker.precision,
        )

    def reset(self):
        """Start a fresh tracking session (IDs restart at 1)."""
        if self.tracker is not None:
            self.tracker.reset()
        self.ids.reset()
        self.frame_count = 0

    # ------------------------------------------------------------------
    def _downscale(self, frame):
        h, w = frame.shape[:2]
        longest = max(h, w)
        if self.infer_max_side and longest > self.infer_max_side:
            scale = self.infer_max_side / longest
            frame = cv2.resize(
                frame,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return frame

    def _run_yolo(self, frame):
        results = self.yolo.predict(
            source=frame,
            conf=self.detect_conf,
            imgsz=self.yolo_imgsz,
            device=str(self.device),
            verbose=False,
        )

        detections = []
        if not results or results[0].boxes is None:
            return detections

        h, w = frame.shape[:2]
        zx1, zy1, zx2, zy2 = self.exclude_zone
        xyxy = results[0].boxes.xyxy.detach().cpu().numpy()
        conf = results[0].boxes.conf.detach().cpu().numpy()

        for box, c in zip(xyxy, conf):
            cx = (box[0] + box[2]) / 2 / w
            cy = (box[1] + box[3]) / 2 / h
            if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                continue  # excluded zone (e.g. overlay / watermark)

            detections.append({
                "box": tuple(float(v) for v in box),
                "confidence": float(c),
            })

        return detections

    # ------------------------------------------------------------------
    def process_frame(self, frame, capture_ts=None):
        if self.yolo is None or self.tracker is None:
            raise RuntimeError("Call load_models() first.")
        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
        ):
            raise ValueError("Expected a 3-channel BGR image.")

        t_start = time.perf_counter()
        frame = self._downscale(frame)

        # 1) SAM2 updates all existing objects.
        t0 = time.perf_counter()
        masks = self.tracker.track(frame)
        sam_ms = (time.perf_counter() - t0) * 1000.0

        self.ids.note_masks(masks)

        # 2) Periodic / triggered YOLO.
        run_yolo = (
            self.frame_count % self.yolo_interval == 0
            or self.ids.needs_detection()
        )

        detections, new_ids, yolo_ms = [], [], 0.0

        if run_yolo:
            t0 = time.perf_counter()
            detections = self._run_yolo(frame)
            yolo_ms = (time.perf_counter() - t0) * 1000.0

            new_objects = self.ids.update(detections, masks)

            # 3) Add confirmed new objects without replaying anything.
            if new_objects:
                t0 = time.perf_counter()
                masks.update(self.tracker.seed(new_objects))
                sam_ms += (time.perf_counter() - t0) * 1000.0
                new_ids = [obj_id for obj_id, _ in new_objects]

        boxes = {}
        for obj_id, mask in masks.items():
            box = mask_to_box(mask)
            if box is not None:
                boxes[obj_id] = box

        result = {
            "frame": frame,                 # (possibly downscaled) frame the masks align to
            "masks": masks,
            "boxes": boxes,
            "detections": detections,
            "frame_index": self.frame_count,
            "capture_ts": capture_ts,
            "yolo_ran": run_yolo,
            "new_ids": new_ids,
            "active_ids": list(self.ids.active),
            "lost_ids": [i for i, n in self.ids.lost.items() if n > 0],
            "yolo_ms": yolo_ms,
            "sam_ms": sam_ms,
            "total_ms": (time.perf_counter() - t_start) * 1000.0,
        }

        self.frame_count += 1
        return result

    # ------------------------------------------------------------------
    def get_status(self):
        return {
            "models_loaded": self.yolo is not None and self.tracker is not None,
            "device": self.device_manager.get_status(),
            "active_ids": list(self.ids.active),
            "next_id": self.ids.next_id,
            "frames_processed": self.frame_count,
        }


def draw_result(result, hud_lines=()):
    """Return a new BGR image with masks, boxes, IDs and HUD text drawn."""
    out = result["frame"].copy()

    for obj_id, mask in result["masks"].items():
        color = _PALETTE[(obj_id - 1) % len(_PALETTE)]
        m = mask.astype(bool)
        if m.any():
            out[m] = (0.5 * out[m] + 0.5 * np.array(color)).astype(np.uint8)

    for obj_id, (x1, y1, x2, y2) in result["boxes"].items():
        color = _PALETTE[(obj_id - 1) % len(_PALETTE)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, color, 2)
        cv2.putText(
            out, f"ID {obj_id}", (p1[0], max(18, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
        )

    for i, line in enumerate(hud_lines):
        cv2.putText(
            out, line, (10, 24 + 22 * i),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

    return out