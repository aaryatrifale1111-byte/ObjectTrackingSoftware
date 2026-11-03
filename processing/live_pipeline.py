
import cv2
import numpy as np
import torch

from ultralytics import YOLO
from processing.device_manager import DeviceManager


class LiveTrackingPipeline:
    """
    Live YOLOv8 + SAM2 tracking pipeline.

    Supports automatic CPU/GPU selection.
    """

    def __init__(
        self,
        yolo_weights,
        sam2_checkpoint,
        sam2_config,
        device_mode="auto",
        detection_confidence=0.10,
        new_object_confidence=0.25,
        max_objects=6,
        match_iou=0.30,
        chain_iou=0.30,
        new_persistence=3,
    ):

        # Hardware selection
        self.device_manager = DeviceManager(device_mode)
        self.device = self.device_manager.get_device()

        # Model paths
        self.yolo_weights = yolo_weights
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config

        # Detection and tracking settings
        self.detection_confidence = detection_confidence
        self.new_object_confidence = new_object_confidence
        self.max_objects = max_objects
        self.match_iou = match_iou
        self.chain_iou = chain_iou
        self.new_persistence = new_persistence

        # Runtime state
        self.yolo = None
        self.sam2_predictor = None
        self.tracking_started = False

        self.object_ids = {}
        self.next_object_id = 1

        print(
            "Selected device:",
            self.device_manager.device_name
        )

    def load_models(self):
        """Load YOLOv8 and SAM2 on the selected device."""

        print("Loading YOLOv8 model...")

        self.yolo = YOLO(self.yolo_weights)
        self.yolo.to(str(self.device))

        print("YOLOv8 loaded successfully.")

        print("Loading SAM2 model...")

        from sam2.build_sam import build_sam2_video_predictor

        self.sam2_predictor = build_sam2_video_predictor(
            self.sam2_config,
            self.sam2_checkpoint,
            device=str(self.device),
        )

        self.sam2_predictor.eval()

        print("SAM2 loaded successfully.")

        print(
            "Inference device:",
            self.device_manager.device_name
        )

    def get_status(self):
        """Return current pipeline and hardware status."""

        return {
            "models_loaded": (
                self.yolo is not None
                and self.sam2_predictor is not None
            ),
            "tracking_started": self.tracking_started,
            "device": self.device_manager.get_status(),
            "next_object_id": self.next_object_id,
            "tracked_objects": len(self.object_ids),
        }
