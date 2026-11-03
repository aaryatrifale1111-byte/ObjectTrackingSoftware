
import time

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from transformers import Sam2VideoModel, Sam2VideoProcessor

from processing.device_manager import DeviceManager


class LiveTrackingPipeline:
    """
    Live YOLOv8 + SAM2.1 streaming pipeline.

    Input:
        One OpenCV BGR frame at a time.

    Output:
        SAM2 masks, persistent object IDs, bounding boxes and FPS.
    """

    def __init__(
        self,
        yolo_weights,
        sam2_checkpoint=None,
        sam2_config=None,
        device_mode="auto",
        detection_confidence=0.10,
        new_object_confidence=0.25,
        max_objects=6,
        match_iou=0.30,
        chain_iou=0.30,
        new_persistence=3,
        sam2_model_id="facebook/sam2.1-hiera-small",
        image_size=640,
    ):
        self.device_manager = DeviceManager(device_mode)
        self.device = self.device_manager.get_device()

        self.yolo_weights = yolo_weights

        # Transformers uses its own model format.
        # A Meta SAM2 .pt checkpoint cannot be loaded directly here.
        if sam2_checkpoint and str(sam2_checkpoint).lower().endswith(".pt"):
            raise ValueError(
                "The old Meta SAM2 .pt checkpoint is not directly "
                "compatible with the Transformers implementation. "
                "Use a Hugging Face SAM2 model ID or local model folder."
            )

        self.sam2_model_id = sam2_checkpoint or sam2_model_id

        self.detection_confidence = detection_confidence
        self.new_object_confidence = new_object_confidence
        self.max_objects = max_objects
        self.match_iou = match_iou
        self.chain_iou = chain_iou
        self.new_persistence = new_persistence
        self.image_size = image_size

        self.yolo = None
        self.sam2_model = None
        self.sam2_processor = None
        self.sam2_session = None

        self.tracking_started = False
        self.frame_index = 0

        self.object_ids = {}
        self.candidates = []
        self.next_object_id = 1

        self.current_fps = 0.0
        self._last_process_time = None

        print("Selected device:", self.device_manager.device_name)

    def load_models(self):
        """Load YOLOv8 and the Hugging Face SAM2.1 model."""

        print("Loading YOLOv8...")
        self.yolo = YOLO(self.yolo_weights)
        self.yolo.to(str(self.device))
        print("YOLOv8 loaded.")

        print("Loading SAM2.1:", self.sam2_model_id)

        self.sam2_processor = Sam2VideoProcessor.from_pretrained(
            self.sam2_model_id
        )

        self.sam2_model = Sam2VideoModel.from_pretrained(
            self.sam2_model_id
        )

        self.sam2_model.to(self.device)
        self.sam2_model.eval()

        self._create_streaming_session()

        print("SAM2.1 loaded.")
        print("Inference device:", self.device_manager.device_name)

    def _create_streaming_session(self):
        """Create an empty SAM2 session for incoming camera frames."""

        self.sam2_session = self.sam2_processor.init_video_session(
            inference_device=str(self.device),
            video_storage_device="cpu",
        )

    @staticmethod
    def box_iou(box_a, box_b):
        """Calculate intersection over union between two boxes."""

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    @staticmethod
    def mask_to_box(mask):
        """Convert a binary segmentation mask into an XYXY box."""

        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            return None

        return (
            float(xs.min()),
            float(ys.min()),
            float(xs.max() + 1),
            float(ys.max() + 1),
        )

    def _run_yolo(self, frame):
        """Run YOLO detection on the incoming frame."""

        results = self.yolo.predict(
            source=frame,
            conf=self.detection_confidence,
            imgsz=self.image_size,
            device=str(self.device),
            verbose=False,
        )

        detections = []

        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()

        for box, confidence in zip(xyxy, confidences):
            detections.append({
                "box": tuple(float(v) for v in box),
                "confidence": float(confidence),
            })

        return detections

    def _add_new_object(self, box, original_size):
        """Give a confirmed new object a permanent ID and SAM2 prompt."""

        if len(self.object_ids) >= self.max_objects:
            return None

        object_id = self.next_object_id
        self.next_object_id += 1

        x1, y1, x2, y2 = box

        self.sam2_processor.add_inputs_to_inference_session(
            inference_session=self.sam2_session,
            frame_idx=self.frame_index,
            obj_ids=object_id,
            input_boxes=[[
                [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                ]
            ]],
            original_size=original_size,
        )

        self.object_ids[object_id] = {
            "box": box,
            "last_seen": self.frame_index,
        }

        print(
            f"New object ID {object_id} "
            f"seeded at frame {self.frame_index}"
        )

        return object_id

    def _update_new_object_candidates(
        self,
        detections,
        original_size,
    ):
        """
        Confirm new objects over multiple consecutive frames.

        Existing SAM2 objects are matched first.
        Unmatched detections must persist before receiving a new ID.
        """

        used_object_ids = set()
        used_candidates = set()

        for detection in detections:
            box = detection["box"]
            confidence = detection["confidence"]

            if confidence < self.new_object_confidence:
                continue

            # Check whether this detection belongs to an existing ID.
            matched_existing = False

            for object_id, state in self.object_ids.items():
                if object_id in used_object_ids:
                    continue

                overlap = self.box_iou(box, state["box"])

                if overlap >= self.match_iou:
                    used_object_ids.add(object_id)
                    matched_existing = True
                    break

            if matched_existing:
                continue

            # Look for a matching candidate from the previous frame.
            best_index = None
            best_iou = 0.0

            for index, candidate in enumerate(self.candidates):
                if index in used_candidates:
                    continue

                if candidate["last_frame"] != self.frame_index - 1:
                    continue

                overlap = self.box_iou(box, candidate["box"])

                if overlap >= self.chain_iou and overlap > best_iou:
                    best_iou = overlap
                    best_index = index

            if best_index is None:
                self.candidates.append({
                    "box": box,
                    "hits": 1,
                    "last_frame": self.frame_index,
                })
                used_candidates.add(len(self.candidates) - 1)

            else:
                candidate = self.candidates[best_index]

                candidate["box"] = box
                candidate["hits"] += 1
                candidate["last_frame"] = self.frame_index

                used_candidates.add(best_index)

                if candidate["hits"] >= self.new_persistence:
                    self._add_new_object(box, original_size)

                    candidate["hits"] = 0

        # Remove candidates that were not observed in the latest frame.
        self.candidates = [
            candidate
            for candidate in self.candidates
            if candidate["last_frame"] == self.frame_index
            and candidate["hits"] > 0
        ]

    def process_frame(self, frame):
        """
        Process one live camera frame.

        Returns a dictionary containing:
            frame
            masks
            boxes
            detections
            fps
            frame_index
        """

        if self.yolo is None or self.sam2_model is None:
            raise RuntimeError("Load the models before processing frames.")

        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Invalid camera frame.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a 3-channel BGR image.")

        start_time = time.perf_counter()

        height, width = frame.shape[:2]

        # Convert OpenCV BGR into RGB for SAM2.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        inputs = self.sam2_processor(
            images=rgb_frame,
            return_tensors="pt",
        )

        pixel_values = inputs.pixel_values[0].to(self.device)

        original_size = tuple(
            int(value) for value in inputs.original_sizes[0]
        )

        # Detect objects and identify new-object candidates.
        with torch.inference_mode():
            detections = self._run_yolo(frame)

            self._update_new_object_candidates(
                detections,
                original_size,
            )

            # Process this frame using the persistent SAM2 session.
            outputs = self.sam2_model(
                inference_session=self.sam2_session,
                frame=pixel_values,
            )

        masks_by_id = {}
        boxes_by_id = {}

        if outputs.pred_masks is not None:
            processed_masks = self.sam2_processor.post_process_masks(
                [outputs.pred_masks],
                original_sizes=inputs.original_sizes,
                binarize=True,
            )[0]

            processed_masks = processed_masks.detach().cpu().numpy()

            output_ids = getattr(outputs, "object_ids", None)

            if output_ids is None:
                output_ids = self.sam2_session.obj_ids

            output_ids = [int(value) for value in output_ids]

            for index, object_id in enumerate(output_ids):
                if index >= len(processed_masks):
                    break

                mask = np.squeeze(processed_masks[index]).astype(
                    np.uint8
                )

                if mask.ndim != 2:
                    continue

                box = self.mask_to_box(mask)

                masks_by_id[object_id] = mask

                if box is not None:
                    boxes_by_id[object_id] = box

                    if object_id in self.object_ids:
                        self.object_ids[object_id]["box"] = box
                        self.object_ids[object_id]["last_seen"] = (
                            self.frame_index
                        )

        elapsed = time.perf_counter() - start_time

        instantaneous_fps = 1.0 / max(elapsed, 1e-9)

        if self._last_process_time is None:
            self.current_fps = instantaneous_fps
        else:
            self.current_fps = (
                0.8 * self.current_fps
                + 0.2 * instantaneous_fps
            )

        self._last_process_time = time.perf_counter()

        result = {
            "frame": frame,
            "masks": masks_by_id,
            "boxes": boxes_by_id,
            "detections": detections,
            "fps": self.current_fps,
            "frame_index": self.frame_index,
        }

        self.frame_index += 1
        self.tracking_started = len(self.object_ids) > 0

        return result

    def reset(self):
        """Reset tracking and start a fresh streaming session."""

        if self.sam2_session is not None:
            self.sam2_session.reset_inference_session()

        self._create_streaming_session()

        self.frame_index = 0
        self.object_ids.clear()
        self.candidates.clear()

        self.next_object_id = 1
        self.tracking_started = False
        self.current_fps = 0.0
        self._last_process_time = None

    def get_status(self):
        return {
            "models_loaded": (
                self.yolo is not None
                and self.sam2_model is not None
            ),
            "tracking_started": self.tracking_started,
            "device": self.device_manager.get_status(),
            "next_object_id": self.next_object_id,
            "tracked_objects": len(self.object_ids),
            "frame_index": self.frame_index,
            "fps": self.current_fps,
        }
