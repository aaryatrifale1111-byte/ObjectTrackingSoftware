
import cv2
import torch
import numpy as np

from pathlib import Path
from ultralytics import YOLO
from sam2.build_sam import build_sam2_video_predictor


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO = "video1.mp4"

YOLO_WEIGHTS = "models/spark_yolov8n_best.pt"
SAM2_CHECKPOINT = "models/sam2.1_hiera_small.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

OUTPUT = "output/yolo_sam2_full_video.mp4"
FRAME_DIR = Path("output/sam2_current_window")

WINDOW_SIZE = 60

DETECT_CONF = 0.10
YOLO_IMGSZ = 640

NEW_CONF = 0.25
NEW_PERSIST = 3

MATCH_IOU = 0.30
CHAIN_IOU = 0.30

MAX_OBJECTS = 6

# Normalised bottom-right exclusion zone
EXCLUDE_ZONE = (0.75, 0.88, 1.0, 1.0)


# ============================================================
# GEOMETRY AND MASK FUNCTIONS
# ============================================================

def box_iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = (
        max(0, ix2 - ix1) *
        max(0, iy2 - iy1)
    )

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def mask_box_iou(mask, box):

    h, w = mask.shape

    x1, y1, x2, y2 = box.astype(int)

    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    box_mask = np.zeros((h, w), dtype=bool)
    box_mask[y1:y2, x1:x2] = True

    intersection = np.logical_and(mask, box_mask).sum()
    union = np.logical_or(mask, box_mask).sum()

    return intersection / union if union > 0 else 0.0


def in_exclusion_zone(box, width, height):

    x1, y1, x2, y2 = box

    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height

    zx1, zy1, zx2, zy2 = EXCLUDE_ZONE

    return (
        zx1 <= cx <= zx2
        and zy1 <= cy <= zy2
    )


def extract_mask(logits):

    return (
        logits.squeeze()
        .detach()
        .float()
        .cpu()
        .numpy() > 0.0
    )


def match_detection_to_masks(box, masks):

    best_id = None
    best_score = 0.0

    for object_id, mask in masks.items():

        score = mask_box_iou(mask, box)

        if score > best_score:
            best_score = score
            best_id = object_id

    if best_score >= MATCH_IOU:
        return best_id

    return None


# ============================================================
# SAM2 FUNCTIONS
# ============================================================

def create_sam2_session(predictor):

    return predictor.init_state(
        video_path=str(FRAME_DIR),
        offload_video_to_cpu=True
    )


def collect_propagation(predictor, state):

    masks_by_frame = {}

    for frame_idx, obj_ids, mask_logits in (
        predictor.propagate_in_video(state)
    ):

        frame_masks = {}

        for i, object_id in enumerate(obj_ids):

            object_id = int(object_id)

            frame_masks[object_id] = extract_mask(
                mask_logits[i]
            )

        masks_by_frame[int(frame_idx)] = frame_masks

    return masks_by_frame


def add_existing_objects(predictor, state, previous_masks):

    for object_id, mask in previous_masks.items():

        if mask.sum() < 20:
            continue

        predictor.add_new_mask(
            state,
            frame_idx=0,
            obj_id=object_id,
            mask=mask.astype(np.uint8)
        )


# ============================================================
# NEW OBJECT DETECTION AND ID MANAGEMENT
# ============================================================

def build_new_object_seeds(
    detections_by_frame,
    existing_masks_by_frame,
    next_id,
    existing_count,
    width,
    height
):

    active_candidates = []
    new_seeds = []

    for frame_idx, detections in enumerate(detections_by_frame):

        unmatched = []

        # Remove detections already covered by existing SAM2 masks
        existing_masks = existing_masks_by_frame.get(
            frame_idx, {}
        )

        for detection in detections:

            box = detection["box"]
            confidence = detection["confidence"]

            if in_exclusion_zone(box, width, height):
                continue

            existing_id = match_detection_to_masks(
                box, existing_masks
            )

            if existing_id is not None:
                continue

            if confidence >= NEW_CONF:
                unmatched.append(detection)

        # Match detections to consecutive-frame candidates
        pairs = []

        for ci, candidate in enumerate(active_candidates):

            if candidate["last_frame"] != frame_idx - 1:
                continue

            for di, detection in enumerate(unmatched):

                score = box_iou(
                    candidate["last_box"],
                    detection["box"]
                )

                if score >= CHAIN_IOU:
                    pairs.append((score, ci, di))

        pairs.sort(reverse=True)

        used_candidates = set()
        used_detections = set()

        next_candidates = []

        for score, ci, di in pairs:

            if ci in used_candidates or di in used_detections:
                continue

            candidate = active_candidates[ci]
            detection = unmatched[di]

            candidate["last_box"] = detection["box"]
            candidate["last_frame"] = frame_idx
            candidate["hits"] += 1

            used_candidates.add(ci)
            used_detections.add(di)

            next_candidates.append(candidate)

            if (
                candidate["hits"] >= NEW_PERSIST
                and not candidate["confirmed"]
            ):

                candidate["confirmed"] = True

                if existing_count + len(new_seeds) < MAX_OBJECTS:

                    new_seeds.append({
                        "id": next_id,
                        "frame_idx": candidate["start_frame"],
                        "box": candidate["start_box"].copy()
                    })

                    print(
                        f"New object ID {next_id} "
                        f"confirmed at local frame "
                        f"{candidate['start_frame']}"
                    )

                    next_id += 1

        # Start new candidate chains
        for di, detection in enumerate(unmatched):

            if di in used_detections:
                continue

            next_candidates.append({
                "start_frame": frame_idx,
                "start_box": detection["box"].copy(),
                "last_box": detection["box"].copy(),
                "last_frame": frame_idx,
                "hits": 1,
                "confirmed": False
            })

        active_candidates = next_candidates

    return new_seeds, next_id


# ============================================================
# YOLO DETECTION
# ============================================================

def run_yolo_on_window(yolo, frames):

    detections_by_frame = []

    for local_idx, frame in enumerate(frames):

        result = yolo.predict(
            frame,
            conf=DETECT_CONF,
            imgsz=YOLO_IMGSZ,
            verbose=False
        )[0]

        detections = []

        if result.boxes is not None:

            boxes = (
                result.boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )

            confidences = (
                result.boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

            for box, confidence in zip(boxes, confidences):

                detections.append({
                    "box": box.astype(np.float32),
                    "confidence": float(confidence)
                })

        detections_by_frame.append(detections)

    return detections_by_frame


# ============================================================
# ANNOTATION
# ============================================================

def annotate_frame(frame, masks):

    output = frame.copy()

    for object_id, mask in masks.items():

        if mask.shape != frame.shape[:2]:
            continue

        if mask.sum() < 20:
            continue

        color = (
            (object_id * 83) % 255,
            (object_id * 151) % 255,
            (object_id * 211) % 255
        )

        overlay = output.copy()
        overlay[mask] = color

        output = cv2.addWeighted(
            overlay, 0.40, output, 0.60, 0
        )

        ys, xs = np.where(mask)

        if len(xs) == 0:
            continue

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            output,
            f"ID {object_id}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    return output


# ============================================================
# SAVE CURRENT WINDOW FRAMES
# ============================================================

def save_window_frames(frames):

    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    for old_file in FRAME_DIR.glob("*.jpg"):
        old_file.unlink()

    for i, frame in enumerate(frames):

        success = cv2.imwrite(
            str(FRAME_DIR / f"{i:05d}.jpg"),
            frame
        )

        if not success:
            raise RuntimeError(
                f"Could not save frame {i}"
            )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    Path("output").mkdir(exist_ok=True)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Selected device:", device)

    # Load YOLO
    print("\nLoading trained YOLOv8 model...")

    yolo = YOLO(YOLO_WEIGHTS)

    print("YOLO classes:", yolo.names)

    # Load official Meta SAM2.1
    print("\nLoading official SAM2.1 predictor...")

    predictor = build_sam2_video_predictor(
        SAM2_CONFIG,
        SAM2_CHECKPOINT,
        device=device
    )

    print("SAM2.1 loaded successfully.")

    cap = cv2.VideoCapture(VIDEO)

    if not cap.isOpened():
        raise RuntimeError("Cannot open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    writer = cv2.VideoWriter(
        OUTPUT,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    # Persistent tracking state
    previous_masks = {}
    next_id = 1

    total_processed = 0
    total_new_ids = 0
    window_number = 0

    try:

        while True:

            # ----------------------------------------
            # Read next window
            # ----------------------------------------

            frames = []

            for _ in range(WINDOW_SIZE):

                success, frame = cap.read()

                if not success:
                    break

                frames.append(frame)

            if not frames:
                break

            window_number += 1

            window_start = total_processed

            print(
                f"\nWINDOW {window_number}: "
                f"frames {window_start} to "
                f"{window_start + len(frames) - 1}"
            )

            # Save frames for SAM2
            save_window_frames(frames)

            # ----------------------------------------
            # Run YOLO on every frame
            # ----------------------------------------

            detections_by_frame = run_yolo_on_window(
                yolo, frames
            )

            # ----------------------------------------
            # First propagate existing objects
            # ----------------------------------------

            existing_masks_by_frame = {}
            state = None

            if previous_masks:

                print(
                    "Existing IDs:",
                    sorted(previous_masks.keys())
                )

                state = create_sam2_session(predictor)

                try:

                    add_existing_objects(
                        predictor, state, previous_masks
                    )

                    existing_masks_by_frame = (
                        collect_propagation(predictor, state)
                    )

                except Exception:
                    predictor.reset_state(state)
                    state = None
                    raise

                predictor.reset_state(state)
                state = None

            # ----------------------------------------
            # Discover new objects
            # ----------------------------------------

            new_seeds, next_id = build_new_object_seeds(
                detections_by_frame,
                existing_masks_by_frame,
                next_id,
                len(previous_masks),
                width,
                height
            )

            total_new_ids += len(new_seeds)

            # ----------------------------------------
            # Final SAM2 session
            # ----------------------------------------

            final_masks_by_frame = existing_masks_by_frame

            if new_seeds:

                print(
                    "Starting fresh SAM2 session "
                    "with existing and new objects."
                )

                state = create_sam2_session(predictor)

                try:

                    # Re-seed existing objects from previous window
                    add_existing_objects(
                        predictor, state, previous_masks
                    )

                    # Add all newly confirmed objects
                    for seed in new_seeds:

                        predictor.add_new_points_or_box(
                            state,
                            frame_idx=seed["frame_idx"],
                            obj_id=seed["id"],
                            box=seed["box"].astype(np.float32)
                        )

                    all_ids = (
                        sorted(previous_masks.keys())
                        + [seed["id"] for seed in new_seeds]
                    )

                    print("Propagating IDs:", all_ids)

                    final_masks_by_frame = collect_propagation(
                        predictor, state
                    )

                except Exception:
                    predictor.reset_state(state)
                    state = None
                    raise

                predictor.reset_state(state)
                state = None

            # ----------------------------------------
            # Write annotated frames
            # ----------------------------------------

            latest_masks = {}

            for local_idx, frame in enumerate(frames):

                masks = final_masks_by_frame.get(
                    local_idx, {}
                )

                for object_id, mask in masks.items():
                    latest_masks[object_id] = mask

                annotated = annotate_frame(frame, masks)

                writer.write(annotated)

                total_processed += 1

            # ----------------------------------------
            # Handoff to next window
            # ----------------------------------------

            previous_masks = {
                object_id: mask.copy()
                for object_id, mask in latest_masks.items()
                if mask.sum() >= 20
            }

            print(
                f"Window completed: "
                f"{total_processed}/{total_frames} frames"
            )

            print(
                "IDs carried forward:",
                sorted(previous_masks.keys())
            )

    finally:

        cap.release()
        writer.release()

    print("\n===================================")
    print("FULL VIDEO PROCESSING COMPLETED")
    print("===================================")

    print("Frames processed:", total_processed)
    print("Expected frames:", total_frames)
    print("New IDs confirmed:", total_new_ids)
    print("Next available ID:", next_id)
    print("Output video:", OUTPUT)


if __name__ == "__main__":
    main()
