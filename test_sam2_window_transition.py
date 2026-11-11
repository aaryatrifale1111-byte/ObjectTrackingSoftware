
import cv2
import torch
import numpy as np
from pathlib import Path

from sam2.build_sam import build_sam2_video_predictor

# Paths
CHECKPOINT = Path("models/sam2.1_hiera_small.pt").resolve()
CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

WINDOW_1 = Path("output/window_1")
WINDOW_2 = Path("output/window_2")

OUTPUT = Path("output/sam2_window_transition.mp4")

OBJECT_ID = 1
BOX = np.array([568, 292, 742, 545], dtype=np.float32)

print("Loading SAM2.1...")
predictor = build_sam2_video_predictor(
    CONFIG,
    str(CHECKPOINT),
    device="cpu"
)

def annotate(frame, mask, obj_id):
    result = frame.copy()

    mask = np.asarray(mask).astype(bool)

    result[mask] = (
        0.4 * result[mask] +
        0.6 * np.array([0, 255, 0])
    ).astype(np.uint8)

    ys, xs = np.where(mask)

    if len(xs):
        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()

        cv2.rectangle(
            result,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

        cv2.putText(
            result,
            f"ID: {obj_id}",
            (int(x1), max(25, int(y1) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    return result


last_mask = None
writer = None

# -----------------------------------------
# WINDOW 1: Start tracking with box prompt
# -----------------------------------------

print("\nProcessing window 1...")

with torch.inference_mode():

    state_1 = predictor.init_state(
        video_path=str(WINDOW_1.resolve()),
        offload_video_to_cpu=True
    )

    predictor.add_new_points_or_box(
        inference_state=state_1,
        frame_idx=0,
        obj_id=OBJECT_ID,
        box=BOX
    )

    for frame_idx, obj_ids, logits in predictor.propagate_in_video(state_1):

        frame = cv2.imread(
            str(WINDOW_1 / f"{frame_idx:05d}.jpg")
        )

        for i, obj_id in enumerate(obj_ids):
            mask = (
                logits[i].squeeze().detach().cpu().numpy() > 0
            )

            if obj_id == OBJECT_ID:
                last_mask = mask.copy()

            annotated = annotate(frame, mask, obj_id)

            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(OUTPUT),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30,
                    (w, h)
                )

            writer.write(annotated)

    if last_mask is None:
        raise RuntimeError("No final mask found in window 1.")

    print("Window 1 completed.")
    print("Previous mask shape:", last_mask.shape)

    # Release first session before initializing the next.
    predictor.reset_state(state_1)
    del state_1

    # -----------------------------------------
    # WINDOW 2: Re-seed with previous mask
    # -----------------------------------------

    print("\nInitializing window 2...")

    state_2 = predictor.init_state(
        video_path=str(WINDOW_2.resolve()),
        offload_video_to_cpu=True
    )

    predictor.add_new_mask(
        inference_state=state_2,
        frame_idx=0,
        obj_id=OBJECT_ID,
        mask=last_mask
    )

    print("Previous mask supplied as prompt.")
    print("Continuing with object ID:", OBJECT_ID)

    for frame_idx, obj_ids, logits in predictor.propagate_in_video(state_2):

        frame = cv2.imread(
            str(WINDOW_2 / f"{frame_idx:05d}.jpg")
        )

        for i, obj_id in enumerate(obj_ids):
            mask = (
                logits[i].squeeze().detach().cpu().numpy() > 0
            )

            annotated = annotate(frame, mask, obj_id)

            writer.write(annotated)

    predictor.reset_state(state_2)

if writer is not None:
    writer.release()

print("\nTransition test completed.")
print("Output:", OUTPUT)
