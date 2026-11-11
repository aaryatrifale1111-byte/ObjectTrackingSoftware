
import cv2
import torch
import numpy as np
from pathlib import Path

from sam2.build_sam import build_sam2_video_predictor

# --------------------------------------------------
# PATHS
# --------------------------------------------------

FRAME_DIR = Path("output/sam2_cups_60")
CHECKPOINT = Path("models/sam2.1_hiera_small.pt")
CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

OUTPUT_DIR = Path("output/sam2_propagation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

OBJECT_ID = 1

# Middle red cup in the first frame
BOX = np.array([568, 292, 742, 545], dtype=np.float32)

FPS = 30

# --------------------------------------------------
# LOAD SAM2.1
# --------------------------------------------------

print("Loading official Meta SAM2.1 predictor...")

predictor = build_sam2_video_predictor(
    CONFIG,
    str(CHECKPOINT.resolve()),
    device="cpu"
)

print("SAM2.1 loaded.")

# --------------------------------------------------
# INITIALIZE VIDEO
# --------------------------------------------------

print("Initializing video frames...")

with torch.inference_mode():

    inference_state = predictor.init_state(
        video_path=str(FRAME_DIR.resolve()),
        offload_video_to_cpu=True
    )

    # Add the first object prompt
    print("Adding object ID:", OBJECT_ID)

    _, out_obj_ids, out_mask_logits = (
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=OBJECT_ID,
            box=BOX
        )
    )

    print("Initial prompt accepted.")
    print("Object IDs:", out_obj_ids)

    # --------------------------------------------------
    # PROPAGATE THROUGH VIDEO
    # --------------------------------------------------

    frame_count = 0
    writer = None

    print("Starting forward propagation...")

    for frame_idx, obj_ids, mask_logits in (
        predictor.propagate_in_video(inference_state)
    ):

        frame_path = FRAME_DIR / f"{frame_idx:05d}.jpg"
        frame = cv2.imread(str(frame_path))

        if frame is None:
            print("Could not read frame:", frame_idx)
            continue

        overlay = frame.copy()

        for i, obj_id in enumerate(obj_ids):

            mask = (
                mask_logits[i]
                .squeeze()
                .detach()
                .cpu()
                .numpy() > 0.0
            )

            # Overlay mask
            overlay[mask] = (
                0.4 * overlay[mask] +
                0.6 * np.array([0, 255, 0])
            ).astype(np.uint8)

            # Calculate mask bounding box
            ys, xs = np.where(mask)

            if len(xs) > 0:

                x1, y1 = xs.min(), ys.min()
                x2, y2 = xs.max(), ys.max()

                cv2.rectangle(
                    overlay,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    overlay,
                    f"ID: {obj_id}",
                    (int(x1), max(25, int(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        # Initialize output video writer
        if writer is None:
            height, width = overlay.shape[:2]

            writer = cv2.VideoWriter(
                str(OUTPUT_DIR / "sam2_tracking.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                FPS,
                (width, height)
            )

        writer.write(overlay)
        frame_count += 1

        print(
            f"Processed frame {frame_idx + 1}/30 | "
            f"Object IDs: {list(obj_ids)}"
        )

    if writer is not None:
        writer.release()

    predictor.reset_state(inference_state)

print("\nPropagation finished.")
print("Frames processed:", frame_count)
print("Output:", OUTPUT_DIR / "sam2_tracking.mp4")
