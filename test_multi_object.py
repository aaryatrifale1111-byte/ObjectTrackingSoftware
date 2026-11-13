"""
Multi-object mechanism test (tracker only, no ID manager).

  video frame idx (stride 10):   0   10   20   30   40   50
  processed frame:               0    1    2    3    4    5
  object 1 (YOLO box):        seed
  object 2 (manual box):                         seed
  expect 1 object on frames 0-2, 2 objects on frames 3-5, no exception.

Object 2's box is a hand-placed region (NOT a real detection). Its mask will
be whatever SAM2 makes of that region; the test only checks the mechanism
(no dtype error, new object added mid-stream without replay, object 1 is not
disturbed, both objects keep their IDs).

    OMP_NUM_THREADS=2 python test_multi_object.py
"""

import argparse
import os
import time
import traceback

import cv2
from ultralytics import YOLO

from processing.device_manager import DeviceManager
from processing.id_manager import box_iou, mask_to_box
from processing.live_pipeline import draw_result
from processing.sam2_stream import SAM2StreamTracker


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="video1.mp4")
    p.add_argument("--yolo", default="models/spark_yolov8n_best.pt")
    p.add_argument("--sam2-checkpoint", default="models/sam2.1_hiera_small.pt")
    p.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp32"])
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--add-at", type=int, default=3, help="processed-frame index where object 2 is added")
    p.add_argument("--out", default="output/multi_object_last.jpg")
    args = p.parse_args()

    device = DeviceManager(args.device).get_device()
    yolo = YOLO(args.yolo)
    tracker = SAM2StreamTracker(
        args.sam2_checkpoint, args.sam2_config, device, precision=args.precision
    )
    print("device:", device, "| precision:", tracker.precision)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")

    def yolo_best_box(frame):
        r = yolo.predict(frame, conf=0.10, imgsz=640, device=str(device), verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None
        i = int(r.boxes.conf.argmax())
        return tuple(float(v) for v in r.boxes.xyxy[i].tolist())

    checks = []
    last_result = None
    px1_history = []

    try:
        for k in range(args.frames):
            for _ in range(0 if k == 0 else args.stride - 1):
                cap.grab()
            ok, frame = cap.read()
            if not ok:
                break

            t0 = time.perf_counter()
            masks = tracker.track(frame)

            if k == 0:
                box = yolo_best_box(frame)
                assert box is not None, "YOLO found nothing on the first frame"
                masks.update(tracker.seed([(1, box)]))
                print(f"frame {k}: seeded ID 1 from YOLO box {tuple(round(v) for v in box)}")

            if k == args.add_at:
                h, w = frame.shape[:2]
                ref = yolo_best_box(frame) or (0, 0, 120, 90)
                bw, bh = ref[2] - ref[0], ref[3] - ref[1]
                manual = (0.08 * w, 0.10 * h, 0.08 * w + bw, 0.10 * h + bh)
                masks.update(tracker.seed([(2, manual)]))
                print(f"frame {k}: seeded ID 2 from MANUAL box {tuple(round(v) for v in manual)}")

            dt = (time.perf_counter() - t0) * 1000
            px = {i: int(m.sum()) for i, m in masks.items()}
            ybox = yolo_best_box(frame)
            iou1 = None
            if 1 in masks and ybox is not None:
                b1 = mask_to_box(masks[1])
                iou1 = round(box_iou(b1, ybox), 2) if b1 else 0.0
            print(f"frame {k} | sam {dt:6.0f} ms | ids {sorted(masks)} | mask px {px} | ID1 vs YOLO IoU {iou1}")

            expected = {1} if k < args.add_at else {1, 2}
            checks.append((f"frame {k} has ids {sorted(expected)}", set(masks) == expected))
            if k >= args.add_at:
                checks.append((f"frame {k} ID 2 mask non-empty", px.get(2, 0) > 0))
            if iou1 is not None:
                checks.append((f"frame {k} ID 1 still on spacecraft (IoU>0.5)", iou1 > 0.5))

            boxes = {i: mask_to_box(m) for i, m in masks.items()}
            last_result = {
                "frame": frame, "masks": masks,
                "boxes": {i: b for i, b in boxes.items() if b is not None},
            }
    except Exception:
        traceback.print_exc()
        checks.append(("no exception", False))

    if last_result is not None:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        cv2.imwrite(args.out, draw_result(last_result, ["multi-object test (ID 2 = manual box)"]))
        print("saved", args.out)

    print("\n=== CHECKS ===")
    for name, passed in checks:
        print(("PASS  " if passed else "FAIL  ") + name)
    print("RESULT:", "ALL PASS" if checks and all(p for _, p in checks) else "SOMETHING FAILED")


if __name__ == "__main__":
    main()