"""
Live prototype on a saved video.

Smoke test (CPU, a few frames, deterministic, no frame dropping):
    python test_live_prototype.py --mode sequential --max-frames 6 --yolo-interval 1

Simulated live stream (producer paced at the video's FPS, consumer always takes
the newest frame and drops the rest):
    python test_live_prototype.py --mode live --max-frames 120 --yolo-interval 5
"""

import argparse
import os
import threading
import time

import cv2
import numpy as np

from camera.frame_buffer import FrameBuffer
from processing.id_manager import box_iou
from processing.live_pipeline import LiveTrackingPipeline, draw_result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="video1.mp4")
    p.add_argument("--yolo", default="models/spark_yolov8n_best.pt")
    p.add_argument("--sam2-checkpoint", default="models/sam2.1_hiera_small.pt")
    p.add_argument(
        "--sam2-config", default="configs/sam2.1/sam2.1_hiera_s.yaml"
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--mode", default="sequential", choices=["sequential", "live"])
    p.add_argument("--start", type=int, default=0, help="first video frame")
    p.add_argument("--max-frames", type=int, default=6)
    p.add_argument(
        "--stride", type=int, default=1,
        help="sequential mode: process every Nth video frame (simulates dropped frames)",
    )
    p.add_argument("--yolo-interval", type=int, default=5)
    p.add_argument("--infer-max-side", type=int, default=1280)
    p.add_argument("--save", default="output/live_proto.mp4")
    p.add_argument(
        "--precision", default="auto", choices=["auto", "bf16", "fp32"],
        help="SAM2 precision (auto: bf16 on CUDA, fp32 on CPU)",
    )
    return p.parse_args()


def open_video(path, start):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    return cap


class Recorder:
    def __init__(self, path, fps):
        self.path, self.fps, self.writer = path, fps, None

    def write(self, image):
        if not self.path:
            return
        if self.writer is None:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            h, w = image.shape[:2]
            self.writer = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
            )
        self.writer.write(image)

    def close(self):
        if self.writer is not None:
            self.writer.release()


def report(rows, dropped, captured, wall, extra=""):
    print("\n=== SUMMARY ===")
    if not rows:
        print("No frames processed.")
        return
    steady = rows[1:] if len(rows) > 1 else rows  # frame 0 includes warm-up
    tot = np.array([r["total_ms"] for r in steady])
    yo = np.array([r["yolo_ms"] for r in steady if r["yolo_ran"]] or [0.0])
    sam = np.array([r["sam_ms"] for r in steady])
    print(f"processed frames : {len(rows)}   (first-frame warm-up: {rows[0]['total_ms']:.0f} ms)")
    print(f"captured / dropped: {captured} / {dropped}")
    print(f"processed FPS     : {len(rows) / wall:.2f}")
    print(f"total ms/frame    : mean {tot.mean():.0f}  p95 {np.percentile(tot, 95):.0f}")
    print(f"YOLO ms (when run): mean {yo.mean():.0f}")
    print(f"SAM2 ms/frame     : mean {sam.mean():.0f}")
    lat = [r["latency_ms"] for r in steady if r.get("latency_ms") is not None]
    if lat:
        print(f"end-to-end latency: mean {np.mean(lat):.0f} ms  p95 {np.percentile(lat, 95):.0f} ms  max {np.max(lat):.0f} ms")
    print(f"active IDs        : {rows[-1]['active_ids']}")
    if extra:
        print(extra)


def mask_vs_yolo(r):
    """Best IoU between each tracked mask's box and any YOLO box (only when YOLO ran)."""
    if not r["detections"]:
        return {}
    return {
        i: round(max(box_iou(b, d["box"]) for d in r["detections"]), 2)
        for i, b in r["boxes"].items()
    }


def run_sequential(pipe, args):
    cap = open_video(args.video, args.start)
    rec = Recorder(args.save, 10)
    rows = []
    t_begin = time.perf_counter()

    video_pos = args.start
    while len(rows) < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        video_pos += 1
        if (video_pos - 1 - args.start) % args.stride != 0:
            continue
        ts = time.perf_counter()
        r = pipe.process_frame(frame, capture_ts=ts)
        r["latency_ms"] = (time.perf_counter() - ts) * 1000.0
        rows.append(r)

        print(
            f"frame {r['frame_index']:3d} | yolo {'Y' if r['yolo_ran'] else '-'} "
            f"{r['yolo_ms']:6.0f} ms | sam {r['sam_ms']:6.0f} ms | "
            f"total {r['total_ms']:6.0f} ms | dets {len(r['detections'])} | "
            f"active {r['active_ids']} | new {r['new_ids']} | lost {r['lost_ids']} | "
            f"mask px { {i: int(m.sum()) for i, m in r['masks'].items()} } | "
            f"vid {video_pos - 1} | mask-vs-yolo IoU {mask_vs_yolo(r)}"
        )
        rec.write(draw_result(r, [f"frame {r['frame_index']}  IDs {r['active_ids']}"]))

    rec.close()
    cap.release()
    wall = time.perf_counter() - t_begin
    report(rows, 0, len(rows), wall)


def run_live(pipe, args):
    cap = open_video(args.video, args.start)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    buf = FrameBuffer()
    stats = {"captured": 0}

    def producer():
        period = 1.0 / video_fps
        next_t = time.perf_counter()
        while stats["captured"] < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            buf.put(frame)
            stats["captured"] += 1
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        buf.close()

    thread = threading.Thread(target=producer, daemon=True)
    rec = Recorder(args.save, 10)
    rows, dropped, last_seq = [], 0, 0

    t_begin = time.perf_counter()
    thread.start()

    while True:
        item = buf.wait_newer(last_seq, timeout=5.0)
        if item is None:
            break
        frame, seq, ts = item
        dropped += seq - last_seq - 1
        last_seq = seq

        r = pipe.process_frame(frame, capture_ts=ts)
        r["latency_ms"] = (time.perf_counter() - ts) * 1000.0
        rows.append(r)

        print(
            f"seq {seq:4d} | dropped so far {dropped:4d} | "
            f"total {r['total_ms']:6.0f} ms | latency {r['latency_ms']:6.0f} ms | "
            f"active {r['active_ids']} | new {r['new_ids']}"
        )
        rec.write(draw_result(r, [
            f"seq {seq}  latency {r['latency_ms']:.0f} ms  IDs {r['active_ids']}"
        ]))

    rec.close()
    cap.release()
    wall = time.perf_counter() - t_begin
    report(rows, dropped, stats["captured"], wall,
           extra=f"video fps (capture rate): {video_fps:.1f}")


def main():
    args = parse_args()

    pipe = LiveTrackingPipeline(
        yolo_weights=args.yolo,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        device_mode=args.device,
        yolo_interval=args.yolo_interval,
        infer_max_side=args.infer_max_side,
        precision=args.precision,
    )
    pipe.load_models()

    if args.mode == "sequential":
        run_sequential(pipe, args)
    else:
        run_live(pipe, args)


if __name__ == "__main__":
    main()