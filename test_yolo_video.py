
import cv2
from pathlib import Path
from ultralytics import YOLO

VIDEO = "video1.mp4"
WEIGHTS = "models/spark_yolov8n_best.pt"
OUTPUT = "output/yolo_full_video.mp4"

Path("output").mkdir(exist_ok=True)

model = YOLO(WEIGHTS)
print("Model classes:", model.names)

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError("Cannot open input video")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

writer = cv2.VideoWriter(
    OUTPUT,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

total_detections = 0
frames_with_detections = 0
processed = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    result = model.predict(
        frame,
        conf=0.10,
        imgsz=640,
        verbose=False
    )[0]

    count = len(result.boxes)
    total_detections += count

    if count > 0:
        frames_with_detections += 1

    writer.write(result.plot())
    processed += 1

    if processed % 100 == 0:
        print(f"Processed {processed}/{total_frames} frames")

cap.release()
writer.release()

print("\nFULL VIDEO YOLO TEST COMPLETED")
print("Frames processed:", processed)
print("Frames with detections:", frames_with_detections)
print("Total detections:", total_detections)
print("Output:", OUTPUT)
