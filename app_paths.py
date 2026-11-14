"""Model/config locations. Works from source and from a PyInstaller bundle."""

import os
import sys

# A packaged app has no pip: stop Ultralytics from trying to auto-install packages.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")


def resource_path(relative):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)


YOLO_WEIGHTS = resource_path(os.path.join("models", "spark_yolov8n_best.pt"))
SAM2_CHECKPOINT = resource_path(os.path.join("models", "sam2.1_hiera_small.pt"))
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
