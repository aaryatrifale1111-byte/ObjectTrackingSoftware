# -*- mode: python ; coding: utf-8 -*-
# Build with:  pyinstaller SpacecraftTracker.spec --noconfirm --clean
# (build_windows.bat does the whole thing.)

import os
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []

# Packages that load data files / plugins dynamically and need everything copied.
for package in (
    "ultralytics", "sam2", "hydra", "omegaconf", "antlr4", "iopath",
    "genicam", "harvesters",
):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h

# Model weights, bundled inside the app.
datas += [("models", "models")]

# SAM2 config YAMLs (Hydra looks them up inside the 'sam2' package).
sam2_configs = os.path.join("third_party", "sam2", "sam2", "configs")
if os.path.isdir(sam2_configs):
    datas.append((sam2_configs, os.path.join("sam2", "configs")))

# Small test clip so --selftest and the video-file mode work on a clean PC.
if os.path.exists("video1.mp4"):
    datas.append(("video1.mp4", "test_video"))

# Modules imported lazily inside functions.
hiddenimports += [
    "app_paths", "selftest",
    "camera.webcam_camera", "camera.gentl_camera", "camera.video_camera",
    "camera.frame_utils", "camera.frame_buffer", "camera.camera_worker",
    "processing.live_pipeline", "processing.sam2_stream",
    "processing.id_manager", "processing.device_manager",
    "processing.inference_worker",
]

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SpacecraftTracker",
    console=True,      # keep the console while testing; set False for release
    upx=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="SpacecraftTracker",
)
