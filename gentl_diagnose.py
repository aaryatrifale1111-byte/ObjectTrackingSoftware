"""
GenTL diagnostic -- run this on the WINDOWS PC (not in Codespaces).

Answers, before any tracking is involved:
  * Which devices does the GenTL producer actually report (vendor, model,
    serial, transport)? Is a real camera present?
  * Is the producer running in simulated mode (SIMCAM_XML_FILE set)?
  * What pixel format / size / bit depth do frames arrive in?
  * How many frames per second does acquisition really deliver?

Close Critical Link's GenTL Viewer first: this script opens the camera itself.

    python gentl_diagnose.py
    python gentl_diagnose.py --cti "C:\\Program Files\\Critical Link LLC\\GenTL Viewer\\bin\\GenTL.dll" --seconds 10
"""

import argparse
import os
import sys
import tempfile
import time

DEFAULT_CTI = r"C:\Program Files\Critical Link LLC\GenTL Viewer\bin\GenTL.dll"


def safe(obj, name):
    try:
        return getattr(obj, name)
    except Exception as error:  # producer may not implement some fields
        return f"<n/a: {type(error).__name__}>"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cti", default=DEFAULT_CTI, help="GenTL producer (.cti or .dll)")
    p.add_argument("--index", type=int, default=0, help="device index to open")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--buffers", type=int, default=4, help="num_buffers (<=5 for large U3V images)")
    p.add_argument("--pixel-format", default=None, help="e.g. Mono8 (optional)")
    p.add_argument("--save", default="gentl_diag_frame.png")
    args = p.parse_args()

    # The Critical Link producer needs its own folder on PATH for helper DLLs.
    bin_dir = os.path.dirname(args.cti)
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory") and os.path.isdir(bin_dir):
        os.add_dll_directory(bin_dir)

    # Avoid the "Permission denied: c:///criticallink_systemgentl.xml" error.
    os.environ.setdefault("HARVESTERS_XML_FILE_DIR", tempfile.gettempdir())

    print("== environment ==")
    print("producer file :", args.cti, "| exists:", os.path.exists(args.cti))
    sim = os.environ.get("SIMCAM_XML_FILE")
    print("SIMCAM_XML_FILE:", sim if sim else "(not set)")
    if sim:
        print("  !! SIMCAM_XML_FILE is set: the producer may show a SIMULATED "
              "camera. Unset it (set SIMCAM_XML_FILE=) and retry.")

    from harvesters.core import Harvester

    h = Harvester()
    if hasattr(h, "add_file"):
        h.add_file(args.cti)
    else:
        h.add_cti_file(args.cti)
    if hasattr(h, "update"):
        h.update()
    else:
        h.update_device_info_list()

    infos = h.device_info_list
    print("\n== devices reported by the producer ==")
    if not infos:
        print("NO DEVICES FOUND. Check the USB/Ethernet cable and power, and "
              "that the GenTL Viewer is closed.")
        h.reset()
        return 1

    for i, d in enumerate(infos):
        print(
            f"[{i}] vendor={safe(d, 'vendor')} model={safe(d, 'model')} "
            f"serial={safe(d, 'serial_number')} transport={safe(d, 'tl_type')} "
            f"id={safe(d, 'id_')}"
        )

    if args.index >= len(infos):
        print(f"--index {args.index} out of range")
        h.reset()
        return 1

    ia = h.create(args.index) if hasattr(h, "create") else h.create_image_acquirer(args.index)

    try:
        try:
            ia.num_buffers = args.buffers
        except Exception as error:
            print("could not set num_buffers:", error)

        nm = ia.remote_device.node_map
        if args.pixel_format:
            nm.PixelFormat.value = args.pixel_format

        print("\n== camera settings ==")
        for name in ("DeviceModelName", "DeviceSerialNumber", "PixelFormat",
                     "Width", "Height", "AcquisitionFrameRate", "ExposureTime"):
            try:
                print(f"{name:22s}: {getattr(nm, name).value}")
            except Exception:
                print(f"{name:22s}: <not available>")

        print(f"\n== acquiring for {args.seconds:.0f} s ==")
        ia.start() if hasattr(ia, "start") else ia.start_acquisition()

        frames, first, fetch_ms = 0, None, []
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < args.seconds:
            t0 = time.perf_counter()
            try:
                with ia.fetch(timeout=3) as buffer:
                    comp = buffer.payload.components[0]
                    img = comp.data.reshape(
                        comp.height, comp.width, int(comp.num_components_per_pixel)
                    ).copy()
                    fmt = getattr(comp, "data_format", "?")
            except Exception as error:
                print("fetch failed:", type(error).__name__, error)
                break
            fetch_ms.append((time.perf_counter() - t0) * 1000)
            frames += 1
            if first is None:
                first = (img, fmt)

        wall = time.perf_counter() - t_start
        ia.stop() if hasattr(ia, "stop") else ia.stop_acquisition()

        print("\n== result ==")
        if frames == 0:
            print("No frames received.")
            return 1

        img, fmt = first
        print(f"frames: {frames}   measured FPS: {frames / wall:.1f}")
        print(f"fetch ms: mean {sum(fetch_ms) / len(fetch_ms):.1f}  max {max(fetch_ms):.1f}")
        print(f"frame shape: {img.shape}  dtype: {img.dtype}  data_format: {fmt}")
        print(f"pixel range: {int(img.min())} .. {int(img.max())}")

        try:
            import cv2
            import numpy as np
            view = img.astype(np.float32)
            view = (255 * (view - view.min()) / max(1.0, view.max() - view.min())).astype(np.uint8)
            cv2.imwrite(args.save, view)
            print("saved preview:", args.save)
        except Exception as error:
            print("preview not saved:", error)
    finally:
        ia.destroy()
        h.reset()

    return 0


if __name__ == "__main__":
    sys.exit(main())