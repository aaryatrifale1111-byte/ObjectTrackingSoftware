
import os
import time
import numpy as np
import cv2

from harvesters.core import Harvester


class GenTLCamera:

    def __init__(self, gentl_dll):

        self.gentl_dll = gentl_dll
        self.harvester = None
        self.acquirer = None
        self.running = False
        self.dll_directory = None

    def connect(self):

        if not os.path.isfile(self.gentl_dll):
            raise FileNotFoundError(
                f"GenTL producer not found: {self.gentl_dll}"
            )

        dll_folder = os.path.dirname(self.gentl_dll)

        os.environ["PATH"] = (
            dll_folder + os.pathsep +
            os.environ.get("PATH", "")
        )

        if hasattr(os, "add_dll_directory"):
            self.dll_directory = os.add_dll_directory(
                dll_folder
            )

        self.harvester = Harvester()

        self.harvester.add_file(self.gentl_dll)
        self.harvester.update()

        devices = self.harvester.device_info_list

        if not devices:
            raise RuntimeError(
                "No GenTL cameras detected."
            )

        print("Available cameras:")

        for index, device in enumerate(devices):
            print(index, device)

        # Connect to the first detected camera
        self.acquirer = self.harvester.create(0)

        # Limit acquisition buffers
        self.acquirer.num_buffers = 5

        return devices

    def start(self):

        if self.acquirer is None:
            raise RuntimeError("Camera is not connected.")

        self.acquirer.start()
        self.running = True

        print("Camera acquisition started.")

    def get_frame(self, timeout=2.0):

        if not self.running:
            raise RuntimeError("Camera is not streaming.")

        with self.acquirer.fetch(timeout=timeout) as buffer:

            component = buffer.payload.components[0]

            width = int(component.width)
            height = int(component.height)

            pixel_format = str(component.data_format)
            channels = int(
                component.num_components_per_pixel
            )

            data = np.asarray(component.data)

            if channels == 1:

                image = data.reshape(height, width)

                if image.dtype == np.uint16:
                    image = (image >> 8).astype(np.uint8)
                else:
                    image = image.astype(
                        np.uint8, copy=False
                    )

                frame = cv2.cvtColor(
                    image,
                    cv2.COLOR_GRAY2BGR
                )

            elif channels == 3:

                image = data.reshape(
                    height, width, 3
                )

                image = image.astype(
                    np.uint8, copy=False
                )

                if "BGR" in pixel_format:
                    frame = image
                else:
                    frame = cv2.cvtColor(
                        image,
                        cv2.COLOR_RGB2BGR
                    )

            else:

                raise ValueError(
                    f"Unsupported format: {pixel_format}, "
                    f"channels: {channels}"
                )

            return np.ascontiguousarray(frame)

    def stop(self):

        self.running = False

        if self.acquirer is not None:

            try:
                self.acquirer.stop()
            except Exception:
                pass

            try:
                self.acquirer.destroy()
            except Exception:
                pass

            self.acquirer = None

        if self.harvester is not None:

            try:
                self.harvester.reset()
            except Exception:
                pass

            self.harvester = None

        if self.dll_directory is not None:
            self.dll_directory.close()
            self.dll_directory = None

        print("Camera disconnected.")
