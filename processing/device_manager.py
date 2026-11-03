
import torch


class DeviceManager:
    """
    Selects the inference device for YOLO and SAM2.

    Modes:
        auto: Uses CUDA GPU if available, otherwise CPU.
        cpu:  Forces CPU.
        gpu:  Requires a compatible CUDA GPU.
    """

    VALID_MODES = ("auto", "cpu", "gpu")

    def __init__(self, mode="auto"):
        mode = mode.lower().strip()

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid device mode: {mode}. "
                f"Choose from {self.VALID_MODES}."
            )

        self.requested_mode = mode
        self.device = self._select_device()

    def _select_device(self):
        cuda_available = torch.cuda.is_available()

        if self.requested_mode == "cpu":
            return torch.device("cpu")

        if self.requested_mode == "gpu":
            if not cuda_available:
                raise RuntimeError(
                    "GPU mode was selected, but CUDA is unavailable. "
                    "Check your NVIDIA GPU, driver and PyTorch installation."
                )

            return torch.device("cuda:0")

        # Automatic mode
        if cuda_available:
            return torch.device("cuda:0")

        return torch.device("cpu")

    @property
    def device_name(self):
        if self.device.type == "cuda":
            return torch.cuda.get_device_name(self.device)

        return "CPU"

    @property
    def is_gpu(self):
        return self.device.type == "cuda"

    def get_device(self):
        return self.device

    def get_status(self):
        return {
            "requested_mode": self.requested_mode,
            "selected_device": str(self.device),
            "device_name": self.device_name,
            "cuda_available": torch.cuda.is_available(),
        }
