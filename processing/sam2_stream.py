"""
Streaming adapter around the OFFICIAL Meta SAM2 video predictor.

Written against the vendored checkout at third_party/sam2, commit 2b90b9f
(Dec 2024), by reading its source:

  * init_state() requires a whole video, so we build the same `inference_state`
    dict ourselves and replace `state["images"]` with a one-frame store.
  * Tracking one frame is exactly what propagate_in_video() does per frame:
    predictor._run_single_frame_inference(...) once per object (batch size 1).
    We call that directly for each incoming frame.
  * New objects can be added at any time (_obj_id_to_idx always allows it):
    add_new_points_or_box() on the current frame + propagate_in_video_preflight()
    to consolidate. No replay of earlier frames.
  * The image backbone runs once per frame (cached by frame index) and is
    shared by all objects.

This relies on private (underscore) predictor methods. That is acceptable
only because the checkout is pinned; do not upgrade third_party/sam2 without
re-running test_live_prototype.py.

Precision:
  * "bf16": all SAM2 work runs under bfloat16 autocast (what Meta's demos do).
    Confirmed on CPU: this removes the earlier "mat1 and mat2 must have the
    same dtype" error, because the predictor stores memory features as
    bfloat16 unconditionally (see _run_single_frame_inference).
  * "fp32": no autocast; stored memory features are converted back to float32
    so nothing is mixed. Intended for CPUs without hardware bfloat16, where
    bf16 autocast measured ~65 s/frame in Codespaces.
  * "auto": bf16 on CUDA, fp32 on CPU.
"""

import contextlib
from collections import OrderedDict

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor

_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)

# Streams have no known length. Only used as an upper bound inside SAM2.
_NUM_FRAMES = 1_000_000


class _OneFrameStore:
    """Stands in for inference_state['images']; holds only the current frame."""

    def __init__(self):
        self.idx = None
        self.tensor = None

    def set(self, idx, tensor):
        self.idx = idx
        self.tensor = tensor

    def __getitem__(self, idx):
        if idx != self.idx:
            raise IndexError(
                f"Frame {idx} is not stored (current frame is {self.idx})."
            )
        return self.tensor


class SAM2StreamTracker:

    def __init__(
        self,
        checkpoint,
        config,
        device,
        keep_frames=32,
        vos_optimized=False,
        precision="auto",
    ):
        self.predictor = build_sam2_video_predictor(
            config,
            checkpoint,
            device=str(device),
            vos_optimized=vos_optimized,
        )
        self.device = self.predictor.device
        self.keep_frames = keep_frames

        if precision == "auto":
            precision = "bf16" if self.device.type == "cuda" else "fp32"
        if precision not in ("bf16", "fp32"):
            raise ValueError("precision must be 'auto', 'bf16' or 'fp32'")
        self.precision = precision

        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.store = _OneFrameStore()
        self.state = None
        self.frame_idx = -1

    def _autocast(self):
        if self.precision == "bf16":
            return torch.autocast(
                device_type=self.device.type, dtype=torch.bfloat16
            )
        return contextlib.nullcontext()

    def _to_fp32(self, out):
        """fp32 mode: undo the predictor's forced bfloat16 memory storage."""
        if self.precision == "fp32" and out.get("maskmem_features") is not None:
            out["maskmem_features"] = out["maskmem_features"].float()

    def _ensure_state(self, height, width):
        if self.state is not None:
            if (
                self.state["video_height"] != height
                or self.state["video_width"] != width
            ):
                raise ValueError(
                    "Frame size changed mid-stream "
                    f"({self.state['video_width']}x{self.state['video_height']} "
                    f"-> {width}x{height}). Call reset() first."
                )
            return

        # Same keys as SAM2VideoPredictor.init_state().
        self.state = {
            "images": self.store,
            "num_frames": _NUM_FRAMES,
            "offload_video_to_cpu": False,
            "offload_state_to_cpu": False,
            "video_height": height,
            "video_width": width,
            "device": self.device,
            "storage_device": self.device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
        }

    def _preprocess(self, frame_bgr):
        """BGR uint8 -> normalised (3, S, S) float tensor, as load_video_frames."""
        size = self.predictor.image_size
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)

        tensor = torch.from_numpy(rgb).float().div_(255.0).permute(2, 0, 1)
        tensor = (tensor - _MEAN) / _STD
        return tensor.to(self.device)

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def track(self, frame_bgr):
        """
        Advance one frame and update every existing object.
        Returns {obj_id: bool mask (H, W)}.
        """
        height, width = frame_bgr.shape[:2]
        self._ensure_state(height, width)

        self.frame_idx += 1
        t = self.frame_idx
        state = self.state

        with self._autocast():
            self.store.set(t, self._preprocess(frame_bgr))

            n_objects = len(state["obj_ids"])
            if n_objects == 0:
                return {}

            for obj_idx in range(n_objects):
                obj_output = state["output_dict_per_obj"][obj_idx]

                current_out, _ = self.predictor._run_single_frame_inference(
                    inference_state=state,
                    output_dict=obj_output,
                    frame_idx=t,
                    batch_size=1,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=False,
                    run_mem_encoder=True,
                )

                self._to_fp32(current_out)
                obj_output["non_cond_frame_outputs"][t] = current_out
                state["frames_tracked_per_obj"][obj_idx][t] = {"reverse": False}

            masks = self._collect(t)
            self._prune(t)

        return masks

    @torch.inference_mode()
    def seed(self, new_objects):
        """
        Add new objects on the CURRENT frame (call after track()).

        new_objects: list of (obj_id, (x1, y1, x2, y2)) in frame pixels.
        Returns {obj_id: bool mask} for the new objects on this frame.
        """
        if not new_objects:
            return {}
        if self.state is None:
            raise RuntimeError("Call track() on a frame before seed().")

        t = self.frame_idx
        state = self.state
        new_ids = {obj_id for obj_id, _ in new_objects}

        with self._autocast():
            for obj_id, box in new_objects:
                self.predictor.add_new_points_or_box(
                    state,
                    frame_idx=t,
                    obj_id=obj_id,
                    box=[float(v) for v in box],
                )

            # Moves the prompt outputs into output_dict and runs the memory
            # encoder for them, so the next frame can use them.
            self.predictor.propagate_in_video_preflight(state)

            for obj_idx, obj_id in state["obj_idx_to_id"].items():
                if obj_id in new_ids:
                    cond = state["output_dict_per_obj"][obj_idx]["cond_frame_outputs"]
                    if t in cond:
                        self._to_fp32(cond[t])

            return self._collect(t, only_ids=new_ids)

    # ------------------------------------------------------------------
    def _collect(self, t, only_ids=None):
        state = self.state
        ids, tensors = [], []

        for obj_idx in range(len(state["obj_ids"])):
            obj_id = state["obj_idx_to_id"][obj_idx]
            if only_ids is not None and obj_id not in only_ids:
                continue

            obj_output = state["output_dict_per_obj"][obj_idx]
            if t in obj_output["cond_frame_outputs"]:
                out = obj_output["cond_frame_outputs"][t]
            elif t in obj_output["non_cond_frame_outputs"]:
                out = obj_output["non_cond_frame_outputs"][t]
            else:
                continue

            ids.append(obj_id)
            tensors.append(out["pred_masks"].to(self.device).float())

        if not tensors:
            return {}

        _, video_res = self.predictor._get_orig_video_res_output(
            state, torch.cat(tensors, dim=0)
        )
        binary = (video_res > 0).squeeze(1).cpu().numpy()

        return {obj_id: binary[i] for i, obj_id in enumerate(ids)}

    def _prune(self, t):
        """Drop per-frame memory older than keep_frames so a live run is bounded."""
        cutoff = t - self.keep_frames
        state = self.state

        for obj_idx, obj_output in state["output_dict_per_obj"].items():
            non_cond = obj_output["non_cond_frame_outputs"]
            for k in [k for k in non_cond if k < cutoff]:
                del non_cond[k]

            tracked = state["frames_tracked_per_obj"][obj_idx]
            for k in [k for k in tracked if k < cutoff]:
                del tracked[k]