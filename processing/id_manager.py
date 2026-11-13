"""
Object-ID bookkeeping for the live pipeline (numpy only, no torch).

Rules (from the project handover):
  * IDs are monotonic and never recycled.
  * A detection is matched to an existing object using the object's actual
    SAM2 mask pixels (mask/box IoU), not box-vs-box overlap.
  * A new object needs NEW_PERSIST consecutive confirming YOLO runs. Chains
    are associated with box IoU. "Consecutive" counts YOLO runs, not camera
    frames, so it still works when YOLO runs every N frames.
  * A temporarily empty mask does not delete an identity.
"""

import numpy as np


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (
        max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        + max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        - inter
    )
    return inter / union if union > 0 else 0.0


def mask_box_iou(mask, box, mask_area=None):
    """
    IoU between a binary mask and a detection box, counted in mask pixels:
        inter = mask pixels inside the box
        union = box area + mask area - inter
    """
    h, w = mask.shape
    x1 = max(0, int(np.floor(box[0])))
    y1 = max(0, int(np.floor(box[1])))
    x2 = min(w, int(np.ceil(box[2])))
    y2 = min(h, int(np.ceil(box[3])))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    if mask_area is None:
        mask_area = int(mask.sum())

    inter = int(mask[y1:y2, x1:x2].sum())
    box_area = (x2 - x1) * (y2 - y1)
    union = box_area + mask_area - inter

    return inter / union if union > 0 else 0.0


def mask_to_box(mask):
    """XYXY box of a binary mask, or None if the mask is empty."""
    cols = np.flatnonzero(mask.any(axis=0))
    if cols.size == 0:
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    return (
        float(cols[0]),
        float(rows[0]),
        float(cols[-1] + 1),
        float(rows[-1] + 1),
    )


class IdManager:

    def __init__(
        self,
        match_iou=0.30,
        chain_iou=0.30,
        new_conf=0.25,
        new_persist=3,
        max_objects=6,
        min_mask_area=16,
    ):
        self.match_iou = match_iou
        self.chain_iou = chain_iou
        self.new_conf = new_conf
        self.new_persist = new_persist
        self.max_objects = max_objects
        self.min_mask_area = min_mask_area

        self.reset()

    def reset(self):
        self.next_id = 1            # monotonic, never reused
        self.active = []            # confirmed IDs, in creation order
        self.lost = {}              # id -> consecutive frames with empty mask
        self.candidates = []        # unconfirmed chains
        self.yolo_run = 0

    # ------------------------------------------------------------------
    def note_masks(self, masks):
        """Update per-object 'lost' counters from this frame's masks."""
        for obj_id in self.active:
            mask = masks.get(obj_id)
            area = 0 if mask is None else int(mask.sum())

            if area < self.min_mask_area:
                self.lost[obj_id] = self.lost.get(obj_id, 0) + 1
            else:
                self.lost[obj_id] = 0

    def needs_detection(self):
        """True when YOLO should run now regardless of the interval."""
        return bool(self.candidates) or any(
            count > 0 for count in self.lost.values()
        )

    # ------------------------------------------------------------------
    def update(self, detections, masks):
        """
        Process one YOLO run.

        detections: list of {"box": (x1,y1,x2,y2), "confidence": float}
        masks:      {obj_id: bool/uint8 mask (H, W)} for existing objects

        Returns a list of (new_obj_id, box) that must be seeded into SAM2
        on the current frame.
        """
        self.yolo_run += 1

        areas = {obj_id: int(m.sum()) for obj_id, m in masks.items()}

        # 1) Detections that land on an existing object's mask pixels.
        unmatched = []
        for det in detections:
            matched = False
            for obj_id, mask in masks.items():
                if areas[obj_id] < self.min_mask_area:
                    continue
                if mask_box_iou(mask, det["box"], areas[obj_id]) >= self.match_iou:
                    matched = True
                    break
            if not matched:
                unmatched.append(det)

        # 2) Chain the rest into candidates (previous YOLO run only).
        previous = [
            c for c in self.candidates
            if c["last_run"] == self.yolo_run - 1
        ]
        used = set()
        next_candidates = []
        confirmed = []

        for det in sorted(unmatched, key=lambda d: -d["confidence"]):
            if det["confidence"] < self.new_conf:
                continue

            best, best_iou = None, 0.0
            for k, cand in enumerate(previous):
                if k in used:
                    continue
                iou = box_iou(det["box"], cand["box"])
                if iou >= self.chain_iou and iou > best_iou:
                    best, best_iou = k, iou

            hits = 1
            if best is not None:
                used.add(best)
                hits = previous[best]["hits"] + 1

            if hits >= self.new_persist:
                if len(self.active) + len(confirmed) < self.max_objects:
                    obj_id = self.next_id
                    self.next_id += 1
                    confirmed.append((obj_id, det["box"]))
                # at capacity: drop the candidate
                continue

            next_candidates.append({
                "box": det["box"],
                "hits": hits,
                "last_run": self.yolo_run,
            })

        self.candidates = next_candidates

        for obj_id, _ in confirmed:
            self.active.append(obj_id)
            self.lost[obj_id] = 0

        return confirmed