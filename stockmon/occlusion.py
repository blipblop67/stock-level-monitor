"""Occlusion checks: skip a tray's reading while hands/customers are over it.

Two detectors, usable together:
  person : YOLO person boxes (needs the model, catches people standing still)
  motion : frame differencing between two frames a moment apart (no model needed)
After anything was seen over a tray, its readings stay blocked for `settle_readings`
more readings so half-finished picking does not count.
"""
import logging
from pathlib import Path

import cv2
import numpy as np

from .config import resolve
from .geometry import box_overlap

log = logging.getLogger(__name__)


class PersonDetector:
    def __init__(self, cfg):
        from ultralytics import YOLO

        c = cfg.get("occlusion", {})
        path = resolve(cfg, c.get("model", "models/yolo11n.engine"))
        if not Path(path).exists():
            log.warning("%s not found, using yolo11n.pt (export a TensorRT engine for speed)", path)
            path = "yolo11n.pt"
        self.model = YOLO(str(path), task="detect")
        self.imgsz = int(c.get("imgsz", 640))
        self.conf = float(c.get("conf", 0.4))

    def boxes(self, frame):
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, classes=[0], verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 4))
        return res.boxes.xyxy.cpu().numpy()


class MotionDetector:
    def __init__(self, cfg):
        c = cfg.get("occlusion", {})
        self.thr = int(c.get("motion_threshold", 25))
        self.dilate = float(c.get("motion_dilate", 0.03))
        self.gap = float(c.get("motion_gap_sec", 0.3))
        self.max_frame_frac = float(c.get("motion_max_frame_frac", 0.4))

    def mask(self, a, b):
        ga = cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        gb = cv2.GaussianBlur(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        # Undo whole-frame brightness changes (light flicker, auto exposure) first.
        gb = cv2.convertScaleAbs(gb, alpha=float(np.median(ga)) / max(float(np.median(gb)), 1.0))
        m = (cv2.absdiff(ga, gb) > self.thr).astype(np.uint8)
        if m.mean() > self.max_frame_frac:
            return np.zeros(m.shape, bool)  # global change, not a hand
        # A moving arm only differs at its edges; dilate to cover the whole arm.
        k = max(3, int(self.dilate * a.shape[1]) | 1)
        return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0


def motion_overlap(tray, mask):
    m, local = tray.crop(mask)
    return float((m & local).sum()) / max(int(local.sum()), 1)


class OcclusionGate:
    """Combines detectors and the settle delay into one 'is this tray blocked?' answer."""

    def __init__(self, cfg, trays):
        c = cfg.get("occlusion", {})
        self.max_overlap = float(c.get("max_overlap", 0.10))
        self.settle = int(c.get("settle_readings", 1))
        self.person = None
        if c.get("person", c.get("enabled", False)):
            try:
                self.person = PersonDetector(cfg)
            except Exception as e:  # noqa: BLE001 - keep running without person checks
                log.warning("Person check disabled: %s", e)
        self.motion = MotionDetector(cfg) if c.get("motion", False) else None
        self._quiet = {t.name: 10 ** 6 for t in trays}  # readings since something was over the tray
        self.person_boxes, self.motion_mask = np.zeros((0, 4)), None

    @property
    def needs_second_frame(self):
        return self.motion is not None

    def update(self, frame, frame2, trays):
        """Returns {tray_name: blocked}."""
        self.person_boxes = self.person.boxes(frame) if self.person else np.zeros((0, 4))
        self.motion_mask = self.motion.mask(frame, frame2) if self.motion and frame2 is not None else None
        blocked = {}
        for t in trays:
            hit = box_overlap(t, self.person_boxes, frame.shape) > self.max_overlap
            if self.motion_mask is not None:
                hit = hit or motion_overlap(t, self.motion_mask) > self.max_overlap
            self._quiet[t.name] = 0 if hit else self._quiet[t.name] + 1
            blocked[t.name] = self._quiet[t.name] <= self.settle
        return blocked
