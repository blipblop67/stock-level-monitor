"""Fill-level estimators. Each returns {tray_name: Reading}."""
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import resolve

log = logging.getLogger(__name__)


@dataclass
class Reading:
    fill: float | None              # share of tray area covered by product, None when unreadable
    mask: np.ndarray | None = None  # product pixels inside the tray bounding box
    occluded: bool = False


def _clean(mask, min_px):
    """Morphological cleanup + removal of blobs smaller than min_px."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[labels]


class ReferenceEstimator:
    """Fill = share of tray pixels that look different from an 'empty tray' photo.

    No training needed. Capture the reference when every tray is in the state you
    call empty (e.g. tray with ice but no fish). Re-capture if the camera or trays move.
    """

    def __init__(self, cfg):
        c = cfg.get("reference", {})
        path = resolve(cfg, c.get("image", "refs/empty.jpg"))
        self.ref = cv2.imread(str(path))
        if self.ref is None:
            raise RuntimeError(f"Reference image not found: {path}. Run: python roi_tool.py reference")
        self.thr = float(c.get("diff_threshold", 22))
        self.lw = float(c.get("lightness_weight", 0.5))
        self.blur = int(c.get("blur", 5)) | 1
        self.min_blob_frac = float(c.get("min_blob_frac", 0.003))
        self.exposure_comp = c.get("exposure_compensation", True)
        self._ref_lab = {}
        self._bg = {}

    def _background(self, shape, trays):
        """Static scene outside the trays (subsampled), used to undo camera exposure changes."""
        key = shape[:2]
        if key not in self._bg:
            m = np.ones(key, np.uint8)
            for t in trays:
                pts, _, _ = t.geometry(*key)
                cv2.fillPoly(m, [pts], 0)
            self._bg[key] = m[::4, ::4] > 0
        return self._bg[key]

    def _lab(self, img):
        img = cv2.GaussianBlur(img, (self.blur, self.blur), 0)
        return cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)

    def _reference_lab(self, shape):
        key = shape[:2]
        if key not in self._ref_lab:
            ref = self.ref
            if ref.shape[:2] != key:
                ref = cv2.resize(ref, (key[1], key[0]), interpolation=cv2.INTER_AREA)
            self._ref_lab[key] = self._lab(ref)
        return self._ref_lab[key]

    def estimate(self, frame, trays):
        cur_lab, ref_lab = self._lab(frame), self._reference_lab(frame.shape)
        if self.exposure_comp:
            bg = self._background(frame.shape, trays)
            cur_l = np.median(cur_lab[::4, ::4, 0][bg])
            if cur_l > 1:
                cur_lab[..., 0] *= np.median(ref_lab[::4, ::4, 0][bg]) / cur_l
        out = {}
        for t in trays:
            cur, local = t.crop(cur_lab)
            ref, _ = t.crop(ref_lab)
            d = cur - ref
            d[..., 0] *= self.lw
            delta = np.sqrt((d ** 2).sum(axis=2))
            min_px = max(int(local.sum() * self.min_blob_frac), 1)
            product = _clean((delta > self.thr) & local, min_px) & local
            out[t.name] = Reading(float(product.sum()) / max(int(local.sum()), 1), product)
        return out


class YoloSegEstimator:
    """Fill = share of tray pixels covered by product masks from a custom YOLO-seg model."""

    def __init__(self, cfg):
        from ultralytics import YOLO

        c = cfg.get("yolo_seg", {})
        path = resolve(cfg, c.get("model", "models/stock-seg.engine"))
        if not Path(path).exists():
            raise RuntimeError(f"Segmentation model not found: {path}")
        self.model = YOLO(str(path), task="segment")
        self.imgsz = int(c.get("imgsz", 960))
        self.conf = float(c.get("conf", 0.35))
        wanted = set(c.get("product_classes") or [])
        self.classes = [i for i, n in self.model.names.items() if n in wanted] if wanted else None

    def estimate(self, frame, trays):
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, classes=self.classes,
                                 retina_masks=True, verbose=False)[0]
        full = np.zeros(frame.shape[:2], bool)
        if res.masks is not None:
            full = (res.masks.data.cpu().numpy() > 0.5).any(axis=0)
            if full.shape != frame.shape[:2]:
                full = cv2.resize(full.astype(np.uint8), (frame.shape[1], frame.shape[0])) > 0
        out = {}
        for t in trays:
            m, local = t.crop(full)
            product = m & local
            out[t.name] = Reading(float(product.sum()) / max(int(local.sum()), 1), product)
        return out


def make_estimator(cfg):
    kind = cfg.get("estimator", "reference")
    if kind == "reference":
        return ReferenceEstimator(cfg)
    if kind == "yolo_seg":
        return YoloSegEstimator(cfg)
    raise ValueError(f"Unknown estimator: {kind}")
