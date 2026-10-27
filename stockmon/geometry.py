import cv2
import numpy as np


class Tray:
    """A display tray defined by a normalised polygon (0-1 coordinates)."""

    def __init__(self, d):
        self.name = d["name"]
        self.label = d.get("label", self.name)
        self.polygon = np.asarray(d["polygon"], dtype=np.float32)
        # Coverage measured on a freshly stocked tray (roi_tool.py calibrate) = 100% full.
        self.full_coverage = float(d.get("full_coverage", 1.0))
        self._cache = {}

    def fill_from_coverage(self, coverage):
        return min(1.0, coverage / max(self.full_coverage, 1e-6))

    def geometry(self, h, w):
        """Pixel polygon, bounding box (x, y, w, h) and the polygon mask inside that box."""
        key = (h, w)
        if key not in self._cache:
            pts = np.round(self.polygon * [w - 1, h - 1]).astype(np.int32)
            x, y, bw, bh = cv2.boundingRect(pts)
            local = np.zeros((bh, bw), np.uint8)
            cv2.fillPoly(local, [pts - [x, y]], 255)
            self._cache[key] = (pts, (x, y, bw, bh), local > 0)
        return self._cache[key]

    def crop(self, img):
        _, (x, y, bw, bh), local = self.geometry(*img.shape[:2])
        return img[y:y + bh, x:x + bw], local


def box_overlap(tray, boxes, shape):
    """Share of the tray polygon covered by any of the given xyxy boxes."""
    if len(boxes) == 0:
        return 0.0
    _, (x, y, bw, bh), local = tray.geometry(*shape[:2])
    covered = np.zeros_like(local)
    for x1, y1, x2, y2 in boxes:
        x1, x2 = int(max(x1 - x, 0)), int(min(x2 - x, bw))
        y1, y2 = int(max(y1 - y, 0)), int(min(y2 - y, bh))
        if x2 > x1 and y2 > y1:
            covered[y1:y2, x1:x2] = True
    return float((covered & local).sum()) / max(int(local.sum()), 1)
