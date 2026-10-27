import cv2
import numpy as np

COLORS = {"OK": (80, 175, 76), "LOW": (0, 152, 255), "EMPTY": (54, 67, 244), None: (160, 160, 160)}


def draw(frame, trays, readings, levels, person_boxes=()):
    img = frame.copy()
    h, w = img.shape[:2]
    scale = w / 1920
    thick = max(2, int(3 * scale))
    overlay = img.copy()
    for t in trays:
        pts, (x, y, bw, bh), _ = t.geometry(h, w)
        r, lvl = readings.get(t.name), levels[t.name]
        color = COLORS[lvl.state]
        if r is not None and r.mask is not None and not r.occluded:
            roi = overlay[y:y + bh, x:x + bw]
            roi[r.mask] = (0.5 * roi[r.mask] + 0.5 * np.array(color)).astype(np.uint8)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    for x1, y1, x2, y2 in person_boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (200, 200, 200), thick)

    for t in trays:
        pts, (x, y, bw, bh), _ = t.geometry(h, w)
        r, lvl = readings.get(t.name), levels[t.name]
        color = COLORS[lvl.state]
        blocked = r is not None and r.occluded
        # Blocked trays get a grey outline; the label keeps the last known level.
        cv2.polylines(img, [pts], True, COLORS[None] if blocked else color, thick)
        pct = f"{lvl.smoothed:.0%}" if lvl.smoothed is not None else "--"
        text = f"{t.label}: {pct} {lvl.state or ''}"
        fs = 0.8 * scale
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
        ty = max(y - 8, th + 8)
        cv2.rectangle(img, (x, ty - th - 8), (x + tw + 10, ty + base), color, -1)
        cv2.putText(img, text, (x + 5, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), thick, cv2.LINE_AA)
    return img


def with_panel(img, trays, levels, readings, alerts, clock="", height=720):
    """Video at `height` px plus a side panel with tray bars and recent alerts (demo output)."""
    h, w = img.shape[:2]
    view = cv2.resize(img, (int(w * height / h), height), interpolation=cv2.INTER_AREA)
    pw = 380
    panel = np.full((height, pw, 3), (39, 31, 25), np.uint8)
    font, white, grey = cv2.FONT_HERSHEY_SIMPLEX, (255, 255, 255), (170, 170, 170)
    cv2.putText(panel, "STOCK LEVELS", (18, 40), font, 0.85, white, 2, cv2.LINE_AA)
    cv2.putText(panel, clock, (18, 68), font, 0.55, grey, 1, cv2.LINE_AA)
    y = 105
    for t in trays:
        lvl, r = levels[t.name], readings.get(t.name)
        pct = lvl.smoothed if lvl.smoothed is not None else 0.0
        status = lvl.state or "--"
        if r is not None and r.occluded:
            status += " (blocked)"
        cv2.putText(panel, t.label, (18, y), font, 0.6, white, 1, cv2.LINE_AA)
        cv2.putText(panel, f"{pct:.0%}", (pw - 70, y), font, 0.6, white, 1, cv2.LINE_AA)
        cv2.rectangle(panel, (18, y + 10), (pw - 18, y + 26), (70, 60, 55), -1)
        cv2.rectangle(panel, (18, y + 10), (18 + int((pw - 36) * pct), y + 26), COLORS[lvl.state], -1)
        cv2.putText(panel, status, (18, y + 46), font, 0.5, COLORS[lvl.state] if lvl.state else grey, 1, cv2.LINE_AA)
        y += 78
    y = max(y + 10, height - 30 - 26 * 7)
    cv2.putText(panel, "ALERTS", (18, y), font, 0.7, white, 2, cv2.LINE_AA)
    for i, (when, text, state) in enumerate(list(alerts)[-7:][::-1]):
        cv2.putText(panel, f"{when}  {text}", (18, y + 30 + 26 * i), font, 0.47, COLORS[state], 1, cv2.LINE_AA)
    return np.hstack([view, panel])


def crop_tray(img, tray, pad=0.15):
    """Crop around one tray (used for alert photos)."""
    h, w = img.shape[:2]
    _, (x, y, bw, bh), _ = tray.geometry(h, w)
    px, py = int(bw * pad), int(bh * pad)
    return img[max(y - py - int(0.06 * h), 0):min(y + bh + py, h), max(x - px, 0):min(x + bw + px, w)]
