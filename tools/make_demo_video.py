#!/usr/bin/env python3
"""Generate a synthetic 'morning at the stall' test video.

    python3 tools/make_demo_video.py
    python3 roi_tool.py calibrate --config demo/config_video.yaml --at 0
    python3 app.py --config demo/config_video.yaml --fast --show --save-video demo/output/annotated.mp4

Customers' hands reach in and take items, one tray is restocked half way, the
lighting drifts and flickers. Writes demo/video/stall_demo.mp4, empty.jpg and trays.yaml.
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "demo" / "video"
W, H, FPS = 1280, 720, 15
TRAYS = [  # name, label, (x, y, w, h), product colour (BGR), items when full, share of picks
    ("tray_1", "Ikan Kembung", (100, 300, 240, 210), (150, 140, 120), 24, 0.30),
    ("tray_2", "Udang", (390, 300, 240, 210), (90, 120, 230), 30, 0.45),
    ("tray_3", "Sotong", (680, 300, 240, 210), (175, 160, 205), 18, 0.15),
    ("tray_4", "Ayam", (970, 300, 240, 210), (140, 175, 235), 20, 0.10),
]
ARM_FRAMES = 24  # one reach in-and-out
SKIN = (120, 150, 200)


def background(rng):
    img = np.full((H, W, 3), (70, 85, 95), np.uint8)
    cv2.rectangle(img, (0, 260), (W, H), (120, 130, 135), -1)
    cv2.putText(img, "GERAI IKAN SEGAR", (430, 120), cv2.FONT_HERSHEY_DUPLEX, 1.6, (200, 210, 220), 3, cv2.LINE_AA)
    img = cv2.add(img, rng.integers(0, 12, img.shape, dtype=np.uint8))
    for _, _, (x, y, w, h), _, _, _ in TRAYS:
        cv2.rectangle(img, (x, y), (x + w, y + h), (235, 238, 240), -1)
        for _ in range(160):
            cx, cy = rng.integers(x + 4, x + w - 4), rng.integers(y + 4, y + h - 4)
            cv2.circle(img, (int(cx), int(cy)), int(rng.integers(2, 7)), (250, 252, 255), -1)
    return img


def item_layout(tray, seed):
    """Fixed positions for every item a full tray holds."""
    rng = np.random.default_rng(seed)
    _, _, (x, y, w, h), color, n, _ = tray
    items = []
    for _ in range(n):
        items.append(((int(rng.integers(x + 22, x + w - 22)), int(rng.integers(y + 15, y + h - 15))),
                      (int(rng.integers(16, 27)), int(rng.integers(7, 11))), float(rng.integers(0, 180)),
                      tuple(int(v + rng.integers(-15, 15)) for v in color)))
    return items


def schedule(rng, total):
    """List of (start_frame, tray_index, delta_items)."""
    events, f = [], 2 * FPS
    weights = np.array([t[5] for t in TRAYS])
    while f < total - ARM_FRAMES - FPS:
        if abs(f - total // 2) < 15:                      # restock Sotong half way
            events.append((f, 2, +99))
        else:
            events.append((f, int(rng.choice(len(TRAYS), p=weights / weights.sum())), -int(rng.integers(1, 3))))
        f += int(rng.integers(26, 36))
    return events


def draw_arm(img, tray, k, rng_seed):
    """Arm reaching from the bottom edge to the tray centre; k = frame in the reach."""
    _, _, (x, y, w, h), _, _, _ = tray
    p = math.sin(math.pi * k / ARM_FRAMES)
    r = np.random.default_rng(rng_seed)
    bx, by = int(x + w / 2 + r.integers(-150, 150)), H + 40
    tx, ty = int(x + w / 2 + r.integers(-40, 40)), int(y + h / 2 + r.integers(-30, 30))
    hx, hy = int(bx + p * (tx - bx)), int(by + p * (ty - by))
    cv2.line(img, (bx, by), (hx, hy), SKIN, 64)
    cv2.circle(img, (hx, hy), 40, (125, 158, 210), -1)
    cv2.line(img, (bx, by), (hx, hy), (105, 135, 185), 6)  # some shading/texture


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=100)
    args = ap.parse_args()
    total = args.seconds * FPS
    rng = np.random.default_rng(11)

    ROOT.mkdir(parents=True, exist_ok=True)
    base = background(rng)
    cv2.imwrite(str(ROOT / "empty.jpg"), base)
    layouts = [item_layout(t, 200 + i) for i, t in enumerate(TRAYS)]
    counts = [t[4] for t in TRAYS]
    events = {start: (ti, d) for start, ti, d in schedule(rng, total)}
    active = None  # (start, tray_index, delta)

    out = cv2.VideoWriter(str(ROOT / "stall_demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in range(total):
        if f in events:
            active = (f, *events[f])
        img = base.copy()
        if active and f - active[0] == ARM_FRAMES // 2:   # items change at full reach
            ti, d = active[1], active[2]
            counts[ti] = min(TRAYS[ti][4], max(0, counts[ti] + d))
        for ti, layout in enumerate(layouts):
            for center, axes, angle, color in layout[:counts[ti]]:
                cv2.ellipse(img, center, axes, angle, 0, 360, color, -1)
        if active:
            k = f - active[0]
            if k < ARM_FRAMES:
                draw_arm(img, TRAYS[active[1]], k, active[0])
            else:
                active = None
        gain = 1 + 0.05 * math.sin(2 * math.pi * f / (40 * FPS))
        if rng.random() < 0.01:                          # fluorescent flicker
            gain *= 0.92
        img = img.astype(np.int16) * gain + rng.integers(-3, 4, img.shape)
        out.write(np.clip(img, 0, 255).astype(np.uint8))
    out.release()

    lines = ["trays:"]
    for name, label, (x, y, w, h), *_ in TRAYS:
        poly = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        norm = ", ".join(f"[{px / (W - 1):.4f}, {py / (H - 1):.4f}]" for px, py in poly)
        lines += [f"  - name: {name}", f"    label: {label}", f"    polygon: [{norm}]"]
    (ROOT / "trays.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {ROOT / 'stall_demo.mp4'} ({args.seconds} s, {len(events)} customer events)")
    print("Final items per tray:", dict(zip([t[1] for t in TRAYS], counts)))


if __name__ == "__main__":
    main()
