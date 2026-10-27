#!/usr/bin/env python3
"""Generate a synthetic stall scene so the pipeline can be tried without a camera.

    python3 tools/make_demo.py
    python3 app.py --config demo/config_demo.yaml --iterations 12

Creates demo/empty.jpg (reference), demo/frames/*.jpg (trays selling out over the
morning, with lighting drift and a hand passing over one tray) and demo/trays.yaml.
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "demo"
W, H = 1920, 1080
TRAYS = [  # name, label, (x, y, w, h) in pixels, product colour (BGR), starting count
    ("tray_1", "Ikan Kembung", (160, 480, 360, 300), (150, 140, 120), 40),
    ("tray_2", "Udang", (580, 480, 360, 300), (90, 120, 230), 60),
    ("tray_3", "Sotong", (1000, 480, 360, 300), (175, 160, 205), 30),
    ("tray_4", "Ayam", (1420, 480, 360, 300), (140, 175, 235), 25),
]
FRAMES = 12


def background(rng):
    img = np.full((H, W, 3), (70, 85, 95), np.uint8)              # wall
    cv2.rectangle(img, (0, 420), (W, H), (120, 130, 135), -1)       # stainless counter
    img = cv2.add(img, rng.integers(0, 12, img.shape, dtype=np.uint8))
    for _, _, (x, y, w, h), _, _ in TRAYS:                          # white trays with ice
        cv2.rectangle(img, (x, y), (x + w, y + h), (235, 238, 240), -1)
        for _ in range(250):
            cx, cy = rng.integers(x + 5, x + w - 5), rng.integers(y + 5, y + h - 5)
            cv2.circle(img, (int(cx), int(cy)), int(rng.integers(3, 9)), (250, 252, 255), -1)
    return img


def draw_products(img, tray, count, rng):
    _, _, (x, y, w, h), color, _ = tray
    for _ in range(count):
        cx, cy = int(rng.integers(x + 30, x + w - 30)), int(rng.integers(y + 20, y + h - 20))
        axes = (int(rng.integers(22, 38)), int(rng.integers(9, 15)))
        c = tuple(int(v + rng.integers(-15, 15)) for v in color)
        cv2.ellipse(img, (cx, cy), axes, float(rng.integers(0, 180)), 0, 360, c, -1)
        cv2.ellipse(img, (cx, cy), axes, 0, 0, 0, (40, 40, 40), 1)


def main():
    rng = np.random.default_rng(7)
    (ROOT / "frames").mkdir(parents=True, exist_ok=True)
    base = background(rng)
    cv2.imwrite(str(ROOT / "empty.jpg"), base)

    for i in range(FRAMES):
        img = base.copy()
        frac = 1 - i / (FRAMES - 1)
        for k, tray in enumerate(TRAYS):
            start = tray[4]
            # Each tray sells at a different speed; tray_3 gets restocked half way.
            speed = [0.9, 1.0, 1.2, 0.4][k]
            remaining = max(0.0, 1 - speed * (1 - frac))
            if k == 2 and i >= 8:
                remaining = 0.9
            draw_products(img, tray, int(start * remaining), np.random.default_rng(100 + k))
        if i == 5:  # a hand reaching over tray 2
            cv2.ellipse(img, (760, 560), (190, 60), 25, 0, 360, (120, 150, 200), -1)
        gain = 1 + 0.06 * np.sin(i / 2)  # lighting drift
        img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        cv2.imwrite(str(ROOT / "frames" / f"frame_{i:03d}.jpg"), img)

    lines = ["trays:"]
    for name, label, (x, y, w, h), _, _ in TRAYS:
        poly = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        norm = ", ".join(f"[{px / (W - 1):.4f}, {py / (H - 1):.4f}]" for px, py in poly)
        lines += [f"  - name: {name}", f"    label: {label}", f"    polygon: [{norm}]"]
    (ROOT / "trays.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Demo written to {ROOT}")


if __name__ == "__main__":
    main()
