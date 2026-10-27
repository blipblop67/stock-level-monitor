#!/usr/bin/env python3
"""Set up tray regions and the empty-tray reference.

    python3 roi_tool.py snapshot            # save a frame to refs/frame.jpg
    python3 roi_tool.py draw                # click tray polygons (needs a display)
    python3 roi_tool.py preview             # draw saved trays onto a frame (headless OK)
    python3 roi_tool.py reference           # save current frame as the empty-tray reference
    python3 roi_tool.py calibrate           # with trays freshly stocked: record "100% full"

Polygons are stored normalised (0-1), so you can draw them on a laptop using a copied
snapshot and then copy trays.yaml back to the Jetson.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from stockmon.camera import make_source
from stockmon.config import load_config, resolve, save_trays
from stockmon.geometry import Tray


def grab(cfg, at=None):
    """One frame from the configured source; `at` = seconds into a video file."""
    src = make_source(cfg)
    try:
        if at is not None and hasattr(src, "seek"):
            src.seek(at)
        return src.read()
    finally:
        src.close()


def cmd_snapshot(cfg, args):
    out = resolve(cfg, args.output or "refs/frame.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    img = grab(cfg, args.at)
    cv2.imwrite(str(out), img)
    print(f"Saved {img.shape[1]}x{img.shape[0]} frame to {out}")


def cmd_reference(cfg, args):
    out = resolve(cfg, args.output or cfg.get("reference", {}).get("image", "refs/empty.jpg"))
    out.parent.mkdir(parents=True, exist_ok=True)
    img = grab(cfg, args.at)
    cv2.imwrite(str(out), img)
    print(f"Saved empty-tray reference to {out}\n"
          "Make sure trays were in their 'empty' state and nobody was in view.")


def cmd_calibrate(cfg, args):
    """Measure product coverage on freshly stocked trays and store it as '100% full'."""
    from stockmon.estimators import make_estimator

    img = load_image(cfg, args) if args.image else grab(cfg, args.at)
    trays = [Tray(t) for t in cfg["trays"]]
    readings = make_estimator(cfg).estimate(img, trays)
    for d in cfg["trays"]:
        cov = readings[d["name"]].fill
        if cov < 0.05:
            print(f"  {d['name']}: coverage {cov:.1%} looks empty - skipped (is it stocked?)")
            continue
        d["full_coverage"] = round(cov, 3)
        print(f"  {d['name']} ({d.get('label', '')}): full_coverage = {cov:.1%}")
    save_trays(cfg["_trays_file"], cfg["trays"])
    print(f"Saved to {cfg['_trays_file']}")


def load_image(cfg, args):
    path = resolve(cfg, args.image or "refs/frame.jpg")
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"Cannot read {path}. Run: python3 roi_tool.py snapshot")
    return img


def render(img, trays, current=None):
    view = img.copy()
    h, w = img.shape[:2]
    thick = max(2, w // 640)
    for t in trays:
        tray = Tray(t)
        pts, (x, y, _, _), _ = tray.geometry(h, w)
        cv2.polylines(view, [pts], True, (0, 200, 0), thick)
        cv2.putText(view, tray.label, (x + 6, y + 30 * thick // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * thick, (0, 255, 0), thick, cv2.LINE_AA)
    if current:
        pts = np.round(np.array(current) * [w - 1, h - 1]).astype(np.int32)
        cv2.polylines(view, [pts], False, (0, 200, 255), thick)
        for p in pts:
            cv2.circle(view, tuple(int(v) for v in p), thick * 3, (0, 200, 255), -1)
    return view


def cmd_preview(cfg, args):
    img = load_image(cfg, args)
    out = resolve(cfg, args.output or "output/trays_preview.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), render(img, cfg["trays"]))
    print(f"Saved preview with {len(cfg['trays'])} trays to {out}")


def cmd_draw(cfg, args):
    img = load_image(cfg, args)
    h, w = img.shape[:2]
    scale = min(1.0, 1600 / w, 900 / h)
    trays = list(cfg["trays"])
    current = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append([round(x / scale / (w - 1), 4), round(y / scale / (h - 1), 4)])
        elif event == cv2.EVENT_RBUTTONDOWN and current:
            current.pop()

    win = "trays: L-click add point | R-click undo | ENTER finish tray | d delete last | s save | q quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = render(img, trays, current)
        cv2.imshow(win, cv2.resize(view, (int(w * scale), int(h * scale))))
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10) and len(current) >= 3:
            n = len(trays) + 1
            name = input(f"Tray id [tray_{n}]: ").strip() or f"tray_{n}"
            label = input(f"Display label (e.g. Ikan Kembung) [{name}]: ").strip() or name
            trays.append({"name": name, "label": label, "polygon": current[:]})
            current.clear()
        elif key == ord("d") and trays:
            print(f"Deleted {trays.pop()['name']}")
        elif key == ord("s"):
            save_trays(cfg["_trays_file"], trays)
            print(f"Saved {len(trays)} trays to {cfg['_trays_file']}")
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["snapshot", "draw", "preview", "reference", "calibrate"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--image", help="frame to draw on / preview (default refs/frame.jpg)")
    ap.add_argument("-o", "--output")
    ap.add_argument("--at", type=float, help="for video files: take the frame at this many seconds")
    args = ap.parse_args()
    cfg = load_config(args.config)
    {"snapshot": cmd_snapshot, "draw": cmd_draw, "preview": cmd_preview, "reference": cmd_reference,
     "calibrate": cmd_calibrate}[args.command](cfg, args)


if __name__ == "__main__":
    main()
