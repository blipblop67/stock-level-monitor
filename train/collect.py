#!/usr/bin/env python3
"""Collect training images from the camera for the YOLO-seg model.

    python3 train/collect.py --every 120 --hours 06:00-14:00

Aim for 300-1000 images covering full, half and empty trays, different products,
morning/afternoon light, wet/dry trays, and hands in view. Then label with polygons.
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stockmon.camera import make_source  # noqa: E402
from stockmon.config import load_config  # noqa: E402


def in_hours(spec):
    if not spec:
        return True
    start, end = spec.split("-")
    now = datetime.now().strftime("%H:%M")
    return start <= now < end if start < end else now >= start or now < end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="dataset/raw")
    ap.add_argument("--every", type=float, default=120, help="seconds between images")
    ap.add_argument("--hours", default="", help="only collect in this window, e.g. 06:00-14:00")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = make_source(cfg)
    print(f"Saving to {out} every {args.every:.0f} s (Ctrl+C to stop)")
    try:
        while True:
            if in_hours(args.hours):
                try:
                    img = src.read()
                    path = out / f"{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    print("saved", path)
                except Exception as e:  # noqa: BLE001
                    print("grab failed:", e)
            time.sleep(args.every)
    except KeyboardInterrupt:
        pass
    finally:
        src.close()


if __name__ == "__main__":
    main()
