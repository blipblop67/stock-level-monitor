#!/usr/bin/env python3
"""Train the stock segmentation model.

Train on a PC with an NVIDIA GPU or Google Colab - the Orin Nano can train,
but it is slow. Then copy runs/segment/stock/weights/best.pt to the Jetson
as models/stock-seg.pt and run scripts/export_engines.sh there.

    pip install ultralytics
    python train/train_seg.py --data train/data.yaml --epochs 100
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train/data.yaml")
    ap.add_argument("--model", default="yolo11n-seg.pt", help="yolo11s-seg.pt for more accuracy")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=960, help="keep equal to yolo_seg.imgsz in config.yaml")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        name="stock", patience=25,
        # Wet market lighting varies a lot; keep colour augmentation moderate so
        # fish/meat colours stay meaningful.
        hsv_h=0.01, hsv_s=0.4, hsv_v=0.4, degrees=3, fliplr=0.5, mosaic=1.0,
    )
    metrics = model.val()
    print("mask mAP50-95:", metrics.seg.map)


if __name__ == "__main__":
    main()
