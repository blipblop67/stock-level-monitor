#!/usr/bin/env bash
# Build TensorRT FP16 engines ON the Jetson (engines are device/TensorRT-version specific).
#   bash scripts/export_engines.sh
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[ -f .venv/bin/activate ] && source .venv/bin/activate
mkdir -p models

# 1) Person detector for occlusion checks (COCO, downloads yolo11n.pt).
if [ ! -f models/yolo11n.engine ]; then
  yolo export model=yolo11n.pt format=engine half=True imgsz=640 device=0
  mv yolo11n.engine models/
fi

# 2) Custom stock segmentation model, if you have trained one (models/stock-seg.pt).
if [ -f models/stock-seg.pt ]; then
  yolo export model=models/stock-seg.pt format=engine half=True imgsz=960 device=0
  echo "Set 'estimator: yolo_seg' in config.yaml to use models/stock-seg.engine"
fi
ls -lh models
