#!/usr/bin/env bash
# Set up the stock monitor on Jetson Orin Nano, JetPack 6.2.x (L4T r36.4, CUDA 12.6).
# Run from the project folder:  bash scripts/setup_jetson.sh
set -euo pipefail

JP_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/"

echo "== Device =="
cat /etc/nv_tegra_release || true
nvcc --version 2>/dev/null | tail -n1 || echo "nvcc not on PATH (fine if /usr/local/cuda exists)"

echo "== System packages =="
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv python3-dev libopenblas-dev \
  gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad

# --system-site-packages keeps JetPack's OpenCV (built with GStreamer) and TensorRT bindings visible.
python3 -m venv --system-site-packages .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip

echo "== PyTorch / torchvision / onnxruntime-gpu built for Orin (sm_87) from Jetson AI Lab =="
pip install --extra-index-url "$JP_INDEX" torch torchvision onnxruntime-gpu

echo "== Ultralytics + app deps =="
pip install "numpy<2" pyyaml ultralytics onnx onnxslim

# Ultralytics pulls opencv-python from PyPI, which has no GStreamer and shadows JetPack's
# OpenCV. Remove it so RTSP hardware decode works. (Snapshot mode works either way.)
pip uninstall -y opencv-python opencv-python-headless || true

echo "== Checks =="
python - <<'PY'
import cv2, numpy, torch
print("numpy", numpy.__version__)
print("opencv", cv2.__version__, "| GStreamer:",
      "YES" if "GStreamer:                   YES" in cv2.getBuildInformation() else "check getBuildInformation()")
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
try:
    import tensorrt
    print("tensorrt", tensorrt.__version__)
except ImportError:
    print("tensorrt python bindings missing: sudo apt install python3-libnvinfer")
PY

echo
echo "If 'import cv2' failed above, install JetPack's OpenCV: sudo apt install nvidia-opencv"
echo "Next: bash scripts/export_engines.sh"
