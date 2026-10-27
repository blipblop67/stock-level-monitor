# Stock & Display Level Monitor
**Jetson Orin Nano (JetPack 6.2.2) + Reolink P437 - wet market stall**

Watches the display trays at a stall, estimates how full each tray is, and alerts the owner (Telegram) when a tray is **LOW** or **EMPTY**, and again when it is restocked. It includes a small web dashboard and a CSV log for daily sales and restock patterns.

```
Reolink P437 ──PoE switch──► Jetson Orin Nano
   HTTP snapshot (4K JPEG every 10 s)  or  RTSP (NVDEC hardware decode)
        │
        ├─ occlusion: motion check (no model) + YOLO11n person detector → skip covered trays
        ├─ fill estimator per tray polygon
        │     reference : compare with "empty tray" photo (no training, day 1)
        │     yolo_seg  : custom YOLO segmentation of product (robust, needs data)
        ├─ median smoothing + OK/LOW/EMPTY state machine (hysteresis, confirmations)
        └─ Telegram alert (with photo) · dashboard :8080 · output/levels.csv
```

Stock changes over minutes, not milliseconds, so the default reads **one full-res snapshot every 10 s**. The Jetson stays cool and nearly idle, and no video decoding is needed.

## 1. Demo without an IP camera (PC or Jetson)
No camera and no AI model needed. Works on Windows, a laptop, or the Jetson itself.

### A. Generated stall video (quickest)
```bash
pip install -r requirements.txt            # on Jetson: run scripts/setup_jetson.sh instead
python3 tools/make_demo_video.py           # demo/video/stall_demo.mp4 (100 s), empty.jpg, trays.yaml
python3 roi_tool.py calibrate --config demo/config_video.yaml --at 0
python3 app.py --config demo/config_video.yaml --fast --show --save-video demo/output/annotated.mp4
```
- `--show` opens a live window: video plus a side panel with tray bars and an alert feed. Press `q` to quit.
- `--save-video` writes the same view to an MP4 you can play in a meeting or send to the customer.
- The dashboard is at http://localhost:8080 while the demo runs.
- What it shows: customers' hands taking items (those readings are marked *blocked*), trays going LOW and EMPTY, a restock, and lighting flicker that doesn't trigger false alerts.

### B. Your own test video (most convincing for customers)
Film a tray of anything (fish, fruit, snacks) from above with a phone, held still or on a tripod:
1. **0–5 s:** tray empty (or ice only)
2. **5–10 s:** put the items in; keep 2–3 s where it is full and nobody is touching it
3. **Then:** take items out a few at a time, like customers buying

```bash
# point the demo config at your video
#   camera.file: video/my_tray.mp4      (path is relative to demo/)
python3 roi_tool.py snapshot  --config demo/config_video.yaml --at 8 -o refs/frame.jpg
python3 roi_tool.py draw      --config demo/config_video.yaml --image refs/frame.jpg   # draw the tray(s)
python3 roi_tool.py reference --config demo/config_video.yaml --at 2     # empty moment
python3 roi_tool.py calibrate --config demo/config_video.yaml --at 8     # full moment
python3 app.py --config demo/config_video.yaml --fast --show --save-video demo/output/my_demo.mp4
```
Tips: use a plain tray that contrasts with the items, avoid shadows moving across it, and set `file_step` so you get 2–5 readings per second of video (phone video is usually 30 fps, so `file_step: 10`).

### C. Live USB webcam
Set `camera.source: webcam` (plus `device: 0`) in a copy of the demo config and leave out `file_loop`. Run `roi_tool.py snapshot`, then `draw`, `reference` and `calibrate` the same way, then `python3 app.py --config <your config> --show`. On the Jetson this is a live demo of the final system, with the webcam standing in for the P437.

### D. Still images
`python3 tools/make_demo.py`, then `python3 app.py --config demo/config_demo.yaml --iterations 12`. This is a minimal check that writes `demo/output/latest.jpg` and `levels.csv`.

## 2. Camera setup (Reolink app / web UI)
1. Give the camera a fixed IP, e.g. `192.168.1.100`.
2. Turn on **HTTP** (snapshot mode) and/or **RTSP** under *Settings → Network → Advanced → Port Settings*. Recent firmware ships with both off.
3. Create a separate **user account** for the Jetson instead of using `admin`.
4. Mount the camera high, looking down at the trays, out of the spray from washing and ice. Use the **5× zoom** to frame just the stall, which also keeps passers-by out (PDPA).
5. Fix exposure and white balance if possible (or at least turn off "auto day/night" during trading hours). The algorithm compensates for brightness, but steady images give better readings.

## 3. Jetson setup (JetPack 6.2.2 = L4T r36.4, CUDA 12.6)
```bash
git clone <this project> ~/stock-level-monitor && cd ~/stock-level-monitor   # or scp the folder
bash scripts/setup_jetson.sh      # venv + torch/torchvision/onnxruntime-gpu from pypi.jetson-ai-lab.io/jp6/cu126
bash scripts/export_engines.sh    # builds models/yolo11n.engine (TensorRT FP16) on the device
source .venv/bin/activate
export REOLINK_USER=jetson REOLINK_PASS='...'
```
Notes
- Install PyTorch from the **Jetson AI Lab index** (built for Orin `sm_87`), not from plain PyPI.
- `numpy<2` is pinned because JetPack's system OpenCV is built against NumPy 1.x.
- The setup script removes the PyPI `opencv-python` wheel so JetPack's OpenCV (with GStreamer) is used. That's only needed for RTSP mode.
- TensorRT engines are tied to the device and TensorRT version. Always build them **on the Jetson**.

## 4. Configure trays
```bash
python3 roi_tool.py snapshot        # refs/frame.jpg
python3 roi_tool.py draw            # click a polygon around each tray, ENTER, name it, 's' to save
python3 roi_tool.py preview         # output/trays_preview.jpg - check the polygons
```
If the Jetson is headless, copy `refs/frame.jpg` to a laptop, run `draw --image frame.jpg` there, and copy `trays.yaml` back. Points are stored normalised, so they also fit the sub stream.

## 5. Teach it "empty" and "full"
```bash
# Before stocking (trays in the state you call empty, e.g. ice only, nobody in view):
python3 roi_tool.py reference
# Right after stocking every tray to what the owner calls "full":
python3 roi_tool.py calibrate
```
`calibrate` stores each tray's `full_coverage` in `trays.yaml`, because a "full" tray of fish never covers 100% of the tray.

## 6. Run
```bash
python3 app.py
# dashboard: http://<jetson-ip>:8080
```
Telegram alerts: create a bot with @BotFather, get your chat id, set `alerts.telegram.enabled: true`, then `export TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=...`.

Run at boot with `scripts/stockmon.service`; instructions are inside that file.

## 7. Upgrade to the trained model (recommended for production)
The reference method is quick to deploy but has known weaknesses. The trained model fixes them:

| Problem with `reference` | Why it happens |
|---|---|
| Pale products (squid, chicken, tofu) on white trays read low | Colour too close to the empty tray |
| Water, blood, ice melt and price tags count as stock | Anything that differs from the reference counts |
| Trays moved or swapped need a new reference | Pixel-by-pixel comparison |

Steps:
1. **Collect** about 300–1000 images over 1–2 weeks: `python3 train/collect.py --every 120 --hours 06:00-14:00`. Cover full, half and empty trays, every product, wet and dry trays, and hands in view.
2. **Label** product regions with polygons (CVAT, Label Studio or Roboflow). One `product` class is enough for fill level. Export as *YOLO segmentation* into `dataset/`.
3. **Train** on a GPU PC or Colab: `python train/train_seg.py --data train/data.yaml`.
4. Copy `best.pt` to the Jetson as `models/stock-seg.pt`, run `bash scripts/export_engines.sh`, set `estimator: yolo_seg`, and run `roi_tool.py calibrate` again.

## Tuning (config.yaml)
| Setting | Raise it if... | Lower it if... |
|---|---|---|
| `reference.diff_threshold` | wet trays or reflections count as stock | pale products are missed |
| `levels.low` / `levels.empty` | owner wants earlier warnings | too many alerts |
| `levels.confirm_readings`, `smoothing.window` | alerts flicker | alerts come too late |
| `occlusion.max_overlap` | readings are skipped too often | arms are counted as stock |
| `interval_sec` | Jetson is busy with other work | owner needs faster alerts |

## Project layout
```
app.py                main loop
roi_tool.py           snapshot / draw / preview / reference / calibrate
config.yaml, trays.yaml
stockmon/             camera, estimators, occlusion, levels, notify, dashboard, annotate
train/                collect.py, data.yaml, train_seg.py
scripts/              setup_jetson.sh, export_engines.sh, stockmon.service
tools/                make_demo_video.py (test stall video), make_demo.py (still frames)
demo/                 config_video.yaml, config_demo.yaml, generated media and output
```

## Known limitations
- **Fill % is based on area, not item count.** Overlapping items hide each other, so a heaped tray can read higher than its item count suggests (in the demo, Sotong reads 83% with 67% of its items left). For true counts, train `yolo_seg` with per-item labels.
- **The motion check needs movement.** A hand held perfectly still over a tray is only caught by the YOLO person check (`occlusion.person: true`).

## Verified vs. not verified
- Verified on a PC: the generated-video demo end to end (fill levels, blocking readings while hands move, ignoring flicker, alerts, side panel, MP4 output), the still-image demo, and the dashboard endpoints.
- **Not yet tested:** the webcam source, Reolink `cmd=Snap` on the P437, RTSP/GStreamer on JetPack 6.2.2, TensorRT export, the YOLO person and segmentation paths, Telegram delivery, and the `--show` window (I only checked the saved video). Test these before a customer demo.
