#!/usr/bin/env python3
"""Stock & display level monitor - main loop.

    python3 app.py                                   # run with config.yaml
    python3 app.py --config demo/config_video.yaml --show --save-video demo/output/annotated.mp4
"""
import argparse
import csv
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2

from stockmon.annotate import crop_tray, draw, with_panel
from stockmon.camera import EndOfVideo, make_source, read_motion_pair
from stockmon.config import load_config, resolve
from stockmon.dashboard import SharedState, start_dashboard
from stockmon.estimators import make_estimator
from stockmon.geometry import Tray
from stockmon.levels import TrayLevel
from stockmon.notify import Notifier
from stockmon.occlusion import OcclusionGate

log = logging.getLogger("stockmon")


def clock_label(source):
    pos = getattr(source, "position_sec", None)
    if pos is not None:
        return f"video {int(pos // 60):02d}:{pos % 60:04.1f}"
    return datetime.now().strftime("%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--iterations", type=int, default=0, help="stop after N readings (0 = forever)")
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("--show", action="store_true", help="live window with the side panel (q to quit)")
    ap.add_argument("--save-video", help="write annotated output with side panel, e.g. output/annotated.mp4")
    ap.add_argument("--fast", action="store_true", help="ignore interval_sec (process files as fast as possible)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    trays = [Tray(d) for d in cfg["trays"]]
    if not trays:
        raise SystemExit(f"No trays in {cfg['_trays_file']}. Run: python3 roi_tool.py draw")

    source = make_source(cfg)
    estimator = make_estimator(cfg)
    gate = OcclusionGate(cfg, trays)
    levels = {t.name: TrayLevel(cfg.get("levels", {}), cfg.get("smoothing", {})) for t in trays}
    notifier = Notifier(cfg)
    interval = 0.0 if args.fast else float(cfg.get("interval_sec", 10))
    alerts = deque(maxlen=20)

    out_cfg = cfg.get("output", {})
    out_dir = resolve(cfg, out_cfg.get("dir", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = resolve(cfg, out_cfg.get("csv", "output/levels.csv"))
    new_csv = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if new_csv:
        writer.writerow(["timestamp", "tray", "label", "raw_fill", "smoothed_fill", "state", "occluded"])
    video_out = None

    shared = SharedState()
    dash = cfg.get("dashboard", {})
    if dash.get("enabled", True) and not args.no_dashboard:
        start_dashboard(shared, dash.get("host", "0.0.0.0"), int(dash.get("port", 8080)))

    log.info("Monitoring %d trays every %.0f s with '%s' estimator", len(trays), interval, cfg.get("estimator"))
    n = 0
    try:
        while not args.iterations or n < args.iterations:
            t0 = time.time()
            n += 1
            try:
                frame = source.read()
                clock = clock_label(source)
                frame2 = read_motion_pair(source, gate.motion.gap) if gate.needs_second_frame else None
            except EndOfVideo:
                log.info("End of video")
                break
            except Exception as e:  # noqa: BLE001 - camera hiccups must not kill the service
                log.error("Frame grab failed: %s", e)
                time.sleep(min(interval, 10))
                continue

            readings = estimator.estimate(frame, trays)
            blocked = gate.update(frame, frame2, trays)
            now = datetime.now().isoformat(timespec="seconds")
            changes = []
            for t in trays:
                r = readings[t.name]
                r.fill = t.fill_from_coverage(r.fill)
                if blocked[t.name]:
                    r.occluded, r.fill = True, None
                lvl = levels[t.name]
                change = lvl.update(r.fill)
                if change:
                    changes.append((t, change))
                writer.writerow([now, t.name, t.label,
                                 "" if r.fill is None else f"{r.fill:.3f}",
                                 "" if lvl.smoothed is None else f"{lvl.smoothed:.3f}",
                                 lvl.state or "", int(r.occluded)])
            csv_file.flush()

            annotated = draw(frame, trays, readings, levels, gate.person_boxes)
            for t, (old, new) in changes:
                ok, jpg = cv2.imencode(".jpg", crop_tray(annotated, t))
                text = notifier.on_change(t, old, new, levels[t.name].smoothed, jpg.tobytes() if ok else None)
                if text:
                    alerts.append((clock.split()[-1][:5], text, new))

            if args.show or args.save_video:
                demo = with_panel(annotated, trays, levels, readings, alerts, clock)
                if args.save_video:
                    if video_out is None:
                        path = Path(args.save_video)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        video_out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                                                    (demo.shape[1], demo.shape[0]))
                        log.info("Writing annotated video to %s", path)
                    video_out.write(demo)
                if args.show:
                    cv2.imshow("Stock levels", demo)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            preview = cv2.resize(annotated, (1280, int(1280 * annotated.shape[0] / annotated.shape[1])))
            ok, jpg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if out_cfg.get("save_annotated", True):
                cv2.imwrite(str(out_dir / "latest.jpg"), annotated)
            shared.set({
                "updated": clock,
                "trays": [{
                    "name": t.name, "label": t.label, "fill": levels[t.name].smoothed,
                    "state": levels[t.name].state, "occluded": readings[t.name].occluded,
                } for t in trays],
            }, jpg.tobytes() if ok else b"")

            summary = ", ".join(
                f"{t.label}={'blocked' if readings[t.name].occluded else f'{levels[t.name].smoothed:.0%}'}"
                f"[{levels[t.name].state}]" for t in trays)
            log.info("#%d %s %s (%.2f s)", n, clock, summary, time.time() - t0)
            time.sleep(max(0.0, interval - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        csv_file.close()
        if video_out is not None:
            video_out.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
