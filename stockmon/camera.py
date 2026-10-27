"""Frame sources: Reolink HTTP snapshot, RTSP stream, USB webcam, or local files."""
import logging
import secrets
import ssl
import threading
import time
from urllib.parse import quote, urlencode
from urllib.request import urlopen

import cv2
import numpy as np

from .config import resolve, secret

log = logging.getLogger(__name__)
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


class EndOfVideo(Exception):
    """Raised by FileSource when a non-looping video has finished."""


class SnapshotSource:
    """Polls full-resolution JPEGs via the Reolink HTTP API (cmd=Snap).

    No video decoding at all, which keeps the Jetson almost idle between readings.
    Enable HTTP (or HTTPS) under Settings > Network > Advanced > Port Settings.
    """

    def __init__(self, c):
        scheme = "https" if c.get("https") else "http"
        port = c.get("http_port", 443 if scheme == "https" else 80)
        self.base = f"{scheme}://{c['host']}:{port}/cgi-bin/api.cgi"
        self.channel = c.get("channel", 0)
        self.user = secret(c.get("user_env"), "admin")
        self.password = secret(c.get("password_env"))
        self.timeout = c.get("timeout_sec", 10)
        # Reolink cameras use self-signed certificates.
        self.ctx = ssl._create_unverified_context() if scheme == "https" else None

    def read(self):
        query = urlencode({
            "cmd": "Snap", "channel": self.channel, "rs": secrets.token_hex(6),
            "user": self.user, "password": self.password,
        })
        with urlopen(f"{self.base}?{query}", timeout=self.timeout, context=self.ctx) as r:
            data = r.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            # The API answers with JSON on errors (bad login, HTTP disabled, ...).
            raise RuntimeError(f"Snapshot failed, camera replied: {data[:200]!r}")
        return img

    def close(self):
        pass


class _ThreadedCapture:
    """Keeps the latest frame of a live stream in a background thread."""

    name = "stream"

    def __init__(self, timeout):
        self.timeout = timeout
        self._frame, self._seq, self._last_read = None, 0, 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _open(self):
        raise NotImplementedError

    def _loop(self):
        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                log.error("Cannot open %s, retrying in 5 s", self.name)
                time.sleep(5)
                continue
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    log.warning("%s read failed, reconnecting", self.name)
                    break
                with self._lock:
                    self._frame, self._seq = frame, self._seq + 1
            cap.release()
            time.sleep(2)

    def read(self):
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None and self._seq != self._last_read:
                    self._last_read = self._seq
                    return self._frame.copy()
            time.sleep(0.01)
        raise RuntimeError(f"No new {self.name} frame within timeout")

    def close(self):
        self._stop.set()


class RtspSource(_ThreadedCapture):
    name = "RTSP stream"

    def __init__(self, c):
        user = quote(secret(c.get("user_env"), "admin"), safe="")
        password = quote(secret(c.get("password_env")), safe="")
        self.url = f"rtsp://{user}:{password}@{c['host']}:{c.get('rtsp_port', 554)}/{c.get('rtsp_path', 'h264Preview_01_main')}"
        self.use_gst = c.get("use_gstreamer", True)
        super().__init__(c.get("timeout_sec", 10))

    def _open(self):
        if self.use_gst:
            # decodebin picks nvv4l2decoder (NVDEC) on Jetson for both H.264 and H.265.
            pipeline = (
                f'rtspsrc location="{self.url}" protocols=tcp latency=200 ! decodebin ! '
                "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=true max-buffers=1 sync=false"
            )
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                return cap
            log.warning("GStreamer pipeline failed to open, falling back to FFmpeg (CPU decode)")
        return cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)


class WebcamSource(_ThreadedCapture):
    """USB webcam (or /dev/videoN) - handy for live demos without an IP camera."""

    name = "webcam"

    def __init__(self, c):
        self.device = c.get("device", 0)
        self.width, self.height = c.get("width", 1280), c.get("height", 720)
        super().__init__(c.get("timeout_sec", 10))

    def _open(self):
        cap = cv2.VideoCapture(self.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap


class FileSource:
    """An image, a folder of images (cycled in name order), or a video file.

    For videos each read() advances `file_step` frames, so a long recording plays
    back as a time-lapse of readings. Set `file_loop: false` to stop at the end.
    """

    def __init__(self, c, cfg):
        self.path = resolve(cfg, c.get("file", "demo/frames"))
        self.step = max(1, int(c.get("file_step", 1)))
        self.loop = c.get("file_loop", True)
        self.images, self.cap, self.idx = [], None, 0
        self.fps = 0.0
        if self.path.is_dir():
            self.images = sorted(p for p in self.path.iterdir() if p.suffix.lower() in IMAGE_EXT)
        elif self.path.suffix.lower() in IMAGE_EXT:
            self.images = [self.path]
        else:
            self.cap = cv2.VideoCapture(str(self.path))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 15.0
        if not self.images and (self.cap is None or not self.cap.isOpened()):
            raise RuntimeError(f"No images or video found at {self.path}")

    @property
    def position_sec(self):
        """Current position in the video (None for images)."""
        if self.cap is None:
            return None
        return self.cap.get(cv2.CAP_PROP_POS_FRAMES) / self.fps

    def seek(self, sec):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * self.fps))

    def _grab(self):
        ok, frame = self.cap.read()
        if not ok:
            if not self.loop:
                raise EndOfVideo()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame

    def read(self):
        if self.images:
            img = cv2.imread(str(self.images[self.idx % len(self.images)]))
            self.idx += 1
            return img
        for _ in range(self.step - 1):
            if not self.cap.grab():
                if not self.loop:
                    raise EndOfVideo()
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self._grab()

    def read_next(self):
        """The very next frame (used for motion detection). Images have no motion."""
        if self.images:
            return cv2.imread(str(self.images[(self.idx - 1) % len(self.images)]))
        return self._grab()

    def close(self):
        if self.cap is not None:
            self.cap.release()


def read_motion_pair(source, gap_sec):
    """A second frame shortly after the last read, for frame-difference motion checks."""
    if hasattr(source, "read_next"):
        return source.read_next()
    time.sleep(gap_sec)
    return source.read()


def make_source(cfg):
    c = cfg["camera"]
    kind = c.get("source", "snapshot")
    if kind == "snapshot":
        return SnapshotSource(c)
    if kind == "rtsp":
        return RtspSource(c)
    if kind == "webcam":
        return WebcamSource(c)
    if kind == "file":
        return FileSource(c, cfg)
    raise ValueError(f"Unknown camera.source: {kind}")
