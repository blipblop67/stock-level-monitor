"""Alerts: always logged, optionally sent to Telegram."""
import json
import logging
import time
import uuid
from urllib.request import Request, urlopen

from .config import secret

log = logging.getLogger(__name__)


class Notifier:
    def __init__(self, cfg):
        c = cfg.get("alerts", {})
        self.cooldown = float(c.get("cooldown_min", 30)) * 60
        self.notify_restock = c.get("notify_restock", True)
        tg = c.get("telegram", {})
        self.token = secret(tg.get("token_env")) if tg.get("enabled") else ""
        self.chat_id = secret(tg.get("chat_id_env")) if tg.get("enabled") else ""
        self.send_photo = tg.get("send_photo", True)
        if tg.get("enabled") and not (self.token and self.chat_id):
            log.warning("Telegram enabled but %s / %s are not set", tg.get("token_env"), tg.get("chat_id_env"))
        self._last = {}

    def on_change(self, tray, old, new, fill, photo_jpeg=None):
        """Logs/sends the alert and returns its text (None if suppressed)."""
        if new == "OK" and not (old in ("LOW", "EMPTY") and self.notify_restock):
            return None
        key = (tray.name, new)
        now = time.time()
        if now - self._last.get(key, 0) < self.cooldown:
            return None
        self._last[key] = now
        icon = {"EMPTY": "\U0001F534", "LOW": "\U0001F7E0", "OK": "\U0001F7E2"}[new]
        verb = "restocked" if new == "OK" else f"is {new}"
        text = f"{icon} {tray.label} {verb} ({fill:.0%})"
        log.info("ALERT: %s", text)
        if self.token and self.chat_id:
            try:
                if photo_jpeg is not None and self.send_photo:
                    self._send_photo(text, photo_jpeg)
                else:
                    self._post("sendMessage", json.dumps({"chat_id": self.chat_id, "text": text}).encode(),
                               "application/json")
            except Exception as e:  # noqa: BLE001 - never crash the loop on a failed alert
                log.error("Telegram send failed: %s", e)
        return f"{tray.label} {verb} ({fill:.0%})"

    def _post(self, method, body, content_type):
        req = Request(f"https://api.telegram.org/bot{self.token}/{method}", data=body,
                      headers={"Content-Type": content_type})
        with urlopen(req, timeout=15) as r:
            r.read()

    def _send_photo(self, caption, jpeg):
        boundary = uuid.uuid4().hex
        parts = []
        for name, value in (("chat_id", self.chat_id), ("caption", caption)):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="photo"; filename="tray.jpg"\r\n'
                     f"Content-Type: image/jpeg\r\n\r\n".encode() + jpeg + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        self._post("sendPhoto", b"".join(parts), f"multipart/form-data; boundary={boundary}")
