"""Smoothing and OK / LOW / EMPTY state machine with hysteresis."""
from collections import deque
from statistics import median

ORDER = {"EMPTY": 0, "LOW": 1, "OK": 2}


class TrayLevel:
    def __init__(self, levels_cfg, smoothing_cfg):
        self.low = float(levels_cfg.get("low", 0.35))
        self.empty = float(levels_cfg.get("empty", 0.10))
        self.h = float(levels_cfg.get("hysteresis", 0.05))
        self.confirm = int(levels_cfg.get("confirm_readings", 3))
        self.history = deque(maxlen=int(smoothing_cfg.get("window", 5)))
        self.state = None
        self.smoothed = None
        self._pending, self._count = None, 0

    def _classify(self, v):
        raw = "EMPTY" if v < self.empty else "LOW" if v < self.low else "OK"
        if self.state is None or raw == self.state:
            return raw
        bounds = [self.empty, self.low]
        cur = ORDER[self.state]
        if ORDER[raw] > cur and v < bounds[cur] + self.h:        # rising: clear upper bound + h
            return self.state
        if ORDER[raw] < cur and v > bounds[cur - 1] - self.h:    # falling: clear lower bound - h
            return self.state
        return raw

    def update(self, fill):
        """Feed one reading (None = skipped). Returns (old_state, new_state) on a change."""
        if fill is None:
            return None
        self.history.append(fill)
        self.smoothed = median(self.history)
        target = self._classify(self.smoothed)
        if self.state is None:
            self.state = target
            return (None, target) if target != "OK" else None
        if target == self.state:
            self._pending, self._count = None, 0
            return None
        if target == self._pending:
            self._count += 1
        else:
            self._pending, self._count = target, 1
        if self._count >= self.confirm:
            old, self.state = self.state, target
            self._pending, self._count = None, 0
            return old, target
        return None
