"""Small thread-safe metric helpers. Every published metric carries units and a source."""

import threading
import time
from collections import deque

import numpy as np


class RateMeter:
    """Events per second over a sliding window."""

    def __init__(self, window: int = 30):
        self._times: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()

    def tick(self) -> None:
        with self._lock:
            self._times.append(time.perf_counter())

    def rate(self) -> float | None:
        with self._lock:
            if len(self._times) < 2:
                return None
            span = self._times[-1] - self._times[0]
            if time.perf_counter() - self._times[-1] > 2.0 or span <= 0:
                return 0.0
            return round((len(self._times) - 1) / span, 2)


class LatencyStats:
    """Rolling P50/P95 of a duration in milliseconds."""

    def __init__(self, window: int = 200):
        self._values: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()

    def add(self, ms: float) -> None:
        with self._lock:
            self._values.append(ms)

    def summary(self) -> dict:
        with self._lock:
            values = np.array(self._values)
        if values.size == 0:
            return {"p50_ms": None, "p95_ms": None, "samples": 0}
        return {
            "p50_ms": round(float(np.percentile(values, 50)), 2),
            "p95_ms": round(float(np.percentile(values, 95)), 2),
            "samples": int(values.size),
        }
