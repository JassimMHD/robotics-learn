"""Capture thread: keeps only the newest frames (bounded buffer) and reports camera health."""

import threading
import time
from collections import deque

from app.camera.source import CameraError, Frame, FrameSource
from app.schemas.events import now_ms
from app.telemetry.metrics import RateMeter


class CaptureWorker:
    def __init__(self, source: FrameSource, buffer_frames: int = 2, stale_after_s: float = 2.0):
        self.source = source
        self.stale_after_s = stale_after_s
        self._buffer: deque[Frame] = deque(maxlen=buffer_frames)  # old frames dropped under load
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)
        self._frame_id = 0
        self.error: str | None = None
        self.fps = RateMeter()
        self.last_frame_wall = 0.0
        self.dropped_frames = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)
        self.source.close()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                image = self.source.read()
            except CameraError as exc:
                self.error = str(exc)
                time.sleep(0.5)
                continue
            except Exception as exc:  # driver-level failure: report, keep the process alive
                self.error = f"camera failure: {exc}"
                time.sleep(0.5)
                continue
            self.error = None
            self._frame_id += 1
            frame = Frame(self._frame_id, now_ms(), image, self.source.kind)
            with self._cond:
                if len(self._buffer) == self._buffer.maxlen:
                    self.dropped_frames += 1
                self._buffer.append(frame)
                self._cond.notify_all()
            self.last_frame_wall = time.time()
            self.fps.tick()

    def latest(self, after_frame_id: int = 0, timeout_s: float = 0.5) -> Frame | None:
        """Newest frame with id > after_frame_id, waiting up to timeout_s."""
        deadline = time.time() + timeout_s
        with self._cond:
            while not self._buffer or self._buffer[-1].frame_id <= after_frame_id:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return self._buffer[-1]

    @property
    def stale(self) -> bool:
        return self.last_frame_wall == 0.0 or time.time() - self.last_frame_wall > self.stale_after_s

    def health(self) -> dict:
        age = None if self.last_frame_wall == 0 else round(time.time() - self.last_frame_wall, 3)
        if self.error:
            status = "error"
        elif self.stale:
            status = "stale" if self.last_frame_wall else "starting"
        else:
            status = "ok"
        return {
            "status": status, "error": self.error, "source_kind": self.source.kind,
            "fps": self.fps.rate(), "last_frame_age_s": age, "dropped_frames": self.dropped_frames,
        }
