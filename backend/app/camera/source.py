"""Frame sources: live camera (USB index / RTSP) and recorded replay (video or still image file).

Replay sources are always labeled `replay`; they are never presented as a live camera.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from saltsort_common.paths import resolve

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".webp"}


class CameraError(Exception):
    pass


@dataclass(frozen=True)
class Frame:
    frame_id: int
    capture_ts_ms: int
    image: np.ndarray  # BGR
    source_kind: str  # live | replay

    @property
    def resolution(self) -> tuple[int, int]:
        return int(self.image.shape[1]), int(self.image.shape[0])


class FrameSource:
    kind = "live"

    def read(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class CameraSource(FrameSource):
    """USB/laptop camera index or rtsp:// URL."""

    kind = "live"

    def __init__(self, source: int | str, width: int, height: int):
        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # DirectShow: fast open on Windows
        else:
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise CameraError(f"cannot open camera source {source!r}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # prefer fresh frames over queued ones

    def read(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise CameraError("camera returned no frame (unplugged or busy?)")
        return frame

    def close(self) -> None:
        self.cap.release()


class VideoFileSource(FrameSource):
    """Recorded footage, looped and paced at the file's frame rate."""

    kind = "replay"

    def __init__(self, path: Path, loop: bool = True):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise CameraError(f"cannot open video file {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.period = 1.0 / fps if fps and fps > 0 else 1 / 15
        self.loop = loop
        self._next = time.perf_counter()

    def read(self) -> np.ndarray:
        delay = self._next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        self._next = max(self._next + self.period, time.perf_counter() - self.period)
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok or frame is None:
            raise CameraError(f"replay file ended or unreadable: {self.path}")
        return frame

    def close(self) -> None:
        self.cap.release()


class ImageFileSource(FrameSource):
    """A still image served repeatedly (deterministic replay for tests and demos)."""

    kind = "replay"

    def __init__(self, path: Path, fps: float = 10.0):
        image = cv2.imread(str(path))
        if image is None:
            raise CameraError(f"cannot read image file {path}")
        self.image = image
        self.period = 1.0 / fps

    def read(self) -> np.ndarray:
        time.sleep(self.period)
        return self.image.copy()


def open_source(source: int | str, width: int, height: int) -> FrameSource:
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        return CameraSource(int(source), width, height)
    if source.lower().startswith(("rtsp://", "http://", "https://")):
        return CameraSource(source, width, height)
    path = resolve(source)
    if not path.is_file():
        raise CameraError(f"camera source file not found: {path}")
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return ImageFileSource(path)
    return VideoFileSource(path)
