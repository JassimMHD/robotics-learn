"""M1: frame sources, bounded capture buffer, camera failure reporting."""

import time

import cv2
import numpy as np
import pytest

from app.camera.source import CameraError, FrameSource, ImageFileSource, VideoFileSource, open_source
from app.camera.worker import CaptureWorker


def test_missing_file_source_is_an_error(tmp_path):
    with pytest.raises(CameraError):
        open_source(str(tmp_path / "missing.mp4"), 640, 480)


def test_image_replay_source_is_labeled_replay(tmp_path):
    path = tmp_path / "tray.png"
    cv2.imwrite(str(path), np.full((480, 640, 3), 128, np.uint8))
    src = open_source(str(path), 640, 480)
    assert isinstance(src, ImageFileSource) and src.kind == "replay"


def test_video_replay_loops(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    for i in range(3):
        writer.write(np.full((48, 64, 3), i * 60, np.uint8))
    writer.release()
    src = VideoFileSource(path)
    frames = [src.read() for _ in range(7)]  # more than the 3 recorded frames
    assert all(f.shape == (48, 64, 3) for f in frames)
    src.close()


def test_worker_delivers_newest_frames_with_increasing_ids(tmp_path):
    path = tmp_path / "tray.png"
    cv2.imwrite(str(path), np.full((48, 64, 3), 128, np.uint8))
    worker = CaptureWorker(ImageFileSource(path, fps=50), buffer_frames=2)
    worker.start()
    try:
        f1 = worker.latest(timeout_s=2)
        f2 = worker.latest(after_frame_id=f1.frame_id, timeout_s=2)
        assert f2.frame_id > f1.frame_id and f2.source_kind == "replay"
        assert worker.health()["status"] == "ok"
    finally:
        worker.stop()


class UnpluggedCamera(FrameSource):
    kind = "live"

    def __init__(self, good_frames: int):
        self.left = good_frames

    def read(self):
        if self.left <= 0:
            time.sleep(0.05)
            raise CameraError("camera returned no frame (unplugged or busy?)")
        self.left -= 1
        time.sleep(0.02)
        return np.zeros((48, 64, 3), np.uint8)


def test_unplugged_camera_reports_error_and_goes_stale():
    worker = CaptureWorker(UnpluggedCamera(good_frames=3), stale_after_s=0.3)
    worker.start()
    try:
        time.sleep(0.8)
        health = worker.health()
        assert health["status"] == "error" and "unplugged" in health["error"]
        assert worker.stale
    finally:
        worker.stop()
