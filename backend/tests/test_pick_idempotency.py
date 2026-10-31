"""Unique pick queue and duplicate suppression (spec sections 8 and 12)."""

import numpy as np
import pytest

from app.calibration.homography import Calibration, build_calibration
from app.camera.source import Frame
from app.orchestrator.scheduler import PickScheduler
from app.orchestrator.session import Session
from app.perception.interface import Detection
from app.settings import Settings
from saltsort_common.frames import TRAY, FramePoint


def P(x, y):
    return FramePoint(TRAY, (x, y, 0.0))


@pytest.fixture
def events():
    return []


@pytest.fixture
def sched(scene, fake_bridge, events):
    return PickScheduler(scene, fake_bridge, events.append)


def test_same_track_submitted_20_times_queues_once(sched):
    results = [sched.submit("salt-1", P(0.2, 0.15), "real_model", f, "valid") for f in range(20)]
    assert results[0].accepted and not any(r.accepted for r in results[1:])
    assert len(sched.entries) == 1


def test_reassigned_track_id_at_same_position_is_duplicate(sched):
    assert sched.submit("salt-1", P(0.2, 0.15), "real_model", 1, "valid").accepted
    adm = sched.submit("salt-9", P(0.205, 0.152), "real_model", 50, "valid")
    assert not adm.accepted and "duplicate of salt-1" in adm.reason


def test_processed_chunk_stays_suppressed_until_reset(sched, fake_bridge):
    a = sched.submit("salt-1", P(0.2, 0.15), "real_model", 1, "valid").entry
    sched.tick()
    fake_bridge.finish(a.event_id)
    sched.tick()
    assert a.status == "completed_sim"
    # The real chunk is still on the tray: a fresh track at the same spot must not be picked again.
    assert not sched.submit("salt-2", P(0.2, 0.15), "real_model", 99, "valid").accepted
    sched.reset()
    assert fake_bridge.resets == 1
    assert sched.submit("salt-2", P(0.2, 0.15), "real_model", 100, "valid").accepted


def test_failed_pick_is_not_requeued_automatically(sched, fake_bridge):
    a = sched.submit("salt-1", P(0.2, 0.15), "real_model", 1, "valid").entry
    sched.tick()
    fake_bridge.finish(a.event_id, "unreachable", "IK failed")
    sched.tick()
    assert a.status == "unreachable"
    assert not sched.submit("salt-1", P(0.2, 0.15), "real_model", 2, "valid").accepted


def test_two_targets_dispatch_sequentially(sched, fake_bridge):
    a = sched.submit("salt-1", P(0.1, 0.1), "real_model", 1, "valid").entry
    b = sched.submit("salt-2", P(0.3, 0.2), "real_model", 1, "valid").entry
    sched.tick()
    assert [r["event_id"] for r in fake_bridge.requests] == [a.event_id]
    sched.tick()  # robot busy: b must wait
    assert len(fake_bridge.requests) == 1 and b.status == "queued"
    fake_bridge.finish(a.event_id)
    sched.tick()
    assert [r["event_id"] for r in fake_bridge.requests] == [a.event_id, b.event_id]
    assert a.event_id != b.event_id


def test_busy_or_unavailable_simulator_never_drops_a_pick(sched, fake_bridge):
    a = sched.submit("salt-1", P(0.2, 0.15), "real_model", 1, "valid").entry
    fake_bridge.busy_responses = 2
    sched.tick()
    sched.tick()
    assert a.status == "queued" and list(sched.queue) == [a]
    fake_bridge.available = False
    sched.tick()
    assert a.status == "queued"
    fake_bridge.available = True
    sched.tick()
    assert a.status == "accepted"


def test_obstruction_and_pause_hold_dispatch(sched, fake_bridge):
    sched.submit("salt-1", P(0.2, 0.15), "real_model", 1, "valid")
    sched.obstructed = True
    sched.tick()
    sched.paused = True
    sched.obstructed = False
    sched.tick()
    assert fake_bridge.requests == []
    sched.paused = False
    sched.tick()
    assert len(fake_bridge.requests) == 1


def test_target_outside_tray_and_full_queue_refused(scene, fake_bridge):
    s = PickScheduler(scene, fake_bridge, lambda e: None, max_queue=2)
    assert not s.submit("salt-0", P(0.5, 0.1), "manual", None, "valid").accepted
    assert s.submit("salt-1", P(0.05, 0.05), "manual", None, "valid").accepted
    assert s.submit("salt-2", P(0.15, 0.05), "manual", None, "valid").accepted
    adm = s.submit("salt-3", P(0.25, 0.05), "manual", None, "valid")
    assert not adm.accepted and "queue full" in adm.reason


def test_pick_request_is_in_robot_frame_with_estimated_height(sched, scene, events):
    entry = sched.submit("salt-1", P(0.2, 0.15), "real_model", 7, "valid").entry
    req = entry.request
    assert req.target.frame == "robot_base"
    expected = scene.tray.robot_from_tray().apply(P(0.2, 0.15)).xyz
    assert req.target.position_m[:2] == pytest.approx(expected[:2])
    assert req.height_estimated is True
    assert events[0]["type"] == "sim.pick.requested" and events[0]["vision_mode"] == "real_model"


# --- full gate: a stationary real chunk seen on 20 frames -> exactly one pick -----------------------
class StaticDetector:
    vision_mode = "real_model"

    def __init__(self, detections):
        self.detections = detections

    def predict(self, frame_bgr, frame_id):
        return list(self.detections)


def _session(scene, fake_bridge, tmp_path, detections, validated=True):
    settings = Settings(app={"mode": "real_model"}, calibration={"path": str(tmp_path / "cal.json")})
    sched = PickScheduler(scene, fake_bridge, lambda e: None)
    s = Session(settings, scene, sched, lambda e, persist=True: None)
    s.detector = StaticDetector(detections)
    corners = [(40.0, 40.0), (600.0, 40.0), (600.0, 440.0), (40.0, 440.0)]
    cal = Calibration(build_calibration("cam-01", (640, 480), corners, 0.40, 0.30))
    if validated:
        for x, y in [(0.05, 0.05), (0.2, 0.15), (0.35, 0.25), (0.1, 0.25), (0.3, 0.05)]:
            cal.add_holdout(cal.tray_to_image(x, y), (x, y))
    s.calibration = cal
    return s, sched


def _frames(n):
    img = np.zeros((480, 640, 3), np.uint8)
    return [Frame(i, 0, img, "live") for i in range(1, n + 1)]


def test_stationary_chunk_seen_20_times_gives_one_pick(scene, fake_bridge, tmp_path):
    chunk = Detection(0, "impure", "reject", 0.9, (300, 220, 340, 260))
    s, sched = _session(scene, fake_bridge, tmp_path, [chunk])
    for f in _frames(20):
        s.process_frame(f)
    assert len(sched.entries) == 1
    assert s.tracks[0].status == "queued"


def test_unstable_or_accept_or_unvalidated_never_queues(scene, fake_bridge, tmp_path):
    chunk = Detection(0, "impure", "reject", 0.9, (300, 220, 340, 260))
    s, sched = _session(scene, fake_bridge, tmp_path, [chunk])
    for f in _frames(2):  # fewer than min_stable_frames=3
        s.process_frame(f)
    assert sched.entries == {}

    s, sched = _session(scene, fake_bridge, tmp_path, [Detection(1, "pure", "accept", 0.9, (300, 220, 340, 260))])
    for f in _frames(10):
        s.process_frame(f)
    assert sched.entries == {} and s.tracks[0].status == "ignored"

    s, sched = _session(scene, fake_bridge, tmp_path, [chunk], validated=False)
    for f in _frames(10):
        s.process_frame(f)
    assert sched.entries == {} and "calibration unvalidated" in s.tracks[0].reason


def test_unwanted_object_in_tray_pauses_dispatch(scene, fake_bridge, tmp_path):
    hand = Detection(2, "unwanted", "obstruction", 0.8, (100, 100, 200, 220))
    chunk = Detection(0, "impure", "reject", 0.9, (300, 220, 340, 260))
    s, sched = _session(scene, fake_bridge, tmp_path, [hand, chunk])
    for f in _frames(5):
        s.process_frame(f)
    assert sched.obstructed is True
    assert len(sched.entries) == 1  # queued...
    sched.tick()
    assert fake_bridge.requests == []  # ...but not dispatched while a hand is in the tray
