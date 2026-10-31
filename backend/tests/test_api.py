"""API tests, including an end-to-end manual pick against the real (in-process) simulator."""

import time

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings


def _app(tmp_path, bridge, **overrides):
    settings = Settings(calibration={"path": str(tmp_path / "cal.json")}, **overrides)
    return create_app(settings, bridge=bridge, event_log=False)


def _wait(predicate, timeout_s=60.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError("condition not reached")


def test_health_labels_and_blockers(tmp_path, fake_bridge):
    with TestClient(_app(tmp_path, fake_bridge)) as c:
        _wait(lambda: c.get("/api/health").json()["components"]["simulator"]["status"] == "connected")
        body = c.get("/api/health").json()
        assert body["labels"]["vision_mode"] == "MANUAL" and body["labels"]["robot"] == "SIMULATED"
        assert body["components"]["model"]["label"] == "NO REAL MODEL"
        assert body["autonomous_picking_enabled"] is False
        assert "calibration missing" in body["blockers"]
        assert body["manual_picking_enabled"] is True


def test_real_model_mode_without_weights_reports_no_real_model(tmp_path, fake_bridge):
    app = _app(tmp_path, fake_bridge, app={"mode": "real_model"},
               model={"backend": "onnx", "weights_path": str(tmp_path / "missing.onnx")})
    with TestClient(app) as c:
        body = c.get("/api/health").json()
        assert body["labels"]["vision_mode"] == "NO REAL MODEL"
        assert "not found" in body["components"]["model"]["detail"]


def test_pixel_target_requires_calibration(tmp_path, fake_bridge):
    with TestClient(_app(tmp_path, fake_bridge)) as c:
        r = c.post("/api/manual-target", json={"image_px": [320, 240]})
        assert r.status_code == 422 and "calibration missing" in r.json()["detail"]
        assert c.post("/api/manual-target", json={}).status_code == 422


def test_calibration_api_flow(tmp_path, fake_bridge):
    with TestClient(_app(tmp_path, fake_bridge)) as c:
        corners = [[40, 40], [600, 40], [600, 440], [40, 440]]
        assert c.post("/api/calibration", json={"image_corners_px": corners[::-1], "resolution": [640, 480]}
                      ).status_code == 422  # reversed order
        info = c.post("/api/calibration", json={"image_corners_px": corners, "resolution": [640, 480]}).json()
        assert info["status"] == "unvalidated"
        # Holdout ground truth: this synthetic camera maps the tray linearly onto the corner rectangle.
        for x, y in [(0.05, 0.05), (0.2, 0.15), (0.35, 0.25), (0.1, 0.25), (0.3, 0.05)]:
            u, v = 40 + x / 0.40 * 560, 440 - y / 0.30 * 400
            info = c.post("/api/calibration/holdout", json={"image_px": [u, v], "tray_m": [x, y]}).json()
        assert info["status"] == "valid" and info["stats"]["max_error_mm"] < 0.01
        # Manual pixel target now maps through the calibration.
        r = c.post("/api/manual-target", json={"image_px": [320, 240]}).json()
        assert r["accepted"] and r["vision_mode"] == "manual"
        assert r["tray_m"] == pytest.approx([0.2, 0.15], abs=1e-4)


def test_end_to_end_manual_pick_with_real_simulator(tmp_path, live_sim):
    bridge, rt = live_sim
    with TestClient(_app(tmp_path, bridge)) as c:
        r = c.post("/api/manual-target", json={"tray_m": [0.2, 0.15]}).json()
        assert r["accepted"]
        event_id = r["pick"]["event_id"]
        dup = c.post("/api/manual-target", json={"tray_m": [0.205, 0.15]}).json()
        assert not dup["accepted"] and "duplicate" in dup["reason"]

        def done():
            picks = c.get("/api/state").json()["scheduler"]["picks"]
            p = next(p for p in picks if p["event_id"] == event_id)
            return p if p["status"] not in ("queued", "accepted", "approaching", "grasping", "verifying",
                                             "lifting", "transferring", "releasing") else None

        pick = _wait(done, timeout_s=90)
        assert pick["status"] == "completed_sim" and pick["simulated"] is True
        assert pick["cycle_time_sim_s"] > 0
        events = c.get("/api/events?limit=300").json()
        statuses = [e["status"] for e in events if e["type"] == "sim.pick.status" and e["pick_event_id"] == event_id]
        assert statuses[-1] == "completed_sim" and "lifting" in statuses
        metrics = c.get("/api/metrics").json()
        assert metrics["picks"]["counts"]["completed_kinematic_demo"] == 1
        assert c.post("/api/session/reset").json() == {"reset": True}
        assert c.post("/api/manual-target", json={"tray_m": [0.2, 0.15]}).json()["accepted"]  # re-armed


def test_grasp_mode_switch_and_pause(tmp_path, fake_bridge):
    with TestClient(_app(tmp_path, fake_bridge)) as c:
        assert c.post("/api/session/grasp-mode", json={"grasp_mode": "contact_physics"}).json()["grasp_mode"] == \
            "contact_physics"
        assert c.post("/api/session/grasp-mode", json={"grasp_mode": "teleport"}).status_code == 422
        assert c.post("/api/session/pause").json() == {"paused": True}
        r = c.post("/api/manual-target", json={"tray_m": [0.1, 0.1]}).json()
        assert r["pick"]["grasp_mode"] == "contact_physics"
        time.sleep(0.6)
        assert fake_bridge.requests == []  # paused: queued, not dispatched


def test_camera_start_failure_is_reported(tmp_path, fake_bridge):
    app = _app(tmp_path, fake_bridge, camera={"source": str(tmp_path / "no-such-video.mp4")})
    with TestClient(app) as c:
        status = c.post("/api/session/start").json()
        assert status["state"] == "error" and "not found" in status["camera"]["error"]
        alerts = [e for e in c.get("/api/events").json() if e["type"] == "system.alert"]
        assert any(a["level"] == "error" and "camera unavailable" in a["message"] for a in alerts)
