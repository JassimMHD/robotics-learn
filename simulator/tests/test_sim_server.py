"""Simulator HTTP API with the physics loop running in a background thread."""

import threading
import time

import pytest
from fastapi.testclient import TestClient

from saltsim.runtime import SimRuntime
from saltsim.server import create_sim_app
from saltsim.testing import pick_request


@pytest.fixture
def client(scene_cfg):
    rt = SimRuntime(scene_cfg)
    stop = threading.Event()
    thread = threading.Thread(target=rt.run_headless, args=(stop,), daemon=True)
    thread.start()
    yield TestClient(create_sim_app(rt)), rt
    stop.set()
    thread.join(timeout=5)


def test_pick_lifecycle_over_http(client, scene_cfg):
    c, _ = client
    req = pick_request(scene_cfg, "pick-000001", "salt-1", 0.2, 0.15)
    r = c.post("/pick", json=req)
    assert r.status_code == 202
    again = c.post("/pick", json=req)
    assert again.status_code == 200 and again.json()["duplicate_request"] is True
    status = c.get("/status").json()
    assert status["robot"]["pick_event_id"] == "pick-000001"
    assert status["robot"]["simulated"] is True
    assert c.get("/picks/pick-000001").json()["status"] in {"accepted", "approaching"}


def test_wrong_frame_and_bad_payload_rejected(client, scene_cfg):
    c, _ = client
    req = pick_request(scene_cfg, "pick-1", "salt-1", 0.2, 0.15)
    req["target"]["frame"] = "tray"
    assert c.post("/pick", json=req).status_code == 422
    assert c.post("/pick", json={"event_id": "x"}).status_code == 422
    req = pick_request(scene_cfg, "pick-2", "salt-2", 0.2, 0.15)
    req["grasp_mode"] = "magic"
    assert c.post("/pick", json=req).status_code == 422


def test_scene_reset_and_frame(client):
    c, _ = client
    scene = c.get("/scene").json()
    assert scene["frame"] == "robot_base" and "tray" in scene
    assert c.post("/reset").json() == {"reset": True}
    deadline = time.time() + 10
    r = c.get("/frame.jpg")
    while r.status_code != 200 and time.time() < deadline:
        r = c.get("/frame.jpg")
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"  # JPEG
    m = c.get("/health").json()["metrics"]
    assert m["real_time_factor"] is not None
