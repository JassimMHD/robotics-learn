import threading

import pytest
from fastapi.testclient import TestClient

from app.orchestrator.bridge import SimBridge, SimUnavailable
from saltsort_common.scene_config import load_scene_config


@pytest.fixture(scope="session")
def scene():
    return load_scene_config()


class FakeBridge:
    """Scriptable stand-in for the simulator API."""

    def __init__(self):
        self.robot_state = "IDLE"
        self.picks: dict[str, dict] = {}
        self.requests: list[dict] = []
        self.available = True
        self.busy_responses = 0
        self.resets = 0

    def status(self) -> dict:
        if not self.available:
            raise SimUnavailable("fake simulator down")
        return {"robot": {"robot_state": self.robot_state}, "picks": list(self.picks.values()), "objects": [],
                "counts": {}, "metrics": {}}

    def pick(self, request: dict):
        if not self.available:
            raise SimUnavailable("fake simulator down")
        if self.busy_responses:
            self.busy_responses -= 1
            return 409, {"error": "robot busy"}
        self.requests.append(request)
        rec = {"event_id": request["event_id"], "status": "accepted", "detail": "", "cycle_time_sim_s": None}
        self.picks[request["event_id"]] = rec
        self.robot_state = "VALIDATE_TARGET"
        return 202, rec

    def finish(self, event_id: str, status: str = "completed_sim", detail: str = "") -> None:
        self.picks[event_id].update(status=status, detail=detail, cycle_time_sim_s=12.0)
        self.robot_state = "IDLE"

    def reset(self) -> dict:
        self.resets += 1
        self.picks.clear()
        self.robot_state = "IDLE"
        return {"reset": True}


@pytest.fixture
def fake_bridge():
    return FakeBridge()


@pytest.fixture
def live_sim(scene):
    """Real simulator runtime in a background thread (5x real time), reached over its HTTP app."""
    from saltsim.runtime import SimRuntime
    from saltsim.server import create_sim_app

    rt = SimRuntime(scene)
    stop = threading.Event()
    thread = threading.Thread(target=rt.run_headless, args=(stop,), kwargs={"time_scale": 5.0}, daemon=True)
    thread.start()
    bridge = SimBridge(client=TestClient(create_sim_app(rt)))
    yield bridge, rt
    stop.set()
    thread.join(timeout=5)
