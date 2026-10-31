"""Event contracts (spec section 6). Every event has a type, immutable event_id, timestamp and source."""

import itertools
import threading
import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

VisionMode = Literal["real_model", "manual", "replay"]
GraspMode = Literal["kinematic_demo", "contact_physics"]
DetectionStatus = Literal["observed", "stable", "eligible", "ignored"]
PickStatus = Literal[
    "queued", "accepted", "approaching", "grasping", "verifying", "lifting", "transferring", "releasing",
    "completed_sim", "rejected_invalid", "unreachable", "timeout", "grasp_failed_sim", "collision_blocked",
    "cancelled",
]
TERMINAL_PICK_STATUSES = {
    "completed_sim", "rejected_invalid", "unreachable", "timeout", "grasp_failed_sim", "collision_blocked",
    "cancelled",
}

RUN_ID = uuid.uuid4().hex[:4]  # distinguishes IDs across backend restarts (the simulator keeps history)


class IdFactory:
    """Monotonic, immutable IDs: '<prefix>-<run>-<n:06d>'."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self._counter = itertools.count(1)
        self._lock = threading.Lock()

    def next(self) -> str:
        with self._lock:
            return f"{self.prefix}-{RUN_ID}-{next(self._counter):06d}"


def now_ms() -> int:
    return int(time.time() * 1000)


class Event(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: str
    event_id: str
    timestamp_ms: int = Field(default_factory=now_ms)
    source: str


class DetectionEvent(Event):
    type: Literal["vision.detection"] = "vision.detection"
    source: str = "vision"
    frame_id: int
    camera_id: str
    vision_mode: VisionMode
    track_id: str
    class_name: str
    role: str
    confidence: float
    bbox_xyxy_px: tuple[float, float, float, float]
    center_px: tuple[float, float]
    stable: bool
    status: DetectionStatus
    reason: str = ""


class Target(BaseModel):
    frame: Literal["robot_base"] = "robot_base"
    position_m: tuple[float, float, float]
    yaw_rad: float = 0.0


class PickRequestEvent(Event):
    type: Literal["sim.pick.requested"] = "sim.pick.requested"
    source: str = "orchestrator"
    track_id: str
    source_frame_id: int | None
    target: Target
    grasp_mode: GraspMode
    vision_mode: VisionMode
    queued_at_ms: int
    calibration_status: str
    height_estimated: bool = True  # one RGB view cannot measure chunk height


class PickStatusEvent(Event):
    type: Literal["sim.pick.status"] = "sim.pick.status"
    source: str = "simulator"
    pick_event_id: str
    track_id: str
    status: PickStatus
    detail: str = ""
    grasp_mode: GraspMode
    vision_mode: VisionMode
    simulated: Literal[True] = True
    cycle_time_sim_s: float | None = None


class AlertEvent(Event):
    type: Literal["system.alert"] = "system.alert"
    level: Literal["info", "warning", "error"]
    message: str
