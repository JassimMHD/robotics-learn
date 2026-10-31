"""Unique pick queue: target admission, duplicate suppression, dispatch to the simulator, status tracking.

Rules (spec sections 5, 8):
- every pick has an immutable event ID; one active pick at a time; others wait in a bounded queue;
- a target is refused if its track already has a pick, or it lies within `duplicate_radius_m` (tray
  frame) of any queued, active, completed or failed pick: the REAL chunk stays on the tray after a
  simulated removal, so it must not be picked again until an explicit reset;
- accepted pick commands are never silently dropped: each ends in a terminal status.
"""

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from app.orchestrator.bridge import SimBridge, SimUnavailable
from app.schemas.events import (
    TERMINAL_PICK_STATUSES, AlertEvent, IdFactory, PickRequestEvent, PickStatusEvent, Target, now_ms,
)
from saltsort_common.frames import TRAY, FramePoint
from saltsort_common.scene_config import SceneConfig

Publish = Callable[[dict], None]


@dataclass
class PickEntry:
    request: PickRequestEvent
    tray_xy: tuple[float, float]
    status: str = "queued"
    detail: str = ""
    cycle_time_sim_s: float | None = None
    history: list[dict] = field(default_factory=list)

    @property
    def event_id(self) -> str:
        return self.request.event_id

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_PICK_STATUSES

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id, "track_id": self.request.track_id, "status": self.status,
            "detail": self.detail, "grasp_mode": self.request.grasp_mode, "vision_mode": self.request.vision_mode,
            "target": self.request.target.model_dump(), "tray_xy_m": [round(v, 4) for v in self.tray_xy],
            "queued_at_ms": self.request.queued_at_ms, "calibration_status": self.request.calibration_status,
            "cycle_time_sim_s": self.cycle_time_sim_s, "simulated": True,
        }


@dataclass
class Admission:
    accepted: bool
    reason: str
    entry: PickEntry | None = None


class PickScheduler:
    def __init__(self, scene: SceneConfig, bridge: SimBridge, publish: Publish, duplicate_radius_m: float = 0.025,
                 max_queue: int = 10, grasp_mode: str = "kinematic_demo"):
        self.scene = scene
        self.bridge = bridge
        self.publish = publish
        self.duplicate_radius_m = duplicate_radius_m
        self.max_queue = max_queue
        self.grasp_mode = grasp_mode
        self.ids = IdFactory("pick")
        self.alert_ids = IdFactory("alert")
        self.robot_from_tray = scene.tray.robot_from_tray()
        self.entries: dict[str, PickEntry] = {}
        self.queue: deque[PickEntry] = deque()
        self.active: PickEntry | None = None
        self.paused = False
        self.obstructed = False
        self.sim_status: dict | None = None
        self.sim_error: str | None = None
        self.counts: Counter = Counter()
        self._lock = threading.RLock()

    # --- admission ------------------------------------------------------------------------------
    def entry_for_track(self, track_id: str) -> PickEntry | None:
        with self._lock:
            return next((e for e in self.entries.values() if e.request.track_id == track_id), None)

    def submit(self, track_id: str, tray_point: FramePoint, vision_mode: str, source_frame_id: int | None,
               calibration_status: str, yaw_tray_rad: float = 0.0) -> Admission:
        if tray_point.frame != TRAY:
            raise ValueError(f"tray point expected, got frame {tray_point.frame}")
        x, y, _ = tray_point.xyz
        with self._lock:
            existing = self.entry_for_track(track_id)
            if existing:
                return Admission(False, f"track already has pick {existing.event_id} ({existing.status})", existing)
            for e in self.entries.values():
                if e.status == "cancelled":
                    continue
                if np.hypot(e.tray_xy[0] - x, e.tray_xy[1] - y) < self.duplicate_radius_m:
                    return Admission(False, f"duplicate of {e.request.track_id} ({e.event_id}, {e.status}) "
                                            f"within {self.duplicate_radius_m * 1000:.0f} mm", e)
            tray = self.scene.tray
            if not (0.0 <= x <= tray.width_m and 0.0 <= y <= tray.depth_m):
                return Admission(False, f"outside tray ({x:.3f}, {y:.3f}) m")
            if len(self.queue) >= self.max_queue:
                return Admission(False, f"pick queue full ({self.max_queue})")
            # Height is an ESTIMATE (chunk half-height), not measured: one RGB view has no depth.
            z = self.scene.salt_proxy.half_size_m[2]
            robot = self.robot_from_tray.apply(FramePoint(TRAY, (x, y, z)))
            request = PickRequestEvent(
                event_id=self.ids.next(), track_id=track_id, source_frame_id=source_frame_id,
                target=Target(position_m=tuple(round(v, 5) for v in robot.xyz),
                              yaw_rad=round(self.robot_from_tray.apply_yaw(yaw_tray_rad), 5)),
                grasp_mode=self.grasp_mode, vision_mode=vision_mode, queued_at_ms=now_ms(),
                calibration_status=calibration_status,
            )
            entry = PickEntry(request, (x, y))
            entry.history.append({"status": "queued", "t_ms": request.queued_at_ms})
            self.entries[entry.event_id] = entry
            self.queue.append(entry)
            self.counts["queued"] += 1
        self.publish(request.model_dump())
        return Admission(True, "queued", entry)

    # --- dispatch / status ------------------------------------------------------------------------
    def tick(self) -> dict | None:
        """Poll the simulator, update the active pick, dispatch the next one. Returns sim status."""
        try:
            status = self.bridge.status()
        except (SimUnavailable, Exception) as exc:
            if self.sim_error is None:
                self._alert("error", f"simulator unavailable: {exc}")
            self.sim_error = str(exc)
            self.sim_status = None
            return None
        if self.sim_error is not None:
            self._alert("info", "simulator connected")
        self.sim_error = None
        self.sim_status = status
        with self._lock:
            self._update_active(status)
            self._dispatch(status)
        return status

    def _update_active(self, status: dict) -> None:
        if not self.active:
            return
        rec = next((p for p in status.get("picks", []) if p["event_id"] == self.active.event_id), None)
        if rec is None:
            if status["robot"]["robot_state"] == "IDLE" and self.active.status != "queued":
                self._set_status(self.active, "cancelled", "simulator lost the pick (reset or restart)")
            return
        if rec["status"] != self.active.status or rec["detail"] != self.active.detail:
            self.active.cycle_time_sim_s = rec.get("cycle_time_sim_s")
            self._set_status(self.active, rec["status"], rec["detail"])

    def _dispatch(self, status: dict) -> None:
        if self.active and not self.active.terminal:
            return
        self.active = None
        if not self.queue or self.paused or self.obstructed:
            return
        if status["robot"]["robot_state"] != "IDLE":
            return
        entry = self.queue.popleft()
        try:
            code, body = self.bridge.pick(entry.request.model_dump(mode="json"))
        except SimUnavailable as exc:
            self.queue.appendleft(entry)  # never drop an accepted command
            self._alert("warning", f"dispatch of {entry.event_id} deferred: {exc}")
            return
        if code in (200, 202):
            self.active = entry
            self._set_status(entry, body.get("status", "accepted"), body.get("detail", ""))
        elif code == 409 and body.get("error") == "robot busy":
            self.queue.appendleft(entry)
        else:
            self._set_status(entry, "rejected_invalid", f"simulator refused ({code}): {body.get('error', body)}")

    def _set_status(self, entry: PickEntry, status: str, detail: str) -> None:
        entry.status, entry.detail = status, detail
        entry.history.append({"status": status, "t_ms": now_ms(), "detail": detail})
        if entry.terminal:
            self.counts[status] += 1
            if status == "completed_sim":
                self.counts[f"completed_{entry.request.grasp_mode}"] += 1
        self.publish(PickStatusEvent(
            event_id=f"{entry.event_id}:{len(entry.history)}", pick_event_id=entry.event_id,
            track_id=entry.request.track_id, status=status, detail=detail, grasp_mode=entry.request.grasp_mode,
            vision_mode=entry.request.vision_mode, cycle_time_sim_s=entry.cycle_time_sim_s,
        ).model_dump())

    def _alert(self, level: str, message: str) -> None:
        self.publish(AlertEvent(event_id=self.alert_ids.next(), source="orchestrator", level=level,
                                message=message).model_dump())

    # --- operator -------------------------------------------------------------------------------
    def reset(self) -> None:
        with self._lock:
            for e in list(self.queue):
                self._set_status(e, "cancelled", "session reset by operator")
            self.queue.clear()
            if self.active and not self.active.terminal:
                self._set_status(self.active, "cancelled", "session reset by operator")
            self.active = None
            self.entries.clear()
            self.counts.clear()
        try:
            self.bridge.reset()
        except SimUnavailable as exc:
            self._alert("warning", f"simulator reset failed: {exc}")

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "active": self.active.to_dict() if self.active else None,
                "queue": [e.to_dict() for e in self.queue],
                "picks": [e.to_dict() for e in list(self.entries.values())[-50:]],
                "paused": self.paused, "obstructed": self.obstructed, "grasp_mode": self.grasp_mode,
                "counts": dict(self.counts), "sim_connected": self.sim_error is None and self.sim_status is not None,
                "sim_error": self.sim_error, "timestamp_ms": now_ms(),
            }


class SchedulerLoop:
    """Background thread: ticks the scheduler (~5 Hz) and publishes robot status snapshots."""

    def __init__(self, scheduler: PickScheduler, publish_snapshot: Publish, period_s: float = 0.2):
        self.scheduler = scheduler
        self.publish_snapshot = publish_snapshot
        self.period_s = period_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pick-scheduler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            status = self.scheduler.tick()
            self.publish_snapshot({
                "type": "sim.snapshot", "timestamp_ms": now_ms(), "simulated": True,
                "robot": status["robot"] if status else None,
                "sim_metrics": status["metrics"] if status else None,
                "sim_counts": status["counts"] if status else None,
                "objects": status["objects"] if status else [],
                "scheduler": self.scheduler.snapshot(),
            })
            self._stop.wait(max(0.0, self.period_s - (time.perf_counter() - t0)))
