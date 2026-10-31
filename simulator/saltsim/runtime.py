"""Simulator runtime: owns MjModel/MjData and the physics loop. Only the sim thread mutates state;
other threads submit commands through `call()`.
"""

import queue
import threading
import time
from collections import deque
from concurrent.futures import Future
from typing import Any, Callable

import cv2
import mujoco
import numpy as np

from saltsim.controller import PickController, PickRecord
from saltsim.object_manager import ObjectManager
from saltsim.planner import Planner
from saltsim.scene import HOME_QPOS, compile_scene, proxy_slots
from saltsort_common.scene_config import SceneConfig

MAX_PICK_HISTORY = 200


class SimRuntime:
    def __init__(self, cfg: SceneConfig, timestep_s: float = 0.002, render_size: tuple[int, int] = (480, 640)):
        self.cfg = cfg
        self.spec, self.model = compile_scene(cfg)
        self.model.opt.timestep = timestep_s
        self.data = mujoco.MjData(self.model)
        self.planner = Planner(self.model, cfg)
        self.objects = ObjectManager(self.model, self.data, proxy_slots(cfg))
        self._go_home()
        self.controller = PickController(self.model, self.data, cfg, self.planner, self.objects)
        self.picks: dict[str, PickRecord] = {}
        self._commands: queue.Queue[tuple[Callable[[], Any], Future]] = queue.Queue()
        self._step_times: deque[tuple[float, float]] = deque(maxlen=600)  # (wall, sim)
        self._steps_total = 0
        self.render_size = render_size
        self._renderer: mujoco.Renderer | None = None
        self._jpeg: bytes | None = None
        self._jpeg_time = 0.0
        self._jpeg_lock = threading.Lock()
        self._render_wanted_until = 0.0
        self._last_render = 0.0
        self.render_fps_target = 12.0
        self._render_times: deque[float] = deque(maxlen=30)
        self.started_wall = time.time()

    def _go_home(self) -> None:
        self.data.qpos[self.planner.geo.arm_qadr] = HOME_QPOS
        mujoco.mj_forward(self.model, self.data)

    # --- cross-thread command interface ----------------------------------------------------------
    def call(self, fn: Callable[[], Any], timeout_s: float = 5.0) -> Any:
        fut: Future = Future()
        self._commands.put((fn, fut))
        return fut.result(timeout=timeout_s)

    def _drain_commands(self) -> None:
        while True:
            try:
                fn, fut = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                fut.set_result(fn())
            except Exception as exc:  # deliver to caller
                fut.set_exception(exc)

    # --- commands (run on sim thread) ------------------------------------------------------------
    def cmd_pick(self, req: dict) -> tuple[int, dict]:
        """Idempotent by event_id. Returns (http_status, body)."""
        existing = self.picks.get(req["event_id"])
        if existing:
            return 200, {**existing.to_dict(), "duplicate_request": True}
        for rec in self.picks.values():
            if rec.track_id == req["track_id"] and rec.status == "completed_sim":
                return 409, {"error": "track already processed in simulation; reset to re-arm", "track_id": rec.track_id}
        if self.controller.busy:
            return 409, {"error": "robot busy", "robot_state": self.controller.state.value}
        target = req["target"]
        if target.get("frame") != "robot_base":
            return 422, {"error": f"target frame must be robot_base, got {target.get('frame')!r}"}
        rec = PickRecord(
            event_id=req["event_id"], track_id=req["track_id"],
            target_position_m=[float(v) for v in target["position_m"]], target_yaw_rad=float(target.get("yaw_rad", 0.0)),
            grasp_mode=req["grasp_mode"], vision_mode=req["vision_mode"], source_frame_id=req.get("source_frame_id"),
        )
        self.picks[rec.event_id] = rec
        self._trim_history()
        self.controller.start_pick(rec)
        return 202, rec.to_dict()

    def cmd_place(self, track_id: str, position_m: list[float], yaw_rad: float) -> dict:
        pos = np.array(position_m, dtype=float)
        self.planner.validate_target(pos)
        proxy = self.objects.place(track_id, pos, yaw_rad, self.planner.geo.robot_geoms)
        return {"track_id": track_id, "proxy_body": proxy.slot.body_name, "status": proxy.status}

    def cmd_reset(self) -> dict:
        self.controller.cancel("session reset by operator")
        if self.controller.busy:
            # Abort motion immediately: reset the robot to home (simulation only).
            self.controller._detach()
            self.controller.state = self.controller.state.IDLE
            self.controller.active = None
        self.objects.reset()
        self._go_home()
        self.data.qvel[:] = 0
        self.controller.hold_arm(np.array(HOME_QPOS))
        self.data.ctrl[self.controller.grip_act] = 0.0
        self.controller.collision_flag = False
        self.controller.target_reachable = None
        self.controller.reset_counts()
        self.picks.clear()
        return {"reset": True}

    def _trim_history(self) -> None:
        while len(self.picks) > MAX_PICK_HISTORY:
            oldest = next(iter(self.picks))
            if not self.picks[oldest].terminal:
                break
            del self.picks[oldest]

    # --- status ----------------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "robot": self.controller.status(),
            "picks": [r.to_dict() for r in list(self.picks.values())[-50:]],
            "objects": self.objects.summary(),
            "counts": dict(self.controller.counts),
            "metrics": self.metrics(),
        }

    def metrics(self) -> dict:
        rtf = steps_per_s = None
        if len(self._step_times) >= 2:
            (w0, s0), (w1, s1) = self._step_times[0], self._step_times[-1]
            if w1 > w0:
                rtf = (s1 - s0) / (w1 - w0)
                steps_per_s = (s1 - s0) / self.model.opt.timestep / (w1 - w0)
        render_fps = None
        if len(self._render_times) >= 2 and self._render_times[-1] > self._render_times[0]:
            render_fps = (len(self._render_times) - 1) / (self._render_times[-1] - self._render_times[0])
        return {
            "simulation_time_s": round(self.data.time, 3),
            "real_time_factor": None if rtf is None else round(rtf, 3),
            "physics_steps_per_s": None if steps_per_s is None else round(steps_per_s),
            "render_fps": None if render_fps is None else round(render_fps, 1),
            "timestep_s": self.model.opt.timestep,
            "source": "simulator wall clock over the last ~600 loop iterations",
        }

    # --- loop --------------------------------------------------------------------------------------------
    def step(self) -> None:
        self.controller.step()
        mujoco.mj_step(self.model, self.data)
        self.controller.after_step()
        self._steps_total += 1

    def tick(self, wall_start: float, sim_start: float, max_lag_s: float = 1 / 30, time_scale: float = 1.0) -> None:
        """Advance physics to wall-clock time x time_scale (bounded per tick), drain commands, maybe render.

        time_scale > 1 is for automated tests only; interactive runs use 1.0 (real time)."""
        self._drain_commands()
        target = (time.perf_counter() - wall_start) * time_scale
        tick_start = time.perf_counter()
        while self.data.time - sim_start < target:
            self.step()
            if time.perf_counter() - tick_start > max_lag_s:
                break  # falling behind: real_time_factor will report < 1
        self._step_times.append((time.perf_counter(), self.data.time))
        self._maybe_render()

    def run_headless(self, stop: threading.Event, time_scale: float = 1.0) -> None:
        wall_start, sim_start = time.perf_counter(), self.data.time
        while not stop.is_set():
            self.tick(wall_start, sim_start, time_scale=time_scale)
            time.sleep(0.002)

    # --- offscreen render stream (sim thread only: GL context is thread-bound) ------------------
    def request_frames(self) -> None:
        self._render_wanted_until = time.time() + 3.0

    def latest_jpeg(self, newer_than: float = 0.0) -> bytes | None:
        """Most recent render, or None if it is not newer than `newer_than` (wall time)."""
        with self._jpeg_lock:
            return self._jpeg if self._jpeg_time > newer_than else None

    def _maybe_render(self) -> None:
        now = time.time()
        if now > self._render_wanted_until or now - self._last_render < 1.0 / self.render_fps_target:
            return
        self._last_render = now
        try:
            if self._renderer is None:
                h, w = self.render_size
                self._renderer = mujoco.Renderer(self.model, h, w)
                # Shadows/reflections cost ~50 ms per frame on integrated Intel graphics (measured
                # 67 -> 15 ms at 640x480); the stream renders on the sim thread, so keep it cheap.
                self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                self._cam = mujoco.MjvCamera()
                self._cam.lookat[:] = [0.30, 0.10, 0.10]
                self._cam.distance, self._cam.azimuth, self._cam.elevation = 1.6, 200.0, -35.0
            self._renderer.update_scene(self.data, self._cam)
            rgb = self._renderer.render()
        except Exception as exc:  # rendering must never block robot control
            self.render_error = str(exc)
            return
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        st = self.controller.status()
        label = f"SIMULATED | {st['robot_state']}"
        if st["pick_event_id"]:
            label += f" | {st['pick_event_id']} ({st['grasp_mode']})"
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 26), (40, 20, 60), -1)
        cv2.putText(bgr, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._jpeg_lock:
                self._jpeg, self._jpeg_time = buf.tobytes(), now
            self._render_times.append(now)
