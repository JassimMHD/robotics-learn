"""Robot pick state machine (spec section 9). Stepped once per physics step from the sim thread.

IDLE -> VALIDATE_TARGET -> PLAN_APPROACH -> MOVE_PREGRASP -> MOVE_TO_GRASP -> CLOSE_GRIPPER
     -> VERIFY_GRASP -> LIFT -> TRANSFER_TO_REJECT_BIN -> OPEN_GRIPPER -> VERIFY_RELEASE
     -> RETURN_HOME -> COMPLETED_SIM -> IDLE
Any stage -> FAILED_SIM -> SAFE_RECOVERY -> IDLE

Grasp modes:
  kinematic_demo   the chunk is ATTACHED to the gripper after closing (scripted, clearly labeled);
                   never counted as a physics grasp.
  contact_physics  the chunk is held only by simulated finger contact and friction; success requires
                   both pads in contact and the chunk staying with the gripper through lift/transfer.
"""

import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

import mujoco
import numpy as np

from saltsim.object_manager import ObjectError, ObjectManager, ProxyState
from saltsim.planner import Planner, PlanningError, Segment
from saltsim.scene import ARM_JOINTS, GRIPPER_ACTUATOR, TCP_SITE
from saltsort_common.scene_config import SceneConfig

GRIPPER_OPEN, GRIPPER_CLOSED = 0.0, 255.0
# Measured in this scene: inner pad gap ~= 85.2 mm - 0.336 mm per control unit (0..255).
GAP_AT_OPEN_M, GAP_PER_CTRL_M = 0.0852, 0.000336
PRESHAPE_MARGIN_M = 0.020
PRESHAPE_S = 0.5
SETTLE_S = 0.15
CLOSE_S = 0.8
OPEN_S = 0.6
RELEASE_VERIFY_S = 2.0
RETENTION_LIMIT_M = 0.03


def gripper_ctrl_for_gap(gap_m: float) -> float:
    return float(np.clip((GAP_AT_OPEN_M - gap_m) / GAP_PER_CTRL_M, GRIPPER_OPEN, GRIPPER_CLOSED))


class RobotState(str, Enum):
    IDLE = "IDLE"
    VALIDATE_TARGET = "VALIDATE_TARGET"
    PLAN_APPROACH = "PLAN_APPROACH"
    MOVE_PREGRASP = "MOVE_PREGRASP"
    MOVE_TO_GRASP = "MOVE_TO_GRASP"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    VERIFY_GRASP = "VERIFY_GRASP"
    LIFT = "LIFT"
    TRANSFER_TO_REJECT_BIN = "TRANSFER_TO_REJECT_BIN"
    OPEN_GRIPPER = "OPEN_GRIPPER"
    VERIFY_RELEASE = "VERIFY_RELEASE"
    RETURN_HOME = "RETURN_HOME"
    COMPLETED_SIM = "COMPLETED_SIM"
    FAILED_SIM = "FAILED_SIM"
    SAFE_RECOVERY = "SAFE_RECOVERY"


PICK_STATUS = {
    RobotState.VALIDATE_TARGET: "accepted",
    RobotState.PLAN_APPROACH: "accepted",
    RobotState.MOVE_PREGRASP: "approaching",
    RobotState.MOVE_TO_GRASP: "approaching",
    RobotState.CLOSE_GRIPPER: "grasping",
    RobotState.VERIFY_GRASP: "verifying",
    RobotState.LIFT: "lifting",
    RobotState.TRANSFER_TO_REJECT_BIN: "transferring",
    RobotState.OPEN_GRIPPER: "releasing",
    RobotState.VERIFY_RELEASE: "releasing",
}
TERMINAL_STATUSES = {
    "completed_sim", "rejected_invalid", "unreachable", "timeout", "grasp_failed_sim", "collision_blocked", "cancelled",
}
MOTION_STATES = {
    RobotState.MOVE_PREGRASP, RobotState.MOVE_TO_GRASP, RobotState.LIFT, RobotState.TRANSFER_TO_REJECT_BIN,
    RobotState.RETURN_HOME,
}


@dataclass
class PickRecord:
    event_id: str
    track_id: str
    target_position_m: list[float]
    target_yaw_rad: float
    grasp_mode: str
    vision_mode: str
    source_frame_id: int | None = None
    status: str = "accepted"
    detail: str = ""
    proxy_body: str | None = None
    accepted_sim_time_s: float = 0.0
    finished_sim_time_s: float | None = None
    accepted_wall_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    finished_wall_ms: int | None = None
    history: list[dict] = field(default_factory=list)

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def to_dict(self) -> dict:
        cycle = None if self.finished_sim_time_s is None else self.finished_sim_time_s - self.accepted_sim_time_s
        return {
            "event_id": self.event_id, "track_id": self.track_id, "status": self.status, "detail": self.detail,
            "grasp_mode": self.grasp_mode, "vision_mode": self.vision_mode, "source_frame_id": self.source_frame_id,
            "target": {"frame": "robot_base", "position_m": self.target_position_m, "yaw_rad": self.target_yaw_rad},
            "proxy_body": self.proxy_body, "simulated": True,
            "accepted_sim_time_s": round(self.accepted_sim_time_s, 3),
            "finished_sim_time_s": None if self.finished_sim_time_s is None else round(self.finished_sim_time_s, 3),
            "cycle_time_sim_s": None if cycle is None else round(cycle, 3),
            "accepted_wall_ms": self.accepted_wall_ms, "finished_wall_ms": self.finished_wall_ms,
            "history": self.history,
        }


class PickController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, cfg: SceneConfig, planner: Planner,
                 objects: ObjectManager):
        self.model, self.data, self.cfg = model, data, cfg
        self.planner, self.objects = planner, objects
        self.geo = planner.geo
        self.arm_act = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j.removesuffix("_joint"))
                                 for j in ARM_JOINTS])
        self.grip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR)
        self.tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        self.state = RobotState.IDLE
        self.state_t0 = 0.0
        self.active: PickRecord | None = None
        self.proxy: ProxyState | None = None
        self.segments: list[Segment] = []
        self.plan = None
        self.held_rel: tuple[np.ndarray, np.ndarray] | None = None  # kinematic attachment (pos, quat) in TCP frame
        self.collision_flag = False
        self.target_reachable: bool | None = None
        self.last_collision: str = ""
        self.counts: Counter = Counter()
        self.drops = 0  # spreads release points over the bin so chunks do not stack into a tower
        half = cfg.salt_proxy.half_size_m
        self.preshape_ctrl = gripper_ctrl_for_gap(2 * max(half[0], half[1]) + PRESHAPE_MARGIN_M)
        self.hold_arm(self.arm_q())

    # --- public API (called from the sim thread only) ---------------------------------------------
    def arm_q(self) -> np.ndarray:
        return self.data.qpos[self.geo.arm_qadr].copy()

    def hold_arm(self, q: np.ndarray) -> None:
        self.data.ctrl[self.arm_act] = q

    @property
    def busy(self) -> bool:
        return self.state != RobotState.IDLE

    def start_pick(self, record: PickRecord) -> None:
        if self.busy:
            raise RuntimeError("robot busy")
        self.active = record
        self.proxy = None
        self.collision_flag = False
        self.target_reachable = None
        record.accepted_sim_time_s = self.data.time
        self._enter(RobotState.VALIDATE_TARGET)

    def cancel(self, reason: str = "cancelled by operator") -> None:
        if self.active and not self.active.terminal:
            self._fail("cancelled", reason)

    def reset_counts(self) -> None:
        self.counts.clear()
        self.drops = 0

    # --- per physics step --------------------------------------------------------------------------
    def step(self) -> None:
        t = self.data.time
        dt = t - self.state_t0
        st = self.state
        rec = self.active

        if rec and not rec.terminal and t - rec.accepted_sim_time_s > self.cfg.robot.pick_timeout_s:
            self._fail("timeout", f"pick exceeded {self.cfg.robot.pick_timeout_s:.0f} s simulated time in {st.value}")
            return

        if st == RobotState.IDLE:
            return
        if st == RobotState.VALIDATE_TARGET:
            pos = np.array(rec.target_position_m)
            try:
                self.planner.validate_target(pos)
                self.proxy = self.objects.place(rec.track_id, pos, rec.target_yaw_rad, self.geo.robot_geoms)
                rec.proxy_body = self.proxy.slot.body_name
            except PlanningError as e:
                return self._fail(e.outcome, e.detail)
            except ObjectError as e:
                return self._fail("rejected_invalid", str(e))
            # Pre-shape the fingers to the chunk size (+margin) so planning checks the real finger span.
            self.data.ctrl[self.grip_act] = self.preshape_ctrl
            self._enter(RobotState.PLAN_APPROACH)
        elif st == RobotState.PLAN_APPROACH:
            if dt < PRESHAPE_S:
                return
            try:
                self.plan = self.planner.plan_pick(
                    self.objects.position(self.proxy), rec.target_yaw_rad, self.arm_q(), self.data.qpos.copy(),
                    {self.proxy.geom_id}, drop_index=self.drops,
                )
            except PlanningError as e:
                self.target_reachable = e.outcome != "unreachable"
                return self._fail(e.outcome, e.detail)
            self.target_reachable = True
            self._enter(RobotState.MOVE_PREGRASP)
        elif st in MOTION_STATES:
            self._follow(st, dt)
        elif st == RobotState.CLOSE_GRIPPER:
            self.data.ctrl[self.grip_act] = GRIPPER_CLOSED
            if dt >= CLOSE_S:
                self._enter(RobotState.VERIFY_GRASP)
        elif st == RobotState.VERIFY_GRASP:
            if rec.grasp_mode == "kinematic_demo":
                self._attach()
            elif not self._pads_touching():
                return self._fail("grasp_failed_sim", "fingers closed without contact on both sides of the chunk")
            self._enter(RobotState.LIFT)
        elif st == RobotState.OPEN_GRIPPER:
            self._detach()
            self.data.ctrl[self.grip_act] = self.preshape_ctrl
            if dt >= OPEN_S:
                self._enter(RobotState.VERIFY_RELEASE)
        elif st == RobotState.VERIFY_RELEASE:
            if self._in_bin():
                self._complete()
            elif dt >= RELEASE_VERIFY_S:
                self._fail("grasp_failed_sim", "chunk did not land in the reject bin")
        elif st == RobotState.FAILED_SIM:
            self._detach()
            self.data.ctrl[self.grip_act] = GRIPPER_OPEN
            self.segments = self.planner.plan_retreat(self.arm_q(), self.data.qpos.copy())
            self._enter(RobotState.SAFE_RECOVERY)
        elif st == RobotState.SAFE_RECOVERY:
            if not self.segments:
                self._enter(RobotState.IDLE)
                return
            seg = self.segments[0]
            self.hold_arm(seg.sample(dt))
            if dt >= seg.duration_s + SETTLE_S:
                self.segments.pop(0)
                self.state_t0 = t
        elif st == RobotState.COMPLETED_SIM:
            self._enter(RobotState.IDLE)

        if self.held_rel is not None:
            self._carry()

    def after_step(self) -> None:
        """Checks that need post-step contacts: collisions and grasp retention."""
        if self.state in MOTION_STATES or self.state == RobotState.SAFE_RECOVERY:
            ignore = {self.proxy.geom_id} if self.proxy else set()
            hits = self.geo.robot_collisions(self.data, ignore)
            if hits:
                a, b = hits[0]
                self.collision_flag = True
                self.last_collision = f"{self.model.geom(a).name or a} / {self.model.geom(b).name or b}"
                if self.state != RobotState.SAFE_RECOVERY and self.active and not self.active.terminal:
                    self._fail("collision_blocked", f"unexpected robot contact: {self.last_collision}")
                    return
        if (
            self.active and self.active.grasp_mode == "contact_physics" and self.proxy
            and self.state in (RobotState.LIFT, RobotState.TRANSFER_TO_REJECT_BIN)
        ):
            gap = np.linalg.norm(self.objects.position(self.proxy) - self.data.site_xpos[self.tcp])
            if gap > RETENTION_LIMIT_M:
                self._fail("grasp_failed_sim", f"chunk slipped/dropped during {self.state.value} ({gap * 1000:.0f} mm)")

    # --- helpers ---------------------------------------------------------------------------------------
    NEXT = {
        RobotState.MOVE_PREGRASP: (RobotState.MOVE_TO_GRASP, "move_to_grasp"),
        RobotState.MOVE_TO_GRASP: (RobotState.CLOSE_GRIPPER, None),
        RobotState.LIFT: (RobotState.TRANSFER_TO_REJECT_BIN, "transfer"),
        RobotState.TRANSFER_TO_REJECT_BIN: (RobotState.OPEN_GRIPPER, None),
        RobotState.RETURN_HOME: (RobotState.IDLE, None),
    }
    SEGMENT_OF = {
        RobotState.MOVE_PREGRASP: "move_pregrasp", RobotState.MOVE_TO_GRASP: "move_to_grasp",
        RobotState.LIFT: "lift", RobotState.TRANSFER_TO_REJECT_BIN: "transfer", RobotState.RETURN_HOME: "return_home",
    }

    def _follow(self, st: RobotState, dt: float) -> None:
        seg = self.plan.segments[self.SEGMENT_OF[st]]
        self.hold_arm(seg.sample(dt))
        if dt >= seg.duration_s + SETTLE_S:
            self._enter(self.NEXT[st][0])

    def _enter(self, state: RobotState) -> None:
        self.state = state
        self.state_t0 = self.data.time
        rec = self.active
        if rec and not rec.terminal and state in PICK_STATUS:
            status = PICK_STATUS[state]
            if status != rec.status or not rec.history:
                rec.status = status
            rec.history.append({"state": state.value, "sim_time_s": round(self.data.time, 3)})
        elif rec and state in (RobotState.RETURN_HOME, RobotState.IDLE, RobotState.SAFE_RECOVERY):
            rec.history.append({"state": state.value, "sim_time_s": round(self.data.time, 3)})
        if state == RobotState.IDLE:
            self.active = None
            self.plan = None

    def _finish(self, status: str, detail: str) -> None:
        rec = self.active
        rec.status, rec.detail = status, detail
        rec.finished_sim_time_s = self.data.time
        rec.finished_wall_ms = int(time.time() * 1000)
        rec.history.append({"state": status, "sim_time_s": round(self.data.time, 3), "detail": detail})

    def _fail(self, outcome: str, detail: str) -> None:
        mode = self.active.grasp_mode
        self._finish(outcome, detail)
        self.counts[f"failed_{outcome}"] += 1
        if outcome == "grasp_failed_sim":
            self.counts[f"grasp_failed_{mode}"] += 1
        self.state = RobotState.FAILED_SIM
        self.state_t0 = self.data.time

    def _complete(self) -> None:
        self.proxy.status = "deposited"
        self.drops += 1
        self._finish("completed_sim", f"simulated pick completed ({self.active.grasp_mode})")
        self.counts[f"completed_{self.active.grasp_mode}"] += 1
        self._enter(RobotState.RETURN_HOME)

    def _pads_touching(self) -> bool:
        touched = {"left": False, "right": False}
        for c in self.data.contact[: self.data.ncon]:
            pair = {int(c.geom1), int(c.geom2)}
            if self.proxy.geom_id in pair:
                for side, pads in self.geo.pad_geoms.items():
                    if pair & pads:
                        touched[side] = True
        return all(touched.values())

    def _tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.tcp])
        return self.data.site_xpos[self.tcp].copy(), quat

    def _attach(self) -> None:
        tcp_pos, tcp_quat = self._tcp_pose()
        obj_pos = self.objects.position(self.proxy)
        obj_quat = self.data.xquat[self.proxy.body_id].copy()
        inv = np.zeros(4)
        mujoco.mju_negQuat(inv, tcp_quat)
        rel_pos = np.zeros(3)
        mujoco.mju_rotVecQuat(rel_pos, obj_pos - tcp_pos, inv)
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, inv, obj_quat)
        self.held_rel = (rel_pos, rel_quat)
        self.objects.set_collision(self.proxy, False)
        self.model.body_gravcomp[self.proxy.body_id] = 1.0
        self.proxy.status = "held"

    def _carry(self) -> None:
        tcp_pos, tcp_quat = self._tcp_pose()
        rel_pos, rel_quat = self.held_rel
        world = np.zeros(3)
        mujoco.mju_rotVecQuat(world, rel_pos, tcp_quat)
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, tcp_quat, rel_quat)
        self.objects.set_world_pose(self.proxy, tcp_pos + world, quat)

    def _detach(self) -> None:
        if self.held_rel is not None and self.proxy is not None:
            self.held_rel = None
            self.objects.set_collision(self.proxy, True)
            self.model.body_gravcomp[self.proxy.body_id] = 0.0
            self.proxy.status = "active"

    def _in_bin(self) -> bool:
        b = self.cfg.reject_bin
        p = self.objects.position(self.proxy)
        v = np.linalg.norm(self.data.qvel[self.proxy.dadr : self.proxy.dadr + 3])
        inside = (abs(p[0] - b.center_in_robot.x_m) < b.inner_size_m[0] / 2
                  and abs(p[1] - b.center_in_robot.y_m) < b.inner_size_m[1] / 2)
        return inside and p[2] < b.wall_height_m and v < 0.05

    def status(self) -> dict:
        return {
            "type": "sim.robot.status",
            "pick_event_id": self.active.event_id if self.active else None,
            "robot_state": self.state.value,
            "grasp_mode": self.active.grasp_mode if self.active else None,
            "simulation_time_s": round(self.data.time, 3),
            "joint_positions_rad": np.round(self.arm_q(), 4).tolist(),
            "target_reachable": self.target_reachable,
            "collision_flag": self.collision_flag,
            "last_collision": self.last_collision,
            "simulated": True,
        }
