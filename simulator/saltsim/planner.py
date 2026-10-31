"""Pick-and-place planning: IK waypoints, smooth joint trajectories, collision checking.

All poses are in the robot_base frame (== MuJoCo world). Planning runs on a private MjData copy.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

from saltsim.ik import IkSolver, top_down_rotation
from saltsim.scene import ARM_JOINTS, GRIPPER_PREFIX, HOME_QPOS, TCP_SITE
from saltsort_common.scene_config import SceneConfig


# Closed 2F-85 pads reach 17.7 mm below the pinch site (measured in this scene); keep them off the tray floor.
FINGERTIP_DROP_M = 0.018
FLOOR_CLEARANCE_M = 0.003


class PlanningError(Exception):
    def __init__(self, outcome: str, detail: str):
        super().__init__(detail)
        self.outcome = outcome  # unreachable | collision_blocked | rejected_invalid
        self.detail = detail


@dataclass
class Segment:
    name: str
    q_from: np.ndarray
    q_to: np.ndarray
    duration_s: float

    def sample(self, t: float) -> np.ndarray:
        s = min(max(t / self.duration_s, 0.0), 1.0)
        blend = 10 * s**3 - 15 * s**4 + 6 * s**5  # minimum-jerk profile: zero vel/acc at both ends
        return self.q_from + (self.q_to - self.q_from) * blend


@dataclass
class PickPlan:
    target_pos: np.ndarray
    grasp_yaw_rad: float
    waypoints: dict[str, np.ndarray] = field(default_factory=dict)
    segments: dict[str, Segment] = field(default_factory=dict)


def _wrap_half_turn(yaw: float) -> float:
    """The parallel gripper is symmetric under 180 deg: map yaw into [-pi/2, pi/2] to keep wrist motion small."""
    yaw = float(np.arctan2(np.sin(yaw), np.cos(yaw)))
    if yaw > np.pi / 2:
        yaw -= np.pi
    elif yaw < -np.pi / 2:
        yaw += np.pi
    return yaw


class RobotGeometry:
    """Index helpers: which geoms belong to the robot / gripper."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        robot_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        gripper_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_PREFIX + "base_mount")
        self.robot_geoms = self._subtree_geoms(robot_root)
        self.gripper_geoms = self._subtree_geoms(gripper_root)
        self.pad_geoms = {
            "left": {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, GRIPPER_PREFIX + f"left_pad{i}") for i in (1, 2)},
            "right": {
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, GRIPPER_PREFIX + f"right_pad{i}") for i in (1, 2)
            },
        }
        self.arm_qadr = np.array(
            [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS]
        )
        self.arm_dadr = np.array(
            [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS]
        )

    def _subtree_geoms(self, root: int) -> set[int]:
        def under(body: int) -> bool:
            while body > 0:
                if body == root:
                    return True
                body = self.model.body_parentid[body]
            return False

        return {g for g in range(self.model.ngeom) if under(self.model.geom_bodyid[g])}

    def robot_collisions(self, data: mujoco.MjData, ignore_geoms: set[int] = frozenset()) -> list[tuple[int, int]]:
        """Penetrating contacts between the robot and anything else (or non-gripper self-contacts)."""
        hits = []
        for c in data.contact[: data.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if c.dist >= 0 or g1 in ignore_geoms or g2 in ignore_geoms:
                continue
            r1, r2 = g1 in self.robot_geoms, g2 in self.robot_geoms
            if r1 and r2:
                if g1 in self.gripper_geoms and g2 in self.gripper_geoms:
                    continue  # internal gripper linkage contacts are part of the mechanism
                hits.append((g1, g2))
            elif r1 or r2:
                hits.append((g1, g2))
        return hits


class Planner:
    def __init__(self, model: mujoco.MjModel, cfg: SceneConfig):
        self.model = model
        self.cfg = cfg
        self.geo = RobotGeometry(model)
        self.ik = IkSolver(model, TCP_SITE, self.geo.arm_qadr, self.geo.arm_dadr)
        self._scratch = mujoco.MjData(model)
        self.home = np.array(HOME_QPOS)
        self.vmax = cfg.robot.max_joint_speed_rad_s

    # --- workspace -------------------------------------------------------------------------
    def validate_target(self, pos_robot: np.ndarray) -> None:
        tray = self.cfg.tray
        p_tray = np.linalg.inv(tray.robot_from_tray().matrix) @ np.append(pos_robot, 1.0)
        # A chunk centered closer than its own half-size to a wall would intersect the wall.
        half = self.cfg.salt_proxy.half_size_m
        margin = max(half[0], half[1]) + 0.002
        if not (margin <= p_tray[0] <= tray.width_m - margin and margin <= p_tray[1] <= tray.depth_m - margin):
            raise PlanningError(
                "rejected_invalid", f"target outside tray interior (tray-frame x={p_tray[0]:.3f}, y={p_tray[1]:.3f} m)"
            )
        if not (-0.005 <= p_tray[2] <= 0.10):
            raise PlanningError("rejected_invalid", f"target height {p_tray[2]:.3f} m above tray is implausible")

    # --- planning --------------------------------------------------------------------------
    def drop_point(self, drop_index: int) -> np.ndarray:
        """Release TCP position: cycles over a 3x3 grid inside the bin so chunks spread out."""
        b = self.cfg.reject_bin
        step = np.array(b.inner_size_m) / 3.3
        i, j = divmod(drop_index % 9, 3)
        offsets = np.array([0, -1, 1])
        return np.array([
            b.center_in_robot.x_m + offsets[i] * step[0],
            b.center_in_robot.y_m + offsets[j] * step[1],
            self.cfg.robot.release_height_m,
        ])

    def plan_pick(
        self, target_pos: np.ndarray, object_yaw_rad: float, q_start: np.ndarray, live_qpos: np.ndarray,
        target_geoms: set[int], drop_index: int = 0,
    ) -> PickPlan:
        self.validate_target(target_pos)
        # Fingers close along TCP y. Prefer closing across the object's shorter (y) side; if that is
        # blocked (e.g. open fingers would hit a tray wall), try the gripper rotated by 90 degrees.
        errors = []
        for offset in (np.pi / 2, 0.0):
            yaw = _wrap_half_turn(object_yaw_rad + offset)
            try:
                return self._plan_pick_with_yaw(target_pos, yaw, q_start, live_qpos, target_geoms, drop_index)
            except PlanningError as e:
                errors.append(e)
        unreachable = [e for e in errors if e.outcome == "unreachable"]
        worst = unreachable[0] if len(unreachable) == len(errors) else next(
            e for e in errors if e.outcome != "unreachable")
        raise PlanningError(worst.outcome, " | ".join(e.detail for e in errors))

    def _plan_pick_with_yaw(
        self, target_pos: np.ndarray, yaw: float, q_start: np.ndarray, live_qpos: np.ndarray, target_geoms: set[int],
        drop_index: int,
    ) -> PickPlan:
        r = self.cfg.robot
        tray_z = self.cfg.tray.pose_in_robot.z_m
        obj_top = tray_z + self.cfg.salt_proxy.height_m
        grasp_z = max(target_pos[2], tray_z + FINGERTIP_DROP_M + FLOOR_CLEARANCE_M)
        if grasp_z > obj_top - 0.004:
            raise PlanningError("rejected_invalid", f"object top {obj_top:.3f} m too low for the fingertips to grasp")

        grasp = target_pos.copy()
        grasp[2] = grasp_z
        pre = grasp.copy()
        pre[2] = obj_top + r.pregrasp_clearance_m
        lift = grasp.copy()
        lift[2] = tray_z + r.lift_height_m
        bin_top = self.drop_point(drop_index)

        plan = PickPlan(target_pos=target_pos, grasp_yaw_rad=yaw)
        seed = q_start
        for name, pos in (("pregrasp", pre), ("grasp", grasp), ("lift", lift), ("release", bin_top)):
            res = self.ik.solve(pos, top_down_rotation(yaw), seed)
            if not res.success:
                raise PlanningError(
                    "unreachable", f"IK failed for {name} at {np.round(pos, 3).tolist()} "
                    f"(pos err {res.pos_error_m * 1000:.1f} mm, rot err {res.rot_error_rad:.3f} rad)"
                )
            plan.waypoints[name] = res.qpos
            seed = res.qpos

        w = plan.waypoints
        order = [
            ("move_pregrasp", q_start, w["pregrasp"], set()),
            ("move_to_grasp", w["pregrasp"], w["grasp"], target_geoms),
            ("lift", w["grasp"], w["lift"], target_geoms),
            ("transfer", w["lift"], w["release"], target_geoms),
            ("return_home", w["release"], self.home, set()),
        ]
        for name, qa, qb, ignore in order:
            seg = Segment(name, qa, qb, self._duration(qa, qb, slow=name in ("move_to_grasp", "lift")))
            self._check_segment(seg, live_qpos, ignore)
            plan.segments[name] = seg
        return plan

    def plan_retreat(self, q_now: np.ndarray, live_qpos: np.ndarray) -> list[Segment]:
        """Safe recovery: straight up to lift height, then home. Not collision-checked against the
        carried object; robot-vs-environment is checked and reported by the controller at runtime."""
        pos, rot = self.ik.forward(q_now)
        up = pos.copy()
        up[2] = max(pos[2], self.cfg.tray.pose_in_robot.z_m + self.cfg.robot.lift_height_m)
        res = self.ik.solve(up, rot, q_now)
        q_up = res.qpos if res.success else q_now
        return [
            Segment("recover_up", q_now, q_up, self._duration(q_now, q_up, slow=True)),
            Segment("recover_home", q_up, self.home, self._duration(q_up, self.home)),
        ]

    def _duration(self, qa: np.ndarray, qb: np.ndarray, slow: bool = False) -> float:
        # Minimum-jerk peak speed is 1.875x the average speed.
        vmax = self.vmax * (0.4 if slow else 1.0)
        return max(0.6, 1.875 * float(np.max(np.abs(qb - qa))) / vmax)

    def _check_segment(self, seg: Segment, live_qpos: np.ndarray, ignore: set[int], samples: int = 25) -> None:
        d = self._scratch
        d.qpos[:] = live_qpos
        for i in range(samples + 1):
            d.qpos[self.geo.arm_qadr] = seg.sample(seg.duration_s * i / samples)
            mujoco.mj_fwdPosition(self.model, d)
            hits = self.geo.robot_collisions(d, ignore)
            if hits:
                names = [(self.model.geom(a).name or str(a), self.model.geom(b).name or str(b)) for a, b in hits[:3]]
                raise PlanningError("collision_blocked", f"segment '{seg.name}' collides at {i}/{samples}: {names}")
