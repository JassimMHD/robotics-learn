"""Build the SaltSort scene with MjSpec: Menagerie UR5e + Robotiq 2F-85, tray, reject bin, salt proxy pool.

The proxy pool is created once at build time; detections never trigger a recompile.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from saltsim.paths import menagerie_dir, ur5e_scene_path
from saltsort_common.scene_config import SceneConfig

GRIPPER_PREFIX = "gripper/"
TCP_SITE = GRIPPER_PREFIX + "pinch"
GRIPPER_ACTUATOR = GRIPPER_PREFIX + "fingers_actuator"
GRIPPER_PAD_GEOMS = tuple(GRIPPER_PREFIX + n for n in ("left_pad1", "left_pad2", "right_pad1", "right_pad2"))
TRAY_BODY = "tray"
BIN_BODY = "reject_bin"
PROXY_PREFIX = "salt_"
PARKED_RGBA = (0.55, 0.55, 0.6, 0.35)
ACTIVE_RGBA = (0.95, 0.55, 0.45, 1.0)  # reject-class proxy (impure)

WALL_T = 0.01  # wall thickness, m
ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
HOME_QPOS = (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)


@dataclass(frozen=True)
class ProxySlot:
    index: int
    body_name: str
    parked_pos: tuple[float, float, float]


def proxy_name(i: int) -> str:
    return f"{PROXY_PREFIX}{i:02d}"


def parked_position(i: int, half_z: float) -> tuple[float, float, float]:
    # A grid behind the robot, outside its working area.
    return (-0.55 - 0.08 * (i % 5), -0.30 + 0.08 * (i // 5), half_z)


def build_spec(cfg: SceneConfig) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(ur5e_scene_path()))
    for key in list(spec.keys):  # arm-only keyframe no longer matches qpos size
        spec.delete(key)
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC  # as recommended by the 2F-85 asset
    spec.option.impratio = 10

    gripper = mujoco.MjSpec.from_file(str(menagerie_dir() / "robotiq_2f85" / "2f85.xml"))
    spec.attach(gripper, prefix=GRIPPER_PREFIX, site=spec.site("attachment_site"))

    # Ideal gravity compensation for the robot, as a real UR controller provides. Routed through
    # the arm actuators (actgravcomp) so actuator force limits still apply.
    for body in spec.bodies:
        if body.name and body.name != "world":
            body.gravcomp = 1.0
    for joint in spec.joints:
        if joint.name in ARM_JOINTS:
            joint.actgravcomp = 1

    world = spec.worldbody
    _add_tray(world, cfg)
    _add_bin(world, cfg)
    _add_proxies(world, cfg)
    return spec


def _box(body, name, pos, half, rgba, collide=True):
    g = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=half, rgba=rgba)
    if not collide:
        g.contype = 0
        g.conaffinity = 0
    return g


def _add_tray(world, cfg: SceneConfig) -> None:
    t = cfg.tray
    p = t.pose_in_robot
    tray = world.add_body(name=TRAY_BODY, pos=[p.x_m, p.y_m, p.z_m], euler=[0, 0, p.yaw_rad])
    w, d, h = t.width_m, t.depth_m, t.wall_height_m
    _box(tray, "tray_floor", [w / 2, d / 2, 0.0005], [w / 2, d / 2, 0.0005], (0.78, 0.66, 0.48, 1), collide=False)
    wall = (0.45, 0.35, 0.25, 1)
    _box(tray, "tray_wall_y0", [w / 2, -WALL_T / 2, h / 2], [w / 2 + WALL_T, WALL_T / 2, h / 2], wall)
    _box(tray, "tray_wall_yd", [w / 2, d + WALL_T / 2, h / 2], [w / 2 + WALL_T, WALL_T / 2, h / 2], wall)
    _box(tray, "tray_wall_x0", [-WALL_T / 2, d / 2, h / 2], [WALL_T / 2, d / 2, h / 2], wall)
    _box(tray, "tray_wall_xw", [w + WALL_T / 2, d / 2, h / 2], [WALL_T / 2, d / 2, h / 2], wall)
    # Debug axes of the tray frame: red = +x, green = +y (visual only).
    _box(tray, "tray_axis_x", [0.04, 0, 0.002], [0.04, 0.002, 0.002], (1, 0, 0, 1), collide=False)
    _box(tray, "tray_axis_y", [0, 0.04, 0.002], [0.002, 0.04, 0.002], (0, 1, 0, 1), collide=False)
    tray.add_site(name="tray_origin", size=[0.006, 0, 0], rgba=(1, 1, 0, 1))


def _add_bin(world, cfg: SceneConfig) -> None:
    b = cfg.reject_bin
    sx, sy = b.inner_size_m[0] / 2, b.inner_size_m[1] / 2
    h = b.wall_height_m
    body = world.add_body(name=BIN_BODY, pos=[b.center_in_robot.x_m, b.center_in_robot.y_m, 0])
    red = (0.7, 0.15, 0.15, 1)
    _box(body, "bin_floor", [0, 0, 0.004], [sx + WALL_T, sy + WALL_T, 0.004], (0.35, 0.1, 0.1, 1))
    _box(body, "bin_wall_n", [0, sy + WALL_T / 2, h / 2], [sx + WALL_T, WALL_T / 2, h / 2], red)
    _box(body, "bin_wall_s", [0, -sy - WALL_T / 2, h / 2], [sx + WALL_T, WALL_T / 2, h / 2], red)
    _box(body, "bin_wall_e", [sx + WALL_T / 2, 0, h / 2], [WALL_T / 2, sy, h / 2], red)
    _box(body, "bin_wall_w", [-sx - WALL_T / 2, 0, h / 2], [WALL_T / 2, sy, h / 2], red)


def _add_proxies(world, cfg: SceneConfig) -> None:
    p = cfg.salt_proxy
    for i in range(p.count):
        body = world.add_body(name=proxy_name(i), pos=list(parked_position(i, p.half_size_m[2])))
        body.add_freejoint(name=proxy_name(i) + "_free")
        g = _box(body, proxy_name(i) + "_geom", [0, 0, 0], list(p.half_size_m), PARKED_RGBA)
        g.mass = p.mass_kg
        g.friction = [p.friction, 0.01, 0.001]
        # The 2F-85 pads have priority 1; outrank them so the configured (estimated) salt friction is the
        # one used for pad-salt contacts instead of being silently replaced by the pad friction.
        g.priority = 2
        g.condim = 4  # torsional friction helps a two-finger grasp


def proxy_slots(cfg: SceneConfig) -> list[ProxySlot]:
    hz = cfg.salt_proxy.half_size_m[2]
    return [ProxySlot(i, proxy_name(i), parked_position(i, hz)) for i in range(cfg.salt_proxy.count)]


def compile_scene(cfg: SceneConfig) -> tuple[mujoco.MjSpec, mujoco.MjModel]:
    spec = build_spec(cfg)
    return spec, spec.compile()


def geom_ids_of_body_subtree(model: mujoco.MjModel, root_body: int) -> np.ndarray:
    bodies = {root_body}
    for b in range(model.nbody):
        if b != root_body and model.body_rootid[b] == model.body_rootid[root_body]:
            # walk ancestors
            a = b
            while a > 0:
                if a == root_body:
                    bodies.add(b)
                    break
                a = model.body_parentid[a]
    return np.array([g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies])
