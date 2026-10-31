"""Load the baseline UR5e scene and expose arm joint/actuator handles.

M0 scope: the official Menagerie UR5e scene only (arm, no gripper). Tray, reject
bin, gripper and salt proxies are added in later milestones.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from saltsim.paths import ur5e_scene_path

ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
ARM_ACTUATOR_NAMES = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
HOME_KEYFRAME = "home"


@dataclass
class World:
    model: mujoco.MjModel
    data: mujoco.MjData
    qpos_adr: np.ndarray  # qpos index of each arm joint, in ARM_JOINT_NAMES order
    actuator_ids: np.ndarray  # actuator id of each arm joint, in ARM_ACTUATOR_NAMES order

    def arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    def set_arm_ctrl(self, targets_rad: np.ndarray) -> None:
        targets = np.asarray(targets_rad, dtype=float)
        lo, hi = self.model.actuator_ctrlrange[self.actuator_ids].T
        if np.any(targets < lo) or np.any(targets > hi):
            raise ValueError(f"arm control target outside actuator ctrlrange: {targets}")
        self.data.ctrl[self.actuator_ids] = targets

    def reset_home(self) -> None:
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, HOME_KEYFRAME)
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)


def load_world(scene_path: Path | None = None, timestep_s: float = 0.002) -> World:
    scene_path = scene_path or ur5e_scene_path()
    if not scene_path.is_file():
        raise FileNotFoundError(
            f"MuJoCo scene not found: {scene_path}. Clone Menagerie first (see simulator/assets/README.md)."
        )
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.timestep = timestep_s
    data = mujoco.MjData(model)

    qpos_adr = np.array(
        [model.jnt_qposadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINT_NAMES]
    )
    actuator_ids = np.array([_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_ACTUATOR_NAMES])

    world = World(model=model, data=data, qpos_adr=qpos_adr, actuator_ids=actuator_ids)
    world.reset_home()
    return world


def _id(model: mujoco.MjModel, obj: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj, name)
    if obj_id < 0:
        raise KeyError(f"{obj.name} '{name}' not found in model")
    return obj_id
