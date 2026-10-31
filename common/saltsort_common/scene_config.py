"""Physical layout schema (config/scene.example.yaml), shared by backend and simulator."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from saltsort_common.frames import ROBOT_BASE, TRAY, RigidTransform
from saltsort_common.paths import config_file


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Pose(_Strict):
    x_m: float
    y_m: float
    z_m: float = 0.0
    yaw_rad: float = 0.0


class XY(_Strict):
    x_m: float
    y_m: float


class TrayConfig(_Strict):
    width_m: float = Field(gt=0)
    depth_m: float = Field(gt=0)
    wall_height_m: float = Field(gt=0)
    pose_in_robot: Pose

    def robot_from_tray(self) -> RigidTransform:
        p = self.pose_in_robot
        return RigidTransform.from_xyz_yaw(ROBOT_BASE, TRAY, p.x_m, p.y_m, p.z_m, p.yaw_rad)


class BinConfig(_Strict):
    center_in_robot: XY
    inner_size_m: tuple[float, float]
    wall_height_m: float = Field(gt=0)


class ProxyConfig(_Strict):
    count: int = Field(gt=0, le=64)
    half_size_m: tuple[float, float, float]
    mass_kg: float = Field(gt=0)
    friction: float = Field(gt=0)

    @property
    def height_m(self) -> float:
        return 2 * self.half_size_m[2]


class RobotMotionConfig(_Strict):
    pregrasp_clearance_m: float = Field(gt=0)
    lift_height_m: float = Field(gt=0)
    release_height_m: float = Field(gt=0)
    max_joint_speed_rad_s: float = Field(gt=0)
    pick_timeout_s: float = Field(gt=0)


class SceneConfig(_Strict):
    tray: TrayConfig
    reject_bin: BinConfig
    salt_proxy: ProxyConfig
    robot: RobotMotionConfig
    source_path: str | None = None


def scene_config_path() -> Path:
    return config_file("SALTSORT_SCENE", "scene")


def load_scene_config(path: Path | None = None) -> SceneConfig:
    path = path or scene_config_path()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return SceneConfig(**raw, source_path=str(path))
