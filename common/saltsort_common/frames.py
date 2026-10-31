"""Frame-tagged points and rigid transforms. SI units only.

Frames:
  image_px    pixel (u right, v down) in a specific camera resolution
  tray        origin at the tray's inner corner shown at the image bottom-left;
              x along image-right, y along image-up, z up out of the tray floor
  robot_base  x forward, y left, z up (MuJoCo world frame == robot_base)
"""

from dataclasses import dataclass

import numpy as np

IMAGE_PX = "image_px"
TRAY = "tray"
ROBOT_BASE = "robot_base"


@dataclass(frozen=True)
class FramePoint:
    frame: str
    xyz: tuple[float, float, float]

    def as_array(self) -> np.ndarray:
        return np.array(self.xyz, dtype=float)


@dataclass(frozen=True)
class RigidTransform:
    """T_parent_from_child: maps points expressed in `child` into `parent`."""

    parent: str
    child: str
    matrix: np.ndarray  # 4x4 homogeneous

    @classmethod
    def from_xyz_yaw(cls, parent: str, child: str, x: float, y: float, z: float, yaw: float) -> "RigidTransform":
        c, s = np.cos(yaw), np.sin(yaw)
        m = np.eye(4)
        m[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        m[:3, 3] = [x, y, z]
        return cls(parent, child, m)

    @property
    def yaw(self) -> float:
        return float(np.arctan2(self.matrix[1, 0], self.matrix[0, 0]))

    def apply(self, point: FramePoint) -> FramePoint:
        if point.frame != self.child:
            raise ValueError(f"transform expects a point in '{self.child}', got '{point.frame}'")
        p = self.matrix @ np.append(point.as_array(), 1.0)
        return FramePoint(self.parent, tuple(float(v) for v in p[:3]))

    def apply_yaw(self, yaw_in_child: float) -> float:
        return float(np.arctan2(np.sin(yaw_in_child + self.yaw), np.cos(yaw_in_child + self.yaw)))
