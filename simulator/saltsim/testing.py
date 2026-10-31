"""Helpers shared by simulator tests and scripts (deterministic manual targets)."""

import numpy as np

from saltsim.runtime import SimRuntime
from saltsort_common.scene_config import SceneConfig


def tray_point(cfg: SceneConfig, tx: float, ty: float) -> list[float]:
    """Tray-frame (x, y) in meters -> robot_base position at chunk-center height."""
    p = cfg.tray.robot_from_tray().matrix @ np.array([tx, ty, 0.0, 1.0])
    return [float(p[0]), float(p[1]), cfg.tray.pose_in_robot.z_m + cfg.salt_proxy.half_size_m[2]]


def pick_request(cfg: SceneConfig, event_id: str, track_id: str, tx: float, ty: float,
                 mode: str = "kinematic_demo", yaw: float = 0.0) -> dict:
    return {
        "event_id": event_id, "track_id": track_id,
        "target": {"frame": "robot_base", "position_m": tray_point(cfg, tx, ty), "yaw_rad": yaw},
        "grasp_mode": mode, "vision_mode": "manual",
    }


def run_until_idle(rt: SimRuntime, max_sim_s: float = 120.0) -> None:
    t_end = rt.data.time + max_sim_s
    while rt.controller.busy and rt.data.time < t_end:
        rt.step()
    if rt.controller.busy:
        raise TimeoutError(f"robot still busy in {rt.controller.state}")
