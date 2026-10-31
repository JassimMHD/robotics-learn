"""Pool of pre-created salt proxy bodies and the track_id -> body mapping.

States: parked -> active (placed at a mapped target) -> attached/grasped -> deposited (in the
virtual bin). The REAL chunk stays on the real tray; only the proxy moves.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

from saltsim.scene import ACTIVE_RGBA, PARKED_RGBA, ProxySlot


class ObjectError(Exception):
    pass


@dataclass
class ProxyState:
    slot: ProxySlot
    body_id: int
    geom_id: int
    qadr: int
    dadr: int
    track_id: str | None = None
    status: str = "parked"  # parked | active | held | deposited
    order: int = 0  # for recycling the oldest deposited proxy


@dataclass
class ObjectManager:
    model: mujoco.MjModel
    data: mujoco.MjData
    slots: list[ProxySlot]
    min_separation_m: float = 0.03
    proxies: list[ProxyState] = field(default_factory=list)
    _counter: int = 0

    def __post_init__(self):
        m = self.model
        for s in self.slots:
            body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, s.body_name)
            geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, s.body_name + "_geom")
            jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, s.body_name + "_free")
            self.proxies.append(ProxyState(s, body, geom, int(m.jnt_qposadr[jnt]), int(m.jnt_dofadr[jnt])))

    def by_track(self, track_id: str) -> ProxyState | None:
        return next((p for p in self.proxies if p.track_id == track_id and p.status != "parked"), None)

    def place(self, track_id: str, pos: np.ndarray, yaw_rad: float, robot_geoms: set[int]) -> ProxyState:
        """Place (or return the existing) proxy for track_id. Idempotent per track."""
        existing = self.by_track(track_id)
        if existing:
            return existing
        for other in self.proxies:
            if other.status == "active" and np.linalg.norm(self.position(other)[:2] - pos[:2]) < self.min_separation_m:
                raise ObjectError(f"target overlaps virtual object of track {other.track_id}")
        proxy = self._free_proxy()
        if self._touching(proxy, robot_geoms):
            raise ObjectError(f"proxy {proxy.slot.body_name} is in contact with the robot; not teleporting")
        half = self.model.geom_size[proxy.geom_id]
        z = pos[2] if pos[2] > half[2] else half[2]
        self._set_pose(proxy, np.array([pos[0], pos[1], z + 0.0005]), yaw_rad)
        self.model.geom_rgba[proxy.geom_id] = ACTIVE_RGBA
        self._counter += 1
        proxy.track_id, proxy.status, proxy.order = track_id, "active", self._counter
        mujoco.mj_forward(self.model, self.data)
        return proxy

    def position(self, proxy: ProxyState) -> np.ndarray:
        return self.data.xpos[proxy.body_id].copy()

    def set_collision(self, proxy: ProxyState, enabled: bool) -> None:
        self.model.geom_contype[proxy.geom_id] = 1 if enabled else 0
        self.model.geom_conaffinity[proxy.geom_id] = 1 if enabled else 0

    def set_world_pose(self, proxy: ProxyState, pos: np.ndarray, quat: np.ndarray) -> None:
        self.data.qpos[proxy.qadr : proxy.qadr + 3] = pos
        self.data.qpos[proxy.qadr + 3 : proxy.qadr + 7] = quat
        self.data.qvel[proxy.dadr : proxy.dadr + 6] = 0.0

    def park(self, proxy: ProxyState) -> None:
        self.set_collision(proxy, True)
        self._set_pose(proxy, np.array(proxy.slot.parked_pos), 0.0)
        self.model.geom_rgba[proxy.geom_id] = PARKED_RGBA
        proxy.track_id, proxy.status = None, "parked"

    def reset(self) -> None:
        for p in self.proxies:
            self.park(p)
        mujoco.mj_forward(self.model, self.data)

    def summary(self) -> list[dict]:
        return [
            {"body": p.slot.body_name, "track_id": p.track_id, "status": p.status,
             "position_m": np.round(self.position(p), 4).tolist(), "frame": "robot_base"}
            for p in self.proxies if p.status != "parked"
        ]

    def _free_proxy(self) -> ProxyState:
        for p in self.proxies:
            if p.status == "parked":
                return p
        deposited = sorted((p for p in self.proxies if p.status == "deposited"), key=lambda p: p.order)
        if deposited:
            self.park(deposited[0])  # recycle the oldest proxy already in the bin
            return deposited[0]
        raise ObjectError("virtual object pool exhausted; reset the session")

    def _set_pose(self, proxy: ProxyState, pos: np.ndarray, yaw: float) -> None:
        quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        self.set_world_pose(proxy, pos, quat)

    def _touching(self, proxy: ProxyState, robot_geoms: set[int]) -> bool:
        for c in self.data.contact[: self.data.ncon]:
            pair = {int(c.geom1), int(c.geom2)}
            if proxy.geom_id in pair and pair & robot_geoms:
                return True
        return False
