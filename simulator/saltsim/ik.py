"""Damped-least-squares inverse kinematics for the UR5e TCP (gripper pinch site).

Works on a private MjData so the live simulation is never disturbed.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

# TCP orientation at the home pose: gripper approach axis (site z) points straight down.
# Fingers close along site y. Top-down grasps are R_DOWN rotated about world z by a yaw angle.
R_DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def top_down_rotation(yaw_rad: float) -> np.ndarray:
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rz @ R_DOWN


@dataclass(frozen=True)
class IkResult:
    success: bool
    qpos: np.ndarray
    pos_error_m: float
    rot_error_rad: float
    iterations: int


class IkSolver:
    def __init__(self, model: mujoco.MjModel, site_name: str, arm_qpos_adr: np.ndarray, arm_dof_adr: np.ndarray):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self.qadr = arm_qpos_adr
        self.dadr = arm_dof_adr
        jnt_ids = [int(np.where(model.jnt_qposadr == a)[0][0]) for a in arm_qpos_adr]
        self.lo = model.jnt_range[jnt_ids, 0].copy()
        self.hi = model.jnt_range[jnt_ids, 1].copy()
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site_id].copy(), self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def solve(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q_seed: np.ndarray,
        max_iter: int = 300,
        pos_tol_m: float = 5e-4,
        rot_tol_rad: float = 5e-3,
        damping: float = 1e-2,
        max_step_rad: float = 0.2,
    ) -> IkResult:
        q = np.clip(np.asarray(q_seed, dtype=float).copy(), self.lo, self.hi)
        pos_err = rot_err = np.inf
        for it in range(1, max_iter + 1):
            pos, rot = self.forward(q)
            e_pos = target_pos - pos
            # Orientation error: rotation vector taking current to target (small-angle approximation via
            # the cross-product form, exact enough near convergence).
            e_rot = 0.5 * sum(np.cross(rot[:, i], target_rot[:, i]) for i in range(3))
            pos_err, rot_err = float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot))
            if pos_err < pos_tol_m and rot_err < rot_tol_rad:
                return IkResult(True, q, pos_err, rot_err, it)
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
            jac = np.vstack([self._jacp[:, self.dadr], self._jacr[:, self.dadr]])
            err = np.concatenate([e_pos, e_rot])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), err)
            scale = max(1.0, np.max(np.abs(dq)) / max_step_rad)
            q = np.clip(q + dq / scale, self.lo, self.hi)
        return IkResult(False, q, pos_err, rot_err, max_iter)
