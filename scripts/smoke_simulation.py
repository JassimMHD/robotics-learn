"""Headless MuJoCo smoke test.

1. Baseline UR5e: command one joint, report tracking and real-time factor (M0).
2. Full SaltSort scene: one kinematic-demo pick from the tray into the reject bin (M5).

    python scripts/smoke_simulation.py

Exit code 0 on success. All results are SIMULATED.
"""

import sys
import time

import mujoco
import numpy as np

from saltsim.runtime import SimRuntime
from saltsim.testing import pick_request, run_until_idle
from saltsim.world import load_world
from saltsort_common.scene_config import load_scene_config

TOLERANCE_RAD = 0.02


def baseline() -> bool:
    world = load_world(timestep_s=0.002)
    model, data = world.model, world.data
    print(f"mujoco {mujoco.__version__}; timestep {model.opt.timestep} s; nq={model.nq} nu={model.nu}")
    target = world.arm_qpos()
    target[0] += 0.5
    world.set_arm_ctrl(target)
    steps = int(3.0 / model.opt.timestep)
    t0 = time.perf_counter()
    for _ in range(steps):
        mujoco.mj_step(model, data)
    wall = time.perf_counter() - t0
    error = abs(world.arm_qpos()[0] - target[0])
    print(f"[baseline] shoulder_pan error {error:.4f} rad; {steps / wall:,.0f} steps/s headless")
    return error < TOLERANCE_RAD and bool(np.all(np.isfinite(data.qpos)))


def full_scene_pick() -> bool:
    cfg = load_scene_config()
    rt = SimRuntime(cfg)
    rt.cmd_pick(pick_request(cfg, "smoke-1", "smoke-track", 0.2, 0.15))
    t0 = time.perf_counter()
    run_until_idle(rt)
    rec = rt.picks["smoke-1"].to_dict()
    print(f"[scene] pick {rec['status']} ({rec['grasp_mode']}), cycle {rec['cycle_time_sim_s']} s sim, "
          f"{time.perf_counter() - t0:.1f} s wall headless")
    return rec["status"] == "completed_sim"


def main() -> int:
    ok = baseline() and full_scene_pick()
    print("SMOKE TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
