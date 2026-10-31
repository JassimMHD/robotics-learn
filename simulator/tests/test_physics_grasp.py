"""M7: contact-physics grasping. Success requires sustained finger contact; failures are reported.

Measured in this scene: each closed finger presses ~43 N on a 2.6 cm chunk, so the holdable mass is
roughly mu * 86 N / g. The cases below sit clearly on either side of that line.
"""

import numpy as np
import pytest

from saltsim.controller import RobotState
from saltsim.runtime import SimRuntime
from saltsim.testing import pick_request, run_until_idle


def _run(cfg, mass, friction, mode="contact_physics"):
    cfg = cfg.model_copy(deep=True)
    cfg.salt_proxy.mass_kg = mass
    cfg.salt_proxy.friction = friction
    rt = SimRuntime(cfg)
    rt.cmd_pick(pick_request(cfg, "pick-1", "salt-1", 0.2, 0.15, mode=mode))
    run_until_idle(rt)
    return rt


@pytest.mark.parametrize("mass,friction", [(0.01, 1.0), (0.035, 1.0), (0.1, 0.5), (0.5, 0.1)])
def test_physics_grasp_succeeds_within_friction_capacity(scene_cfg, mass, friction):
    rt = _run(scene_cfg, mass, friction)
    assert rt.picks["pick-1"].status == "completed_sim"
    assert rt.controller.counts["completed_contact_physics"] == 1
    assert rt.controller.counts["completed_kinematic_demo"] == 0


@pytest.mark.parametrize("mass,friction", [(0.2, 0.01), (1.0, 0.05)])
def test_physics_grasp_slip_is_reported_not_counted(scene_cfg, mass, friction):
    rt = _run(scene_cfg, mass, friction)
    rec = rt.picks["pick-1"]
    assert rec.status == "grasp_failed_sim"
    assert "slipped" in rec.detail
    assert rt.controller.counts["completed_contact_physics"] == 0
    assert rt.controller.counts["grasp_failed_contact_physics"] == 1
    assert rt.controller.state == RobotState.IDLE  # recovered


def test_kinematic_demo_ignores_physics_and_is_labeled(scene_cfg):
    # The same impossible case succeeds in kinematic_demo mode, which is exactly why the
    # two modes must never be reported together.
    rt = _run(scene_cfg, 1.0, 0.05, mode="kinematic_demo")
    assert rt.picks["pick-1"].status == "completed_sim"
    assert rt.picks["pick-1"].grasp_mode == "kinematic_demo"
    assert rt.controller.counts["completed_contact_physics"] == 0


def test_closing_on_nothing_is_a_grasp_failure(runtime, scene_cfg):
    rt = runtime
    rt.cmd_pick(pick_request(scene_cfg, "pick-1", "salt-1", 0.2, 0.15, mode="contact_physics"))
    while rt.controller.state != RobotState.CLOSE_GRIPPER:
        rt.step()
    # The chunk vanishes before the fingers close (e.g. it was moved): park it far away.
    proxy = rt.controller.proxy
    rt.objects.set_world_pose(proxy, np.array([-1.5, -1.5, 0.02]), np.array([1.0, 0, 0, 0]))
    run_until_idle(rt)
    rec = rt.picks["pick-1"]
    assert rec.status == "grasp_failed_sim" and "without contact" in rec.detail
