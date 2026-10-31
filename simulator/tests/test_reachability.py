"""M4: IK accuracy, joint limits, workspace validation, collision-checked planning."""

import numpy as np
import pytest

from saltsim.ik import top_down_rotation
from saltsim.planner import PlanningError, Segment
from saltsim.scene import HOME_QPOS

from saltsim.testing import tray_point


@pytest.mark.parametrize("tx", [0.03, 0.2, 0.37])
@pytest.mark.parametrize("ty", [0.03, 0.15, 0.27])
@pytest.mark.parametrize("height", [0.021, 0.12])
def test_ik_reaches_tray_grid_within_tolerance(runtime, scene_cfg, tx, ty, height):
    ik = runtime.planner.ik
    target = np.array(tray_point(scene_cfg, tx, ty))
    target[2] = height
    res = ik.solve(target, top_down_rotation(0.4), np.array(HOME_QPOS))
    assert res.success
    pos, rot = ik.forward(res.qpos)
    assert np.linalg.norm(pos - target) < 1e-3  # < 1 mm
    assert np.all(res.qpos >= ik.lo) and np.all(res.qpos <= ik.hi)


def test_far_target_is_unreachable(runtime):
    res = runtime.planner.ik.solve(np.array([1.5, 0.0, 0.1]), top_down_rotation(0.0), np.array(HOME_QPOS))
    assert not res.success


def test_target_outside_tray_rejected(runtime, scene_cfg):
    with pytest.raises(PlanningError) as e:
        runtime.planner.validate_target(np.array(tray_point(scene_cfg, 0.5, 0.1)))
    assert e.value.outcome == "rejected_invalid"


def test_min_jerk_segment_hits_endpoints():
    seg = Segment("s", np.zeros(6), np.ones(6), 2.0)
    assert seg.sample(0.0) == pytest.approx(np.zeros(6))
    assert seg.sample(2.0) == pytest.approx(np.ones(6))
    assert seg.sample(5.0) == pytest.approx(np.ones(6))
    assert seg.sample(1.0) == pytest.approx(0.5 * np.ones(6))


def test_planned_pick_is_collision_free_and_within_limits(runtime, scene_cfg):
    rt = runtime
    proxy = rt.objects.place("t", np.array(tray_point(scene_cfg, 0.2, 0.15)), 0.0, rt.planner.geo.robot_geoms)
    plan = rt.planner.plan_pick(rt.objects.position(proxy), 0.0, rt.controller.arm_q(), rt.data.qpos.copy(),
                                {proxy.geom_id})
    assert list(plan.segments) == ["move_pregrasp", "move_to_grasp", "lift", "transfer", "return_home"]
    for q in plan.waypoints.values():
        assert np.all(q >= rt.planner.ik.lo) and np.all(q <= rt.planner.ik.hi)


def test_chunk_inside_tray_wall_is_rejected(runtime, scene_cfg):
    with pytest.raises(PlanningError) as e:
        runtime.planner.validate_target(np.array(tray_point(scene_cfg, 0.39, 0.15)))
    assert e.value.outcome == "rejected_invalid"


def test_crowded_target_is_collision_blocked(runtime, scene_cfg):
    # Neighbours 45 mm away on all four sides: the fingers would land on a neighbour in both
    # gripper rotations, so the planner must refuse rather than plough through them.
    rt = runtime
    geoms = rt.planner.geo.robot_geoms
    for i, (dx, dy) in enumerate([(0.045, 0), (-0.045, 0), (0, 0.045), (0, -0.045)]):
        rt.objects.place(f"n{i}", np.array(tray_point(scene_cfg, 0.2 + dx, 0.15 + dy)), 0.0, geoms)
    proxy = rt.objects.place("t", np.array(tray_point(scene_cfg, 0.2, 0.15)), 0.0, geoms)
    with pytest.raises(PlanningError) as e:
        rt.planner.plan_pick(rt.objects.position(proxy), 0.0, rt.controller.arm_q(), rt.data.qpos.copy(),
                             {proxy.geom_id})
    assert e.value.outcome == "collision_blocked"
