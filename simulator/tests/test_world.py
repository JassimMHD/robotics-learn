"""M0 simulator smoke tests: the official UR5e scene loads and its joints are controllable."""

import mujoco
import numpy as np
import pytest

from saltsim.world import ARM_ACTUATOR_NAMES, ARM_JOINT_NAMES, load_world


@pytest.fixture
def world():
    return load_world(timestep_s=0.002)


def test_scene_loads_with_expected_physics(world):
    assert world.model.opt.timestep == pytest.approx(0.002)
    assert world.model.opt.gravity == pytest.approx([0.0, 0.0, -9.81])
    assert len(world.qpos_adr) == len(ARM_JOINT_NAMES) == 6
    assert len(world.actuator_ids) == len(ARM_ACTUATOR_NAMES) == 6


def test_baseline_scene_has_no_gripper(world):
    # The UR5e asset is arm-only; a gripper must be added explicitly in a later milestone.
    assert world.model.nu == 6


def test_home_pose_is_stable(world):
    home = world.arm_qpos()
    for _ in range(500):  # 1.0 s simulated
        mujoco.mj_step(world.model, world.data)
    assert np.max(np.abs(world.arm_qpos() - home)) < 0.01
    assert np.all(np.isfinite(world.data.qpos))


@pytest.mark.parametrize("joint_index", range(6))
def test_each_joint_tracks_a_position_command(world, joint_index):
    target = world.arm_qpos()
    target[joint_index] += 0.3
    world.set_arm_ctrl(target)
    for _ in range(1500):  # 3.0 s simulated
        mujoco.mj_step(world.model, world.data)
    error = abs(world.arm_qpos()[joint_index] - target[joint_index])
    assert error < 0.02, f"{ARM_JOINT_NAMES[joint_index]} error {error:.4f} rad"


def test_out_of_range_command_is_rejected(world):
    target = world.arm_qpos()
    target[2] = 4.0  # elbow ctrlrange is +/-pi
    with pytest.raises(ValueError):
        world.set_arm_ctrl(target)
