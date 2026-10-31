"""M3: scene assets, proxy pool, track -> virtual object mapping."""

import mujoco
import numpy as np
import pytest

from saltsim.object_manager import ObjectError
from saltsim.scene import BIN_BODY, GRIPPER_ACTUATOR, TCP_SITE, TRAY_BODY

from saltsim.testing import tray_point


def test_scene_contains_robot_gripper_tray_bin_and_pool(runtime, scene_cfg):
    m = runtime.model
    assert m.nu == 7  # 6 arm + 1 gripper
    for name, kind in ((TRAY_BODY, mujoco.mjtObj.mjOBJ_BODY), (BIN_BODY, mujoco.mjtObj.mjOBJ_BODY),
                       (TCP_SITE, mujoco.mjtObj.mjOBJ_SITE), (GRIPPER_ACTUATOR, mujoco.mjtObj.mjOBJ_ACTUATOR)):
        assert mujoco.mj_name2id(m, kind, name) >= 0, name
    assert len(runtime.objects.proxies) == scene_cfg.salt_proxy.count
    assert all(p.status == "parked" for p in runtime.objects.proxies)
    assert m.opt.gravity == pytest.approx([0, 0, -9.81])


def test_tray_body_matches_configured_transform(runtime, scene_cfg):
    tray_id = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_BODY, TRAY_BODY)
    expected = scene_cfg.tray.robot_from_tray().matrix[:3, 3]
    assert runtime.data.xpos[tray_id] == pytest.approx(expected, abs=1e-9)


def test_place_puts_proxy_at_mapped_position(runtime, scene_cfg):
    pos = np.array(tray_point(scene_cfg, 0.2, 0.15))
    proxy = runtime.cmd_place("salt-1", pos.tolist(), 0.0)
    p = runtime.objects.by_track("salt-1")
    assert proxy["status"] == "active"
    assert runtime.objects.position(p)[:2] == pytest.approx(pos[:2], abs=1e-6)


def test_place_is_idempotent_per_track(runtime, scene_cfg):
    a = runtime.cmd_place("salt-1", tray_point(scene_cfg, 0.2, 0.15), 0.0)
    b = runtime.cmd_place("salt-1", tray_point(scene_cfg, 0.1, 0.1), 0.0)
    assert a["proxy_body"] == b["proxy_body"]
    assert len(runtime.objects.summary()) == 1


def test_overlapping_target_rejected(runtime, scene_cfg):
    runtime.cmd_place("salt-1", tray_point(scene_cfg, 0.2, 0.15), 0.0)
    with pytest.raises(ObjectError):
        runtime.cmd_place("salt-2", tray_point(scene_cfg, 0.205, 0.15), 0.0)


def test_pool_exhaustion_reported(runtime, scene_cfg):
    n = scene_cfg.salt_proxy.count
    xs, ys = np.linspace(0.03, 0.37, 5), np.linspace(0.03, 0.27, 5)  # 85 x 60 mm grid
    points = [(x, y) for x in xs for y in ys][:n]
    for i, (x, y) in enumerate(points):
        runtime.cmd_place(f"salt-{i}", tray_point(scene_cfg, x, y), 0.0)
    with pytest.raises(ObjectError, match="exhausted"):
        runtime.cmd_place("salt-extra", tray_point(scene_cfg, 0.0725, 0.06), 0.0)  # between grid points


def test_reset_parks_everything(runtime, scene_cfg):
    runtime.cmd_place("salt-1", tray_point(scene_cfg, 0.2, 0.15), 0.0)
    runtime.cmd_reset()
    assert runtime.objects.summary() == []
