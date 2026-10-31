import numpy as np
import pytest

from saltsort_common.frames import ROBOT_BASE, TRAY, FramePoint, RigidTransform
from saltsort_common.scene_config import load_scene_config


def test_transform_applies_rotation_then_translation():
    t = RigidTransform.from_xyz_yaw(ROBOT_BASE, TRAY, 0.3, -0.15, 0.0, np.pi / 2)
    p = t.apply(FramePoint(TRAY, (0.1, 0.0, 0.02)))
    assert p.frame == ROBOT_BASE
    assert p.xyz == pytest.approx((0.3, -0.05, 0.02))
    assert t.apply_yaw(0.0) == pytest.approx(np.pi / 2)


def test_transform_rejects_wrong_frame():
    t = RigidTransform.from_xyz_yaw(ROBOT_BASE, TRAY, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        t.apply(FramePoint(ROBOT_BASE, (0, 0, 0)))


def test_example_scene_config_is_consistent():
    cfg = load_scene_config()
    assert cfg.tray.robot_from_tray().parent == ROBOT_BASE
    assert cfg.salt_proxy.height_m == pytest.approx(2 * cfg.salt_proxy.half_size_m[2])
