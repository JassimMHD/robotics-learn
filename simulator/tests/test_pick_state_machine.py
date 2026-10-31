"""M5: deterministic kinematic-demo pick cycle, idempotency, reset, timeout, cancel."""

import numpy as np

from saltsim.controller import RobotState
from saltsim.testing import pick_request, run_until_idle

EXPECTED_STATES = [
    "VALIDATE_TARGET", "PLAN_APPROACH", "MOVE_PREGRASP", "MOVE_TO_GRASP", "CLOSE_GRIPPER", "VERIFY_GRASP",
    "LIFT", "TRANSFER_TO_REJECT_BIN", "OPEN_GRIPPER", "VERIFY_RELEASE", "completed_sim", "RETURN_HOME", "IDLE",
]


def test_kinematic_demo_full_cycle(runtime, scene_cfg):
    code, body = runtime.cmd_pick(pick_request(scene_cfg, "pick-1", "salt-1", 0.2, 0.15))
    assert code == 202 and body["status"] == "accepted"
    run_until_idle(runtime)
    rec = runtime.picks["pick-1"].to_dict()
    assert rec["status"] == "completed_sim"
    assert rec["simulated"] is True and rec["grasp_mode"] == "kinematic_demo"
    assert [h["state"] for h in rec["history"]] == EXPECTED_STATES
    assert rec["cycle_time_sim_s"] > 0
    assert runtime.controller.counts["completed_kinematic_demo"] == 1
    assert runtime.controller.counts["completed_contact_physics"] == 0  # never mixed
    assert not runtime.controller.collision_flag
    proxy = runtime.objects.by_track("salt-1")
    assert proxy.status == "deposited"


def test_same_event_id_is_idempotent(runtime, scene_cfg):
    req = pick_request(scene_cfg, "pick-1", "salt-1", 0.2, 0.15)
    assert runtime.cmd_pick(req)[0] == 202
    code, body = runtime.cmd_pick(req)
    assert code == 200 and body["duplicate_request"] is True
    assert len(runtime.picks) == 1


def test_second_pick_while_busy_is_rejected_not_dropped(runtime, scene_cfg):
    runtime.cmd_pick(pick_request(scene_cfg, "pick-1", "salt-1", 0.2, 0.15))
    code, body = runtime.cmd_pick(pick_request(scene_cfg, "pick-2", "salt-2", 0.1, 0.1))
    assert code == 409 and body["error"] == "robot busy"
    assert "pick-2" not in runtime.picks


def test_completed_track_cannot_be_picked_again_until_reset(runtime, scene_cfg):
    runtime.cmd_pick(pick_request(scene_cfg, "pick-1", "salt-1", 0.2, 0.15))
    run_until_idle(runtime)
    code, body = runtime.cmd_pick(pick_request(scene_cfg, "pick-2", "salt-1", 0.2, 0.15))
    assert code == 409 and "already processed" in body["error"]
    runtime.cmd_reset()
    assert runtime.cmd_pick(pick_request(scene_cfg, "pick-3", "salt-1", 0.2, 0.15))[0] == 202


def test_two_targets_sequentially(runtime, scene_cfg):
    for i, (x, y) in enumerate([(0.1, 0.08), (0.3, 0.2)]):
        assert runtime.cmd_pick(pick_request(scene_cfg, f"pick-{i}", f"salt-{i}", x, y))[0] == 202
        run_until_idle(runtime)
    assert [r.status for r in runtime.picks.values()] == ["completed_sim", "completed_sim"]
    assert {p["track_id"] for p in runtime.objects.summary()} == {"salt-0", "salt-1"}


def test_unreachable_or_invalid_target_fails_without_motion(runtime, scene_cfg):
    q0 = runtime.controller.arm_q()
    runtime.cmd_pick(pick_request(scene_cfg, "pick-1", "salt-1", 0.60, 0.15))  # outside tray
    run_until_idle(runtime)
    rec = runtime.picks["pick-1"]
    assert rec.status == "rejected_invalid"
    assert np.max(np.abs(runtime.controller.arm_q() - q0)) < 0.02
    assert not runtime.controller.collision_flag


def test_reset_mid_pick_returns_home_and_clears(runtime, scene_cfg):
    runtime.cmd_pick(pick_request(scene_cfg, "pick-1", "salt-1", 0.2, 0.15))
    for _ in range(2000):
        runtime.step()
    runtime.cmd_reset()
    assert runtime.controller.state == RobotState.IDLE
    assert runtime.picks == {} and runtime.objects.summary() == []
    assert dict(runtime.controller.counts) == {}


def test_timeout_is_reported(scene_cfg):
    from saltsim.runtime import SimRuntime

    cfg = scene_cfg.model_copy(deep=True)
    cfg.robot.pick_timeout_s = 2.0
    rt = SimRuntime(cfg)
    rt.cmd_pick(pick_request(cfg, "pick-1", "salt-1", 0.2, 0.15))
    run_until_idle(rt)
    assert rt.picks["pick-1"].status == "timeout"
    assert rt.controller.counts["failed_timeout"] == 1


def test_cancel_is_reported(runtime, scene_cfg):
    runtime.cmd_pick(pick_request(scene_cfg, "pick-1", "salt-1", 0.2, 0.15))
    for _ in range(1000):
        runtime.step()
    runtime.controller.cancel()
    run_until_idle(runtime)
    assert runtime.picks["pick-1"].status == "cancelled"
