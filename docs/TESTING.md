# Testing

```powershell
python -m pytest                     # 122 tests (~1 min)
python scripts/smoke_simulation.py   # baseline joint control + one full simulated pick
python scripts/check_environment.py  # deps, assets, scene compile, OpenGL, node
python scripts/benchmark.py          # measured numbers for this machine -> benchmarks/
cd frontend; npm run build           # TypeScript type-check + production build
```

## Coverage by milestone

| Milestone | Tests |
|---|---|
| M0 environment | `simulator/tests/test_world.py`, `backend/tests/test_settings.py` (loopback-only hosts and simulator URL) |
| M1 camera + manual | `test_camera.py` (replay sources, newest-frame buffer, unplugged camera → error + stale), `test_api.py` (manual target, camera start failure alert) |
| M2 calibration | `test_calibration.py` (corner mapping, round trip, holdout mm error, too few / out-of-tolerance, reversed / crossed / mirrored / collinear corners rejected, resolution change fails closed, tray-size mismatch, malformed file, ArUco on synthetic image) |
| M3 virtual objects | `test_scene_objects.py` (assets, tray pose = config transform, placement, per-track idempotency, overlap refusal, pool exhaustion, reset) |
| M4 motion | `test_reachability.py` (IK < 1 mm over the tray grid within joint limits, unreachable, outside tray, chunk inside a wall, crowded target → `collision_blocked`, min-jerk endpoints) |
| M5 demo picking | `test_pick_state_machine.py` (exact state sequence, idempotent event IDs, busy refusal, processed track refused until reset, two sequential targets, invalid target without motion, reset mid-pick, timeout, cancel) |
| M6 real model | `test_onnx_adapter.py` (letterbox geometry, decode + NMS back to image px, class mismatch refused, missing / garbage weights, the supplied bird model refused); `test_tracker.py`; `test_pick_idempotency.py` (stationary chunk seen 20× → one pick, ID reassignment → duplicate, failed not re-queued, busy / unreachable sim never drops, obstruction + pause hold dispatch, unstable / accept / unvalidated never queue) |
| M7 physics grasp | `test_physics_grasp.py` (mass/friction inside capacity succeed, outside slip → `grasp_failed_sim` and not counted, kinematic mode labeled and counted separately, closing on nothing fails) |
| M8 integration | `test_sim_server.py` (HTTP lifecycle, validation, JPEG frame, metrics), `test_api.py::test_end_to_end_manual_pick_with_real_simulator` |

## Spec §12 pre-pitch checklist

| Check | Status |
|---|---|
| Camera unplugged → visible error, safe stop | automated (`test_unplugged_camera_reports_error_and_goes_stale`); verify once with your webcam |
| Weights missing → `NO REAL MODEL`, manual usable | automated |
| Calibration missing → autonomous picking disabled | automated |
| Same chunk detected 20× → one pick until re-armed | automated |
| Unreachable target rejected without collision | automated |
| Two targets sequential, stable IDs | automated |
| Simulation slowdown shows real-time factor | dashboard metric; `RTF < 0.95` flagged |
| Grasp failure reported, not counted | automated |
| Reset clears queue, proxies, processed IDs | automated |
| REAL vs SIMULATED labels | banner on camera stream, badges on both panes |
| Metrics not mixed | kinematic vs physics counted separately; no accuracy number shown |

Manual checks still needed on the real rig: webcam unplug during a session, real tray calibration
with measured holdouts, and the supplied salt model on live footage.
