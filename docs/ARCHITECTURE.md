# Architecture

```text
camera / replay file ─► CaptureWorker (newest frames only) ─► OnnxDetector (REAL MODEL / REPLAY)
                                                              │  or operator clicks (MANUAL)
                                                              ▼
                     Tracker (IoU + centroid, stable IDs) ─► stability gate ─► Calibration (px → tray)
                                                                                   │
                                     PickScheduler: duplicate suppression, bounded queue, one active pick
                                                                                   │ HTTP (loopback)
                                                                                   ▼
             Simulator process: SimRuntime ─► PickController state machine ─► Planner (IK + collision)
                                 │                         └─► ObjectManager (20 pre-built proxies)
                                 └─► offscreen render ─► /stream.mjpg
backend EventHub ─► /ws/telemetry, JSONL log ◄─ status polling (5 Hz)          React dashboard (Vite proxy)
```

## Processes

| Process | Owns | Address |
|---|---|---|
| Simulator (`python -m saltsim.run`) | `MjModel`/`MjData`, physics loop, controller, viewer, offscreen stream. Only the sim thread mutates state; HTTP handlers submit closures through `SimRuntime.call()`. | `127.0.0.1:8001` (hard-coded) |
| Backend (`uvicorn app.main:app`) | camera, inference, tracking, calibration, scheduling, UI API, event log | `127.0.0.1:8000` (validated loopback) |
| Dashboard (`npm run dev`) | display + operator commands; proxies `/api`, `/streams`, `/ws` → 8000 and `/sim` → 8001 | `127.0.0.1:5173` |

## Decisions

1. **Backend → simulator transport:** loopback HTTP. `POST /pick` is idempotent by `event_id`
   (repeat → 200 + `duplicate_request`), returns 202 accepted, 409 busy / already processed, 422 invalid.
   The backend polls `GET /status` at 5 Hz. A busy or unreachable simulator leaves the pick queued; it
   is never dropped.
2. **Inference:** ONNX Runtime CPU with Ultralytics-style letterbox preprocessing. Class names come
   from the model's metadata and must equal `config/classes.*.yaml`, otherwise `NO REAL MODEL`.
3. **Class roles:** `impure` = reject, `pure` = accept, `unwanted` = obstruction (hands, keys…). An
   obstruction inside the tray pauses dispatch of new picks.
4. **Duplicate suppression** is positional in the tray frame (`duplicate_radius_m`) across queued,
   active, completed and failed picks, plus per-track. It survives tracker ID switches. Only an
   explicit Reset re-arms.
5. **Pick IDs:** `pick-<run>-<n>` where `<run>` is random per backend start, so IDs never collide
   with history the simulator kept from an earlier backend run.
6. **Grasp modes:** `kinematic_demo` attaches the chunk to the TCP after closing (collisions off while
   held); `contact_physics` relies only on finger contact and friction, needs both pads touching, and
   fails with `grasp_failed_sim` on slip (> 30 mm from TCP) or an empty grasp. Counted separately.
7. **Gravity compensation** on robot bodies via `actgravcomp`, as a real UR controller does; without
   it the gripper weight made the arm sag ~1 cm.
8. **Contact priority:** salt proxies have geom priority 2 so the configured (estimated) salt friction
   governs pad contacts instead of being silently replaced by the pads' friction.
9. **Stream rendering** disables shadows/reflections (67 → 15 ms per frame on Intel iGPU); the native
   viewer keeps full quality.

## Deviations from the spec's suggested layout

| Spec | Actual | Reason |
|---|---|---|
| `backend/pyproject.toml` | root `pyproject.toml`, editable install of `app`, `saltsim`, `saltsort_common` | one venv, clean imports for all three packages |
| `simulator/src/` | `simulator/saltsim/` | `src` is not an importable name |
| `simulator/scenes/*.xml` | scene built in code with `MjSpec` (`saltsim/scene.py`) from Menagerie MJCF | attaches the 2F-85 and adds the proxy pool without hand-merged XML |
| `ik.py`, `planner.py`, `controller.py`, `gripper.py`, `bridge.py` | gripper logic lives in `controller.py`; HTTP API in `server.py` | fewer, cohesive modules |
| `perception/manual.py`, `ultralytics_adapter.py` | manual targets in `orchestrator/session.py`; only the ONNX adapter | spec: implement only the supplied backend |
| `calibration/frames.py` | `common/saltsort_common/frames.py` | shared by backend and simulator |
| `/streams/simulator.mjpg` on the backend | simulator serves `/stream.mjpg`; dashboard proxies it as `/sim/stream.mjpg` | avoids re-streaming every frame through the backend |
| `pydantic-settings` | Pydantic models + PyYAML | config is YAML |
