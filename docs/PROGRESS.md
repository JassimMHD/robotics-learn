# Progress log

Machine: Lenovo IdeaPad Slim 3, i7-13620H, 16 GB, Intel iGPU, Windows 11. Python 3.12.7, MuJoCo 3.17.0,
ONNX Runtime 1.31.1, OpenCV 5.0. All numbers below were measured here (`scripts/benchmark.py`, live runs);
they are not guarantees.

## Status at a glance (2026-11-01)

| Milestone | State | Evidence |
|---|---|---|
| M0 Environment | done | env check, smoke test, viewer opened |
| M1 Camera + manual targets | done (replay-tested; **webcam not yet tried by Claude**) | tests + live run on sample image |
| M2 Calibration | done in software; **real tray not calibrated** | tests; live run left `unvalidated` (no measured holdouts) |
| M3 Virtual objects | done | tests, live run |
| M4 Robot motion | done | tests, live run |
| M5 Demo picking | done | tests, 3/3 live picks |
| M6 Real model | code done; **blocked: supplied model is the wrong model** | tests incl. refusal of the supplied file |
| M7 Physics grasp | done | tests, 3/3 live contact-physics picks |
| M8 Pitch dashboard | done; **not yet viewed in a browser by a human** | build, proxy + WebSocket checks, stream frames inspected |

**Acceptance statement (spec §1):** met in MANUAL mode. Three different targets in one session were
each picked in simulation with unique IDs; a repeated click on the same chunk was refused
(`duplicate ... within 25 mm`); Reset re-armed it. Not yet demonstrated with REAL MODEL detections.

## 2026-11-01: M0 Environment

- Python 3.12.7 venv (`py -3.12`; the machine default `python` is 3.13.9), Menagerie pinned at `b0655057`.
- UR5e baseline: joint commands tracked; native viewer ran at real-time factor 1.00.

## 2026-11-01: M1–M8

### Received model files: not usable for salt sorting

`../best.onnx`, `../best.pt` and `../data.yaml` (received 2026-11-01) are a **YOLO26n, 1-class `bird`**
model (imgsz 960, output `[1, 5, 18900]`, trained 2026-10-28, Ultralytics 8.4.126). On the salt sample
images it placed small low-confidence boxes on parts of chunks and on paper scribbles. The salt
detector described in `vision-model-training` is YOLOv8, classes `impure/pure/unwanted`, input 320, last
seen at `C:/Development/python/yolo/inference3/best.pt` on the collaborator's machine.
The adapter refuses the supplied file (class-metadata mismatch), and a test checks that.
Note: these Ultralytics exports carry `license = AGPL-3.0` metadata; review before commercial delivery.

### Measurements

| Metric | Value | Source |
|---|---|---|
| Physics + controller, headless | ~5,500 steps/s (RTF ≈ 11) | benchmark |
| Live sim with stream + backend polling | RTF 0.999, 500 steps/s at 0.002 s | live `/api/metrics` |
| IK solve (6-DOF, top-down) | p50 2.6 ms, p95 3.6 ms | benchmark, 30 tray points |
| Offscreen render 640×480 | 67 ms with shadows → 15 ms without (stream config) | benchmark |
| Simulated pick cycle | p50 15.0 s sim time (both grasp modes) | benchmark, 3 picks each |
| Camera (replay of a still image) | 9.6 fps (source paced at 10 fps) | live; webcam not measured |
| Detector inference | **not measured**: no salt model available | — |
| Calibration error on the real tray | **not measured**: real rig not calibrated | — |
| Contact grip force | ~43 N per finger on a 2.6 cm chunk | contact-force probe |

### Findings worth knowing

- **Gravity sag**: without gravity compensation the 2F-85 weight made the arm sag ~1 cm. Fixed with
  `actgravcomp` (as a real UR controller would).
- **Fingertips vs floor**: closed pads reach 17.7 mm below the TCP, so grasps are planned at ≥ 21 mm
  above the tray floor.
- **Gripper pre-shaping**: opening only to chunk size + 20 mm (instead of fully, 85 mm) unblocked
  targets near tray walls. Chunks closer to a wall than about half the finger span are still refused as
  `collision_blocked`, which is correct for this gripper.
- **Bin stacking**: dropping every chunk at the bin center built a tower that overflowed the bin wall on
  the 4th pick (physics reported it honestly). Drop points now cycle over a 3×3 grid.
- **Friction was being ignored**: Robotiq pads have contact priority 1, so the salt friction setting had
  no effect. Proxies now have priority 2. With ~43 N grip per finger, the holdable mass is about
  μ × 86 N / g; tests confirm success/slip on either side of that line (e.g. 0.2 kg at μ 0.01 slips).
  The real salt–rubber friction is unknown: the configured 1.0 is an estimate.
- **Stale stream frame**: `/frame.jpg` could return an old render after the stream had been idle. It now
  waits for a render newer than the request.

### Not done / limitations

- REAL MODEL mode cannot be demonstrated until the 3-class salt model is supplied.
- Webcam path is implemented and unit-tested with fakes; Claude only ran a replay of a sample photo.
  Try `Start camera` with your webcam and unplug it once to see the error path.
- Real tray, real tray-to-robot pose and holdout measurements are still to be done (all scene values
  are examples/estimates).
- Detector gives axis-aligned boxes: chunk yaw unknown (0 used) and proxy size is fixed, not scaled
  to the box. Height is an estimate.
- Only one camera, stationary tray, one active pick (as specified for MVP 1).
- Dashboard was built and its API/stream/WebSocket proxies verified with scripts, but not visually
  reviewed in a browser by a person.
- Starlette prints a deprecation warning about `httpx` in the test client (filtered in pytest).

### Next steps

1. Get the real salt `best.pt` / `best.onnx` (3 classes, `nms=False`) and run
   `python scripts/benchmark.py --model models/best.onnx --images <your tray photos>`.
2. Build the physical tray rig, measure it, calibrate with ≥ 5 holdouts to `valid`.
3. Record backup footage: `python scripts/record_camera.py --seconds 120`.
4. Label ~50 held-out frames from this camera to measure detector precision/recall honestly.
