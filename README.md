# SaltSort Digital Twin

A real camera views real salt chunks; a **virtual** UR5e arm with a Robotiq 2F-85 gripper in MuJoCo
picks the matching virtual chunk and drops it in a reject bin. Everything the robot does is
**SIMULATED**: the real salt stays on the real tray.

- Spec: `../SaltSort_MuJoCo_Live_Camera_Claude_Code_Spec.md`
- Status, measurements and open issues: [docs/PROGRESS.md](docs/PROGRESS.md)
- Detector: trained in the sibling `vision-model-training` repo; weights are supplied separately and never committed.

## What you can run today

| Mode | Works now? | What it shows |
|---|---|---|
| `MANUAL` | yes | Click a chunk on the live camera (or type tray coordinates) and the simulated robot picks it |
| `REAL MODEL` | **needs the salt model** | The supplied `best.onnx` is a 1-class `bird` detector and is refused (see PROGRESS.md) |
| `REPLAY` | yes (with a model) | Same pipeline on recorded footage, labeled REPLAY |

## Requirements

Windows 10/11, PowerShell, Python **3.12** (`py -3.12`; the default `python` here is 3.13), Node.js 22+, git.

## Setup (PowerShell, repo root)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m pip install -e . --no-deps

# Robot assets (pinned commit: simulator/assets/README.md)
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git vendor/mujoco_menagerie
git -C vendor/mujoco_menagerie sparse-checkout set universal_robots_ur5e robotiq_2f85

cd frontend; npm ci; cd ..
python scripts/check_environment.py
```

## Run the demo

One command (opens three windows and the dashboard):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1
```

Or manually, each in its own PowerShell window with `.venv` active:

```powershell
python -m saltsim.run                                         # 1. simulator + native viewer (API :8001)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000   # 2. backend (API :8000)
cd frontend; npm run dev                                       # 3. dashboard http://127.0.0.1:5173
```

In the dashboard:

1. **Start camera** (default: webcam 0 at 640×480; change `camera.source` in `config/app.local.yaml`).
2. **Calibration → Click 4 corners** of the tray's inner rectangle in the order top-left, top-right,
   bottom-right, bottom-left. Then **Add holdout points** (≥ 5 measured points) until status is `valid`.
   See [docs/CALIBRATION.md](docs/CALIBRATION.md).
3. Click a chunk in the camera view: a `MANUAL` target is queued, a virtual chunk appears, and the robot picks it.
4. **Reset** re-arms processed chunks (needed because the real chunk is still on the tray).

No camera? Send tray-frame targets from the Controls panel, or run the scripted demo:

```powershell
python scripts/demo_targets.py --reset                          # kinematic demo grasps
python scripts/demo_targets.py --reset --grasp-mode contact_physics
```

## Using the real salt model (when supplied)

1. Put the 3-class YOLOv8 export at `models/best.onnx` (git-ignored). Export with `nms=False`, e.g.
   `yolo export model=best.pt format=onnx imgsz=320 opset=12`.
2. Create `config/app.local.yaml`:

   ```yaml
   app: {mode: "real_model"}
   model: {backend: "onnx", weights_path: "models/best.onnx"}
   ```

3. Check `config/classes.example.yaml` matches the model (`0 impure, 1 pure, 2 unwanted`). The adapter
   compares the model's own class metadata and **refuses to run** on any mismatch.
4. Calibrate to `valid`; autonomous picks are blocked otherwise.

## Tests

```powershell
python -m pytest                    # 122 tests: backend, simulator, shared
python scripts/smoke_simulation.py
python scripts/benchmark.py         # measures this machine; saves benchmarks/*.json
cd frontend; npm run build
```

## Layout

```text
backend/app/        FastAPI: camera, perception (ONNX), tracking, calibration, orchestrator, API
simulator/saltsim/  MuJoCo scene, IK, planner, pick state machine, runtime, loopback API
common/             shared config schema and frame math
frontend/           React + TypeScript + Tailwind dashboard
config/             app / scene / classes example configs
scripts/            env check, smoke, benchmark, demo, calibration and recording tools
docs/               architecture, calibration, testing, progress
vendor/             (git-ignored) MuJoCo Menagerie checkout
```
