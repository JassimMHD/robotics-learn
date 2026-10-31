"""Measure this machine: physics rate, IK time, offscreen render time, full simulated pick cycle and
(optionally) detector inference latency. Results are printed and saved to benchmarks/ (git-ignored).

    python scripts/benchmark.py
    python scripts/benchmark.py --model models/best.onnx --images path/to/img1.jpg path/to/img2.jpg

Every number is a measurement on this machine under current load, not a guarantee.
"""

import argparse
import json
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from saltsim.ik import top_down_rotation
from saltsim.runtime import SimRuntime
from saltsim.scene import HOME_QPOS
from saltsim.testing import pick_request, run_until_idle, tray_point
from saltsort_common.paths import REPO_ROOT, resolve
from saltsort_common.scene_config import load_scene_config


def pct(values: list[float]) -> dict:
    a = np.array(values)
    return {"p50": round(float(np.percentile(a, 50)), 3), "p95": round(float(np.percentile(a, 95)), 3), "n": len(a)}


def bench_physics(rt: SimRuntime, seconds: float = 5.0) -> dict:
    steps = int(seconds / rt.model.opt.timestep)
    t0 = time.perf_counter()
    for _ in range(steps):
        rt.step()
    wall = time.perf_counter() - t0
    return {"physics_steps_per_s": round(steps / wall), "headless_real_time_factor": round(seconds / wall, 2),
            "note": "physics + controller only; no rendering, camera or inference"}


def bench_ik(rt: SimRuntime, cfg) -> dict:
    times = []
    for tx in np.linspace(0.03, 0.37, 6):
        for ty in np.linspace(0.03, 0.27, 5):
            p = np.array(tray_point(cfg, tx, ty))
            t0 = time.perf_counter()
            rt.planner.ik.solve(p, top_down_rotation(0.3), np.array(HOME_QPOS))
            times.append((time.perf_counter() - t0) * 1000)
    return {"ik_solve_ms": pct(times)}


def bench_render(rt: SimRuntime) -> dict:
    out = {}
    for label, effects in (("full_effects", True), ("dashboard_stream_no_shadows", False)):
        times = []
        with mujoco.Renderer(rt.model, 480, 640) as r:
            r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = effects
            r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = effects
            for _ in range(30):
                t0 = time.perf_counter()
                r.update_scene(rt.data)
                r.render()
                times.append((time.perf_counter() - t0) * 1000)
        out[label] = pct(times)
    return {"offscreen_render_640x480_ms": out}


def bench_picks(cfg) -> dict:
    out = {}
    for mode in ("kinematic_demo", "contact_physics"):
        rt = SimRuntime(cfg)
        cycles, walls, outcomes = [], [], []
        for i, (x, y) in enumerate([(0.1, 0.08), (0.2, 0.15), (0.3, 0.22)]):
            rt.cmd_pick(pick_request(cfg, f"b-{i}", f"t-{i}", x, y, mode=mode))
            t0 = time.perf_counter()
            run_until_idle(rt)
            walls.append(time.perf_counter() - t0)
            rec = rt.picks[f"b-{i}"].to_dict()
            outcomes.append(rec["status"])
            if rec["cycle_time_sim_s"]:
                cycles.append(rec["cycle_time_sim_s"])
        out[mode] = {"outcomes": outcomes, "cycle_time_sim_s": pct(cycles) if cycles else None,
                     "headless_wall_s_per_pick": pct(walls)}
    return {"simulated_pick_cycles": out}


def bench_model(model: Path, classes: Path, images: list[Path]) -> dict:
    import cv2

    from app.perception.interface import load_class_config
    from app.perception.onnx_adapter import OnnxDetector

    det = OnnxDetector(model, load_class_config(classes), 0.5, 0.45)
    frames = [cv2.imread(str(p)) for p in images] if images else [np.zeros((480, 640, 3), np.uint8)]
    for f in frames:  # warm-up
        det.predict(f, 0)
    times, counts = [], []
    for i in range(60):
        f = frames[i % len(frames)]
        t0 = time.perf_counter()
        counts.append(len(det.predict(f, i)))
        times.append((time.perf_counter() - t0) * 1000)
    return {"inference_ms": pct(times), "model": det.metadata, "images": [p.name for p in images],
            "mean_detections": round(statistics.mean(counts), 2), "provider": "ONNX Runtime CPU"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--classes", type=Path, default=Path("config/classes.example.yaml"))
    ap.add_argument("--images", type=Path, nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_scene_config()
    rt = SimRuntime(cfg)
    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "machine": {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version(),
                    "mujoco": mujoco.__version__},
        **bench_physics(rt), **bench_ik(rt, cfg), **bench_render(rt), **bench_picks(cfg),
    }
    if args.model:
        try:
            result["detector"] = bench_model(resolve(args.model), resolve(args.classes), [resolve(p) for p in args.images])
        except Exception as exc:
            result["detector"] = {"error": str(exc)}
    out_dir = REPO_ROOT / "benchmarks"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"benchmark-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
