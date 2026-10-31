"""Run the SaltSort simulator: physics loop + loopback HTTP API (+ optional native viewer).

    python -m saltsim.run                 # native viewer + API on 127.0.0.1:8001
    python -m saltsim.run --no-viewer     # headless (API + MJPEG stream only)
    python -m saltsim.run --baseline      # M0 view: bare UR5e scene, optional --sweep

All robot motion is SIMULATED. The virtual robot does not move real salt.
"""

import argparse
import math
import threading
import time

import mujoco
import mujoco.viewer
import uvicorn

from saltsim.server import create_sim_app
from saltsim.runtime import SimRuntime
from saltsort_common.scene_config import load_scene_config, scene_config_path
from saltsort_common.paths import display

HOST = "127.0.0.1"  # loopback only (spec rule 7); intentionally not configurable


def run_baseline(sweep: bool) -> None:
    from saltsim.world import load_world

    world = load_world()
    home = world.arm_qpos()
    model, data = world.model, world.data
    print("SIMULATED UR5e baseline scene. Close the viewer window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall_start, sim_start = time.perf_counter(), data.time
        while viewer.is_running():
            while data.time - sim_start < time.perf_counter() - wall_start:
                if sweep:
                    target = home.copy()
                    target[0] += 0.6 * math.sin(2 * math.pi * 0.1 * data.time)
                    world.set_arm_ctrl(target)
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1 / 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-viewer", action="store_true", help="run without the native MuJoCo viewer")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--timestep", type=float, default=0.002)
    parser.add_argument("--baseline", action="store_true", help="M0 bare UR5e scene in the viewer")
    parser.add_argument("--sweep", action="store_true", help="with --baseline: sweep the shoulder joint")
    args = parser.parse_args()

    if args.baseline:
        run_baseline(args.sweep)
        return

    cfg = load_scene_config()
    rt = SimRuntime(cfg, timestep_s=args.timestep)
    server = uvicorn.Server(uvicorn.Config(create_sim_app(rt), host=HOST, port=args.port, log_level="warning"))
    threading.Thread(target=server.run, name="sim-api", daemon=True).start()
    print(f"SIMULATED SaltSort scene from {display(scene_config_path())}")
    print(f"Simulator API: http://{HOST}:{args.port}  (stream: /stream.mjpg)")

    stop = threading.Event()
    try:
        if args.no_viewer:
            print("Headless. Ctrl+C to stop.")
            rt.run_headless(stop)
        else:
            with mujoco.viewer.launch_passive(rt.model, rt.data) as viewer:
                viewer.cam.lookat[:] = [0.30, 0.10, 0.10]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.6, 200.0, -35.0
                wall_start, sim_start = time.perf_counter(), rt.data.time
                print("Native viewer open. Close it (or Ctrl+C) to stop.")
                while viewer.is_running():
                    rt.tick(wall_start, sim_start)
                    viewer.sync()
                    time.sleep(1 / 120)
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True
        print(f"Stopped. Metrics: {rt.metrics()}")


if __name__ == "__main__":
    main()
