"""Check the local development environment for SaltSort.

    python scripts/check_environment.py            # required checks + viewer/OpenGL check
    python scripts/check_environment.py --camera   # also try to open the configured camera

Exit code 0 when every REQUIRED check passes.
"""

import argparse
import importlib
import shutil
import subprocess
import sys

PACKAGES = ["mujoco", "numpy", "cv2", "onnxruntime", "fastapi", "uvicorn", "pydantic", "yaml", "pytest", "httpx",
            "app", "saltsim", "saltsort_common"]
results: list[tuple[str, bool, bool, str]] = []  # (name, required, ok, detail)


def check(name: str, required: bool = True):
    def wrap(fn):
        try:
            ok, detail = fn()
        except Exception as exc:  # report, don't crash the checker
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        results.append((name, required, ok, detail))
        return fn

    return wrap


def run_checks(camera: bool) -> None:
    @check("python 3.12")
    def _():
        return sys.version_info[:2] == (3, 12), sys.version.split()[0]

    for pkg in PACKAGES:

        @check(f"package {pkg}")
        def _(pkg=pkg):
            mod = importlib.import_module(pkg)
            return True, getattr(mod, "__version__", "installed")

    @check("menagerie UR5e scene")
    def _():
        from saltsim.paths import menagerie_dir, ur5e_scene_path

        rev = subprocess.run(
            ["git", "-C", str(menagerie_dir()), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        return ur5e_scene_path().is_file(), f"{ur5e_scene_path()} @ {rev or 'unknown commit'}"

    @check("UR5e model compiles")
    def _():
        from saltsim.world import load_world

        world = load_world()
        return True, f"nq={world.model.nq} nu={world.model.nu}"

    @check("SaltSort scene compiles (UR5e + 2F-85 + tray + bin + proxies)")
    def _():
        from saltsim.scene import compile_scene
        from saltsort_common.scene_config import load_scene_config

        _, model = compile_scene(load_scene_config())
        return model.nu == 7, f"nq={model.nq} nu={model.nu} bodies={model.nbody}"

    @check("config loads")
    def _():
        from app.settings import load_settings
        from saltsort_common.scene_config import load_scene_config

        settings = load_settings()
        scene = load_scene_config()
        return True, (f"{settings.source_path} (mode={settings.app.mode}, host={settings.app.host}); "
                      f"scene tray {scene.tray.width_m}x{scene.tray.depth_m} m")

    @check("model weights", required=False)
    def _():
        from app.settings import load_settings
        from saltsort_common.paths import resolve

        m = load_settings().model
        if m.backend == "none" or not m.weights_path:
            return False, "none configured: REAL MODEL mode unavailable (MANUAL mode works)"
        return resolve(m.weights_path).is_file(), m.weights_path

    @check("offscreen OpenGL render", required=False)
    def _():
        import mujoco

        from saltsim.world import load_world

        world = load_world()
        with mujoco.Renderer(world.model, 240, 320) as renderer:
            renderer.update_scene(world.data)
            pixels = renderer.render()
        return pixels.std() > 0, f"rendered {pixels.shape}"

    for tool in ("node", "npm"):

        @check(f"{tool} on PATH", required=False)
        def _(tool=tool):
            exe = shutil.which(tool)
            if not exe:
                return False, "not found"
            out = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip()
            return True, out

    if camera:

        @check("camera opens and returns a frame", required=False)
        def _():
            import cv2

            from app.settings import load_settings

            settings = load_settings()
            source = settings.camera.source
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if isinstance(source, int) else cv2.VideoCapture(source)
            try:
                ok, frame = cap.read()
                return ok, f"source={source!r} frame={None if frame is None else frame.shape}"
            finally:
                cap.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", action="store_true", help="try to open the configured camera")
    args = parser.parse_args()
    run_checks(args.camera)

    failed_required = False
    for name, required, ok, detail in results:
        tag = "PASS" if ok else ("FAIL" if required else "WARN")
        failed_required |= required and not ok
        print(f"[{tag}] {name}: {detail}")
    print("\nREQUIRED CHECKS", "FAILED" if failed_required else "PASSED")
    return 1 if failed_required else 0


if __name__ == "__main__":
    sys.exit(main())
