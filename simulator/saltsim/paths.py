"""Filesystem locations for simulator assets."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def menagerie_dir() -> Path:
    override = os.environ.get("SALTSORT_MENAGERIE_DIR")
    path = Path(override) if override else REPO_ROOT / "vendor" / "mujoco_menagerie"
    return path if path.is_absolute() else REPO_ROOT / path


def ur5e_scene_path() -> Path:
    return menagerie_dir() / "universal_robots_ur5e" / "scene.xml"
