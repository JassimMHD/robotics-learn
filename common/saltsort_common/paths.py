"""Repository-relative paths."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def config_file(env_var: str, name: str) -> Path:
    """`$env_var` if set, else config/<name>.local.yaml if present, else config/<name>.example.yaml."""
    override = os.environ.get(env_var)
    if override:
        return resolve(override)
    local = REPO_ROOT / "config" / f"{name}.local.yaml"
    return local if local.is_file() else REPO_ROOT / "config" / f"{name}.example.yaml"


def display(path: Path) -> str:
    path = path.resolve()
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)
