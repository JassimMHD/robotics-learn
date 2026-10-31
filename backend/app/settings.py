"""Runtime configuration loaded from YAML (see config/app.example.yaml)."""

import ipaddress
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from saltsort_common.paths import REPO_ROOT, config_file, display

EXAMPLE_CONFIG = REPO_ROOT / "config" / "app.example.yaml"


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AppSection(_Section):
    host: str = "127.0.0.1"
    port: int = 8000
    mode: Literal["manual", "real_model", "replay"] = "manual"

    @field_validator("host")
    @classmethod
    def host_must_be_loopback(cls, host: str) -> str:
        # MVP 1 is local-only: never expose robot control on a network interface.
        if not _is_loopback(host):
            raise ValueError(f"app.host must be a loopback address for MVP 1, got {host!r}")
        return host


class CameraSection(_Section):
    source: int | str = 0
    camera_id: str = "cam-01"
    width: int = 640
    height: int = 480
    max_capture_buffer_frames: int = Field(2, ge=1, le=10)
    stale_after_s: float = Field(2.0, gt=0)


class ModelSection(_Section):
    backend: Literal["none", "onnx"] = "none"
    weights_path: str | None = None
    classes_path: str = "config/classes.example.yaml"
    confidence_threshold: float = Field(0.70, ge=0, le=1)
    iou_threshold: float = Field(0.45, ge=0, le=1)
    detect_every_n_frames: int = Field(1, ge=1)


class TrackingSection(_Section):
    min_stable_frames: int = Field(3, ge=1)
    max_missed_frames: int = Field(15, ge=1)
    iou_match_threshold: float = Field(0.3, gt=0, le=1)
    min_bbox_size_px: int = Field(12, ge=0)
    duplicate_radius_m: float = Field(0.025, gt=0)
    max_queue: int = Field(10, ge=1)
    max_active_pick_requests: Literal[1] = 1
    prevent_repeat_picks_until_reset: bool = True


class CalibrationSection(_Section):
    path: str = "config/calibration.local.json"
    intrinsics_path: str | None = None  # from scripts/calibrate_intrinsics.py; None = distortion not modeled
    min_holdout_points: int = Field(5, ge=1)
    max_holdout_error_mm: float = Field(10.0, gt=0)


class RobotSection(_Section):
    grasp_mode: Literal["kinematic_demo", "contact_physics"] = "kinematic_demo"


class SimulationSection(_Section):
    url: str = "http://127.0.0.1:8001"

    @field_validator("url")
    @classmethod
    def url_must_be_loopback(cls, url: str) -> str:
        if not _is_loopback(urlparse(url).hostname or ""):
            raise ValueError(f"simulation.url must point at a loopback address, got {url!r}")
        return url.rstrip("/")


class TelemetrySection(_Section):
    event_log_dir: str = "logs"


class Settings(_Section):
    app: AppSection = AppSection()
    camera: CameraSection = CameraSection()
    model: ModelSection = ModelSection()
    tracking: TrackingSection = TrackingSection()
    calibration: CalibrationSection = CalibrationSection()
    robot: RobotSection = RobotSection()
    simulation: SimulationSection = SimulationSection()
    telemetry: TelemetrySection = TelemetrySection()
    source_path: str | None = None


def config_path() -> Path:
    return config_file("SALTSORT_CONFIG", "app")


def load_settings(path: Path | None = None) -> Settings:
    path = path or config_path()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Settings(**raw, source_path=display(path))
