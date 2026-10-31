"""Detector protocol and normalized detection record. No robot logic lives in perception."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

ROLES = ("reject", "accept", "obstruction")


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    role: str  # reject | accept | obstruction | unmapped
    confidence: float
    bbox_xyxy_px: tuple[float, float, float, float]

    @property
    def center_px(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy_px
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def min_side_px(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy_px
        return min(x2 - x1, y2 - y1)


class Detector(Protocol):
    vision_mode: str

    def predict(self, frame_bgr: np.ndarray, frame_id: int) -> list[Detection]:
        """Return normalized detections in image pixels."""


class ClassConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_classes: dict[int, str]
    roles: dict[str, list[str]]

    def role_of(self, class_name: str) -> str:
        for role, names in self.roles.items():
            if class_name in names:
                return role
        return "unmapped"


def load_class_config(path: Path) -> ClassConfig:
    cfg = ClassConfig(**yaml.safe_load(path.read_text(encoding="utf-8")))
    unknown = set(cfg.roles) - set(ROLES)
    if unknown:
        raise ValueError(f"unknown class roles {sorted(unknown)}; allowed {ROLES}")
    names = set(cfg.model_classes.values())
    for role, members in cfg.roles.items():
        missing = set(members) - names
        if missing:
            raise ValueError(f"role '{role}' lists classes not in model_classes: {sorted(missing)}")
    return cfg
