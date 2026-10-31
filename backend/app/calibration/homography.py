"""Image -> tray planar calibration (spec section 7).

Corner order is as seen in the camera image: top-left, top-right, bottom-right, bottom-left.
Tray frame (see saltsort_common.frames): origin at the image-bottom-left inner corner, x toward
image-right, y toward image-top, z up. So TL->(0,D), TR->(W,D), BR->(W,0), BL->(0,0).

A homography maps the tray *plane* only; chunk height is a separate, labeled estimate.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field

from saltsort_common.frames import TRAY, FramePoint

CORNER_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")
ARUCO_DICT = cv2.aruco.DICT_4X4_50
ARUCO_CORNER_IDS = (0, 1, 2, 3)  # marker centers placed on TL, TR, BR, BL inner tray corners


class CalibrationError(Exception):
    pass


class HoldoutPoint(BaseModel):
    image_px: tuple[float, float]
    tray_m: tuple[float, float]  # measured ground truth in the tray frame
    mapped_m: tuple[float, float] | None = None
    error_mm: float | None = None


class CalibrationData(BaseModel):
    version: int = 1
    camera_id: str
    resolution: tuple[int, int]  # (width, height) the calibration is valid for
    image_corners_px: list[tuple[float, float]] = Field(min_length=4, max_length=4)
    tray_width_m: float = Field(gt=0)
    tray_depth_m: float = Field(gt=0)
    homography: list[list[float]]
    camera_matrix: list[list[float]] | None = None
    dist_coeffs: list[float] | None = None
    distortion_note: str = "lens distortion not modeled; verify with holdout errors"
    method: str = "manual_corners"
    created_at: str
    holdout: list[HoldoutPoint] = []

    def stats(self) -> dict:
        errs = [h.error_mm for h in self.holdout if h.error_mm is not None]
        if not errs:
            return {"n": 0, "mean_error_mm": None, "max_error_mm": None}
        return {"n": len(errs), "mean_error_mm": round(float(np.mean(errs)), 2), "max_error_mm": round(max(errs), 2)}


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def validate_corner_order(corners: np.ndarray) -> None:
    """Corners must form a convex quad in TL, TR, BR, BL (clockwise on screen) order."""
    if corners.shape != (4, 2):
        raise CalibrationError("exactly four (u, v) corners are required")
    crosses = []
    for i in range(4):
        a, b, c = corners[i], corners[(i + 1) % 4], corners[(i + 2) % 4]
        crosses.append(_cross2(b - a, c - b))
    if not (all(x > 0 for x in crosses)):
        raise CalibrationError(
            "corners are not a convex quad in top-left, top-right, bottom-right, bottom-left order "
            "(reversed, crossed or collinear)"
        )
    area = 0.5 * abs(sum(_cross2(corners[i], corners[(i + 1) % 4]) for i in range(4)))
    if area < 100.0:
        raise CalibrationError(f"corner quad too small ({area:.0f} px^2)")


def _undistort(points: np.ndarray, data: CalibrationData) -> np.ndarray:
    if data.camera_matrix is None or data.dist_coeffs is None:
        return points
    k = np.array(data.camera_matrix, dtype=float)
    d = np.array(data.dist_coeffs, dtype=float)
    return cv2.undistortPoints(points.reshape(-1, 1, 2).astype(np.float64), k, d, P=k).reshape(-1, 2)


def _homography_4pt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Exact homography through 4 correspondences (float64; OpenCV's fit works in float32)."""
    a, b = [], []
    for (u, v), (x, y) in zip(src, dst):
        a.append([u, v, 1, 0, 0, 0, -u * x, -v * x])
        a.append([0, 0, 0, u, v, 1, -u * y, -v * y])
        b.extend([x, y])
    h = np.linalg.solve(np.array(a, dtype=float), np.array(b, dtype=float))
    return np.append(h, 1.0).reshape(3, 3)


def tray_corner_coords(width_m: float, depth_m: float) -> np.ndarray:
    return np.array([[0.0, depth_m], [width_m, depth_m], [width_m, 0.0], [0.0, 0.0]])


def build_calibration(
    camera_id: str, resolution: tuple[int, int], corners_px: list[tuple[float, float]], width_m: float,
    depth_m: float, camera_matrix=None, dist_coeffs=None, method: str = "manual_corners",
) -> CalibrationData:
    corners = np.array(corners_px, dtype=float)
    validate_corner_order(corners)
    w, h = resolution
    if np.any(corners < 0) or np.any(corners[:, 0] > w) or np.any(corners[:, 1] > h):
        raise CalibrationError(f"corners outside the {w}x{h} image")
    draft = CalibrationData(
        camera_id=camera_id, resolution=resolution, image_corners_px=[tuple(c) for c in corners_px],
        tray_width_m=width_m, tray_depth_m=depth_m, homography=np.eye(3).tolist(),
        camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, method=method,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        distortion_note=("undistorted with supplied intrinsics" if camera_matrix is not None
                         else "lens distortion not modeled; verify with holdout errors"),
    )
    src = _undistort(corners, draft).astype(np.float64)
    dst = tray_corner_coords(width_m, depth_m).astype(np.float64)
    try:
        hmat = _homography_4pt(src, dst)
    except np.linalg.LinAlgError as exc:
        raise CalibrationError("degenerate homography") from exc
    if not np.all(np.isfinite(hmat)) or abs(np.linalg.det(hmat)) < 1e-12:
        raise CalibrationError("degenerate homography")
    return draft.model_copy(update={"homography": hmat.tolist()})


@dataclass
class Calibration:
    data: CalibrationData

    @property
    def h(self) -> np.ndarray:
        return np.array(self.data.homography, dtype=float)

    def image_to_tray(self, u: float, v: float) -> FramePoint:
        p = _undistort(np.array([[u, v]], dtype=float), self.data)[0]
        q = self.h @ np.array([p[0], p[1], 1.0])
        return FramePoint(TRAY, (float(q[0] / q[2]), float(q[1] / q[2]), 0.0))

    def tray_to_image(self, x: float, y: float) -> tuple[float, float]:
        q = np.linalg.inv(self.h) @ np.array([x, y, 1.0])
        return float(q[0] / q[2]), float(q[1] / q[2])  # undistorted pixel; fine for overlays

    def inside_tray(self, point: FramePoint, margin_m: float = 0.0) -> bool:
        x, y, _ = point.xyz
        return (-margin_m <= x <= self.data.tray_width_m + margin_m
                and -margin_m <= y <= self.data.tray_depth_m + margin_m)

    def add_holdout(self, image_px: tuple[float, float], tray_m: tuple[float, float]) -> HoldoutPoint:
        mapped = self.image_to_tray(*image_px).xyz
        err = float(np.hypot(mapped[0] - tray_m[0], mapped[1] - tray_m[1]) * 1000)
        point = HoldoutPoint(image_px=image_px, tray_m=tray_m, mapped_m=(mapped[0], mapped[1]), error_mm=round(err, 2))
        self.data.holdout.append(point)
        return point

    def status(self, current_resolution: tuple[int, int] | None, tray_dims_m: tuple[float, float],
               min_holdout: int, max_error_mm: float) -> tuple[str, str]:
        d = self.data
        if abs(d.tray_width_m - tray_dims_m[0]) > 1e-6 or abs(d.tray_depth_m - tray_dims_m[1]) > 1e-6:
            return "invalid", f"tray size {d.tray_width_m}x{d.tray_depth_m} m differs from scene {tray_dims_m}"
        if current_resolution is not None and tuple(current_resolution) != tuple(d.resolution):
            return "resolution_mismatch", f"calibrated for {d.resolution}, camera is {current_resolution}; recalibrate"
        s = d.stats()
        if s["n"] < min_holdout:
            return "unvalidated", f"{s['n']}/{min_holdout} holdout points measured"
        if s["max_error_mm"] > max_error_mm:
            return "out_of_tolerance", f"max holdout error {s['max_error_mm']} mm > {max_error_mm} mm"
        return "valid", f"{s['n']} holdout points, mean {s['mean_error_mm']} mm, max {s['max_error_mm']} mm"


def save_calibration(cal: CalibrationData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cal.model_dump_json(indent=2), encoding="utf-8")


def load_calibration(path: Path) -> Calibration | None:
    """None if missing. Raises CalibrationError if the file is malformed or geometrically invalid."""
    if not path.is_file():
        return None
    try:
        data = CalibrationData(**json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise CalibrationError(f"malformed calibration file {path.name}: {exc}") from exc
    validate_corner_order(np.array(data.image_corners_px, dtype=float))
    return Calibration(data)


def detect_aruco_corners(image_bgr: np.ndarray) -> list[tuple[float, float]]:
    """Centers of ArUco markers 0..3 (DICT_4X4_50) as TL, TR, BR, BL tray corners."""
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))
    found = {} if ids is None else {int(i): c.reshape(4, 2).mean(axis=0) for i, c in zip(ids.flatten(), corners)}
    missing = [i for i in ARUCO_CORNER_IDS if i not in found]
    if missing:
        raise CalibrationError(f"ArUco markers not found: {missing} (need ids 0=TL, 1=TR, 2=BR, 3=BL)")
    return [tuple(map(float, found[i])) for i in ARUCO_CORNER_IDS]
