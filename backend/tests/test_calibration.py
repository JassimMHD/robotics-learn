"""M2: planar calibration, holdout validation, fail-closed rules (spec section 7)."""

import cv2
import numpy as np
import pytest

from app.calibration.homography import (
    Calibration, CalibrationError, build_calibration, detect_aruco_corners, load_calibration, save_calibration,
)

W, D = 0.40, 0.30
RES = (640, 480)
# Slightly perspective-distorted tray seen by a camera that is not perfectly top-down.
CORNERS = [(102.0, 61.0), (548.0, 70.0), (561.0, 420.0), (90.0, 411.0)]
TRAY_STATUS_ARGS = dict(tray_dims_m=(W, D), min_holdout=5, max_error_mm=10.0)


def _cal() -> Calibration:
    return Calibration(build_calibration("cam-01", RES, CORNERS, W, D))


def test_corners_map_to_tray_coordinates():
    cal = _cal()
    expected = [(0, D), (W, D), (W, 0), (0, 0)]
    for (u, v), (x, y) in zip(CORNERS, expected):
        p = cal.image_to_tray(u, v)
        assert p.frame == "tray"
        assert p.xyz[0] == pytest.approx(x, abs=1e-9) and p.xyz[1] == pytest.approx(y, abs=1e-9)


def test_round_trip_tray_to_image():
    cal = _cal()
    u, v = cal.tray_to_image(0.13, 0.21)
    p = cal.image_to_tray(u, v)
    assert p.xyz[:2] == pytest.approx((0.13, 0.21), abs=1e-9)


def test_holdout_points_report_error_in_mm():
    cal = _cal()
    truth = [(0.05, 0.05), (0.2, 0.15), (0.35, 0.25), (0.1, 0.25), (0.3, 0.05)]
    for i, (x, y) in enumerate(truth):
        u, v = cal.tray_to_image(x, y)
        # Simulated measurement noise of ~1-3 px on the clicked holdout point.
        cal.add_holdout((u + (i % 3), v - (i % 2) * 2), (x, y))
    stats = cal.data.stats()
    assert stats["n"] == 5 and 0 < stats["max_error_mm"] < 5
    status, _ = cal.status(RES, **TRAY_STATUS_ARGS)
    assert status == "valid"


def test_too_few_holdouts_is_unvalidated():
    cal = _cal()
    cal.add_holdout(cal.tray_to_image(0.2, 0.15), (0.2, 0.15))
    assert cal.status(RES, **TRAY_STATUS_ARGS)[0] == "unvalidated"


def test_large_holdout_error_is_out_of_tolerance():
    cal = _cal()
    for x, y in [(0.05, 0.05), (0.2, 0.15), (0.35, 0.25), (0.1, 0.25), (0.3, 0.05)]:
        cal.add_holdout(cal.tray_to_image(x, y), (x + 0.02, y))  # 20 mm off
    assert cal.status(RES, **TRAY_STATUS_ARGS)[0] == "out_of_tolerance"


@pytest.mark.parametrize("corners", [
    list(reversed(CORNERS)),                                     # counter-clockwise
    [CORNERS[0], CORNERS[2], CORNERS[1], CORNERS[3]],            # crossed (bow-tie)
    [CORNERS[1], CORNERS[0], CORNERS[3], CORNERS[2]],            # mirrored
    [(100, 100), (200, 100), (300, 100), (100, 300)],            # collinear
])
def test_bad_corner_order_rejected(corners):
    with pytest.raises(CalibrationError):
        build_calibration("cam-01", RES, corners, W, D)


def test_resolution_change_fails_closed():
    assert _cal().status((1280, 720), **TRAY_STATUS_ARGS)[0] == "resolution_mismatch"


def test_tray_size_mismatch_is_invalid():
    assert _cal().status(RES, tray_dims_m=(0.5, 0.3), min_holdout=5, max_error_mm=10)[0] == "invalid"


def test_save_load_and_malformed_file(tmp_path):
    path = tmp_path / "cal.json"
    assert load_calibration(path) is None
    save_calibration(_cal().data, path)
    loaded = load_calibration(path)
    assert loaded.image_to_tray(*CORNERS[2]).xyz[:2] == pytest.approx((W, 0), abs=1e-9)
    path.write_text('{"camera_id": "x"}')
    with pytest.raises(CalibrationError):
        load_calibration(path)


def test_aruco_corners_detected_on_synthetic_image():
    img = np.full((480, 640, 3), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    centers = [(120, 90), (520, 90), (520, 390), (120, 390)]
    for marker_id, (cx, cy) in enumerate(centers):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 60)
        img[cy - 30 : cy + 30, cx - 30 : cx + 30] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    found = detect_aruco_corners(img)
    assert np.allclose(found, centers, atol=1.0)


def test_aruco_missing_markers_reported():
    with pytest.raises(CalibrationError, match="not found"):
        detect_aruco_corners(np.full((480, 640, 3), 255, np.uint8))
