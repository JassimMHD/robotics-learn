"""Estimate camera intrinsics and lens distortion from chessboard photos (spec section 7, step 3).

    python scripts/calibrate_intrinsics.py --images "calib_imgs/*.jpg" --cols 9 --rows 6 --square-mm 25
        --out config/intrinsics.local.json

Take 15-30 photos at the SAME resolution the system uses, tilting the board across the whole view.
Then set calibration.intrinsics_path in config/app.local.yaml and redo the 4-corner calibration.
"""

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="glob pattern")
    ap.add_argument("--cols", type=int, required=True, help="inner corners per row")
    ap.add_argument("--rows", type=int, required=True, help="inner corners per column")
    ap.add_argument("--square-mm", type=float, required=True)
    ap.add_argument("--out", type=Path, default=Path("config/intrinsics.local.json"))
    args = ap.parse_args()

    grid = np.zeros((args.rows * args.cols, 3), np.float32)
    grid[:, :2] = np.mgrid[0 : args.cols, 0 : args.rows].T.reshape(-1, 2) * args.square_mm / 1000.0
    obj_pts, img_pts, size = [], [], None
    for path in sorted(glob.glob(args.images)):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        if size and gray.shape[::-1] != size:
            raise SystemExit(f"{path}: resolution {gray.shape[::-1]} differs from {size}")
        size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows))
        print(f"{path}: {'found' if ok else 'no board'}")
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
            obj_pts.append(grid)
            img_pts.append(corners)
    if len(obj_pts) < 8:
        raise SystemExit(f"only {len(obj_pts)} usable images; need at least 8")
    rms, k, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "resolution": list(size), "camera_matrix": k.tolist(), "dist_coeffs": dist.flatten().tolist(),
        "rms_reprojection_error_px": round(float(rms), 4), "images_used": len(obj_pts),
    }, indent=2))
    print(f"RMS reprojection error {rms:.3f} px from {len(obj_pts)} images -> {args.out}")


if __name__ == "__main__":
    main()
