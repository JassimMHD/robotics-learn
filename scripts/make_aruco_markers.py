"""Generate printable ArUco markers (DICT_4X4_50, ids 0-3) for tray-corner calibration.

    python scripts/make_aruco_markers.py --size-mm 40 --out markers

Print at 100% scale (no "fit to page"). Place each marker's CENTER exactly on the tray's inner corner:
id 0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left, as seen in the camera image.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

DPI = 300


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size-mm", type=float, default=40.0)
    ap.add_argument("--out", type=Path, default=Path("markers"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    px = int(round(args.size_mm / 25.4 * DPI))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    names = ["top-left", "top-right", "bottom-right", "bottom-left"]
    for marker_id, name in enumerate(names):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, px)
        margin = px // 4
        page = np.full((px + 2 * margin + 60, px + 2 * margin), 255, np.uint8)
        page[margin : margin + px, margin : margin + px] = marker
        cv2.putText(page, f"id {marker_id} {name} | {args.size_mm:.0f} mm", (margin, px + margin + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
        path = args.out / f"aruco_{marker_id}_{name}.png"
        cv2.imwrite(str(path), page)
        print(f"wrote {path} ({px}x{px} px marker at {DPI} dpi)")


if __name__ == "__main__":
    main()
