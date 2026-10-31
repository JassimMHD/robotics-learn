"""Record raw camera footage for the replay fallback (camera.source: recordings/<file>.avi).

    python scripts/record_camera.py --source 0 --seconds 60

Press q in the preview window to stop early. Footage is private data: recordings/ is git-ignored.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out-dir", type=Path, default=Path("recordings"))
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if isinstance(src, int) else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.source!r}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("camera returned no frame")
    h, w = frame.shape[:2]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"tray-{datetime.now():%Y%m%d-%H%M%S}-{w}x{h}.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), args.fps, (w, h))
    t_end, n, period = time.time() + args.seconds, 0, 1.0 / args.fps
    next_t = time.time()
    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok:
            print("camera stopped returning frames")
            break
        if time.time() >= next_t:
            writer.write(frame)
            n += 1
            next_t += period
        cv2.imshow("recording (q to stop)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"saved {n} frames ({w}x{h}) to {path}")


if __name__ == "__main__":
    main()
