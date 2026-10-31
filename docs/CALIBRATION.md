# Calibration

Two transforms connect a camera pixel to the virtual robot:

```text
image_px --(homography H, per camera + resolution)--> tray (m) --(T_robot_from_tray, measured)--> robot_base (m)
```

## Frames

| Frame | Origin | Axes |
|---|---|---|
| `image_px` | top-left pixel | u right, v down |
| `tray` | the tray's inner corner that appears **bottom-left** in the image | x → image right, y → image up, z up |
| `robot_base` | robot base (== MuJoCo world) | x forward, y left, z up |

Corner order, as seen in the image: **top-left → (0, D)**, **top-right → (W, D)**, **bottom-right → (W, 0)**,
**bottom-left → (0, 0)**, with W = `tray.width_m` and D = `tray.depth_m` from `config/scene.*.yaml`.

## Procedure

1. Fix the camera rigidly above the tray, roughly top-down. Do not change resolution afterwards.
2. Measure the tray's **interior** width and depth and write them in `config/scene.local.yaml`
   (copy `scene.example.yaml`). Measure where that tray sits relative to the robot base and set
   `tray.pose_in_robot` (this *is* `T_robot_from_tray`). For a virtual-only robot you choose this pose,
   but keep it consistent with the simulator scene (both read the same file).
3. Optional, recommended: lens intrinsics. Photograph a printed chessboard 15–30 times at the system
   resolution, then
   `python scripts/calibrate_intrinsics.py --images "calib/*.jpg" --cols 9 --rows 6 --square-mm 25`
   and set `calibration.intrinsics_path: config/intrinsics.local.json`. Without it, distortion is not
   modeled and the holdout error tells you whether that is acceptable.
4. Corners, either way:
   - **Click**: dashboard → Calibration → *Click 4 corners* (TL, TR, BR, BL).
   - **ArUco**: `python scripts/make_aruco_markers.py --size-mm 40`, print at 100 %, put each marker's
     **center** on a corner (id 0 TL, 1 TR, 2 BR, 3 BL), then *Detect ArUco*.
5. **Holdout validation** (required): place a small marker at ≥ 5 *other* measured spots (not the
   corners), click each and type its true tray coordinates. The panel shows mean / max error in mm.
6. Status must read `valid` (≥ `min_holdout_points`, max error ≤ `max_holdout_error_mm`).

## Fail-closed rules

| Status | Autonomous (REAL MODEL / REPLAY) picks | MANUAL pixel clicks |
|---|---|---|
| `missing`, `invalid`, `resolution_mismatch` | blocked | blocked |
| `unvalidated`, `out_of_tolerance` | blocked | allowed, labeled with the status |
| `valid` | allowed | allowed |

Reversed / crossed / collinear corners are rejected. A resolution change makes the calibration
`resolution_mismatch` until redone. The tray size in the calibration must match the scene config.

## Height

A homography maps the tray *plane* only. The chunk height used for the grasp is the configured proxy
half-height (`salt_proxy.half_size_m[2]`), marked `height_estimated: true` in every pick request. The
planner also keeps the closed fingertips ≥ 3 mm above the tray floor.

## Grasp point

The detector gives axis-aligned boxes (no masks), so the target is the box center and the chunk yaw
is unknown (0 is used). The gripper first tries to close across the proxy's short side and falls back
to a 90° rotation if the approach would collide.
