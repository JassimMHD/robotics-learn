# External robot assets

Robot models are **not committed**. They come from the official
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), cloned
into the git-ignored `vendor/mujoco_menagerie` folder.

| Item | Value |
|---|---|
| Repository | https://github.com/google-deepmind/mujoco_menagerie |
| Pinned commit | `b0655057a9df2abe39ea7fc989914ba6f248301c` (2026-10-31) |
| Directories used | `universal_robots_ur5e`, `robotiq_2f85` |
| UR5e license | BSD-3-Clause style, (c) 2018 ROS Industrial Consortium (see `universal_robots_ur5e/LICENSE`) |
| Robotiq 2F-85 license | BSD-3-Clause style, (c) 2013 ROS-Industrial (see `robotiq_2f85/LICENSE`) |
| Menagerie repository license | see top-level `LICENSE` in the checkout |

Re-check the licenses before any commercial delivery (spec rule 10).

## Install (PowerShell, from the repo root)

```powershell
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git vendor/mujoco_menagerie
git -C vendor/mujoco_menagerie sparse-checkout set universal_robots_ur5e robotiq_2f85
git -C vendor/mujoco_menagerie fetch --depth 1 origin b0655057a9df2abe39ea7fc989914ba6f248301c
git -C vendor/mujoco_menagerie checkout b0655057a9df2abe39ea7fc989914ba6f248301c
```

To use a checkout elsewhere, set `SALTSORT_MENAGERIE_DIR`.

## Notes

- The UR5e asset is **arm only** (6 position actuators, `home` keyframe). No gripper is assumed.
- The Robotiq 2F-85 is fetched now but not integrated until a later milestone.
