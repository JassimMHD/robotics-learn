# Examples

`manual-targets.json` holds deterministic MANUAL targets in the tray frame (meters) for a repeatable
demo without a camera or model:

```powershell
python scripts/demo_targets.py --reset
python scripts/demo_targets.py --reset --grasp-mode contact_physics
```

Expected: A, B and C are picked in simulation; the B duplicate and the out-of-tray target are refused
at admission. These are not detections and are labeled `MANUAL` everywhere.
