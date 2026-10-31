"""Deterministic demo: send the MANUAL tray-frame targets in examples/manual-targets.json to the backend
and wait for each simulated pick to finish. Needs the simulator and backend running.

    python scripts/demo_targets.py
    python scripts/demo_targets.py --file examples/manual-targets.json --grasp-mode contact_physics
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

TERMINAL = {"completed_sim", "rejected_invalid", "unreachable", "timeout", "grasp_failed_sim", "collision_blocked",
            "cancelled"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, default=Path("examples/manual-targets.json"))
    ap.add_argument("--backend", default="http://127.0.0.1:8000")
    ap.add_argument("--grasp-mode", choices=["kinematic_demo", "contact_physics"])
    ap.add_argument("--reset", action="store_true", help="reset the session first (re-arms processed targets)")
    args = ap.parse_args()

    spec = json.loads(args.file.read_text())
    c = httpx.Client(base_url=args.backend, timeout=10)
    if args.reset:
        c.post("/api/session/reset")
    if args.grasp_mode:
        c.post("/api/session/grasp-mode", json={"grasp_mode": args.grasp_mode})
    ids = []
    for t in spec["targets"]:
        r = c.post("/api/manual-target", json={"tray_m": [t["x_m"], t["y_m"]]}).json()
        tag = "queued" if r["accepted"] else f"refused: {r['reason']}"
        print(f"{t.get('label', '')} tray=({t['x_m']}, {t['y_m']}) m -> {tag}")
        if r["accepted"]:
            ids.append(r["pick"]["event_id"])
    deadline = time.time() + 60 * max(1, len(ids))
    done: dict[str, dict] = {}
    while len(done) < len(ids) and time.time() < deadline:
        for p in c.get("/api/state").json()["scheduler"]["picks"]:
            if p["event_id"] in ids and p["status"] in TERMINAL and p["event_id"] not in done:
                done[p["event_id"]] = p
                cyc = f"{p['cycle_time_sim_s']:.1f} s sim" if p["cycle_time_sim_s"] else ""
                print(f"  {p['event_id']} {p['track_id']}: {p['status']} [{p['grasp_mode']}] {cyc} {p['detail']}")
        time.sleep(0.5)
    ok = sum(p["status"] == "completed_sim" for p in done.values())
    print(f"\n{ok}/{len(ids)} SIMULATED picks completed ({len(spec['targets']) - len(ids)} refused at admission)")
    return 0 if len(done) == len(ids) else 1


if __name__ == "__main__":
    sys.exit(main())
