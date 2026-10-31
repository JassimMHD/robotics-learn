import type { SimSnapshot } from "../types/events";
import { Panel } from "./ui";

export default function RobotStatus({ sim }: { sim: SimSnapshot | null }) {
  const r = sim?.robot;
  const active = sim?.scheduler.active;
  return (
    <Panel title="Robot (simulated)">
      {!r ? (
        <p className="text-sm text-slate-400">No simulator status.</p>
      ) : (
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
          <dt className="text-slate-400">State</dt>
          <dd className="font-mono font-semibold text-amber-300">{r.robot_state}</dd>
          <dt className="text-slate-400">Pick</dt>
          <dd className="truncate font-mono">{active ? `${active.event_id} → ${active.track_id}` : "—"}</dd>
          <dt className="text-slate-400">Target</dt>
          <dd className="font-mono">
            {active ? `[${active.target.position_m.map((v) => v.toFixed(3)).join(", ")}] m robot_base` : "—"}
          </dd>
          <dt className="text-slate-400">Reachable</dt>
          <dd>{r.target_reachable === null ? "—" : r.target_reachable ? "yes" : "no"}</dd>
          <dt className="text-slate-400">Collision</dt>
          <dd className={r.collision_flag ? "text-red-400" : ""}>{r.collision_flag ? `yes (${r.last_collision})` : "no"}</dd>
          <dt className="text-slate-400">Joints</dt>
          <dd className="font-mono text-xs">{r.joint_positions_rad.map((q) => q.toFixed(2)).join(" ")} rad</dd>
          <dt className="text-slate-400">Status</dt>
          <dd>
            {sim?.scheduler.obstructed ? <span className="text-orange-400">workspace obstructed · </span> : null}
            {sim?.scheduler.paused ? <span className="text-amber-300">paused</span> : "running"}
          </dd>
        </dl>
      )}
    </Panel>
  );
}
