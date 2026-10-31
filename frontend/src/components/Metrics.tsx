import type { SimSnapshot, VisionSnapshot } from "../types/events";
import { Panel, fmt } from "./ui";

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="min-w-0 rounded border border-slate-800 bg-slate-950/50 px-3 py-2">
      <div className="truncate text-[11px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className="font-mono text-lg tabular-nums text-slate-100">{value}</div>
      {sub && <div className="truncate text-[11px] text-slate-500">{sub}</div>}
    </div>
  );
}

export default function Metrics({ vision, sim }: { vision: VisionSnapshot | null; sim: SimSnapshot | null }) {
  const c = sim?.scheduler.counts ?? {};
  const m = sim?.sim_metrics;
  const picks = sim?.scheduler.picks ?? [];
  const lastDone = [...picks].reverse().find((p) => p.cycle_time_sim_s !== null);
  const failures = Object.entries(c)
    .filter(([k]) => !["queued", "completed_sim"].includes(k) && !k.startsWith("completed_"))
    .reduce((a, [, v]) => a + v, 0);
  const rtf = m?.real_time_factor;

  return (
    <Panel title="Metrics (measured on this machine)">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-8">
        <Stat label="Tracks" value={String(vision?.tracks.length ?? 0)} sub={`${vision?.tracks.filter((t) => t.role === "reject").length ?? 0} reject`} />
        <Stat label="Queued" value={String(c.queued ?? 0)} sub={`${sim?.scheduler.queue.length ?? 0} waiting`} />
        <Stat label="Sim picks · kinematic" value={String(c.completed_kinematic_demo ?? 0)} sub="scripted attach (demo)" />
        <Stat label="Sim picks · physics" value={String(c.completed_contact_physics ?? 0)} sub="contact grasp" />
        <Stat label="Failures" value={String(failures)} sub="all error outcomes" />
        <Stat label="Cycle time" value={fmt(lastDone?.cycle_time_sim_s, 1, " s")} sub="last pick, sim time" />
        <Stat label="Camera" value={fmt(vision?.camera?.fps as number | null, 1, " fps")} sub={`age ${fmt(vision?.camera?.last_frame_age_s as number | null, 2, " s")}`} />
        <Stat label="Inference" value={fmt(vision?.inference.p50_ms, 0, " ms")} sub={`p95 ${fmt(vision?.inference.p95_ms, 0, " ms")}`} />
        <Stat label="Real-time factor" value={fmt(rtf, 2)} sub={rtf !== null && rtf !== undefined && rtf < 0.95 ? "SIM SLOWER THAN REAL TIME" : "sim / wall clock"} />
        <Stat label="Physics" value={m?.physics_steps_per_s ? `${(m.physics_steps_per_s / 1000).toFixed(1)}k/s` : "—"} sub="steps per second" />
        <Stat label="Sim render" value={fmt(m?.render_fps, 1, " fps")} sub="stream frames" />
        <Stat label="Sim time" value={fmt(m?.simulation_time_s, 0, " s")} />
      </div>
    </Panel>
  );
}
