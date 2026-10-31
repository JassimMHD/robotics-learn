import type { LogEvent, SimSnapshot, VisionSnapshot } from "../types/events";
import { Panel, statusTone } from "./ui";

const ROLE_TONE: Record<string, string> = {
  reject: "text-red-400", accept: "text-emerald-400", obstruction: "text-orange-400", unmapped: "text-slate-400",
};

export function DetectionList({ vision }: { vision: VisionSnapshot | null }) {
  const tracks = vision?.tracks ?? [];
  return (
    <Panel title={`Detections (${vision?.label ?? "—"})`}>
      {tracks.length === 0 ? (
        <p className="text-sm text-slate-400">
          {vision?.label === "MANUAL" ? "Manual mode: no model inference. Click the camera image to add targets." : "No tracks."}
        </p>
      ) : (
        <div className="max-h-56 overflow-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-slate-500"><tr><th>Track</th><th>Class</th><th>Conf</th><th>Status</th><th>Reason</th></tr></thead>
            <tbody>
              {tracks.map((t) => (
                <tr key={t.track_id} className="border-t border-slate-800">
                  <td className="py-1 font-mono">{t.track_id}</td>
                  <td className={ROLE_TONE[t.role] ?? ""}>{t.class_name}</td>
                  <td className="font-mono">{t.confidence.toFixed(2)}</td>
                  <td>{t.status}</td>
                  <td className="max-w-48 truncate text-slate-400" title={t.reason}>{t.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

export function PickList({ sim }: { sim: SimSnapshot | null }) {
  const picks = [...(sim?.scheduler.picks ?? [])].reverse();
  return (
    <Panel title="Simulated picks">
      {picks.length === 0 ? (
        <p className="text-sm text-slate-400">No picks yet.</p>
      ) : (
        <div className="max-h-56 overflow-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-slate-500"><tr><th>Pick</th><th>Track</th><th>Source</th><th>Mode</th><th>Status</th><th>Cycle</th></tr></thead>
            <tbody>
              {picks.map((p) => (
                <tr key={p.event_id} className="border-t border-slate-800" title={p.detail}>
                  <td className="py-1 font-mono">{p.event_id.split("-").slice(-1)[0]}</td>
                  <td className="font-mono">{p.track_id}</td>
                  <td className="uppercase">{p.vision_mode}</td>
                  <td>{p.grasp_mode === "kinematic_demo" ? "kinematic" : "physics"}</td>
                  <td className={statusTone(p.status)}>{p.status}</td>
                  <td className="font-mono">{p.cycle_time_sim_s ? `${p.cycle_time_sim_s.toFixed(1)} s` : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

function describe(e: LogEvent): string {
  switch (e.type) {
    case "system.alert":
      return e.message ?? "";
    case "vision.detection":
      return `${e.track_id} ${e.class_name} → ${e.status}${e.reason ? ` (${e.reason})` : ""}`;
    case "sim.pick.requested":
      return `${e.event_id} queued for ${e.track_id} [${e.vision_mode?.toUpperCase()}, ${e.grasp_mode}]`;
    case "sim.pick.status":
      return `${e.pick_event_id} ${e.status}${e.detail ? `: ${e.detail}` : ""}`;
    default:
      return e.type;
  }
}

export function EventLog({ log }: { log: LogEvent[] }) {
  return (
    <Panel title="Event log">
      <div className="max-h-64 overflow-auto font-mono text-[11px] leading-5">
        {log.length === 0 && <p className="text-slate-500">No events yet.</p>}
        {log.map((e) => (
          <div key={e.event_id} className="flex gap-2 border-t border-slate-900">
            <span className="shrink-0 text-slate-500">{new Date(e.timestamp_ms).toLocaleTimeString()}</span>
            <span className={`shrink-0 ${e.level === "error" ? "text-red-400" : e.level === "warning" ? "text-orange-400" : e.status ? statusTone(e.status) : "text-slate-400"}`}>
              {e.type}
            </span>
            <span className="min-w-0 break-words text-slate-200">{describe(e)}</span>
          </div>
        ))}
      </div>
    </Panel>
  );
}
