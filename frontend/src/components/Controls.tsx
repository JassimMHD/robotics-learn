import { useState } from "react";
import { api } from "../hooks/useTelemetry";
import type { GraspMode, Health, SimSnapshot, VisionSnapshot } from "../types/events";
import { Button, Panel } from "./ui";

interface Props {
  health: Health | null;
  vision: VisionSnapshot | null;
  sim: SimSnapshot | null;
  onError: (msg: string) => void;
}

export default function Controls({ health, vision, sim, onError }: Props) {
  const [tx, setTx] = useState("0.20");
  const [ty, setTy] = useState("0.15");
  const running = vision?.session === "running";
  const paused = sim?.scheduler.paused ?? false;
  const grasp = (sim?.scheduler.grasp_mode ?? health?.labels.grasp_mode ?? "kinematic_demo") as GraspMode;

  const call = (path: string, body?: unknown) => api(path, "POST", body).catch((e: Error) => onError(e.message));

  const trayTarget = async () => {
    const x = Number(tx), y = Number(ty);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return onError("tray coordinates must be numbers (meters)");
    const r = await api<{ accepted: boolean; reason: string }>("/api/manual-target", "POST", { tray_m: [x, y] })
      .catch((e: Error) => { onError(e.message); return null; });
    if (r && !r.accepted) onError(`target refused: ${r.reason}`);
  };

  return (
    <Panel title="Controls">
      <div className="flex flex-wrap items-center gap-2">
        {running ? (
          <Button onClick={() => call("/api/session/stop")}>Stop camera</Button>
        ) : (
          <Button tone="primary" onClick={() => call("/api/session/start")}>Start camera</Button>
        )}
        {paused ? (
          <Button tone="active" onClick={() => call("/api/session/resume")}>Resume picking</Button>
        ) : (
          <Button onClick={() => call("/api/session/pause")}>Pause picking</Button>
        )}
        <Button
          tone="danger"
          title="Clears the pick queue, processed IDs and virtual objects"
          onClick={() => {
            if (confirm("Reset: clear pick queue, processed chunk IDs and virtual objects?")) call("/api/session/reset");
          }}
        >
          Reset
        </Button>
        <span className="mx-1 h-6 w-px bg-slate-700" />
        <span className="text-xs text-slate-400">Grasp mode</span>
        <Button tone={grasp === "kinematic_demo" ? "active" : "default"} onClick={() => call("/api/session/grasp-mode", { grasp_mode: "kinematic_demo" })}>
          Kinematic demo
        </Button>
        <Button tone={grasp === "contact_physics" ? "active" : "default"} onClick={() => call("/api/session/grasp-mode", { grasp_mode: "contact_physics" })}>
          Contact physics
        </Button>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
        <span className="text-xs text-slate-400">Manual target (tray frame, m)</span>
        <label className="flex items-center gap-1">x
          <input value={tx} onChange={(e) => setTx(e.target.value)} className="w-20 rounded bg-slate-800 px-2 py-1 font-mono" />
        </label>
        <label className="flex items-center gap-1">y
          <input value={ty} onChange={(e) => setTy(e.target.value)} className="w-20 rounded bg-slate-800 px-2 py-1 font-mono" />
        </label>
        <Button onClick={trayTarget}>Send MANUAL target</Button>
      </div>
      {health && !health.autonomous_picking_enabled && (
        <p className="mt-3 text-xs text-amber-300">
          Autonomous picking disabled: {health.blockers.join(" · ")}
        </p>
      )}
    </Panel>
  );
}
