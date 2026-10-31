import { useState } from "react";
import type { CalibrationInfo } from "../types/events";
import type { ClickMode } from "./LiveCamera";
import { Button, Panel, fmt } from "./ui";
import { api } from "../hooks/useTelemetry";

interface Props {
  calibration: CalibrationInfo | null;
  clickMode: ClickMode;
  setClickMode: (m: ClickMode) => void;
  pendingCorners: number[][];
  setPendingCorners: (c: number[][]) => void;
  pendingHoldout: number[] | null;
  setPendingHoldout: (p: number[] | null) => void;
  refresh: () => Promise<void>;
  onError: (msg: string) => void;
}

const TONE: Record<string, string> = {
  valid: "text-emerald-400", unvalidated: "text-amber-300", out_of_tolerance: "text-red-400",
  missing: "text-red-400", invalid: "text-red-400", resolution_mismatch: "text-red-400",
};

export default function CalibrationPanel(p: Props) {
  const [hx, setHx] = useState("");
  const [hy, setHy] = useState("");
  const cal = p.calibration;
  const run = async (fn: () => Promise<unknown>) => {
    try {
      await fn();
    } catch (e) {
      p.onError(e instanceof Error ? e.message : String(e));
    }
    await p.refresh();
  };

  const submitHoldout = () =>
    run(async () => {
      if (!p.pendingHoldout) return;
      await api("/api/calibration/holdout", "POST", { image_px: p.pendingHoldout, tray_m: [Number(hx), Number(hy)] });
      p.setPendingHoldout(null);
      setHx("");
      setHy("");
    });

  return (
    <Panel title="Calibration (image → tray)">
      <p className="text-sm">
        Status: <span className={`font-semibold ${TONE[cal?.status ?? "missing"] ?? ""}`}>{cal?.status ?? "—"}</span>
        <span className="ml-2 text-xs text-slate-400">{cal?.detail}</span>
      </p>
      {cal?.stats && (
        <p className="mt-1 text-xs text-slate-400">
          Holdout: {cal.stats.n} pts · mean {fmt(cal.stats.mean_error_mm, 1, " mm")} · max {fmt(cal.stats.max_error_mm, 1, " mm")}
          {cal.resolution ? ` · for ${cal.resolution[0]}×${cal.resolution[1]}` : ""}
        </p>
      )}
      <div className="mt-3 flex flex-wrap gap-2">
        <Button tone={p.clickMode === "corners" ? "active" : "default"}
          onClick={() => { p.setPendingCorners([]); p.setClickMode(p.clickMode === "corners" ? "target" : "corners"); }}>
          {p.clickMode === "corners" ? `Cancel corners (${p.pendingCorners.length}/4)` : "Click 4 corners"}
        </Button>
        <Button onClick={() => run(() => api("/api/calibration/aruco", "POST"))}>Detect ArUco</Button>
        <Button tone={p.clickMode === "holdout" ? "active" : "default"} disabled={!cal || cal.status === "missing"}
          onClick={() => { p.setPendingHoldout(null); p.setClickMode(p.clickMode === "holdout" ? "target" : "holdout"); }}>
          {p.clickMode === "holdout" ? "Done with holdouts" : "Add holdout points"}
        </Button>
        <Button disabled={!cal?.stats?.n} onClick={() => run(() => api("/api/calibration/holdout", "DELETE"))}>
          Clear holdouts
        </Button>
      </div>
      {p.pendingHoldout && (
        <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
          <span className="text-xs text-slate-400">
            Pixel ({p.pendingHoldout.map((v) => v.toFixed(0)).join(", ")}) is at tray (m):
          </span>
          <input placeholder="x" value={hx} onChange={(e) => setHx(e.target.value)} className="w-20 rounded bg-slate-800 px-2 py-1 font-mono" />
          <input placeholder="y" value={hy} onChange={(e) => setHy(e.target.value)} className="w-20 rounded bg-slate-800 px-2 py-1 font-mono" />
          <Button tone="primary" disabled={hx === "" || hy === ""} onClick={submitHoldout}>Save point</Button>
        </div>
      )}
      <p className="mt-3 text-[11px] leading-relaxed text-slate-500">
        Tray frame: origin at the inner corner shown bottom-left in the image, x → image right, y → image up.
        Autonomous picks need status <b>valid</b> (≥ 5 holdout points within tolerance). See docs/CALIBRATION.md.
      </p>
    </Panel>
  );
}
