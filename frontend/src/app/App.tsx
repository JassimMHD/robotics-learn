import { useState } from "react";
import CalibrationPanel from "../components/CalibrationPanel";
import Controls from "../components/Controls";
import { DetectionList, EventLog, PickList } from "../components/Lists";
import LiveCamera, { type ClickMode } from "../components/LiveCamera";
import Metrics from "../components/Metrics";
import RobotStatus from "../components/RobotStatus";
import SimView from "../components/SimView";
import { Badge } from "../components/ui";
import { api, useTelemetry } from "../hooks/useTelemetry";

export default function App() {
  const t = useTelemetry();
  const [clickMode, setClickMode] = useState<ClickMode>("target");
  const [pendingCorners, setPendingCorners] = useState<number[][]>([]);
  const [pendingHoldout, setPendingHoldout] = useState<number[] | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const showError = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast((cur) => (cur === msg ? null : cur)), 6000);
  };

  const onImageClick = async (px: [number, number]) => {
    try {
      if (clickMode === "corners") {
        const next = [...pendingCorners, px];
        if (next.length < 4) return setPendingCorners(next);
        setPendingCorners([]);
        setClickMode("target");
        await api("/api/calibration", "POST", { image_corners_px: next });
        await t.refreshCalibration();
      } else if (clickMode === "holdout") {
        setPendingHoldout(px);
      } else {
        const r = await api<{ accepted: boolean; reason: string }>("/api/manual-target", "POST", { image_px: px });
        if (!r.accepted) showError(`target refused: ${r.reason}`);
      }
    } catch (e) {
      showError(e instanceof Error ? e.message : String(e));
    }
  };

  const label = t.health?.labels.vision_mode;
  const robotState = t.sim?.robot?.robot_state;

  return (
    <div className="min-h-screen">
      <header className="flex flex-wrap items-center gap-2 border-b border-slate-800 bg-slate-950 px-4 py-3">
        <h1 className="mr-auto text-lg font-semibold">SaltSort Digital Twin</h1>
        {label && <Badge text={label} tone={label === "REAL MODEL" ? "green" : label === "REPLAY" ? "sky" : label === "MANUAL" ? "amber" : "red"} />}
        <Badge text="SIMULATED" tone="violet" />
        {t.health && <Badge text={t.health.labels.grasp_mode === "kinematic_demo" ? "KINEMATIC DEMO GRASP" : "CONTACT PHYSICS GRASP"} tone="slate" />}
        <Badge text={robotState ?? "SIM OFFLINE"} tone={robotState ? "slate" : "red"} />
        <Badge text={t.healthError ? "BACKEND OFFLINE" : t.wsConnected ? "LIVE" : "CONNECTING"} tone={t.healthError ? "red" : t.wsConnected ? "green" : "slate"} />
      </header>

      {t.healthError && (
        <div className="border-b border-red-800 bg-red-950 px-4 py-2 text-sm">
          Backend unreachable ({t.healthError}). Start it: <code>python -m uvicorn app.main:app --host 127.0.0.1 --port 8000</code>
        </div>
      )}
      {toast && (
        <div className="fixed right-4 bottom-4 z-10 max-w-md rounded border border-red-700 bg-red-950 px-4 py-3 text-sm shadow-lg">
          {toast}
        </div>
      )}

      <main className="mx-auto grid max-w-[1600px] gap-3 p-3 lg:grid-cols-2">
        <LiveCamera vision={t.vision} clickMode={clickMode} pendingCorners={pendingCorners} onImageClick={onImageClick} />
        <SimView sim={t.sim} />
        <div className="lg:col-span-2">
          <Metrics vision={t.vision} sim={t.sim} />
        </div>
        <Controls health={t.health} vision={t.vision} sim={t.sim} onError={showError} />
        <RobotStatus sim={t.sim} />
        <CalibrationPanel
          calibration={t.calibration} clickMode={clickMode} setClickMode={setClickMode}
          pendingCorners={pendingCorners} setPendingCorners={setPendingCorners}
          pendingHoldout={pendingHoldout} setPendingHoldout={setPendingHoldout}
          refresh={t.refreshCalibration} onError={showError}
        />
        <DetectionList vision={t.vision} />
        <PickList sim={t.sim} />
        <EventLog log={t.log} />
      </main>
      <footer className="px-4 pb-4 text-xs text-slate-500">
        Camera feed is REAL; every robot action is SIMULATED in MuJoCo. Kinematic-demo and contact-physics outcomes are
        counted separately. No detector accuracy is shown: it has not been measured on labeled held-out images.
      </footer>
    </div>
  );
}
