import { useState } from "react";
import type { SimSnapshot } from "../types/events";
import { Badge, Panel } from "./ui";

export default function SimView({ sim }: { sim: SimSnapshot | null }) {
  const [key, setKey] = useState(0);
  const [failed, setFailed] = useState(false);
  const connected = sim?.scheduler.sim_connected ?? false;
  const robot = sim?.robot;

  return (
    <Panel
      title="MuJoCo simulation"
      right={
        <>
          <Badge text="SIMULATED" tone="violet" />
          {robot?.grasp_mode && <Badge text={robot.grasp_mode === "kinematic_demo" ? "KINEMATIC DEMO" : "CONTACT PHYSICS"} tone={robot.grasp_mode === "kinematic_demo" ? "amber" : "sky"} />}
        </>
      }
    >
      <div className="relative aspect-[4/3] w-full overflow-hidden rounded bg-black">
        {connected && !failed ? (
          <img
            key={key}
            src={`/sim/stream.mjpg?k=${key}`}
            alt="MuJoCo simulator render"
            onError={() => {
              setFailed(true);
              setTimeout(() => { setFailed(false); setKey((k) => k + 1); }, 2000);
            }}
            className="h-full w-full object-contain"
          />
        ) : (
          <div className="flex h-full items-center justify-center p-6 text-center text-sm text-slate-400">
            {sim?.scheduler.sim_error ?? "Simulator not connected. Start it with: python -m saltsim.run"}
          </div>
        )}
      </div>
      <p className="mt-2 text-xs text-slate-400">
        Virtual robot only: it does not remove the real salt from the physical tray.
      </p>
    </Panel>
  );
}
