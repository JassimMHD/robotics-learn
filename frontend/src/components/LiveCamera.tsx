import { useRef, useState, type MouseEvent } from "react";
import type { VisionSnapshot } from "../types/events";
import { Badge, Panel } from "./ui";

export type ClickMode = "target" | "corners" | "holdout";

interface Props {
  vision: VisionSnapshot | null;
  clickMode: ClickMode;
  pendingCorners: number[][];
  onImageClick: (px: [number, number]) => void;
}

const CORNER_LABELS = ["top-left", "top-right", "bottom-right", "bottom-left"];

export default function LiveCamera({ vision, clickMode, pendingCorners, onImageClick }: Props) {
  const img = useRef<HTMLImageElement>(null);
  const [streamKey, setStreamKey] = useState(0);
  const [failed, setFailed] = useState(false);
  const running = vision?.session === "running";

  const click = (e: MouseEvent<HTMLImageElement>) => {
    const el = img.current;
    if (!el || !el.naturalWidth) return;
    const rect = el.getBoundingClientRect();
    // Map display coordinates to native camera pixels.
    const u = ((e.clientX - rect.left) / rect.width) * el.naturalWidth;
    const v = ((e.clientY - rect.top) / rect.height) * el.naturalHeight;
    onImageClick([Math.round(u * 10) / 10, Math.round(v * 10) / 10]);
  };

  const hint =
    clickMode === "corners"
      ? `Click the ${CORNER_LABELS[pendingCorners.length]} inner tray corner (${pendingCorners.length + 1}/4)`
      : clickMode === "holdout"
        ? "Click a measured holdout point, then enter its tray coordinates"
        : "Click a salt chunk to send a MANUAL target to the simulated robot";

  return (
    <Panel
      title="Real camera"
      right={
        <>
          <Badge text={vision?.label ?? "NO SESSION"} tone={vision?.label === "REAL MODEL" ? "green" : vision?.label === "REPLAY" ? "sky" : vision?.label === "MANUAL" ? "amber" : "red"} />
          <span className="text-xs text-slate-400">
            {vision?.camera?.resolution ? `${vision.camera.resolution[0]}×${vision.camera.resolution[1]}` : ""}
          </span>
        </>
      }
    >
      <div className="relative aspect-[4/3] w-full overflow-hidden rounded bg-black">
        {running && !failed ? (
          <img
            key={streamKey}
            ref={img}
            src={`/streams/camera.mjpg?k=${streamKey}`}
            alt="Annotated live camera feed"
            onClick={click}
            onError={() => {
              setFailed(true);
              setTimeout(() => { setFailed(false); setStreamKey((k) => k + 1); }, 1500);
            }}
            className={`h-full w-full object-contain ${clickMode === "target" ? "cursor-crosshair" : "cursor-cell"}`}
          />
        ) : (
          <div className="flex h-full items-center justify-center p-6 text-center text-sm text-slate-400">
            {vision?.camera?.error
              ? `Camera error: ${vision.camera.error}`
              : "Camera session not running. Press Start camera."}
          </div>
        )}
      </div>
      <p className={`mt-2 text-xs ${clickMode === "target" ? "text-slate-400" : "text-amber-300"}`}>{hint}</p>
    </Panel>
  );
}
