import { useCallback, useEffect, useRef, useState } from "react";
import type { CalibrationInfo, Health, LogEvent, SimSnapshot, VisionSnapshot } from "../types/events";

const MAX_LOG = 300;

export interface Telemetry {
  health: Health | null;
  healthError: string | null;
  vision: VisionSnapshot | null;
  sim: SimSnapshot | null;
  calibration: CalibrationInfo | null;
  log: LogEvent[];
  wsConnected: boolean;
  refreshCalibration: () => Promise<void>;
}

export async function api<T = unknown>(path: string, method = "GET", body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `HTTP ${res.status}`);
  return data as T;
}

export function useTelemetry(): Telemetry {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [vision, setVision] = useState<VisionSnapshot | null>(null);
  const [sim, setSim] = useState<SimSnapshot | null>(null);
  const [calibration, setCalibration] = useState<CalibrationInfo | null>(null);
  const [log, setLog] = useState<LogEvent[]>([]);
  const [wsConnected, setWsConnected] = useState(false);
  const seen = useRef(new Set<string>());

  const refreshCalibration = useCallback(async () => {
    try {
      setCalibration(await api<CalibrationInfo>("/api/calibration"));
    } catch {
      /* health shows the backend error */
    }
  }, []);

  // Health polling: never keep showing stale "ok" data when the backend is down.
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const h = await api<Health>("/api/health");
        if (!cancelled) {
          setHealth(h);
          setHealthError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setHealth(null);
          setHealthError(err instanceof Error ? err.message : String(err));
        }
      }
    };
    poll();
    refreshCalibration();
    const id = setInterval(poll, 2000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [refreshCalibration]);

  // Telemetry WebSocket with reconnect.
  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let closed = false;
    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws/telemetry`);
      ws.onopen = () => setWsConnected(true);
      ws.onclose = () => {
        setWsConnected(false);
        setVision(null);
        setSim(null);
        if (!closed) retry = setTimeout(connect, 1500);
      };
      ws.onmessage = (msg) => {
        const ev = JSON.parse(msg.data);
        if (ev.type === "vision.snapshot") setVision(ev);
        else if (ev.type === "sim.snapshot") setSim(ev);
        else if (ev.event_id && !seen.current.has(ev.event_id)) {
          seen.current.add(ev.event_id);
          setLog((prev) => [ev as LogEvent, ...prev].slice(0, MAX_LOG));
        }
      };
    };
    connect();
    return () => {
      closed = true;
      clearTimeout(retry);
      ws?.close();
    };
  }, []);

  return { health, healthError, vision, sim, calibration, log, wsConnected, refreshCalibration };
}
