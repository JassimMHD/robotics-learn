// Mirrors backend/app/schemas/events.py and the snapshot messages on /ws/telemetry.

export type VisionLabel = "MANUAL" | "REPLAY" | "REAL MODEL" | "NO REAL MODEL";
export type GraspMode = "kinematic_demo" | "contact_physics";

export interface ComponentStatus {
  status: string;
  label?: string;
  detail?: string | null;
  error?: string | null;
  [key: string]: unknown;
}

export interface Health {
  status: "ok";
  timestamp_ms: number;
  labels: { vision_mode: VisionLabel; robot: "SIMULATED"; grasp_mode: GraspMode };
  components: Record<string, ComponentStatus>;
  autonomous_picking_enabled: boolean;
  manual_picking_enabled: boolean;
  blockers: string[];
  config_source: string | null;
}

export interface TrackView {
  track_id: string;
  class_name: string;
  role: string;
  confidence: number;
  bbox_xyxy_px: number[];
  status: string;
  reason: string;
  hits: number;
}

export interface VisionSnapshot {
  type: "vision.snapshot";
  timestamp_ms: number;
  vision_mode: string;
  label: VisionLabel;
  session: string;
  camera: ComponentStatus & { fps?: number | null; last_frame_age_s?: number | null; resolution?: number[] | null };
  inference: { p50_ms: number | null; p95_ms: number | null; samples: number; fps: number | null };
  processed_fps: number | null;
  tracks: TrackView[];
}

export interface PickView {
  event_id: string;
  track_id: string;
  status: string;
  detail: string;
  grasp_mode: GraspMode;
  vision_mode: string;
  target: { frame: string; position_m: number[]; yaw_rad: number };
  tray_xy_m: number[];
  calibration_status: string;
  cycle_time_sim_s: number | null;
  simulated: true;
}

export interface RobotStatus {
  type: "sim.robot.status";
  pick_event_id: string | null;
  robot_state: string;
  grasp_mode: GraspMode | null;
  simulation_time_s: number;
  joint_positions_rad: number[];
  target_reachable: boolean | null;
  collision_flag: boolean;
  last_collision: string;
  simulated: true;
}

export interface SimSnapshot {
  type: "sim.snapshot";
  timestamp_ms: number;
  robot: RobotStatus | null;
  sim_metrics: {
    real_time_factor: number | null;
    physics_steps_per_s: number | null;
    render_fps: number | null;
    simulation_time_s: number;
  } | null;
  sim_counts: Record<string, number> | null;
  objects: { body: string; track_id: string; status: string; position_m: number[] }[];
  scheduler: {
    active: PickView | null;
    queue: PickView[];
    picks: PickView[];
    paused: boolean;
    obstructed: boolean;
    grasp_mode: GraspMode;
    counts: Record<string, number>;
    sim_connected: boolean;
    sim_error: string | null;
  };
}

export interface LogEvent {
  type: string;
  event_id: string;
  timestamp_ms: number;
  source?: string;
  level?: string;
  message?: string;
  status?: string;
  detail?: string;
  reason?: string;
  track_id?: string;
  pick_event_id?: string;
  class_name?: string;
  vision_mode?: string;
  grasp_mode?: string;
}

export interface CalibrationInfo {
  status: string;
  detail: string;
  path: string;
  resolution?: number[];
  image_corners_px?: number[][];
  method?: string;
  created_at?: string;
  stats?: { n: number; mean_error_mm: number | null; max_error_mm: number | null };
  holdout?: { image_px: number[]; tray_m: number[]; error_mm: number }[];
}
