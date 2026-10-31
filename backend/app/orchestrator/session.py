"""Vision session: camera -> detector -> tracker -> stability gate -> pick scheduler, plus overlays.

Vision modes (never silently swapped):
  manual      no detector; targets come only from operator clicks / tray coordinates (labeled MANUAL)
  real_model  supplied ONNX model on the live camera (labeled REAL MODEL only if it actually loaded)
  replay      supplied model on recorded footage (labeled REPLAY)
"""

import json
import threading
import time

import cv2
import numpy as np

from app.calibration.homography import (
    Calibration, CalibrationError, build_calibration, detect_aruco_corners, load_calibration, save_calibration,
)
from app.camera.source import CameraError, Frame, open_source
from app.camera.worker import CaptureWorker
from app.orchestrator.scheduler import PickScheduler
from app.perception.interface import Detection, load_class_config
from app.perception.onnx_adapter import ModelError, OnnxDetector
from app.schemas.events import AlertEvent, DetectionEvent, IdFactory, now_ms
from app.settings import Settings
from app.telemetry.metrics import LatencyStats, RateMeter
from app.tracking.tracker import Track, Tracker
from saltsort_common.frames import TRAY, FramePoint
from saltsort_common.paths import resolve
from saltsort_common.scene_config import SceneConfig

ROLE_COLORS = {"reject": (60, 60, 230), "accept": (80, 200, 80), "obstruction": (0, 150, 255), "unmapped": (150, 150, 150)}
MANUAL_USABLE = {"valid", "unvalidated", "out_of_tolerance"}


class Session:
    def __init__(self, settings: Settings, scene: SceneConfig, scheduler: PickScheduler, publish):
        self.settings = settings
        self.scene = scene
        self.scheduler = scheduler
        self.publish = publish
        self.mode = settings.app.mode
        self.vision_mode = settings.app.mode  # event label: manual | real_model | replay
        self.alert_ids = IdFactory("alert")
        self.detection_ids = IdFactory("vision")
        self.manual_ids = IdFactory("manual")
        self.tracker = Tracker(settings.tracking.iou_match_threshold, settings.tracking.max_missed_frames)
        self.worker: CaptureWorker | None = None
        self.state = "stopped"  # stopped | running | error
        self.camera_error: str | None = None
        self.inference = LatencyStats()
        self.inference_fps = RateMeter()
        self.processed_fps = RateMeter()
        self.last_frame: Frame | None = None
        self.latest_jpeg: bytes | None = None
        self.manual_markers: list[dict] = []
        self.tracks: list[Track] = []
        self._jpeg_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_snapshot = 0.0
        self.detector: OnnxDetector | None = None
        self.model_error: str | None = None
        self._load_model()
        self.calibration: Calibration | None = None
        self.calibration_error: str | None = None
        self._load_calibration()

    # --- model ------------------------------------------------------------------------------------
    def _load_model(self) -> None:
        m = self.settings.model
        if self.mode == "manual":
            return
        if m.backend == "none" or not m.weights_path:
            self.model_error = "no model configured (model.backend / model.weights_path)"
            return
        try:
            classes = load_class_config(resolve(m.classes_path))
            self.detector = OnnxDetector(resolve(m.weights_path), classes, m.confidence_threshold, m.iou_threshold)
        except (ModelError, ValueError, FileNotFoundError) as exc:
            self.detector = None
            self.model_error = str(exc)

    def model_status(self) -> dict:
        if self.mode == "manual":
            return {"status": "not_used", "label": "NO REAL MODEL", "detail": "manual mode: no inference"}
        if self.detector is None:
            return {"status": "error", "label": "NO REAL MODEL", "detail": self.model_error}
        return {"status": "loaded", "label": "REAL MODEL", **self.detector.metadata}

    def vision_label(self) -> str:
        if self.mode == "manual":
            return "MANUAL"
        if self.detector is None:
            return "NO REAL MODEL"
        return "REPLAY" if self.vision_mode == "replay" else "REAL MODEL"

    # --- calibration --------------------------------------------------------------------------------
    def _cal_path(self):
        return resolve(self.settings.calibration.path)

    def _load_calibration(self) -> None:
        try:
            self.calibration = load_calibration(self._cal_path())
            self.calibration_error = None
        except CalibrationError as exc:
            self.calibration, self.calibration_error = None, str(exc)

    def current_resolution(self) -> tuple[int, int] | None:
        return self.last_frame.resolution if self.last_frame else None

    def calibration_status(self) -> tuple[str, str]:
        if self.calibration_error:
            return "invalid", self.calibration_error
        if self.calibration is None:
            return "missing", "no calibration saved"
        c = self.settings.calibration
        return self.calibration.status(self.current_resolution(), (self.scene.tray.width_m, self.scene.tray.depth_m),
                                       c.min_holdout_points, c.max_holdout_error_mm)

    def calibration_info(self) -> dict:
        status, detail = self.calibration_status()
        info = {"status": status, "detail": detail, "path": self.settings.calibration.path}
        if self.calibration:
            info.update(self.calibration.data.model_dump(exclude={"homography"}))
            info["stats"] = self.calibration.data.stats()
        return info

    def set_calibration(self, corners_px: list[tuple[float, float]], resolution: tuple[int, int] | None = None,
                        method: str = "manual_corners") -> dict:
        resolution = resolution or self.current_resolution()
        if resolution is None:
            raise CalibrationError("no camera frame yet: start the session or pass the image resolution")
        k, dist = self._intrinsics(resolution)
        data = build_calibration(self.settings.camera.camera_id, resolution, corners_px, self.scene.tray.width_m,
                                 self.scene.tray.depth_m, camera_matrix=k, dist_coeffs=dist, method=method)
        save_calibration(data, self._cal_path())
        self._load_calibration()
        self._alert("info", f"calibration saved ({method}); add holdout points to validate it")
        return self.calibration_info()

    def _intrinsics(self, resolution: tuple[int, int]):
        path = self.settings.calibration.intrinsics_path
        if not path:
            return None, None
        try:
            data = json.loads(resolve(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CalibrationError(f"cannot read intrinsics {path}: {exc}") from exc
        if tuple(data["resolution"]) != tuple(resolution):
            raise CalibrationError(f"intrinsics are for {data['resolution']}, camera is {resolution}")
        return data["camera_matrix"], data["dist_coeffs"]

    def calibrate_from_aruco(self) -> dict:
        if self.last_frame is None:
            raise CalibrationError("no camera frame yet")
        corners = detect_aruco_corners(self.last_frame.image)
        return self.set_calibration(corners, self.last_frame.resolution, method="aruco")

    def add_holdout(self, image_px: tuple[float, float], tray_m: tuple[float, float]) -> dict:
        if self.calibration is None:
            raise CalibrationError("no calibration to validate")
        point = self.calibration.add_holdout(image_px, tray_m)
        save_calibration(self.calibration.data, self._cal_path())
        return {"point": point.model_dump(), **self.calibration_info()}

    def clear_holdout(self) -> dict:
        if self.calibration:
            self.calibration.data.holdout.clear()
            save_calibration(self.calibration.data, self._cal_path())
        return self.calibration_info()

    # --- camera session -------------------------------------------------------------------------------
    def start(self) -> dict:
        if self.state == "running":
            return self.status()
        cam = self.settings.camera
        try:
            source = open_source(cam.source, cam.width, cam.height)
        except CameraError as exc:
            self.state, self.camera_error = "error", str(exc)
            self._alert("error", f"camera unavailable: {exc}")
            return self.status()
        # The model running on a recorded file is a replay, never presented as live.
        self.vision_mode = "replay" if source.kind == "replay" and self.mode != "manual" else self.mode
        self.worker = CaptureWorker(source, cam.max_capture_buffer_frames, cam.stale_after_s)
        self.worker.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="vision", daemon=True)
        self._thread.start()
        self.state, self.camera_error = "running", None
        self._alert("info", f"session started ({self.vision_label()}, camera source {cam.source!r})")
        return self.status()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        if self.worker:
            self.worker.stop()
        self.worker, self._thread = None, None
        self.state = "stopped"
        self._alert("info", "session stopped")
        return self.status()

    def reset(self) -> None:
        self.tracker.reset()
        self.tracks = []
        self.manual_markers.clear()
        self.scheduler.reset()
        self._alert("info", "session reset: processed IDs, queue and virtual objects cleared by operator")

    def _run(self) -> None:
        last_id = 0
        warned_stale = False
        n = self.settings.model.detect_every_n_frames
        while not self._stop.is_set():
            frame = self.worker.latest(after_frame_id=last_id, timeout_s=0.5)
            if frame is None:
                if self.worker.stale and not warned_stale:
                    self.camera_error = self.worker.error or "no frames from camera"
                    self._alert("error", f"camera stale: {self.camera_error}; autonomous picking stopped")
                    warned_stale = True
                    self.tracks = []
                continue
            if warned_stale:
                self._alert("info", "camera frames resumed")
                warned_stale, self.camera_error = False, None
            last_id = frame.frame_id
            self.process_frame(frame, run_detector=frame.frame_id % n == 0)
            if time.time() - self._last_snapshot > 0.2:
                self._last_snapshot = time.time()
                self.publish(self.vision_snapshot(), persist=False)

    def process_frame(self, frame: Frame, run_detector: bool = True) -> None:
        self.last_frame = frame
        if self.detector is not None and run_detector:
            t0 = time.perf_counter()
            try:
                detections = self.detector.predict(frame.image, frame.frame_id)
            except Exception as exc:  # do not silently fall back to a stub
                self.model_error = f"inference failed: {exc}"
                self.detector = None
                self._alert("error", f"REAL MODEL disabled: {self.model_error}")
                detections = []
            self.inference.add((time.perf_counter() - t0) * 1000)
            self.inference_fps.tick()
            self.tracks = self.tracker.update(detections, frame.frame_id)
            self._gate(frame)
        self._render(frame)
        self.processed_fps.tick()

    # --- stability gate ------------------------------------------------------------------------------
    def _gate(self, frame: Frame) -> None:
        t = self.settings.tracking
        cal_status, cal_detail = self.calibration_status()
        obstructed = False
        for track in self.tracks:
            det = track.detection
            tray_pt = self._to_tray(det.center_px) if cal_status in MANUAL_USABLE else None
            in_tray = tray_pt is not None and self.calibration.inside_tray(tray_pt, margin_m=0.01)
            if track.role == "obstruction" and (in_tray or tray_pt is None):
                obstructed = True
            entry = self.scheduler.entry_for_track(track.track_id)
            if entry:
                self._set_track(track, frame, _track_status_for(entry.status), entry.detail or entry.status)
                continue
            if track.status in ("ignored", "queued", "processed_sim", "failed_sim"):
                continue
            if track.consecutive_hits < t.min_stable_frames:
                continue
            if track.status == "observed":
                self._set_track(track, frame, "stable", f"{track.consecutive_hits} consecutive observations")
            if track.role != "reject":
                self._set_track(track, frame, "ignored", f"class '{track.class_name}' role '{track.role}'")
            elif det.min_side_px < t.min_bbox_size_px:
                self._set_track(track, frame, "ignored", f"box {det.min_side_px:.0f}px < {t.min_bbox_size_px}px")
            elif cal_status != "valid":
                track.reason = f"waiting: calibration {cal_status} ({cal_detail}); autonomous picking disabled"
            elif not in_tray:
                self._set_track(track, frame, "ignored", "outside calibrated tray area")
            else:
                self._set_track(track, frame, "eligible", "stable reject inside tray")
                adm = self.scheduler.submit(track.track_id, tray_pt, self.vision_mode, frame.frame_id, cal_status)
                self._set_track(track, frame, "queued" if adm.accepted else "ignored",
                                adm.entry.event_id if adm.accepted else adm.reason)
        if obstructed != self.scheduler.obstructed:
            self.scheduler.obstructed = obstructed
            self._alert("warning" if obstructed else "info",
                        "workspace obstructed (unwanted object in tray): new picks paused" if obstructed
                        else "workspace clear: picking resumes")

    def _set_track(self, track: Track, frame: Frame, status: str, reason: str) -> None:
        if track.status == status and track.reason == reason:
            return
        track.status, track.reason = status, reason
        det = track.detection
        if status in ("observed", "stable", "eligible", "ignored"):
            self.publish(DetectionEvent(
                event_id=self.detection_ids.next(), frame_id=frame.frame_id, camera_id=self.settings.camera.camera_id,
                vision_mode=self.vision_mode, track_id=track.track_id, class_name=track.class_name, role=track.role,
                confidence=round(det.confidence, 4), bbox_xyxy_px=tuple(round(v, 1) for v in det.bbox_xyxy_px),
                center_px=tuple(round(v, 1) for v in det.center_px), stable=status != "observed", status=status,
                reason=reason,
            ).model_dump())

    def _to_tray(self, center_px: tuple[float, float]) -> FramePoint | None:
        if self.calibration is None:
            return None
        return self.calibration.image_to_tray(*center_px)

    # --- manual targets ------------------------------------------------------------------------------
    def manual_target_pixel(self, u: float, v: float) -> dict:
        status, detail = self.calibration_status()
        if status not in MANUAL_USABLE:
            raise CalibrationError(f"calibration {status}: {detail}. Calibrate first, or give tray coordinates.")
        res = self.current_resolution()
        if res and not (0 <= u <= res[0] and 0 <= v <= res[1]):
            raise CalibrationError(f"pixel ({u:.0f}, {v:.0f}) outside the {res[0]}x{res[1]} image")
        tray_pt = self.calibration.image_to_tray(u, v)
        result = self._submit_manual(tray_pt, status, self.last_frame.frame_id if self.last_frame else None)
        self.manual_markers.append({"u": u, "v": v, "track_id": result["track_id"]})
        return result

    def manual_target_tray(self, x_m: float, y_m: float) -> dict:
        return self._submit_manual(FramePoint(TRAY, (x_m, y_m, 0.0)), "not_used (tray coordinates)", None)

    def _submit_manual(self, tray_pt: FramePoint, cal_status: str, frame_id: int | None) -> dict:
        track_id = self.manual_ids.next()
        adm = self.scheduler.submit(track_id, tray_pt, "manual", frame_id, cal_status)
        return {
            "accepted": adm.accepted, "reason": adm.reason, "track_id": track_id, "vision_mode": "manual",
            "tray_m": [round(v, 4) for v in tray_pt.xyz[:2]], "frame": "tray",
            "pick": adm.entry.to_dict() if adm.entry else None,
        }

    # --- overlay ------------------------------------------------------------------------------------------
    def _render(self, frame: Frame) -> None:
        img = frame.image.copy()
        if self.calibration and self.calibration_status()[0] != "resolution_mismatch":
            pts = np.array(self.calibration.data.image_corners_px, dtype=np.int32)
            cv2.polylines(img, [pts], True, (255, 200, 0), 1, cv2.LINE_AA)
        for track in self.tracks:
            d = track.detection
            color = ROLE_COLORS.get(track.role, ROLE_COLORS["unmapped"])
            x1, y1, x2, y2 = map(int, d.bbox_xyxy_px)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{track.track_id} {track.class_name} {d.confidence:.2f} {track.status}"
            cv2.putText(img, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        for m in self.manual_markers[-20:]:
            entry = self.scheduler.entry_for_track(m["track_id"])
            u, v = int(m["u"]), int(m["v"])
            cv2.drawMarker(img, (u, v), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
            status = entry.status if entry else "refused"
            cv2.putText(img, f"{m['track_id']} MANUAL {status}", (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 0, 255), 1, cv2.LINE_AA)
        banner = f"REAL CAMERA ({frame.source_kind.upper()}) | {self.vision_label()} | frame {frame.frame_id}"
        cv2.rectangle(img, (0, 0), (img.shape[1], 22), (20, 20, 20), -1)
        cv2.putText(img, banner, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._jpeg_lock:
                self.latest_jpeg = buf.tobytes()

    def jpeg(self) -> bytes | None:
        with self._jpeg_lock:
            return self.latest_jpeg

    # --- status ----------------------------------------------------------------------------------------
    def camera_status(self) -> dict:
        if self.worker is None:
            return {"status": "error" if self.camera_error else "not_started", "error": self.camera_error,
                    "source": str(self.settings.camera.source)}
        h = self.worker.health()
        return {**h, "source": str(self.settings.camera.source), "resolution": self.current_resolution()}

    def vision_snapshot(self) -> dict:
        return {
            "type": "vision.snapshot", "timestamp_ms": now_ms(), "vision_mode": self.vision_mode,
            "label": self.vision_label(), "session": self.state, "camera": self.camera_status(),
            "inference": {**self.inference.summary(), "fps": self.inference_fps.rate(),
                          "source": "wall clock around detector.predict"},
            "processed_fps": self.processed_fps.rate(),
            "tracks": [{
                "track_id": t.track_id, "class_name": t.class_name, "role": t.role,
                "confidence": round(t.detection.confidence, 3),
                "bbox_xyxy_px": [round(v, 1) for v in t.detection.bbox_xyxy_px], "status": t.status,
                "reason": t.reason, "hits": t.hits,
            } for t in self.tracks],
        }

    def status(self) -> dict:
        return {"state": self.state, "mode": self.mode, "vision_mode": self.vision_mode,
                "label": self.vision_label(), "camera": self.camera_status()}

    def _alert(self, level: str, message: str) -> None:
        self.publish(AlertEvent(event_id=self.alert_ids.next(), source="session", level=level,
                                message=message).model_dump())


def _track_status_for(pick_status: str) -> str:
    if pick_status == "completed_sim":
        return "processed_sim"
    if pick_status in ("rejected_invalid", "unreachable", "timeout", "grasp_failed_sim", "collision_blocked",
                       "cancelled"):
        return "failed_sim"
    return "queued"
