"""REST routes (spec section 6)."""

import time
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.calibration.homography import CalibrationError
from app.orchestrator.scheduler import PickScheduler
from app.orchestrator.session import Session
from app.settings import Settings

router = APIRouter(prefix="/api")


def _session(request: Request) -> Session:
    return request.app.state.session


def _scheduler(request: Request) -> PickScheduler:
    return request.app.state.scheduler


@router.get("/health")
def health(request: Request) -> dict:
    session, scheduler = _session(request), _scheduler(request)
    settings: Settings = request.app.state.settings
    model = session.model_status()
    cal_status, cal_detail = session.calibration_status()
    camera = session.camera_status()
    sim_ok = scheduler.sim_error is None and scheduler.sim_status is not None
    components = {
        "api": {"status": "ok"},
        "model": model,
        "camera": camera,
        "calibration": {"status": cal_status, "detail": cal_detail, "path": settings.calibration.path},
        "simulator": {"status": "connected" if sim_ok else "not_connected",
                      "detail": scheduler.sim_error or settings.simulation.url},
        "session": {"status": session.state},
    }
    blockers = []
    if model["status"] != "loaded":
        blockers.append("no real model loaded")
    if cal_status != "valid":
        blockers.append(f"calibration {cal_status}")
    if camera["status"] != "ok":
        blockers.append(f"camera {camera['status']}")
    if not sim_ok:
        blockers.append("simulator not connected")
    if scheduler.obstructed:
        blockers.append("workspace obstructed")
    return {
        "status": "ok",
        "timestamp_ms": int(time.time() * 1000),
        "labels": {"vision_mode": session.vision_label(), "robot": "SIMULATED", "grasp_mode": scheduler.grasp_mode},
        "components": components,
        "autonomous_picking_enabled": not blockers,
        "manual_picking_enabled": sim_ok,
        "blockers": blockers,
        "config_source": settings.source_path,
    }


@router.get("/config")
def config(request: Request) -> dict:
    settings: Settings = request.app.state.settings
    return {
        "settings": settings.model_dump(exclude={"source_path"}),
        "scene": request.app.state.scene.model_dump(exclude={"source_path"}),
        "note": "example/estimated values unless measured on the real rig",
    }


@router.get("/state")
def state(request: Request) -> dict:
    session = _session(request)
    return {
        "session": session.status(), "vision": session.vision_snapshot(),
        "scheduler": _scheduler(request).snapshot(), "calibration": session.calibration_info(),
        "events": list(request.app.state.hub.recent)[-100:],
    }


# --- session -------------------------------------------------------------------------------------------
@router.post("/session/start")
def session_start(request: Request) -> dict:
    return _session(request).start()


@router.post("/session/stop")
def session_stop(request: Request) -> dict:
    return _session(request).stop()


@router.post("/session/pause")
def session_pause(request: Request) -> dict:
    _scheduler(request).paused = True
    return {"paused": True}


@router.post("/session/resume")
def session_resume(request: Request) -> dict:
    _scheduler(request).paused = False
    return {"paused": False}


@router.post("/session/reset")
def session_reset(request: Request) -> dict:
    _session(request).reset()
    return {"reset": True}


class GraspModeIn(BaseModel):
    grasp_mode: Literal["kinematic_demo", "contact_physics"]


@router.post("/session/grasp-mode")
def grasp_mode(body: GraspModeIn, request: Request) -> dict:
    _scheduler(request).grasp_mode = body.grasp_mode  # applies to picks queued from now on
    return {"grasp_mode": body.grasp_mode}


# --- calibration ---------------------------------------------------------------------------------------
class CalibrationIn(BaseModel):
    image_corners_px: list[tuple[float, float]] = Field(min_length=4, max_length=4)
    resolution: tuple[int, int] | None = None


class HoldoutIn(BaseModel):
    image_px: tuple[float, float]
    tray_m: tuple[float, float]


def _cal_call(fn, *args):
    try:
        return fn(*args)
    except CalibrationError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/calibration")
def get_calibration(request: Request) -> dict:
    return _session(request).calibration_info()


@router.post("/calibration")
def post_calibration(body: CalibrationIn, request: Request) -> dict:
    return _cal_call(_session(request).set_calibration, body.image_corners_px, body.resolution)


@router.post("/calibration/aruco")
def calibration_aruco(request: Request) -> dict:
    return _cal_call(_session(request).calibrate_from_aruco)


@router.post("/calibration/holdout")
def calibration_holdout(body: HoldoutIn, request: Request) -> dict:
    return _cal_call(_session(request).add_holdout, body.image_px, body.tray_m)


@router.delete("/calibration/holdout")
def calibration_clear_holdout(request: Request) -> dict:
    return _session(request).clear_holdout()


# --- manual targets ------------------------------------------------------------------------------------
class ManualTargetIn(BaseModel):
    """Exactly one of image_px (needs calibration) or tray_m (tray frame, meters)."""

    image_px: tuple[float, float] | None = None
    tray_m: tuple[float, float] | None = None


@router.post("/manual-target")
def manual_target(body: ManualTargetIn, request: Request) -> dict:
    if (body.image_px is None) == (body.tray_m is None):
        raise HTTPException(422, "give exactly one of image_px or tray_m")
    session = _session(request)
    if body.image_px is not None:
        return _cal_call(session.manual_target_pixel, *body.image_px)
    return session.manual_target_tray(*body.tray_m)


# --- metrics / events ----------------------------------------------------------------------------------
@router.get("/metrics")
def metrics(request: Request) -> dict:
    session, scheduler = _session(request), _scheduler(request)
    sim = scheduler.sim_status or {}
    cal = session.calibration.data.stats() if session.calibration else {}
    return {
        "camera": {**session.camera_status(), "units": {"fps": "frames/s", "last_frame_age_s": "s"},
                   "source": "backend capture thread"},
        "inference": {**session.inference.summary(), "fps": session.inference_fps.rate(), "label": session.vision_label(),
                      "source": "wall clock around detector.predict on this machine"},
        "calibration_holdout_error": {**cal, "units": "mm", "source": "operator-measured holdout points"},
        "picks": {"counts": dict(scheduler.counts),
                  "note": "kinematic_demo and contact_physics completions are counted separately; all SIMULATED"},
        "simulator": {**(sim.get("metrics") or {}), "counts": sim.get("counts")},
        "detector_accuracy": "not measured: needs manually labeled held-out images from this camera",
    }


@router.get("/events")
def events(request: Request, limit: int = 100) -> list[dict]:
    return list(request.app.state.hub.recent)[-limit:]
