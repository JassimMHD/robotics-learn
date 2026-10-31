"""Loopback-only HTTP API of the simulator process (backend -> simulator commands).

Handlers never touch MuJoCo state directly: they submit closures to the sim thread via runtime.call().
"""

import asyncio
import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from saltsim.planner import PlanningError
from saltsim.object_manager import ObjectError
from saltsim.runtime import SimRuntime


class TargetIn(BaseModel):
    frame: str
    position_m: tuple[float, float, float]
    yaw_rad: float = 0.0


class PickIn(BaseModel):
    type: Literal["sim.pick.requested"] = "sim.pick.requested"
    event_id: str = Field(min_length=1, max_length=64)
    track_id: str = Field(min_length=1, max_length=64)
    source_frame_id: int | None = None
    target: TargetIn
    grasp_mode: Literal["kinematic_demo", "contact_physics"]
    vision_mode: Literal["real_model", "manual", "replay"]
    queued_at_ms: int | None = None


class PlaceIn(BaseModel):
    track_id: str = Field(min_length=1, max_length=64)
    target: TargetIn


def create_sim_app(rt: SimRuntime) -> FastAPI:
    app = FastAPI(title="SaltSort simulator (SIMULATED)", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "simulated": True, "metrics": rt.metrics()}

    @app.get("/status")
    def status() -> dict:
        return rt.call(rt.snapshot)

    @app.get("/scene")
    def scene() -> dict:
        cfg = rt.cfg.model_dump(exclude={"source_path"})
        return {"frame": "robot_base", "units": "SI (m, kg, rad)", **cfg}

    @app.post("/pick")
    def pick(req: PickIn) -> JSONResponse:
        code, body = rt.call(lambda: rt.cmd_pick(req.model_dump()))
        return JSONResponse(body, status_code=code)

    @app.get("/picks/{event_id}")
    def get_pick(event_id: str) -> dict:
        rec = rt.call(lambda: rt.picks.get(event_id))
        if rec is None:
            raise HTTPException(404, f"unknown pick {event_id}")
        return rt.call(rec.to_dict)

    @app.post("/objects")
    def place(req: PlaceIn) -> dict:
        if req.target.frame != "robot_base":
            raise HTTPException(422, "target frame must be robot_base")
        try:
            return rt.call(lambda: rt.cmd_place(req.track_id, list(req.target.position_m), req.target.yaw_rad))
        except (PlanningError, ObjectError) as e:
            raise HTTPException(409, str(e)) from e

    @app.post("/cancel")
    def cancel() -> dict:
        rt.call(lambda: rt.controller.cancel("cancelled by operator"))
        return {"cancelled": True}

    @app.post("/reset")
    def reset() -> dict:
        return rt.call(rt.cmd_reset)

    @app.get("/frame.jpg")
    async def frame() -> Response:
        requested = time.time()
        rt.request_frames()
        for _ in range(40):  # wait for a render made after this request, never serve a stale one
            jpeg = rt.latest_jpeg(newer_than=requested)
            if jpeg:
                return Response(jpeg, media_type="image/jpeg")
            await asyncio.sleep(0.05)
        raise HTTPException(503, "no fresh simulator frame rendered")

    @app.get("/stream.mjpg")
    async def stream() -> StreamingResponse:
        started = time.time()

        async def frames():
            last = None
            while True:
                rt.request_frames()
                jpeg = rt.latest_jpeg(newer_than=started)
                if jpeg and jpeg is not last:
                    last = jpeg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                await asyncio.sleep(1 / 15)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app
