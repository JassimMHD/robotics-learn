"""FastAPI entry point.

    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import routes, streams, ws
from app.api.ws import EventHub
from app.orchestrator.bridge import SimBridge
from app.orchestrator.scheduler import PickScheduler, SchedulerLoop
from app.orchestrator.session import Session
from app.settings import Settings, load_settings
from app.telemetry.logger import EventLog
from saltsort_common.paths import resolve
from saltsort_common.scene_config import SceneConfig, load_scene_config


def create_app(settings: Settings | None = None, scene: SceneConfig | None = None, bridge: SimBridge | None = None,
               background: bool = True, event_log: bool = True) -> FastAPI:
    settings = settings or load_settings()
    scene = scene or load_scene_config()
    hub = EventHub(EventLog(resolve(settings.telemetry.event_log_dir)) if event_log else None)
    bridge = bridge or SimBridge(settings.simulation.url)
    scheduler = PickScheduler(scene, bridge, hub.publish, settings.tracking.duplicate_radius_m,
                              settings.tracking.max_queue, settings.robot.grasp_mode)
    session = Session(settings, scene, scheduler, hub.publish)
    loop = SchedulerLoop(scheduler, lambda snap: hub.publish(snap, persist=False))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if background:
            loop.start()
        yield
        if background:
            loop.stop()
        if session.state == "running":
            session.stop()

    app = FastAPI(title="SaltSort Digital Twin", version="0.1.0", lifespan=lifespan)
    app.state.settings, app.state.scene, app.state.hub = settings, scene, hub
    app.state.scheduler, app.state.session = scheduler, session
    app.include_router(routes.router)
    app.include_router(streams.router)
    app.include_router(ws.router)
    return app


app = create_app()
