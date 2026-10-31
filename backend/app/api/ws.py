"""Thread-safe event hub feeding WebSocket clients, the recent-event buffer and the JSONL log."""

import asyncio
import threading
from collections import deque

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.telemetry.logger import EventLog

router = APIRouter()


class EventHub:
    def __init__(self, log: EventLog | None = None, recent: int = 300):
        self.log = log
        self.recent: deque[dict] = deque(maxlen=recent)
        self._clients: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self._lock = threading.Lock()

    def publish(self, event: dict, persist: bool = True) -> None:
        """persist=False for periodic snapshots (not logged, not kept in the recent list)."""
        if persist:
            self.recent.append(event)
            if self.log:
                self.log.write(event)
        with self._lock:
            clients = list(self._clients)
        for loop, queue in clients:
            try:
                loop.call_soon_threadsafe(_offer, queue, event)
            except RuntimeError:  # loop closed
                pass

    def subscribe(self, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        with self._lock:
            self._clients.append((loop, queue))
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self._lock:
            self._clients = [(l, q) for l, q in self._clients if q is not queue]


def _offer(queue: asyncio.Queue, event: dict) -> None:
    if queue.full():
        try:
            queue.get_nowait()  # slow client: drop its oldest message, never block publishers
        except asyncio.QueueEmpty:
            pass
    queue.put_nowait(event)


@router.websocket("/ws/telemetry")
async def telemetry(ws: WebSocket) -> None:
    await ws.accept()
    hub: EventHub = ws.app.state.hub
    queue = hub.subscribe(asyncio.get_running_loop())
    try:
        for event in list(hub.recent)[-50:]:
            await ws.send_json(event)
        while True:
            await ws.send_json(await queue.get())
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.unsubscribe(queue)
