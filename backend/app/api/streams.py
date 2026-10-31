"""MJPEG stream of the annotated REAL camera feed."""

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

router = APIRouter()


@router.get("/streams/camera.mjpg")
async def camera_stream(request: Request) -> StreamingResponse:
    session = request.app.state.session

    async def frames():
        last = None
        while not await request.is_disconnected():
            jpeg = session.jpeg()
            if jpeg is not None and jpeg is not last:
                last = jpeg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(1 / 30)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/streams/camera.jpg")
def camera_frame(request: Request) -> Response:
    jpeg = request.app.state.session.jpeg()
    if jpeg is None:
        return Response(status_code=503, content=b"no camera frame")
    return Response(jpeg, media_type="image/jpeg")
