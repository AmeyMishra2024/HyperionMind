from __future__ import annotations

import asyncio
import base64
import json
import cv2
from threading import Thread
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import APIConnectionError, APITimeoutError, AuthenticationError, PermissionDeniedError, RateLimitError
from pydantic import BaseModel, Field

from app.demo import demo_respond
from app.openai_service import respond, synthesize, transcribe, validate_api_key
from app.sessions import SessionRegistry
from app.slam import HAS_SCIPY, CameraCalibration, SlamRegistry
from app.vision import VisionEngine
from app.live import SlamRunner, encode_jpeg
from app.webcam import SyntheticCapture, WebcamCapture, list_camera_names

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="HyperionMind", version="1.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

sessions = SessionRegistry()
vision = VisionEngine()
slam_sessions = SlamRegistry(BASE_DIR.parent / "slam_output")


@app.on_event("startup")
async def preload_vision_model() -> None:
    """Warm the local pose model without delaying the HTTP server startup."""
    Thread(target=vision.warm_up, name="vision-model-warmup", daemon=True).start()


class ApiKeyRequest(BaseModel):
    api_key: str = Field(min_length=20, max_length=512)


class TextTurnRequest(BaseModel):
    session_id: str
    person_id: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=4000)


def require_session(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Session expired. Enter the API key again.")
    return session


def audio_payload(audio: bytes) -> str:
    return base64.b64encode(audio).decode("ascii")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/slam")
async def slam_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "slam.html")


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True, "vision_mode": vision.mode}


@app.get("/api/slam/{session_id}/export")
async def export_slam_map(session_id: str) -> FileResponse:
    exported = await asyncio.to_thread(slam_sessions.export, session_id)
    if exported is None:
        raise HTTPException(status_code=404, detail="SLAM session not found or expired.")
    map_path, _ = exported
    return FileResponse(map_path, media_type="application/octet-stream", filename=map_path.name)


@app.get("/api/slam/cameras")
async def slam_cameras() -> dict:
    names = await asyncio.to_thread(list_camera_names)
    return {"cameras": [{"index": i, "name": n} for i, n in enumerate(names or [])],
            "names_available": names is not None, "bundle_adjustment": HAS_SCIPY}


@app.get("/api/slam/{session_id}/costmap")
async def export_slam_costmap(session_id: str) -> FileResponse:
    path = await asyncio.to_thread(slam_sessions.export_costmap, session_id)
    if path is None:
        raise HTTPException(status_code=404, detail="No costmap yet. Map a little more first.")
    return FileResponse(path, media_type="image/png", filename=path.name)


@app.get("/api/slam/{session_id}/trajectory")
async def export_slam_trajectory(session_id: str) -> FileResponse:
    exported = await asyncio.to_thread(slam_sessions.export, session_id)
    if exported is None:
        raise HTTPException(status_code=404, detail="SLAM session not found or expired.")
    _, trajectory_path = exported
    return FileResponse(trajectory_path, media_type="text/csv", filename=trajectory_path.name)


@app.post("/api/session")
async def create_session(request: ApiKeyRequest) -> dict:
    key = request.api_key.strip()
    try:
        await asyncio.to_thread(validate_api_key, key)
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail="OpenAI rejected this API key.") from exc
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=403, detail="This API key does not have the required project permission.") from exc
    except RateLimitError as exc:
        raise HTTPException(status_code=429, detail="OpenAI rate-limited the validation request. Try again shortly.") from exc
    except (APIConnectionError, APITimeoutError) as exc:
        raise HTTPException(status_code=503, detail="The server cannot currently reach OpenAI. The key was not rejected.") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="OpenAI validation failed unexpectedly.") from exc
    return {"session_id": sessions.create(key)}


@app.post("/api/demo-session")
async def create_demo_session() -> dict:
    return {"session_id": sessions.create(None, demo=True), "demo": True}


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str) -> dict:
    return {"deleted": sessions.delete(session_id)}


@app.post("/api/greet")
async def greet(request: TextTurnRequest) -> dict:
    session = require_session(request.session_id)
    if not session.try_begin_turn():
        raise HTTPException(status_code=409, detail="The assistant is already handling a turn.")
    greeting = "Hello! How are you?"
    try:
        if session.demo:
            audio = None
        else:
            audio = await asyncio.to_thread(synthesize, session.api_key, greeting)
        session.memory.add(request.person_id, "assistant", greeting)
        return {"person_id": request.person_id, "reply": greeting, "audio": audio_payload(audio) if audio else None, "demo": session.demo}
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Speech generation failed.") from exc
    finally:
        session.end_turn()


@app.post("/api/text-turn")
async def text_turn(request: TextTurnRequest) -> dict:
    session = require_session(request.session_id)
    if not session.try_begin_turn():
        raise HTTPException(status_code=409, detail="The assistant is already handling a turn.")
    try:
        history = session.memory.context(request.person_id)
        if session.demo:
            reply = demo_respond(history, request.text)
            audio = None
        else:
            reply = await asyncio.to_thread(respond, session.api_key, history, request.text)
            audio = await asyncio.to_thread(synthesize, session.api_key, reply)
        session.memory.add(request.person_id, "user", request.text)
        session.memory.add(request.person_id, "assistant", reply)
        return {"person_id": request.person_id, "transcript": request.text, "reply": reply, "audio": audio_payload(audio) if audio else None, "demo": session.demo}
    except Exception as exc:
        raise HTTPException(status_code=502, detail="OpenAI could not complete the turn.") from exc
    finally:
        session.end_turn()


@app.post("/api/audio-turn")
async def audio_turn(
    session_id: str = Form(...),
    person_id: str = Form(...),
    audio: UploadFile = File(...),
) -> dict:
    session = require_session(session_id)
    if session.demo:
        raise HTTPException(status_code=400, detail="Demo mode uses browser speech recognition or typed messages.")
    if not session.try_begin_turn():
        raise HTTPException(status_code=409, detail="The assistant is already handling a turn.")
    try:
        raw_audio = await audio.read()
        if not raw_audio or len(raw_audio) > 15 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Audio must be between 1 byte and 15 MB.")
        transcript = await asyncio.to_thread(transcribe, session.api_key, raw_audio, audio.filename or "speech.webm")
        if not transcript:
            raise HTTPException(status_code=422, detail="No speech was detected.")
        history = session.memory.context(person_id)
        reply = await asyncio.to_thread(respond, session.api_key, history, transcript)
        output_audio = await asyncio.to_thread(synthesize, session.api_key, reply)
        session.memory.add(person_id, "user", transcript)
        session.memory.add(person_id, "assistant", reply)
        return {"person_id": person_id, "transcript": transcript, "reply": reply, "audio": audio_payload(output_audio)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail="The voice turn could not be completed.") from exc
    finally:
        session.end_turn()


@app.delete("/api/memory/{session_id}/{person_id}")
async def forget_person(session_id: str, person_id: str) -> dict:
    session = require_session(session_id)
    return {"forgotten": session.memory.forget(person_id)}


@app.websocket("/ws/vision")
async def vision_socket(websocket: WebSocket) -> None:
    session_id = websocket.query_params.get("session_id", "")
    session = sessions.get(session_id)
    if not session:
        await websocket.close(code=1008, reason="Valid session required")
        return
    await websocket.accept()
    tracker = session.tracker
    try:
        while True:
            frame = await websocket.receive_bytes()
            try:
                result = await asyncio.to_thread(vision.detect_jpeg, frame)
                result["detections"] = tracker.update(result["detections"])
                result["count"] = len(result["detections"])
                await websocket.send_json(result)
            except ValueError as exc:
                await websocket.send_json({"error": str(exc)})
            except Exception:
                await websocket.send_json({"error": "Vision processing failed for this frame."})
    except WebSocketDisconnect:
        return


async def _slam_command(send, slam, command: dict) -> None:
    """Handles reset / calibration / view / resend. `send` is an async callable taking a dict."""
    kind = command.get("type")
    if kind == "reset":
        await asyncio.to_thread(slam.reset)
        await send({"type": "reset"})
    elif kind == "calibration":
        calibration = CameraCalibration.from_dict(command)
        await asyncio.to_thread(slam.set_calibration, calibration)
        await send({"type": "calibration", "ok": True})
    elif kind == "view":
        slam.set_view(str(command.get("mode", "features")))
        await send({"type": "view", "mode": slam.view_mode})
    elif kind == "resend":
        slam.resend_map()


@app.websocket("/ws/slam")
async def slam_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id, slam = slam_sessions.create()
    # the browser shows its own <video>; server-rendered previews only on request (?preview=1)
    want_preview = websocket.query_params.get("preview") == "1"
    await websocket.send_json({"type": "ready", "session_id": session_id})
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                try:
                    result = await asyncio.to_thread(slam.process_jpeg, message["bytes"], want_preview)
                    result["type"] = "map"
                    await websocket.send_json(result)
                except ValueError as exc:
                    await websocket.send_json({"type": "error", "error": str(exc)})
                except Exception:
                    await websocket.send_json({"type": "error", "error": "SLAM processing failed for this frame."})
            elif message.get("text") is not None:
                try:
                    await _slam_command(websocket.send_json, slam, json.loads(message["text"]))
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    await websocket.send_json({"type": "error", "error": f"Invalid SLAM command: {exc}"})
    except WebSocketDisconnect:
        return


@app.websocket("/ws/slam/webcam")
async def direct_webcam_slam_socket(websocket: WebSocket) -> None:
    """Python-side webcam. Two protocols:
    * {"type": "capture"} -> one processed frame + server-rendered preview (request/response)
    * {"type": "stream"}  -> continuous: binary JPEG frames at camera rate + JSON SLAM results
      as they finish (SLAM on its own thread, so the video never waits for SLAM)."""
    await websocket.accept()
    camera = None
    runner = None
    pump = None
    send_lock = asyncio.Lock()

    async def send_json(obj: dict) -> None:
        async with send_lock:
            await websocket.send_json(obj)

    async def send_bytes(data: bytes) -> None:
        async with send_lock:
            await websocket.send_bytes(data)

    async def stream_loop() -> None:
        last_frame, last_result, last_error = -1, -1, None
        while True:
            seq, frame = camera.latest()
            if frame is not None and seq != last_frame:
                last_frame = seq
                jpg = await asyncio.to_thread(encode_jpeg, frame)
                if jpg:
                    await send_bytes(jpg)
            rseq, result = runner.latest()
            if result is not None and rseq != last_result:
                last_result = rseq
                await send_json({**result, "type": "map"})
            if runner.error and runner.error != last_error:
                await send_json({"type": "error", "error": runner.error})
            last_error = runner.error
            await asyncio.sleep(0.008)

    try:
        source = websocket.query_params.get("camera", "0")
        if source == "synthetic":
            camera = SyntheticCapture()
        else:
            index = int(source)
            if index not in range(10):
                raise ValueError("Webcam index must be between 0 and 9.")
            camera = WebcamCapture(index)
        await asyncio.to_thread(camera.open)
        session_id, slam = slam_sessions.create()
        await websocket.send_json({"type": "ready", "session_id": session_id, "source": "direct"})
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("text") is None:
                continue
            try:
                command = json.loads(message["text"])
                if command.get("type") == "stream":
                    if runner is None:
                        runner = SlamRunner(camera, slam)
                        pump = asyncio.create_task(stream_loop())
                elif command.get("type") == "capture":
                    frame = await asyncio.to_thread(camera.read)
                    result = await asyncio.to_thread(slam.process_frame, frame, True)
                    result["type"] = "map"
                    await send_json(result)
                else:
                    await _slam_command(send_json, slam, command)
            except (ValueError, TypeError) as exc:
                await send_json({"type": "error", "error": str(exc)})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "fatal": True, "error": str(exc)})
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        if pump is not None:
            pump.cancel()
        if runner is not None:
            await asyncio.to_thread(runner.stop)
        if camera is not None:
            await asyncio.to_thread(camera.close)
