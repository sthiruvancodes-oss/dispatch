"""HTTP API for Dispatch: run browser tasks, stream frames, handle Twilio calls."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from twilio.request_validator import RequestValidator

from agent_runner import AgentRunner
from voice_agent import VoiceAgentSession, call_registry

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dispatch")

app = FastAPI(title="Dispatch")
runner = AgentRunner()

_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if origin.strip()
]
# No IP or *.trycloudflare.com wildcard: with allow_credentials those made every
# quick tunnel on the internet a permitted credentialed origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TaskRequest(BaseModel):
    instruction: str = Field(min_length=1)


def _allowed_callers() -> list[str]:
    """Numbers permitted to call in. Empty means reject everyone.

    Deny-by-default is deliberate: an unset allowlist must never mean that any
    caller can drive a browser that holds your logins.
    """
    raw = os.getenv("ALLOWED_CALLERS", "")
    return [number.strip() for number in raw.split(",") if number.strip()]


def _is_allowed_caller(number: str) -> bool:
    return bool(number) and number.strip() in _allowed_callers()


def _signature_candidates(request: Request) -> list[str]:
    """URLs Twilio might have signed. It signs the URL exactly as configured,
    so a trailing slash or a proxy-rewritten host changes the signature."""
    path = request.url.path
    query = request.url.query
    bases = []
    env_base = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    if env_base:
        bases.append(env_base)
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    if host:
        proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        bases.append(f"{proto}://{host}")
        bases.append(f"https://{host}")
    seen, urls = set(), []
    for base in bases:
        for suffix in (path, path + "/"):
            for full in ((f"{base}{suffix}?{query}",) if query else ()) + (f"{base}{suffix}",):
                if full not in seen:
                    seen.add(full)
                    urls.append(full)
    return urls


def _signature_ok(request: Request, form: dict[str, str]) -> bool:
    """Verify X-Twilio-Signature so only Twilio can reach the voice endpoints."""
    token = (os.getenv("TWILIO_AUTH_TOKEN") or "").strip()
    if not token:
        logger.warning("TWILIO_AUTH_TOKEN unset; refusing unverified voice request")
        return False
    signature = request.headers.get("X-Twilio-Signature", "")
    validator = RequestValidator(token)
    candidates = _signature_candidates(request)
    for url in candidates:
        if validator.validate(url, form, signature):
            if url != candidates[0]:
                logger.warning(
                    "Twilio signed %s, not %s — set PUBLIC_BASE_URL to match "
                    "the webhook URL exactly.", url, candidates[0]
                )
            return True
    logger.warning(
        "Signature mismatch. signature_present=%s tried=%s form_keys=%s",
        bool(signature), candidates, sorted(form),
    )
    return False


def _public_base(request: Request) -> str:
    env = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    if env:
        return env
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    return f"{proto}://{host}"


def _twiml(body: str) -> Response:
    return Response(
        content=f'<?xml version="1.0" encoding="UTF-8"?>\n<Response>{body}</Response>',
        media_type="application/xml",
    )


def _task_payload(session) -> dict:
    return {
        "task_id": session.task_id,
        "instruction": session.instruction,
        "status": session.status,
        "source": session.source,
        "result": session.result,
    }


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.post("/tasks")
async def create_task(body: TaskRequest):
    try:
        session = await runner.start_task(
            body.instruction,
            source="web",
        )
    except RuntimeError as exc:
        running = runner.current_running()
        return JSONResponse(
            status_code=409,
            content={
                "detail": str(exc),
                "task_id": running.task_id if running else None,
            },
        )
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    return {"task_id": session.task_id}


@app.post("/tasks/stop")
async def stop_task():
    try:
        session = await runner.stop_current()
    except RuntimeError as exc:
        return JSONResponse(status_code=409, content={"detail": str(exc)})
    return {"task_id": session.task_id, "status": "stopping"}


@app.post("/tasks/reset")
async def reset_demo():
    await runner.reset()
    return {"ok": True, "latest_task_id": None}


@app.get("/tasks")
async def list_tasks():
    latest = runner.tasks.get(runner.latest_task_id) if runner.latest_task_id else None
    return {
        "latest_task_id": runner.latest_task_id,
        "current": _task_payload(latest) if latest else None,
        "tasks": [_task_payload(s) for s in runner.tasks.values()],
    }


@app.websocket("/ws/{task_id}")
async def task_ws(websocket: WebSocket, task_id: str):
    session = runner.tasks.get(task_id)
    if session is None:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    queue = session.subscribe()
    try:
        for text in session.logs:
            await websocket.send_json({"type": "log", "text": text})
        if session.latest_frame:
            await websocket.send_json({"type": "frame", "data": session.latest_frame})
        if session.status in ("done", "error"):
            await websocket.send_json({"type": "done", "result": session.result or ""})
            return

        while True:
            message = await queue.get()
            await websocket.send_json(message)
            if message.get("type") == "done":
                break
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for %s", task_id)
    finally:
        session.unsubscribe(queue)


@app.post("/voice")
async def voice_incoming(request: Request):
    """Inbound call: verify it's really Twilio, verify the caller, then hand the
    audio to the Deepgram voice agent over a bidirectional Media Stream."""
    form = dict(await request.form())

    if not _signature_ok(request, form):
        logger.warning("Rejected /voice with a bad Twilio signature")
        return Response(status_code=403)

    caller = str(form.get("From") or "")
    if not _is_allowed_caller(caller):
        logger.warning("Rejected call from %s (not in ALLOWED_CALLERS)", caller)
        return _twiml("<Say>Sorry, this number isn't available.</Say><Hangup/>")

    call_sid = str(form.get("CallSid") or "")
    call_registry.authorize(call_sid)
    logger.info("Accepted call %s from %s", call_sid, caller)

    stream = f"{_public_base(request).replace('https://', 'wss://')}/voice/stream"
    # <Connect>, not <Start>: audio has to flow back to the caller.
    return _twiml(f'<Connect><Stream url="{stream}" /></Connect>')


@app.websocket("/voice/stream")
async def voice_stream(websocket: WebSocket):
    """Twilio Media Stream. Authorization is checked against the CallSid that
    POST /voice registered, since the WebSocket upgrade carries no signature."""
    await websocket.accept()
    session = VoiceAgentSession(websocket, runner)
    try:
        await session.run()
    except WebSocketDisconnect:
        logger.info("Caller hung up")
        await session.aclose()
    except Exception:
        logger.exception("Voice session failed")
        await session.aclose()
