from collections import deque
from datetime import datetime, timezone
from typing import Any, Literal
import os
import sys
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.orchestrator import JarvisCore
from core.hub_source import HubSource
from core.announcer import AnnouncementQueue
from core.tts import NeuralTTS

app = FastAPI(title="Jarvis Workshop API", version="0.10.0")
jarvis = JarvisCore()
hub = HubSource()
announcer = AnnouncementQueue(hub)
tts = NeuralTTS()

EVENTS = deque(maxlen=2000)
DEVICE_STATE: dict[str, dict[str, Any]] = {}
API_KEY = os.getenv("JARVIS_API_KEY", "change-me")
DEVICE_STALE_SECONDS = max(30, int(os.getenv("JARVIS_DEVICE_STALE_SECONDS", "150")))


class Event(BaseModel):
    type: str
    ts: str | None = None
    severity: Literal["debug", "info", "warning", "critical"] = "info"
    camera_id: str | None = None
    device_id: str | None = None
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class AssistantRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    context: dict[str, Any] = Field(default_factory=dict)


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2500)
    voice: Literal["female", "male"] = "male"


class AckRequest(BaseModel):
    id: str = Field(min_length=1, max_length=500)


def authorize(authorization: str | None):
    if API_KEY == "":
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def heartbeat_age_seconds(last_seen: str | None, now: datetime) -> float | None:
    if not last_seen:
        return None
    try:
        parsed = datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (now - parsed.astimezone(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return None


def device_snapshot(item: dict[str, Any], now: datetime) -> dict[str, Any]:
    current = dict(item)
    reported_status = str(current.get("status") or "UNKNOWN").upper()
    age = heartbeat_age_seconds(current.get("last_seen"), now)
    stale = age is None or age > DEVICE_STALE_SECONDS

    current["reported_status"] = reported_status
    current["heartbeat_age_seconds"] = round(age, 1) if age is not None else None
    current["stale"] = stale
    current["status"] = "OFFLINE" if stale else reported_status
    return current


def workshop_snapshot() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    devices = [device_snapshot(x, now) for x in DEVICE_STATE.values()]
    alerts = [x for x in EVENTS if x.get("severity") in {"warning", "critical"}][:20]
    offline = sum(1 for x in devices if x.get("status") == "OFFLINE")
    printing = sum(1 for x in devices if x.get("status") == "PRINTING")
    warnings = sum(1 for x in alerts if x.get("severity") == "warning")
    critical = sum(1 for x in alerts if x.get("severity") == "critical")

    return {
        "devices": devices,
        "alerts": alerts,
        "summary": {
            "total": len(devices),
            "online": len(devices) - offline,
            "offline": offline,
            "printing": printing,
            "warning_alerts": warnings,
            "critical_alerts": critical,
        },
        "heartbeat_stale_seconds": DEVICE_STALE_SECONDS,
        "captured_at": now.isoformat(),
    }


@app.on_event("startup")
async def startup_event():
    print(f"[Jarvis LLM] {jarvis.brain.status()}", flush=True)
    announcer.start()


@app.on_event("shutdown")
async def shutdown_event():
    await announcer.stop()


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/dashboard/")


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "jarvis-workshop",
        "version": "0.10.0",
        "hub_configured": hub.configured,
        "tts_profiles": tts.profiles(),
        "llm": jarvis.brain.status(),
        "device_count": len(DEVICE_STATE),
        "heartbeat_stale_seconds": DEVICE_STALE_SECONDS,
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/v1/events")
def ingest_event(event: Event, authorization: str | None = Header(default=None)):
    authorize(authorization)
    payload = event.model_dump()
    if not payload["ts"]:
        payload["ts"] = datetime.now(timezone.utc).isoformat()
    EVENTS.appendleft(payload)

    if event.device_id:
        state = DEVICE_STATE.setdefault(event.device_id, {})
        state.update({
            "device_id": event.device_id,
            "last_seen": payload["ts"],
            "last_event": event.type,
            "severity": event.severity,
            "message": event.message,
            "data": event.data,
        })
        if event.type == "device.online":
            state["status"] = "ONLINE"
        elif event.type in {"device.offline", "camera.offline"}:
            state["status"] = "OFFLINE"
        elif event.type == "job.started":
            state["status"] = "PRINTING"
        elif event.type == "job.paused":
            state["status"] = "PAUSED"
        elif event.type == "job.completed":
            state["status"] = "DONE"
        elif event.type == "anomaly.detected":
            state["status"] = "ERROR"
        elif event.type == "vision.state":
            state["status"] = str(event.data.get("status") or state.get("status") or "UNKNOWN")
        elif event.type == "device.idle":
            state["status"] = "IDLE"
        elif event.type == "device.printing":
            state["status"] = "PRINTING"

    return {"ok": True}


@app.get("/api/v1/events")
def list_events(limit: int = 100):
    limit = max(1, min(limit, 500))
    return list(EVENTS)[:limit]


@app.get("/api/v1/devices")
def list_devices():
    now = datetime.now(timezone.utc)
    return [device_snapshot(x, now) for x in DEVICE_STATE.values()]


@app.get("/api/v1/devices/{device_id}")
def get_device(device_id: str):
    item = DEVICE_STATE.get(device_id)
    if not item:
        raise HTTPException(status_code=404, detail="Device not found")
    return device_snapshot(item, datetime.now(timezone.utc))


@app.get("/api/v1/workshop/status")
def get_workshop_status():
    return workshop_snapshot()


@app.get("/api/v1/hub/status")
async def get_hub_status(authorization: str | None = Header(default=None)):
    authorize(authorization)
    return await hub.snapshot()


@app.get("/api/v1/announcements")
def get_announcements(limit: int = 20, authorization: str | None = Header(default=None)):
    authorize(authorization)
    return {"items": announcer.pending(limit)}


@app.post("/api/v1/announcements/ack")
def ack_announcement(request: AckRequest, authorization: str | None = Header(default=None)):
    authorize(authorization)
    return {"ok": announcer.ack(request.id)}


@app.post("/api/v1/assistant/query")
async def assistant_query(request: AssistantRequest, authorization: str | None = Header(default=None)):
    authorize(authorization)
    try:
        hub_state = await hub.snapshot()
    except Exception:
        hub_state = {"configured": False, "error": "HUB_SNAPSHOT_UNAVAILABLE"}
    return await jarvis.handle(request.text, workshop_snapshot(), hub_state, request.context)


@app.get("/api/v1/tts/voices")
def tts_voices(authorization: str | None = Header(default=None)):
    authorize(authorization)
    return {"voices": tts.profiles()}


@app.post("/api/v1/tts")
async def synthesize_tts(request: TTSRequest, authorization: str | None = Header(default=None)):
    authorize(authorization)
    try:
        audio = await tts.synthesize(request.text, request.voice)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"TTS_UNAVAILABLE: {type(exc).__name__}") from exc
    return Response(content=audio, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


DASHBOARD_DIR = ROOT / "dashboard"
if DASHBOARD_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")
