from collections import deque
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Jarvis Workshop API", version="0.1.0")

EVENTS = deque(maxlen=2000)
DEVICE_STATE: dict[str, dict[str, Any]] = {}
API_KEY = "change-me"


class Event(BaseModel):
    type: str
    ts: str | None = None
    severity: Literal["debug", "info", "warning", "critical"] = "info"
    camera_id: str | None = None
    device_id: str | None = None
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


def authorize(authorization: str | None):
    if API_KEY == "":
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health():
    return {"ok": True, "service": "jarvis-workshop", "time": datetime.now(timezone.utc).isoformat()}


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

    return {"ok": True}


@app.get("/api/v1/events")
def list_events(limit: int = 100):
    limit = max(1, min(limit, 500))
    return list(EVENTS)[:limit]


@app.get("/api/v1/devices")
def list_devices():
    return list(DEVICE_STATE.values())


@app.get("/api/v1/devices/{device_id}")
def get_device(device_id: str):
    item = DEVICE_STATE.get(device_id)
    if not item:
        raise HTTPException(status_code=404, detail="Device not found")
    return item
