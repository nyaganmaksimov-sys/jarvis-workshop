import argparse
import time
from datetime import datetime, timezone

import cv2
import requests
import yaml

from vision_engine import VisionEngine


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def post_event(base_url, api_key, payload):
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(
        f"{base_url.rstrip('/')}/api/v1/events",
        json=payload,
        headers=headers,
        timeout=10,
    )
    response.raise_for_status()


def safe_post_event(base_url, api_key, payload):
    try:
        post_event(base_url, api_key, payload)
        return True
    except requests.RequestException as exc:
        print(f"[{utcnow()}] Event delivery failed: {type(exc).__name__}: {exc}", flush=True)
        return False


def state_event_type(previous, current):
    if current == "PRINTING" and previous in {"IDLE", "DONE", "PAUSED"}:
        return "job.started"
    if current == "IDLE" and previous == "PRINTING":
        return "job.completed"
    if current == "IDLE":
        return "device.idle"
    if current == "PRINTING":
        return "device.printing"
    if current == "ERROR":
        return "anomaly.detected"
    return "device.unknown"


def build_engine(vision_cfg):
    return VisionEngine(
        motion_threshold=float(vision_cfg.get("motion_threshold", 6.0)),
        printing_window=int(vision_cfg.get("printing_window", 8)),
        idle_window=int(vision_cfg.get("idle_window", 12)),
        frozen_frame_seconds=int(vision_cfg.get("frozen_frame_seconds", 20)),
    )


def open_camera(source):
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap
    cap.release()
    return None


def run(config_path):
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    server = config["server"]
    base_url = server["base_url"]
    api_key = server.get("api_key", "")

    cameras = config.get("cameras", [])
    if not cameras:
        raise RuntimeError("No cameras configured")

    camera = cameras[0]
    source = camera["url"]
    camera_id = camera["id"]
    device_id = camera.get("device_id")
    vision_cfg = camera.get("vision", {})

    analyze_fps = max(float(camera.get("analyze_fps", 1)), 0.1)
    interval = 1.0 / analyze_fps
    heartbeat_seconds = max(float(camera.get("heartbeat_seconds", 60)), 5.0)
    reconnect_seconds = max(float(camera.get("reconnect_seconds", 5)), 1.0)
    read_failure_limit = max(int(camera.get("read_failure_limit", 3)), 1)
    offline_event_seconds = max(float(camera.get("offline_event_seconds", 60)), 10.0)

    cap = None
    engine = None
    last_analyze = 0.0
    last_state = "UNKNOWN"
    last_state_emit = 0.0
    last_anomaly_reason = None
    last_offline_emit = 0.0
    read_failures = 0

    def emit_offline(message, severity="warning"):
        nonlocal last_offline_emit
        now = time.time()
        if now - last_offline_emit < offline_event_seconds:
            return
        safe_post_event(base_url, api_key, {
            "type": "camera.offline",
            "camera_id": camera_id,
            "device_id": device_id,
            "ts": utcnow(),
            "severity": severity,
            "message": message,
        })
        last_offline_emit = now

    try:
        while True:
            if cap is None:
                print(f"[{utcnow()}] Connecting camera {camera_id}...", flush=True)
                cap = open_camera(source)
                if cap is None:
                    emit_offline("Unable to open video stream", "critical")
                    time.sleep(reconnect_seconds)
                    continue

                print(f"[{utcnow()}] Camera {camera_id} connected", flush=True)
                engine = build_engine(vision_cfg)
                read_failures = 0
                last_analyze = 0.0
                last_state = "UNKNOWN"
                last_state_emit = 0.0
                last_anomaly_reason = None
                last_offline_emit = 0.0
                safe_post_event(base_url, api_key, {
                    "type": "device.online",
                    "camera_id": camera_id,
                    "device_id": device_id,
                    "ts": utcnow(),
                    "severity": "info",
                    "message": "Edge Agent connected to camera",
                })

            ok, frame = cap.read()
            if not ok:
                read_failures += 1
                if read_failures < read_failure_limit:
                    time.sleep(0.25)
                    continue

                print(
                    f"[{utcnow()}] Camera {camera_id} stream lost after {read_failures} read failures",
                    flush=True,
                )
                emit_offline("Video stream lost; reconnecting")
                cap.release()
                cap = None
                engine = None
                read_failures = 0
                time.sleep(reconnect_seconds)
                continue

            read_failures = 0
            now = time.time()
            if now - last_analyze < interval:
                continue
            last_analyze = now

            result = engine.analyze(frame)
            changed = result.state != last_state
            heartbeat_due = now - last_state_emit >= heartbeat_seconds

            if result.anomaly and result.anomaly_reason != last_anomaly_reason:
                delivered = safe_post_event(base_url, api_key, {
                    "type": "anomaly.detected",
                    "camera_id": camera_id,
                    "device_id": device_id,
                    "ts": utcnow(),
                    "severity": "critical",
                    "message": f"Vision anomaly: {result.anomaly_reason}",
                    "data": {
                        "state": result.state,
                        "confidence": result.confidence,
                        "motion_score": result.motion_score,
                        "brightness": result.brightness,
                        "sharpness": result.sharpness,
                        **result.metrics,
                    },
                })
                if delivered:
                    last_anomaly_reason = result.anomaly_reason
                    last_state_emit = now
            elif changed or heartbeat_due:
                event_type = state_event_type(last_state, result.state)
                delivered = safe_post_event(base_url, api_key, {
                    "type": event_type,
                    "camera_id": camera_id,
                    "device_id": device_id,
                    "ts": utcnow(),
                    "severity": "warning" if result.state == "ERROR" else "info",
                    "message": f"Vision state: {result.state}",
                    "data": {
                        "state": result.state,
                        "confidence": result.confidence,
                        "motion_score": result.motion_score,
                        "brightness": result.brightness,
                        "sharpness": result.sharpness,
                        **result.metrics,
                    },
                })
                if delivered:
                    last_state_emit = now

            if not result.anomaly:
                last_anomaly_reason = None
            last_state = result.state
    except KeyboardInterrupt:
        print(f"[{utcnow()}] Edge Agent stopped by operator", flush=True)
    finally:
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jarvis Workshop Edge Agent")
    parser.add_argument("--config", default="../config/devices.yaml")
    args = parser.parse_args()
    run(args.config)
