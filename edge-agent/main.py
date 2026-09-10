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


def state_event_type(previous, current):
    if current == "PRINTING" and previous != "PRINTING":
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
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        post_event(base_url, api_key, {
            "type": "camera.offline",
            "camera_id": camera["id"],
            "device_id": camera.get("device_id"),
            "ts": utcnow(),
            "severity": "critical",
            "message": "Unable to open video stream",
        })
        raise RuntimeError(f"Unable to open camera: {camera['id']}")

    post_event(base_url, api_key, {
        "type": "device.online",
        "camera_id": camera["id"],
        "device_id": camera.get("device_id"),
        "ts": utcnow(),
        "severity": "info",
        "message": "Edge Agent connected to camera",
    })

    vision_cfg = camera.get("vision", {})
    engine = VisionEngine(
        motion_threshold=float(vision_cfg.get("motion_threshold", 6.0)),
        printing_window=int(vision_cfg.get("printing_window", 8)),
        idle_window=int(vision_cfg.get("idle_window", 12)),
        frozen_frame_seconds=int(vision_cfg.get("frozen_frame_seconds", 20)),
    )

    analyze_fps = max(float(camera.get("analyze_fps", 1)), 0.1)
    interval = 1.0 / analyze_fps
    last_analyze = 0.0
    last_state = "UNKNOWN"
    last_state_emit = 0.0
    last_anomaly_reason = None
    heartbeat_seconds = float(camera.get("heartbeat_seconds", 60))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                post_event(base_url, api_key, {
                    "type": "camera.offline",
                    "camera_id": camera["id"],
                    "device_id": camera.get("device_id"),
                    "ts": utcnow(),
                    "severity": "warning",
                    "message": "Video frame read failed",
                })
                time.sleep(2)
                continue

            now = time.time()
            if now - last_analyze < interval:
                continue
            last_analyze = now

            result = engine.analyze(frame)
            changed = result.state != last_state
            heartbeat_due = now - last_state_emit >= heartbeat_seconds

            if result.anomaly and result.anomaly_reason != last_anomaly_reason:
                post_event(base_url, api_key, {
                    "type": "anomaly.detected",
                    "camera_id": camera["id"],
                    "device_id": camera.get("device_id"),
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
                last_anomaly_reason = result.anomaly_reason
                last_state_emit = now
            elif changed or heartbeat_due:
                event_type = state_event_type(last_state, result.state)
                post_event(base_url, api_key, {
                    "type": event_type,
                    "camera_id": camera["id"],
                    "device_id": camera.get("device_id"),
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
                last_state_emit = now

            if not result.anomaly:
                last_anomaly_reason = None
            last_state = result.state
    finally:
        cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jarvis Workshop Edge Agent")
    parser.add_argument("--config", default="../config/devices.yaml")
    args = parser.parse_args()
    run(args.config)
