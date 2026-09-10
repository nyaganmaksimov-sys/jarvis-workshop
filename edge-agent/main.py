import argparse
import time
from datetime import datetime, timezone

import cv2
import requests
import yaml


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def post_event(base_url, api_key, payload):
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(f"{base_url.rstrip('/')}/api/v1/events", json=payload, headers=headers, timeout=10)
    response.raise_for_status()


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
            "message": "Unable to open video stream"
        })
        raise RuntimeError(f"Unable to open camera: {camera['id']}")

    post_event(base_url, api_key, {
        "type": "device.online",
        "camera_id": camera["id"],
        "device_id": camera.get("device_id"),
        "ts": utcnow(),
        "severity": "info",
        "message": "Edge Agent connected to camera"
    })

    analyze_fps = max(float(camera.get("analyze_fps", 1)), 0.1)
    interval = 1.0 / analyze_fps
    last_emit = 0.0

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
                    "message": "Video frame read failed"
                })
                time.sleep(2)
                continue

            now = time.time()
            if now - last_emit < interval:
                continue
            last_emit = now

            height, width = frame.shape[:2]
            post_event(base_url, api_key, {
                "type": "observation.frame",
                "camera_id": camera["id"],
                "device_id": camera.get("device_id"),
                "ts": utcnow(),
                "severity": "debug",
                "message": "Frame captured",
                "data": {"width": width, "height": height}
            })
    finally:
        cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jarvis Workshop Edge Agent")
    parser.add_argument("--config", default="../config/devices.yaml")
    args = parser.parse_args()
    run(args.config)
