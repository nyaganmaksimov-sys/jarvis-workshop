from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time

import cv2
import numpy as np


@dataclass
class VisionResult:
    state: str
    confidence: float
    motion_score: float
    brightness: float
    sharpness: float
    anomaly: bool = False
    anomaly_reason: str | None = None
    metrics: dict = field(default_factory=dict)


class VisionEngine:
    """Conservative vision heuristics for workshop monitoring v1.

    This engine does not claim to understand the printer perfectly. It observes
    motion and image quality over time and produces a cautious state estimate.
    Future model-based detectors can be plugged in behind the same interface.
    """

    def __init__(
        self,
        motion_threshold: float = 6.0,
        printing_window: int = 8,
        idle_window: int = 12,
        frozen_frame_seconds: int = 20,
    ) -> None:
        self.motion_threshold = motion_threshold
        self.motion_history: deque[float] = deque(maxlen=max(printing_window, idle_window))
        self.prev_gray: np.ndarray | None = None
        self.prev_hash: int | None = None
        self.same_hash_since: float | None = None
        self.printing_window = printing_window
        self.idle_window = idle_window
        self.frozen_frame_seconds = frozen_frame_seconds

    @staticmethod
    def _frame_hash(gray: np.ndarray) -> int:
        small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        return hash(small.tobytes())

    def analyze(self, frame: np.ndarray) -> VisionResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        motion = 0.0
        if self.prev_gray is not None:
            prev = cv2.resize(self.prev_gray, (320, 180), interpolation=cv2.INTER_AREA)
            curr = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
            motion = float(cv2.absdiff(prev, curr).mean())
        self.prev_gray = gray
        self.motion_history.append(motion)

        now = time.time()
        current_hash = self._frame_hash(gray)
        if current_hash == self.prev_hash:
            if self.same_hash_since is None:
                self.same_hash_since = now
        else:
            self.prev_hash = current_hash
            self.same_hash_since = now

        frozen_for = 0.0 if self.same_hash_since is None else now - self.same_hash_since

        anomaly = False
        anomaly_reason = None
        if brightness < 8:
            anomaly = True
            anomaly_reason = "camera_too_dark"
        elif sharpness < 8:
            anomaly = True
            anomaly_reason = "camera_blurred"
        elif frozen_for >= self.frozen_frame_seconds:
            anomaly = True
            anomaly_reason = "camera_frame_frozen"

        recent = list(self.motion_history)
        printing_samples = recent[-self.printing_window :]
        idle_samples = recent[-self.idle_window :]
        printing_ratio = (
            sum(1 for x in printing_samples if x >= self.motion_threshold) / len(printing_samples)
            if printing_samples
            else 0.0
        )
        idle_ratio = (
            sum(1 for x in idle_samples if x < self.motion_threshold * 0.55) / len(idle_samples)
            if idle_samples
            else 0.0
        )

        if anomaly:
            state, confidence = "ERROR", 0.95
        elif len(printing_samples) >= self.printing_window and printing_ratio >= 0.62:
            state, confidence = "PRINTING", min(0.95, 0.6 + printing_ratio * 0.35)
        elif len(idle_samples) >= self.idle_window and idle_ratio >= 0.8:
            state, confidence = "IDLE", min(0.9, 0.55 + idle_ratio * 0.35)
        else:
            state, confidence = "UNKNOWN", 0.4

        return VisionResult(
            state=state,
            confidence=confidence,
            motion_score=motion,
            brightness=brightness,
            sharpness=sharpness,
            anomaly=anomaly,
            anomaly_reason=anomaly_reason,
            metrics={
                "printing_ratio": printing_ratio,
                "idle_ratio": idle_ratio,
                "frozen_for_s": frozen_for,
            },
        )
