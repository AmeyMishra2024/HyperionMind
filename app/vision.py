from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np

# Keep third-party settings inside this project instead of writing to a user profile.
RUNTIME_DIR = Path(__file__).resolve().parents[1] / ".runtime"
try:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME_DIR))
except OSError:
    # Read-only deployments can let Ultralytics choose its platform default.
    pass


class VisionEngine:
    """OpenCV frame pipeline with YOLO pose inference and a HOG fallback."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._model_error: str | None = None
        self._lock = Lock()
        self._hog: cv2.HOGDescriptor | None = None
        self._warming = False

    @property
    def mode(self) -> str:
        if self._model is not None:
            return "pose"
        if self._warming:
            return "warming"
        if self._model_error is not None:
            return "opencv-hog"
        return "loading"

    def warm_up(self) -> None:
        """Load weights and initialize inference before the first webcam frame."""
        if self._model is not None or self._model_error is not None:
            return
        self._warming = True
        try:
            self._ensure_model()
            if self._model is not None:
                # Ultralytics may defer provider/kernel initialization until the
                # first prediction; a small blank frame pays that cost at boot.
                blank = np.zeros((384, 640, 3), dtype=np.uint8)
                with self._lock:
                    self._model.predict(blank, conf=0.28, iou=0.45, imgsz=640, verbose=False)
        except Exception as exc:
            # Keep normal detection available: it will fall back to HOG if the
            # pose model truly cannot be initialized on this machine.
            self._model = None
            self._model_error = f"Pose model unavailable; raised-hand detection is disabled ({exc})."
        finally:
            self._warming = False

    def detect_jpeg(self, payload: bytes) -> dict:
        encoded = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("The browser sent an invalid image frame.")
        if frame.shape[1] > 768:
            scale = 768 / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        self._ensure_model()
        detections = self._detect_pose(frame) if self._model is not None else self._detect_hog(frame)
        return {
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "mode": self.mode,
            "warning": self._model_error,
            "detections": detections,
        }

    def _ensure_model(self) -> None:
        if self._model is not None or self._model_error is not None:
            return
        with self._lock:
            if self._model is not None or self._model_error is not None:
                return
            try:
                from ultralytics import YOLO

                self._model = YOLO(os.getenv("YOLO_POSE_MODEL", "yolo11n-pose.pt"))
            except Exception as exc:  # A useful fallback for offline first runs.
                self._model_error = f"Pose model unavailable; raised-hand detection is disabled ({exc})."

    def _detect_pose(self, frame: np.ndarray) -> list[dict]:
        with self._lock:
            # A slightly lower confidence floor improves recall for people at
            # the edge of the room; tracker confirmation prevents flicker.
            result = self._model.predict(frame, conf=0.28, iou=0.45, imgsz=640, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
        keypoints = result.keypoints.data.cpu().numpy() if result.keypoints is not None else []
        detections: list[dict] = []
        for index, box_values in enumerate(boxes):
            x1, y1, x2, y2 = [int(value) for value in box_values]
            raised = False
            raised_side = None
            if index < len(keypoints):
                points = keypoints[index]
                # COCO pose indices: left/right shoulder 5/6, wrist 9/10.
                for side, shoulder_index, wrist_index in (("left", 5, 9), ("right", 6, 10)):
                    shoulder, wrist = points[shoulder_index], points[wrist_index]
                    margin = max(12, int((y2 - y1) * 0.06))
                    if shoulder[2] >= 0.35 and wrist[2] >= 0.35 and wrist[1] < shoulder[1] - margin:
                        raised, raised_side = True, side
                        break
            detections.append(
                {
                    "box": [x1, y1, x2, y2],
                    "confidence": round(float(confidences[index]), 3),
                    "hand_raised": raised,
                    "raised_side": raised_side,
                    "signature": self._appearance_signature(frame, (x1, y1, x2, y2)),
                }
            )
        return detections

    def _detect_hog(self, frame: np.ndarray) -> list[dict]:
        if self._hog is None:
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, weights = self._hog.detectMultiScale(
            frame, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        return [
            {
                "box": [int(x), int(y), int(x + width), int(y + height)],
                "confidence": round(float(weights[index]), 3),
                "hand_raised": False,
                "raised_side": None,
                "signature": self._appearance_signature(
                    frame, (int(x), int(y), int(x + width), int(y + height))
                ),
            }
            for index, (x, y, width, height) in enumerate(boxes)
        ]

    @staticmethod
    def _appearance_signature(frame: np.ndarray, box: tuple[int, int, int, int]) -> list[float] | None:
        """Describe the central upper and lower torso for anonymous re-identification."""
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box
        person_width, person_height = max(1, x2 - x1), max(1, y2 - y1)
        left = max(0, min(width, x1 + int(person_width * 0.18)))
        right = max(0, min(width, x2 - int(person_width * 0.18)))
        top = max(0, min(height, y1 + int(person_height * 0.18)))
        middle = max(0, min(height, y1 + int(person_height * 0.48)))
        bottom = max(0, min(height, y1 + int(person_height * 0.76)))
        if right - left < 4 or bottom - top < 8:
            return None

        parts: list[np.ndarray] = []
        for band_top, band_bottom in ((top, middle), (middle, bottom)):
            crop = frame[band_top:band_bottom, left:right]
            if crop.size == 0:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hue_saturation = cv2.calcHist([hsv], [0, 1], None, [16, 6], [0, 180, 0, 256]).flatten()
            brightness = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
            color_total, brightness_total = float(hue_saturation.sum()), float(brightness.sum())
            if color_total and brightness_total:
                # Including brightness makes similarly coloured but differently
                # toned clothes easier to separate in ordinary indoor lighting.
                parts.append(np.concatenate((0.75 * hue_saturation / color_total, 0.25 * brightness / brightness_total)))
        if not parts:
            return None
        signature = np.concatenate(parts)
        signature /= max(float(signature.sum()), 1e-9)
        return [round(float(value), 5) for value in signature]
