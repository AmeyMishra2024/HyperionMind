"""Decoupled live pipeline for direct (Python-side) webcams.

The old loop was strictly serial: browser asks -> read frame -> SLAM -> render preview -> reply,
so the video could never be smoother than SLAM. Now:

  * the camera thread grabs frames at the camera's own rate (WebcamCapture)
  * SlamRunner runs SLAM on its own thread on the newest frame, as fast as it can
  * the websocket pump streams every new camera frame as a binary JPEG (~30 fps) and every new SLAM
    result as JSON; overlays (feature boxes, AR) are drawn by the browser on top of the live video
"""
from __future__ import annotations

import threading
import time

import cv2


class SlamRunner:
    def __init__(self, camera, slam):
        self.camera = camera
        self.slam = slam
        self._lock = threading.Lock()
        self._result = None
        self._seq = 0
        self.error: str | None = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="slam-runner", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            try:
                frame = self.camera.read()          # newest frame (waits for a fresh one)
                result = self.slam.process_frame(frame)
            except ValueError as exc:              # camera unplugged etc.
                self.error = str(exc)
                time.sleep(0.2)
                continue
            except Exception as exc:  # pragma: no cover - keep the stream alive
                self.error = f"SLAM processing failed: {exc}"
                time.sleep(0.05)
                continue
            self.error = None
            with self._lock:
                self._result = result
                self._seq += 1

    def latest(self):
        with self._lock:
            return self._seq, self._result

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2)


def encode_jpeg(frame, quality: int = 75) -> bytes | None:
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpg.tobytes() if ok else None
