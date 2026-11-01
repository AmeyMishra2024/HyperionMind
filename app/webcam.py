from __future__ import annotations

import threading
import time

import cv2
import numpy as np


def list_camera_names() -> list[str] | None:
    """DirectShow device names in index order (needs the optional `pygrabber` package on Windows)."""
    try:
        import comtypes  # type: ignore
        from pygrabber.dshow_graph import FilterGraph  # type: ignore

        comtypes.CoInitialize()  # worker threads need COM initialised before DirectShow calls
        try:
            return list(FilterGraph().get_input_devices())
        finally:
            comtypes.CoUninitialize()
    except Exception:
        return None


class WebcamCapture:
    """Windows USB camera capture with backend fallback and deterministic cleanup.

    Frames are grabbed on a background thread so `read()` always returns the newest frame:
    the SLAM loop never works on stale, buffered images. MJPG is requested so several
    C270s can share one USB 2.0 controller at 640x480 @ 30 fps.
    """

    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.capture = None
        self.index = index
        self.size = (width, height)
        self.fps = fps
        self._frame = None
        self._seq = 0
        self._last_read = 0
        self._error: str | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._cond = threading.Condition()

    def open(self) -> None:
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            capture = cv2.VideoCapture(self.index, backend)
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            capture.set(cv2.CAP_PROP_FPS, self.fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = capture.read()
            if capture.isOpened() and ok and frame is not None:
                self.capture = capture
                self._store(frame)
                self._running = True
                self._thread = threading.Thread(target=self._loop, name=f"webcam-{self.index}", daemon=True)
                self._thread.start()
                return
            capture.release()
        raise ValueError(
            f"Cannot open webcam {self.index}. Close Camera, Teams, Zoom and other camera apps, "
            "check Windows camera privacy settings, or select another webcam index."
        )

    def _store(self, frame) -> None:
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._cond.notify_all()

    def _loop(self) -> None:
        failures = 0
        while self._running:
            capture = self.capture
            if capture is None:
                break
            ok, frame = capture.read()
            if ok and frame is not None:
                failures = 0
                self._store(frame)
            else:
                failures += 1
                if failures > 50:
                    self._error = "Webcam stopped sending frames. Check its USB connection and restart mapping."
                    with self._cond:
                        self._cond.notify_all()
                    break
                time.sleep(0.01)

    def read(self) -> np.ndarray:
        if self.capture is None:
            raise ValueError("Webcam is not open.")
        with self._cond:
            if self._seq <= self._last_read and self._error is None:
                self._cond.wait(timeout=1.0)
            if self._error:
                raise ValueError(self._error)
            if self._frame is None:
                raise ValueError("Webcam stopped sending frames. Check its USB connection and restart mapping.")
            frame, self._last_read = self._frame, self._seq
        height, width = frame.shape[:2]
        if width > 640:
            frame = cv2.resize(frame, (640, round(height * 640 / width)))
        return frame

    def latest(self):
        """(sequence number, newest frame) without consuming it; used for the live preview stream."""
        with self._cond:
            return self._seq, self._frame

    def close(self) -> None:
        self._running = False
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class SyntheticCapture:
    """Simulated camera moving through a textured room: try the mapper without hardware."""

    def __init__(self, fps: float = 30.0):
        from app.slamkit.config import default_intrinsics
        from app.slamkit.synthetic import SyntheticRoom

        K, _ = default_intrinsics(640, 480)
        self.room = SyntheticRoom(K, (640, 480), fps)
        self.fps = fps
        self.index = "synthetic"
        self._i = 0
        self._t_next = None
        self._last = None
        self._lock = threading.Lock()

    def open(self) -> None:
        self._t_next = time.perf_counter()

    def read(self) -> np.ndarray:
        now = time.perf_counter()
        if self._t_next is not None and now < self._t_next:
            time.sleep(self._t_next - now)
        self._t_next = max(now, self._t_next or now) + 1.0 / self.fps
        frame, _ = self.room.frame(self._i)
        with self._lock:
            self._i += 1
            self._last = frame
        return frame

    def latest(self):
        with self._lock:
            return self._i, self._last

    def close(self) -> None:
        pass
