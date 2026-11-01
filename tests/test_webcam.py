from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.webcam import WebcamCapture


def test_camera_backend_fallback_and_cleanup():
    failed = MagicMock()
    failed.read.return_value = (False, None)
    failed.isOpened.return_value = False
    working = MagicMock()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    working.read.return_value = (True, frame)
    working.isOpened.return_value = True
    with patch("app.webcam.cv2.VideoCapture", side_effect=[failed, working]):
        camera = WebcamCapture()
        camera.open()
        assert camera.read().shape == (480, 640, 3)
        failed.release.assert_called_once()
        camera.close()
        working.release.assert_called_once()


def test_direct_webcam_socket_returns_preview_and_releases_camera():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch("app.main.WebcamCapture") as webcam:
        webcam.return_value.read.return_value = frame
        with TestClient(app).websocket_connect("/ws/slam/webcam?camera=0") as socket:
            assert socket.receive_json()["source"] == "direct"
            socket.send_json({"type": "capture"})
            result = socket.receive_json()
            assert result["type"] == "map"
            assert result["preview"]
            assert result["frame"] == 1
        webcam.return_value.close.assert_called_once()
