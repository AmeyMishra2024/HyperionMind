"""Tests for the upgraded Map Lab SLAM engine (synthetic camera, no hardware needed)."""
import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.slam import MonocularSlam
from app.slamkit.config import default_intrinsics
from app.slamkit.synthetic import SyntheticRoom


def _run(frames: int):
    K, _ = default_intrinsics(640, 480)
    room = SyntheticRoom(K)
    slam = MonocularSlam()
    results = [slam.process_frame(room.frame(i)[0], include_preview=(i % 20 == 0)) for i in range(frames)]
    return slam, results


def test_synthetic_room_initializes_tracks_and_builds_costmap() -> None:
    slam, results = _run(120)
    last = results[-1]
    assert last["state"] == "TRACKING"
    assert last["landmarks"] > 300
    assert last["keyframe_count"] >= 3
    assert sum(r["state"] == "TRACKING" for r in results) > 80
    assert "camera" in last and len(last["camera"]["rotation"]) == 9
    assert "costmap" in last and last["costmap"]["width"] > 10
    grid = cv2.imdecode(np.frombuffer(slam.costmap.png, np.uint8), cv2.IMREAD_UNCHANGED)
    assert (grid == 254).any() and (grid == 0).any() and (grid == 255).any()


def test_map_payload_is_only_resent_when_the_map_changes() -> None:
    slam, results = _run(60)
    with_points = [r for r in results if "points" in r]
    assert 0 < len(with_points) < len(results)
    assert all("keyframes" in r and "graph" in r for r in with_points)


def test_preview_modes_render_jpeg() -> None:
    K, _ = default_intrinsics(640, 480)
    frame = SyntheticRoom(K).frame(0)[0]
    slam = MonocularSlam()
    for mode in ("features", "ar", "raw"):
        slam.set_view(mode)
        preview = slam.process_frame(frame, include_preview=True)["preview"]
        assert cv2.imdecode(np.frombuffer(base64.b64decode(preview), np.uint8), cv2.IMREAD_COLOR).shape == (480, 640, 3)


def test_simulated_camera_socket_and_view_command() -> None:
    with TestClient(app).websocket_connect("/ws/slam/webcam?camera=synthetic") as socket:
        assert socket.receive_json()["type"] == "ready"
        socket.send_json({"type": "view", "mode": "ar"})
        assert socket.receive_json() == {"type": "view", "mode": "ar"}
        for _ in range(3):
            socket.send_json({"type": "capture"})
            result = socket.receive_json()
            assert result["type"] == "map" and result["preview"]


def test_camera_list_endpoint() -> None:
    response = TestClient(app).get("/api/slam/cameras")
    assert response.status_code == 200
    assert "cameras" in response.json()


def test_stream_mode_sends_video_frames_and_slam_results() -> None:
    import json

    with TestClient(app).websocket_connect("/ws/slam/webcam?camera=synthetic") as socket:
        assert socket.receive_json()["type"] == "ready"
        socket.send_json({"type": "stream"})
        got_frame = got_map = False
        for _ in range(60):
            message = socket.receive()
            if message.get("bytes"):
                assert message["bytes"][:2] == b"\xff\xd8"          # JPEG
                got_frame = True
            elif message.get("text"):
                data = json.loads(message["text"])
                if data.get("type") == "map":
                    assert "features2d" in data and "intrinsics" in data
                    got_map = True
            if got_frame and got_map:
                break
        assert got_frame and got_map
