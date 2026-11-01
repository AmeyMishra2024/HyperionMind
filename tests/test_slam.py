from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.slam import CameraCalibration, MonocularSlam


def test_calibration_scales_to_frame_size() -> None:
    calibration = CameraCalibration(width=640, height=480, fx=500, fy=510, cx=320, cy=240)
    matrix = calibration.matrix_for(1280, 960)
    np.testing.assert_allclose(matrix, [[1000, 0, 640], [0, 1020, 480], [0, 0, 1]])


def test_blank_first_frame_requests_texture() -> None:
    slam = MonocularSlam()
    result = slam.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    assert result["landmarks"] == 0
    assert "textured" in result["status"].lower()


def test_process_jpeg_rejects_invalid_data() -> None:
    slam = MonocularSlam()
    try:
        slam.process_jpeg(b"not a jpeg")
    except ValueError as exc:
        assert "valid JPEG" in str(exc)
    else:
        raise AssertionError("invalid JPEG should fail")


def test_ply_export_contains_colored_vertices(tmp_path: Path) -> None:
    slam = MonocularSlam()
    slam.map_points = [np.array([1.0, 2.0, 3.0]), np.array([-1.0, 0.5, 4.0])]
    slam.map_colors = [(255, 10, 20), (5, 100, 200)]
    output = slam.export_ply(tmp_path / "map.ply")
    text = output.read_text(encoding="ascii")
    assert "element vertex 2" in text
    assert "1 2 3 255 10 20" in text


def test_synthetic_feature_pair_bootstraps_map() -> None:
    rng = np.random.default_rng(4)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(350):
        x, y = rng.integers(25, 615), rng.integers(25, 455)
        color = tuple(int(v) for v in rng.integers(80, 255, size=3))
        cv2.circle(image, (int(x), int(y)), 3, color, -1)
    shifted = np.zeros_like(image)
    shifted[:, :-10] = image[:, 10:]
    slam = MonocularSlam()
    slam.process_frame(image)
    result = slam.process_frame(shifted)
    assert result["matches"] >= 10


def test_slam_page_and_websocket_are_available() -> None:
    client = TestClient(app)
    response = client.get("/slam")
    assert response.status_code == 200
    assert "C270 Map Lab" in response.text
    with client.websocket_connect("/ws/slam") as websocket:
        ready = websocket.receive_json()
        assert ready["type"] == "ready"
        assert len(ready["session_id"]) == 32
