from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a Logitech C270 using a printed checkerboard.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument("--columns", type=int, default=9, help="Checkerboard inner corners across")
    parser.add_argument("--rows", type=int, default=6, help="Checkerboard inner corners down")
    parser.add_argument("--square-mm", type=float, default=25.0, help="Printed checker square size in mm")
    parser.add_argument("--samples", type=int, default=18, help="Number of accepted views")
    parser.add_argument("--output", type=Path, default=Path("c270-calibration.json"))
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}.")

    pattern = (args.columns, args.rows)
    object_template = np.zeros((args.columns * args.rows, 3), np.float32)
    object_template[:, :2] = np.mgrid[0 : args.columns, 0 : args.rows].T.reshape(-1, 2)
    object_template *= args.square_mm
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    last_center: np.ndarray | None = None
    image_size = (640, 480)

    print("Show the checkerboard at different positions and angles.")
    print("SPACE accepts a detected view, R removes the last view, Q quits.")
    try:
        while len(image_points) < args.samples:
            ok, frame = camera.read()
            if not ok:
                continue
            image_size = (frame.shape[1], frame.shape[0])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern)
            refined = None
            if found:
                refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )
                cv2.drawChessboardCorners(frame, pattern, refined, found)
            cv2.putText(frame, f"Accepted: {len(image_points)}/{args.samples}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 240, 170), 2)
            cv2.putText(frame, "SPACE accept | R undo | Q quit", (16, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
            cv2.imshow("C270 calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r") and image_points:
                image_points.pop(); object_points.pop()
            if key == 32 and refined is not None:
                center = refined.mean(axis=0).ravel()
                if last_center is None or np.linalg.norm(center - last_center) > 15:
                    image_points.append(refined.copy())
                    object_points.append(object_template.copy())
                    last_center = center
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if len(image_points) < 8:
        raise SystemExit("At least 8 varied checkerboard views are required.")
    error, matrix, distortion, _, _ = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    data = {
        "camera": "Logitech C270",
        "width": image_size[0], "height": image_size[1],
        "fx": float(matrix[0, 0]), "fy": float(matrix[1, 1]),
        "cx": float(matrix[0, 2]), "cy": float(matrix[1, 2]),
        "distortion": [float(value) for value in distortion.ravel()[:5]],
        "rms_reprojection_error": float(error),
        "checkerboard": {"columns": args.columns, "rows": args.rows, "square_mm": args.square_mm},
    }
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output.resolve()} (RMS error: {error:.3f} px)")


if __name__ == "__main__":
    main()
