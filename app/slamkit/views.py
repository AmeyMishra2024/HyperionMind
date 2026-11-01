"""Camera-frame renderings: ORB-SLAM3 style feature view (green boxes + status bar)."""
from __future__ import annotations

import cv2
import numpy as np

from .core import INITIALIZING, LOST, NOT_INITIALIZED, TRACKING

GREEN = (0, 255, 0)


def draw_orbslam_frame(frame: np.ndarray, slam) -> np.ndarray:
    """Mimics ORB-SLAM3's FrameDrawer: matched map points as green boxes with a centre dot,
    initialization matches as green lines, and the one-line status bar at the bottom."""
    img = frame.copy()
    state = slam.state
    if state in (NOT_INITIALIZED, INITIALIZING) and slam.init_tracks is not None:
        a, b = slam.init_tracks
        for p, q in zip(np.round(a).astype(int), np.round(b).astype(int)):
            cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), GREEN, 1)
    elif state == TRACKING and slam.current is not None:
        f = slam.current
        r = 5
        for x, y in np.round(f.pts_raw[f.point_ids >= 0]).astype(int):
            cv2.rectangle(img, (int(x) - r, int(y) - r), (int(x) + r, int(y) + r), GREEN, 1)
            cv2.circle(img, (int(x), int(y)), 2, GREEN, -1)
    if state == NOT_INITIALIZED:
        text = "WAITING FOR IMAGES"
    elif state == INITIALIZING:
        text = "TRYING TO INITIALIZE"
    elif state == LOST:
        text = "TRACK LOST. TRYING TO RELOCALIZE"
    else:
        text = (f"SLAM MODE |  KFs: {len(slam.map.keyframes)}, MPs: {len(slam.map.points)}, "
                f"Matches: {slam.stats.get('tracked', 0)}")
    h, w = img.shape[:2]
    bar = 24
    img[h - bar:h] = (img[h - bar:h] * 0.15).astype(np.uint8)
    cv2.putText(img, text, (6, h - 7), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
    return img
