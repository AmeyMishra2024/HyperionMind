"""SLAM parameters and Logitech C270 camera defaults."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Logitech C270: 1280x960 sensor, fixed focus, ~55 deg diagonal FOV. 640x480 is binned (full FOV).
C270_APPROX_FX_640 = 790.0


@dataclass
class SlamConfig:
    n_features: int = 1500
    grid_cols: int = 8
    grid_rows: int = 6
    # initialization
    init_min_matches: int = 100
    init_min_disp_px: float = 25.0
    init_min_inliers: int = 60
    init_min_points: int = 50
    init_min_parallax_deg: float = 2.0
    # tracking
    match_th: int = 64
    search_radius: float = 18.0
    refine_radius: float = 8.0
    min_tracked: int = 25
    pnp_reproj_px: float = 4.0
    # keyframes / mapping
    kf_min_frames: int = 5
    kf_track_ratio: float = 0.75
    kf_min_tracked: int = 150
    kf_baseline_ratio: float = 0.12
    tri_neighbors: int = 3
    min_parallax_deg: float = 1.0
    # bundle adjustment (needs SciPy)
    use_ba: bool = True
    async_ba: bool = True
    ba_window: int = 5
    ba_iters: int = 15
    # recovery
    auto_reset_frames: int = 150
    # environment
    max_planes: int = 3


def default_intrinsics(width: int, height: int):
    fx = C270_APPROX_FX_640 * width / 640.0
    K = np.array([[fx, 0, width / 2.0], [0, fx, height / 2.0], [0, 0, 1]], dtype=np.float64)
    return K, np.zeros(5)
