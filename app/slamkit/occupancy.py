"""Nav2-style 2D occupancy grid + inflated costmap built from the monocular SLAM map.

Ground reference: if SLAM found a floor-like surface (normal close to the first camera's "up"),
that plane is the ground; otherwise the first camera's up axis is used.

  * obstacle cells  = map points standing above the floor (below the ceiling band)
  * free cells      = ray-traced from every keyframe centre to each point it observed,
                      plus floor points themselves
  * cost values     = ROS costmap_2d convention:
                        0 free, 1..252 inflated cost, 253 inscribed, 254 lethal, 255 unknown

The grid is returned as an 8-bit single channel PNG so the browser can colour it and plan on it.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

FREE, INSCRIBED, LETHAL, UNKNOWN = 0, 253, 254, 255


class _SnapKF:
    __slots__ = ("center", "R", "point_ids")

    def __init__(self, center, R, point_ids):
        self.center, self.R, self.point_ids = center, R, point_ids


class _SnapMap:
    def __init__(self, keyframes, ids, P):
        self.keyframes = keyframes
        self._arrays = (ids, P, None)

    def arrays(self):
        return self._arrays


class MapSnapshot:
    """Immutable copy of what the costmap needs, so it can be built on another thread."""

    def __init__(self, slam):
        ids, P, _ = slam.map.arrays()
        kfs = [_SnapKF(kf.center.copy(), kf.R.copy(), kf.point_ids.copy()) for kf in slam.map.keyframes]
        self.map = _SnapMap(kfs, ids.copy(), P.copy())
        self.planes = list(slam.planes)
        self.median_depth = float(slam.median_depth)


class OccupancyMapper:
    def __init__(self, inscribed_cells=2, inflation_cells=9, max_cells=480, cost_scaling=0.35):
        self.inscribed = inscribed_cells
        self.inflation = inflation_cells
        self.max_cells = max_cells
        self.cost_scaling = cost_scaling
        self.reset()

    def reset(self):
        self.version = 0
        self.res = None
        self.grid = None
        self.png = None
        self.meta = None

    # ---------------------------------------------------------------- frame
    def _basis(self, slam):
        kf0 = slam.map.keyframes[0]
        up0 = -kf0.R[1]                       # camera y points down -> world "up" of the first view
        floor = None
        for pl in slam.planes:
            if abs(float(np.dot(pl.n, up0))) > math.cos(math.radians(35)):
                floor = pl
                break
        up = up0 if floor is None else (floor.n if np.dot(floor.n, up0) > 0 else -floor.n)
        right = kf0.R[0] - np.dot(kf0.R[0], up) * up
        right /= np.linalg.norm(right) + 1e-12
        fwd = np.cross(up, right)
        origin = floor.p0 if floor is not None else kf0.center
        return origin, up, right, fwd, floor is not None

    def to_grid(self, X):
        """World points -> (col, row) float pixel coordinates in the current grid."""
        m = self.meta
        if m is None:
            return None
        d = np.atleast_2d(X) - m["origin"]
        u, v = d @ m["right"], d @ m["fwd"]
        return np.stack([(u - m["umin"]) / m["res"], (m["vmax"] - v) / m["res"]], axis=1)

    def heading(self, R):
        """Camera forward axis -> angle in grid pixel space (0 = +col, clockwise positive)."""
        m = self.meta
        f = R[2]
        return math.atan2(-float(f @ m["fwd"]), float(f @ m["right"]))

    @staticmethod
    def _floor_from_histogram(h, cam_h, md):
        below = h[h < cam_h - 0.3 * md]
        if len(below) < max(30, 0.08 * len(h)):
            return None
        bw = 0.04 * md
        edges = np.arange(below.min(), below.max() + bw, bw)
        if len(edges) < 2:
            return None
        counts, edges = np.histogram(below, edges)
        k = int(np.argmax(counts))
        if counts[k] < max(25, 0.08 * len(h)):
            return None
        return float((edges[k] + edges[k + 1]) / 2)

    # --------------------------------------------------------------- update
    def update(self, slam) -> bool:
        kfs = slam.map.keyframes
        ids, P, _ = slam.map.arrays()
        if len(kfs) < 2 or len(P) < 20:
            return False
        origin, up, right, fwd, has_floor = self._basis(slam)
        md = max(slam.median_depth, 1e-6)
        if self.res is None:
            self.res = 0.02 * md
        d = P - origin
        h = d @ up
        u, v = d @ right, d @ fwd
        cams = np.array([kf.center for kf in kfs])
        cu, cv_ = (cams - origin) @ right, (cams - origin) @ fwd
        if has_floor:
            tol = 0.06 * md
            cam_h = float(np.median((cams - origin) @ up))
            is_floor = np.abs(h) < tol
            is_obst = (h >= tol) & (h < max(cam_h, 0) + 0.8 * md)
        else:
            # no fitted floor plane: look for a dense low layer of points below the cameras
            cam_h = float(np.median((cams - origin) @ up))
            floor_h = self._floor_from_histogram(h, cam_h, md)
            if floor_h is not None:
                tol = 0.06 * md
                is_floor = np.abs(h - floor_h) < tol
                is_obst = (h >= floor_h + tol) & (h < cam_h + 0.8 * md)
                self._floor_h = floor_h
            else:
                is_floor = np.zeros(len(P), bool)
                is_obst = np.ones(len(P), bool)
        keep = is_floor | is_obst
        if keep.sum() < 10:
            return False
        uu = np.concatenate([u[keep], cu])
        vv = np.concatenate([v[keep], cv_])
        lo_u, hi_u = np.percentile(uu, 1), np.percentile(uu, 99)
        lo_v, hi_v = np.percentile(vv, 1), np.percentile(vv, 99)
        lo_u, hi_u = min(lo_u, cu.min()), max(hi_u, cu.max())
        lo_v, hi_v = min(lo_v, cv_.min()), max(hi_v, cv_.max())
        span = max(hi_u - lo_u, hi_v - lo_v)
        res = max(self.res, span / (self.max_cells - 2 * (self.inflation + 4)))
        margin = (self.inflation + 4) * res
        umin, vmax = lo_u - margin, hi_v + margin
        W = int(math.ceil((hi_u - lo_u + 2 * margin) / res)) + 1
        H = int(math.ceil((hi_v - lo_v + 2 * margin) / res)) + 1
        meta = {"origin": origin, "up": up, "right": right, "fwd": fwd, "umin": umin, "vmax": vmax,
                "res": res, "width": W, "height": H, "floor": has_floor}

        def px(uv_u, uv_v):
            return np.stack([(uv_u - umin) / res, (vmax - uv_v) / res], axis=1)

        pts_px = px(u, v)
        cam_px = px(cu, cv_)
        inside = (pts_px[:, 0] >= 0) & (pts_px[:, 0] < W) & (pts_px[:, 1] >= 0) & (pts_px[:, 1] < H)
        # free space: rays keyframe -> observed point
        free = np.zeros((H, W), np.uint16)
        id_to_row = {int(i): r for r, i in enumerate(ids)}
        for k, kf in enumerate(kfs):
            rows = [id_to_row.get(int(p)) for p in kf.point_ids[kf.point_ids >= 0]]
            rows = np.array([r for r in rows if r is not None and keep[r] and inside[r]], int)
            if len(rows) == 0:
                continue
            c = np.round(cam_px[k]).astype(np.int32)
            segs = [np.array([c, e], np.int32) for e in np.round(pts_px[rows]).astype(np.int32)]
            mask = np.zeros((H, W), np.uint8)
            cv2.polylines(mask, segs, False, 1, 1)
            free += mask
        hits = np.zeros((H, W), np.uint16)
        ob = is_obst & inside
        if ob.any():
            q = np.floor(pts_px[ob]).astype(int)
            np.add.at(hits, (q[:, 1], q[:, 0]), 1)
        fl = is_floor & inside
        if fl.any():
            q = np.floor(pts_px[fl]).astype(int)
            np.add.at(free, (q[:, 1], q[:, 0]), 3)
        occupied = (hits >= 2) | ((hits >= 1) & (free <= 2 * hits))
        occupied = cv2.morphologyEx(occupied.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)) > 0
        known = cv2.dilate(((free > 0) | occupied).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        grid = np.full((H, W), UNKNOWN, np.uint8)
        grid[known] = FREE
        if occupied.any():
            dist = cv2.distanceTransform((~occupied).astype(np.uint8), cv2.DIST_L2, 5)
            band = (dist > self.inscribed) & (dist <= self.inflation)
            cost = 252.0 * np.exp(-self.cost_scaling * (dist - self.inscribed))
            grid[band] = np.maximum(np.clip(cost[band], 1, 252).astype(np.uint8), 1)
            grid[(dist <= self.inscribed) & ~occupied] = INSCRIBED
            grid[occupied] = LETHAL
        ok, png = cv2.imencode(".png", grid)
        # publish together so readers on other threads never see a half-updated map
        self.grid, self.png, self.meta = grid, (png.tobytes() if ok else None), meta
        self.version += 1
        return True
