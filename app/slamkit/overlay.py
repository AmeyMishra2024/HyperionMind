"""Draws the reconstructed environment on top of the live camera frame (AR style).

The camera image is never modified in place: every layer is drawn on a copy, and translucent
layers are alpha-blended, so the original feed stays visible underneath. Press `o` to see the
raw feed with no overlay at all.
"""
from __future__ import annotations

import cv2
import numpy as np

from .core import INITIALIZING, LOST, NOT_INITIALIZED, TRACKING

STATE_COLORS = {TRACKING: (80, 220, 80), INITIALIZING: (0, 200, 255), NOT_INITIALIZED: (200, 200, 200),
                LOST: (60, 60, 255)}
PLANE_COLORS = [(255, 200, 40), (60, 170, 255), (200, 90, 255)]   # BGR: cyan-ish, orange, magenta
_JET = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)

DEFAULT_FLAGS = {"overlay": True, "grid": True, "surfaces": True, "points": True, "features": True,
                 "trajectory": True, "keyframes": True, "minimap": True, "hud": True}


class OverlayRenderer:
    def __init__(self, K, dist, size, label="CAM"):
        self.K = np.asarray(K, np.float64)
        self.dist = np.zeros(5) if dist is None else np.asarray(dist, np.float64)
        self.size = tuple(size)
        self.label = label
        self.flags = dict(DEFAULT_FLAGS)
        self.near = 0.02

    # ---------------------------------------------------------------- utils
    def _to_cam(self, R, t, X):
        return np.atleast_2d(X) @ R.T + t

    def _proj_cam(self, Xc):
        """Camera-frame points -> distorted pixels. Returns (uv, valid mask)."""
        Xc = np.atleast_2d(Xc)
        z = Xc[:, 2]
        valid = z > self.near
        with np.errstate(divide="ignore", invalid="ignore"):
            nx, ny = Xc[:, 0] / z, Xc[:, 1] / z
        valid &= (np.abs(nx) < 1.2) & (np.abs(ny) < 1.0)
        uv = np.full((len(Xc), 2), -1e4)
        if valid.any():
            p, _ = cv2.projectPoints(Xc[valid].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.dist)
            uv[valid] = p.reshape(-1, 2)
        return uv, valid

    def _polyline3d(self, img, R, t, pts3d, color, thick=1, segments=24):
        self._polylines3d(img, R, t, [pts3d], color, thick, segments)

    def _polylines3d(self, img, R, t, lines, color, thick=1, segments=24):
        """Draws 3D polylines, subdivided so they clip at the near plane and bend with lens distortion.
        All lines are projected in one batch."""
        dense, owner = [], []
        for k, pts3d in enumerate(lines):
            pts3d = np.asarray(pts3d, np.float64)
            if len(pts3d) < 2:
                continue
            s = np.linspace(0, 1, segments, endpoint=False)[None, :, None]
            a, b = pts3d[:-1, None, :], pts3d[1:, None, :]
            d = np.vstack([(a + s * (b - a)).reshape(-1, 3), pts3d[-1:]])
            dense.append(d)
            owner.append(np.full(len(d), k))
        if not dense:
            return
        dense, owner = np.vstack(dense), np.concatenate(owner)
        uv, ok = self._proj_cam(self._to_cam(R, t, dense))
        # break runs where a point is invalid or the owning line changes
        brk = ~ok | np.r_[True, owner[1:] != owner[:-1]]
        run_id = np.cumsum(brk)
        polys = []
        for rid in np.unique(run_id[ok]):
            sel = (run_id == rid) & ok
            if sel.sum() > 1:
                polys.append(np.round(uv[sel] * 16).astype(np.int32))
        if polys:
            cv2.polylines(img, polys, False, color, thick, cv2.LINE_AA, shift=4)

    # --------------------------------------------------------------- layers
    def _draw_planes(self, img, slam, R, t):
        if self.flags["surfaces"]:
            layer = img.copy()
            for k, pl in enumerate(slam.planes):
                if len(pl.hull2d) < 3:
                    continue
                Xc = self._to_cam(R, t, pl.to_world(pl.hull2d))
                if np.all(Xc[:, 2] > self.near):
                    uv, ok = self._proj_cam(Xc)
                    if ok.all():
                        cv2.fillPoly(layer, [np.round(uv).astype(np.int32)], PLANE_COLORS[k % 3], cv2.LINE_AA)
            cv2.addWeighted(layer, 0.25, img, 0.75, 0, dst=img)     # translucent: the feed stays visible
        for k, pl in enumerate(slam.planes):
            color = PLANE_COLORS[k % len(PLANE_COLORS)]
            if self.flags["grid"] and k == 0:           # grid on the dominant surface only
                u0, u1, v0, v1 = pl.extent
                s = pl.spacing
                lines = [pl.to_world(np.array([[uu, v0], [uu, v1]])) for uu in np.arange(u0, u1 + 1e-9, s)]
                lines += [pl.to_world(np.array([[u0, vv], [u1, vv]])) for vv in np.arange(v0, v1 + 1e-9, s)]
                self._polylines3d(img, R, t, lines, (20, 20, 20), 2)       # dark halo for contrast
                self._polylines3d(img, R, t, lines, color, 1)
            if self.flags["surfaces"] and len(pl.hull2d) >= 3:
                h = np.vstack([pl.hull2d, pl.hull2d[:1]])
                self._polyline3d(img, R, t, pl.to_world(h), color, 2, segments=8)
        if slam.planes and self.flags["grid"]:
            pl = slam.planes[0]
            L = pl.spacing * 2
            o = pl.p0
            for axis, col in ((pl.u, (60, 60, 255)), (pl.v, (60, 255, 60)), (pl.n, (255, 120, 60))):
                self._polyline3d(img, R, t, np.array([o, o + L * axis]), col, 3, segments=6)

    def _draw_points(self, img, slam, R, t):
        ids, P, _ = slam.map.arrays()
        if len(P) == 0:
            return
        Xc = self._to_cam(R, t, P)
        uv, ok = self._proj_cam(Xc)
        w, h = self.size
        ok &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        if not ok.any():
            return
        z = Xc[ok, 2]
        zn = np.clip((np.log(z) - np.log(max(slam.median_depth, 1e-6) * 0.3)) / np.log(8.0), 0, 1)
        cols = _JET[(255 * (1 - zn)).astype(int)]         # near = red, far = blue
        xy = np.round(uv[ok]).astype(int)
        h, w = img.shape[:2]
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1)):   # 2-3 px dots, vectorized
            x = np.clip(xy[:, 0] + dx, 0, w - 1)
            y = np.clip(xy[:, 1] + dy, 0, h - 1)
            img[y, x] = cols

    def _draw_features(self, img, slam):
        f = slam.current
        if f is None:
            return
        tracked = f.point_ids >= 0
        xy = np.round(f.pts_raw[~tracked]).astype(int)
        h, w = img.shape[:2]
        img[np.clip(xy[:, 1], 0, h - 1), np.clip(xy[:, 0], 0, w - 1)] = (160, 160, 160)
        for x, y in np.round(f.pts_raw[tracked]).astype(int):
            cv2.circle(img, (x, y), 3, (80, 230, 80), 1)

    def _draw_trajectory(self, img, slam, R, t):
        if len(slam.trajectory) > 1:
            traj = np.array(slam.trajectory[-600:])
            self._polyline3d(img, R, t, traj, (0, 220, 255), 2, segments=2)

    def _draw_keyframes(self, img, slam, R, t):
        s = 0.08 * max(slam.median_depth, 1e-6)
        w, h = self.size
        corners_px = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float64)
        rays = corners_px @ np.linalg.inv(self.K).T * s
        lines = []
        for kf in slam.map.keyframes[-40:]:
            C = kf.center
            cw = rays @ kf.R + C             # camera->world: R^T x + C
            lines += [np.array([C, c]) for c in cw] + [np.vstack([cw, cw[:1]])]
        self._polylines3d(img, R, t, lines, (255, 255, 255), 1, segments=4)

    def _draw_init(self, img, slam):
        tracks = slam.init_tracks
        if tracks is not None:
            for a, b in zip(np.round(tracks[0]).astype(int), np.round(tracks[1]).astype(int)):
                cv2.line(img, tuple(a), tuple(b), (0, 200, 255), 1, cv2.LINE_AA)
                cv2.circle(img, tuple(b), 2, (0, 255, 255), -1)
        msg = "Slide the camera sideways slowly to initialize the map"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        x = (self.size[0] - tw) // 2
        y = self.size[1] - 20
        cv2.rectangle(img, (x - 8, y - th - 8), (x + tw + 8, y + 8), (0, 0, 0), -1)
        cv2.putText(img, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

    def _draw_minimap(self, img, slam):
        S = min(200, self.size[0] // 3, self.size[1] // 2)
        x0, y0 = self.size[0] - S - 10, self.size[1] - S - 10
        roi = img[y0:y0 + S, x0:x0 + S]
        dark = np.zeros_like(roi)
        cv2.addWeighted(dark, 0.6, roi, 0.4, 0, dst=roi)
        _, P, _ = slam.map.arrays()
        if slam.planes:
            pl = slam.planes[0]
            o, a, b = pl.p0, pl.u, pl.v
            label = "top-down (on fitted plane)"
        else:
            o, a, b = np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
            label = "top-down (x-z)"
        traj = np.array(slam.trajectory[-2000:]) if slam.trajectory else np.zeros((0, 3))

        def flat(X):
            X = np.atleast_2d(X)
            return np.stack([(X - o) @ a, (X - o) @ b], axis=1) if len(X) else np.zeros((0, 2))

        p2, t2 = flat(P), flat(traj)
        allp = np.vstack([p2, t2]) if len(p2) or len(t2) else np.zeros((1, 2))
        lo = np.percentile(allp, 3, axis=0) if len(allp) > 10 else allp.min(0)
        hi = np.percentile(allp, 97, axis=0) if len(allp) > 10 else allp.max(0)
        ctr = (lo + hi) / 2
        span = max(float(np.max(hi - lo)), 1e-3) * 1.15
        sc = (S - 16) / span

        def pix(q):
            return np.round((q - ctr) * [sc, -sc] + S / 2).astype(int)

        ids, _, _ = slam.map.arrays()
        cols = [slam.map.points[i].color for i in ids] if len(ids) else []
        for (x, y), c in zip(pix(p2), cols):
            if 0 <= x < S and 0 <= y < S:
                roi[y, x] = c
        if len(t2) > 1:
            cv2.polylines(roi, [pix(t2)], False, (0, 220, 255), 1, cv2.LINE_AA)
        for kf in slam.map.keyframes:
            x, y = pix(flat(kf.center))[0]
            if 0 <= x < S and 0 <= y < S:
                cv2.circle(roi, (int(x), int(y)), 2, (255, 255, 255), -1)
        if slam.R is not None:
            C = slam.camera_center
            fwd = slam.R[2]
            cxy = pix(flat(C))[0]
            d = flat(C + fwd * 1e-3)[0] - flat(C)[0]
            n = np.linalg.norm(d)
            if n > 1e-9:
                d = d / n * np.array([1, -1])
                side = np.array([-d[1], d[0]])
                tri = np.array([cxy + d * 10, cxy - d * 5 + side * 6, cxy - d * 5 - side * 6]).astype(np.int32)
                cv2.fillPoly(roi, [tri], STATE_COLORS.get(slam.state, (255, 255, 255)), cv2.LINE_AA)
        cv2.rectangle(roi, (0, 0), (S - 1, S - 1), (180, 180, 180), 1)
        cv2.putText(roi, label, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (220, 220, 220), 1, cv2.LINE_AA)

    def _draw_hud(self, img, slam, fps_cam, fps_slam, extra=""):
        st = slam.state
        lines = [f"{self.label}  {st}",
                 f"cam {fps_cam:4.1f} fps | slam {fps_slam:4.1f} fps | {slam.stats['proc_ms']:.0f} ms",
                 f"map pts {len(slam.map.points)} | kfs {len(slam.map.keyframes)} | "
                 f"tracked {slam.stats['tracked']}/{slam.stats['features']}",
                 f"surfaces {len(slam.planes)}" + (f" | resets {slam.stats['resets']}" if slam.stats['resets'] else "")]
        if extra:
            lines.append(extra)
        if st == LOST:
            lines.append("LOST - return to a mapped area (r = reset)")
        h = 18 * len(lines) + 10
        roi = img[5:5 + h, 5:5 + 360]
        cv2.addWeighted(np.zeros_like(roi), 0.55, roi, 0.45, 0, dst=roi)
        for i, line in enumerate(lines):
            col = STATE_COLORS.get(st, (255, 255, 255)) if i == 0 else (235, 235, 235)
            cv2.putText(img, line, (12, 22 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45 if i else 0.55, col, 1,
                        cv2.LINE_AA)

    # --------------------------------------------------------------- public
    def render(self, frame, slam, fps_cam=0.0, fps_slam=0.0, extra=""):
        img = frame.copy()
        fl = self.flags
        if not fl["overlay"]:
            cv2.putText(img, f"{self.label} RAW", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)
            return img
        if slam.state in (INITIALIZING, NOT_INITIALIZED):
            self._draw_init(img, slam)
        R, t = (slam.R, slam.t) if slam.R is not None else (None, None)
        if R is not None and slam.state in (TRACKING, LOST):
            if fl["grid"] or fl["surfaces"]:
                self._draw_planes(img, slam, R, t)
            if fl["keyframes"]:
                self._draw_keyframes(img, slam, R, t)
            if fl["trajectory"]:
                self._draw_trajectory(img, slam, R, t)
            if fl["points"]:
                self._draw_points(img, slam, R, t)
        if fl["features"] and slam.state == TRACKING:
            self._draw_features(img, slam)
        if fl["minimap"] and slam.map.keyframes:
            self._draw_minimap(img, slam)
        if fl["hud"]:
            self._draw_hud(img, slam, fps_cam, fps_slam, extra)
        return img
