"""Monocular feature-based visual SLAM.

Pipeline (ORB-SLAM style, simplified):
  * ORB features, bucketed over an image grid so they cover the whole frame
  * Two-view initialization (essential matrix + triangulation, median depth = 1 unit)
  * Tracking: constant-velocity prediction -> guided projection matching -> PnP RANSAC
    -> LM refinement; global descriptor matching as fallback / relocalization
  * Keyframes: new points triangulated against neighbour keyframes (epipolar-checked)
  * Local bundle adjustment over a sliding keyframe window (scipy, sparse Jacobian)
  * Environment: dominant planes (floor / walls) fitted with RANSAC for the AR overlay

Poses are world->camera: x_cam = R @ x_world + t. Monocular => arbitrary global scale.
"""
from __future__ import annotations

import itertools
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from .config import SlamConfig

try:  # SciPy powers local bundle adjustment and fast neighbour search; both degrade gracefully.
    from scipy.optimize import least_squares
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:  # pragma: no cover - exercised only on installs without SciPy
    least_squares = csr_matrix = None
    HAS_SCIPY = False

    class cKDTree:  # minimal grid-bucket replacement for query_ball_point
        def __init__(self, pts, cell=16.0):
            self.pts = np.asarray(pts, dtype=np.float64)
            self.cell = cell
            self.buckets = {}
            for i, (x, y) in enumerate(self.pts):
                self.buckets.setdefault((int(x // cell), int(y // cell)), []).append(i)

        def query_ball_point(self, queries, r):
            out = []
            reach = int(np.ceil(r / self.cell))
            for qx, qy in np.asarray(queries, dtype=np.float64):
                cx, cy = int(qx // self.cell), int(qy // self.cell)
                found = []
                for gx in range(cx - reach, cx + reach + 1):
                    for gy in range(cy - reach, cy + reach + 1):
                        for i in self.buckets.get((gx, gy), ()):
                            dx, dy = self.pts[i, 0] - qx, self.pts[i, 1] - qy
                            if dx * dx + dy * dy <= r * r:
                                found.append(i)
                out.append(found)
            return out

LM_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1e-7)
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

NOT_INITIALIZED = "WAITING"
INITIALIZING = "INITIALIZING"
TRACKING = "TRACKING"
LOST = "LOST"


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return POPCOUNT[np.bitwise_xor(a, b)].sum(axis=1, dtype=np.int32)


def rotate(points: np.ndarray, rvecs: np.ndarray) -> np.ndarray:
    """Rodrigues rotation of points[i] by rvecs[i] (vectorized)."""
    theta = np.linalg.norm(rvecs, axis=1)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.nan_to_num(rvecs / theta)
    dot = np.sum(points * v, axis=1)[:, None]
    c, s = np.cos(theta), np.sin(theta)
    return c * points + s * np.cross(v, points) + dot * (1 - c) * v


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def project(K, R, t, X):
    Xc = X @ R.T + t
    z = Xc[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    uv = np.empty((len(X), 2))
    uv[:, 0] = K[0, 0] * Xc[:, 0] / zs + K[0, 2]
    uv[:, 1] = K[1, 1] * Xc[:, 1] / zs + K[1, 2]
    return uv, z


class Frame:
    def __init__(self, idx, ts, image, pts_raw, pts, desc, colors, sigma):
        self.idx = idx
        self.sigma = sigma              # keypoint position std-dev in px (from pyramid level)
        self.ts = ts
        self.image = image
        self.pts_raw = pts_raw          # distorted pixel coords (for drawing)
        self.pts = pts                  # undistorted pixel coords (for geometry)
        self.desc = desc
        self.colors = colors
        self.point_ids = np.full(len(pts), -1, dtype=np.int64)
        self.R = None
        self.t = None
        self._tree = None

    @property
    def tree(self):
        if self._tree is None:
            self._tree = cKDTree(self.pts)
        return self._tree


class Keyframe:
    def __init__(self, kf_id, frame: Frame):
        self.id = kf_id
        self.frame_idx = frame.idx
        self.R = frame.R.copy()
        self.t = frame.t.copy()
        self.pts = frame.pts
        self.pts_raw = frame.pts_raw
        self.desc = frame.desc
        self.colors = frame.colors
        self.sigma = frame.sigma
        self.point_ids = frame.point_ids.copy()

    @property
    def center(self):
        return -self.R.T @ self.t


class MapPoint:
    __slots__ = ("id", "pos", "desc", "obs", "visible", "found", "color", "created_kf")

    def __init__(self, pid, pos, desc, color, created_kf):
        self.id = pid
        self.pos = np.asarray(pos, dtype=np.float64)
        self.desc = desc
        self.obs = {}           # kf_id -> keypoint index
        self.visible = 1
        self.found = 1
        self.color = color
        self.created_kf = created_kf


class Map:
    def __init__(self):
        self.points: dict[int, MapPoint] = {}
        self.keyframes: list[Keyframe] = []
        self.kf_by_id: dict[int, Keyframe] = {}
        self._next_pid = 0
        self._next_kf = 0
        self.version = 0
        self._cache = (None, None)

    def add_keyframe(self, frame) -> Keyframe:
        kf = Keyframe(self._next_kf, frame)
        self._next_kf += 1
        self.keyframes.append(kf)
        self.kf_by_id[kf.id] = kf
        return kf

    def add_point(self, pos, desc, color, kf_id) -> MapPoint:
        mp = MapPoint(self._next_pid, pos, desc, color, kf_id)
        self._next_pid += 1
        self.points[mp.id] = mp
        self.version += 1
        return mp

    def remove_point(self, pid):
        mp = self.points.pop(pid, None)
        if mp is None:
            return
        for kf_id, kpi in mp.obs.items():
            kf = self.kf_by_id.get(kf_id)
            if kf is not None and kf.point_ids[kpi] == pid:
                kf.point_ids[kpi] = -1
        self.version += 1

    def arrays(self):
        """(ids, positions, descriptors) of all map points, cached per map version."""
        ver, data = self._cache
        if ver == self.version and data is not None:
            return data
        if not self.points:
            data = (np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros((0, 32), np.uint8))
        else:
            mps = list(self.points.values())
            data = (np.fromiter((m.id for m in mps), np.int64, len(mps)),
                    np.array([m.pos for m in mps]),
                    np.array([m.desc for m in mps], dtype=np.uint8))
        self._cache = (self.version, data)
        return data


class Plane:
    def __init__(self, p0, n, u, v, hull2d, extent, spacing, n_inliers):
        self.p0, self.n, self.u, self.v = p0, n, u, v
        self.hull2d = hull2d        # (H,2) convex hull in plane coordinates
        self.extent = extent        # (umin, umax, vmin, vmax)
        self.spacing = spacing
        self.n_inliers = n_inliers

    def to_world(self, uv):
        uv = np.atleast_2d(uv)
        return self.p0 + uv[:, :1] * self.u + uv[:, 1:2] * self.v


def _nice_step(x):
    if x <= 0:
        return 1.0
    e = 10 ** np.floor(np.log10(x))
    for m in (1, 2, 5, 10):
        if x <= m * e:
            return m * e
    return 10 * e


class MonoSLAM:
    def __init__(self, K, dist=None, image_size=(640, 480), config: SlamConfig | None = None):
        self.cfg = config or SlamConfig()
        self.K = np.asarray(K, dtype=np.float64)
        self.Kinv = np.linalg.inv(self.K)
        self.dist = None if dist is None or not np.any(dist) else np.asarray(dist, dtype=np.float64)
        self.image_size = tuple(image_size)
        self.orb = cv2.ORB_create(nfeatures=self.cfg.n_features * 2, scaleFactor=1.2, nlevels=8,
                                  fastThreshold=12, edgeThreshold=19, patchSize=31)
        self.orb_relaxed = cv2.ORB_create(nfeatures=self.cfg.n_features * 2, scaleFactor=1.2, nlevels=8,
                                          fastThreshold=5, edgeThreshold=19, patchSize=31)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        if not HAS_SCIPY:
            self.cfg.use_ba = False
        self._ba_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ba")
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.map = Map()
        self.state = NOT_INITIALIZED
        self.frame_idx = 0
        self.init_ref: Frame | None = None
        self.init_tracks = None
        self.R = None
        self.t = None
        self.velocity = None
        self.last_kf: Keyframe | None = None
        self.last_kf_tracked = 0
        self.median_depth = 1.0
        self.lost_count = 0
        self.planes: list[Plane] = []
        self.trajectory: list[np.ndarray] = []
        self.current: Frame | None = None
        self.last_view_idx = np.zeros(0, np.int64)
        self.stats = {"tracked": 0, "matches": 0, "features": 0, "proc_ms": 0.0, "ba_ms": 0.0, "resets": 0}
        self._epoch = getattr(self, "_epoch", 0) + 1     # invalidates in-flight BA results after reset
        self._ba_future = None

    @property
    def has_pose(self):
        return self.state == TRACKING and self.R is not None

    @property
    def camera_center(self):
        return None if self.R is None else -self.R.T @ self.t

    # ------------------------------------------------------------- front-end
    def _detect(self, gray):
        kps = self.orb.detect(gray, None)
        if len(kps) < self.cfg.n_features // 3:          # low texture / dim light
            kps = self.orb_relaxed.detect(gray, None)
        if not kps:
            return [], None
        h, w = gray.shape[:2]
        gc, gr = self.cfg.grid_cols, self.cfg.grid_rows
        cap = max(4, int(2 * self.cfg.n_features / (gc * gr)))
        pts = cv2.KeyPoint_convert(kps)
        resp = np.fromiter((k.response for k in kps), np.float32, len(kps))
        cell = np.minimum((pts[:, 0] * gc / w).astype(int), gc - 1) + gc * np.minimum((pts[:, 1] * gr / h).astype(int), gr - 1)
        order = np.lexsort((-resp, cell))                 # by cell, strongest first
        c_sorted = cell[order]
        first = np.searchsorted(c_sorted, c_sorted, side="left")
        rank = np.arange(len(order)) - first               # rank of each keypoint inside its cell
        kept = order[rank < cap]
        kept = kept[np.argsort(-resp[kept])][: self.cfg.n_features]
        kept_kps, desc = self.orb.compute(gray, [kps[i] for i in kept])
        return kept_kps, desc

    def make_frame(self, image_bgr, ts=None) -> Frame:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
        kps, desc = self._detect(gray)
        if desc is None or len(kps) == 0:
            pts_raw = np.zeros((0, 2), np.float64)
            desc = np.zeros((0, 32), np.uint8)
            octaves = np.zeros(0, int)
        else:
            pts_raw = np.array([k.pt for k in kps], dtype=np.float64)
            octaves = np.array([k.octave & 0xFF for k in kps], dtype=int)
        return self.make_frame_from_features(pts_raw, desc, image_bgr, ts, octaves)

    def make_frame_from_features(self, pts_raw, desc, image=None, ts=None, octaves=None) -> Frame:
        pts_raw = np.asarray(pts_raw, dtype=np.float64).reshape(-1, 2)
        if self.dist is not None and len(pts_raw):
            pts = cv2.undistortPoints(pts_raw.reshape(-1, 1, 2), self.K, self.dist, P=self.K).reshape(-1, 2)
        else:
            pts = pts_raw.copy()
        if image is not None and image.ndim == 3 and len(pts_raw):
            h, w = image.shape[:2]
            xi = np.clip(pts_raw[:, 0].astype(int), 0, w - 1)
            yi = np.clip(pts_raw[:, 1].astype(int), 0, h - 1)
            colors = image[yi, xi].astype(np.uint8)
        else:
            colors = np.full((len(pts_raw), 3), 200, np.uint8)
        octaves = np.zeros(len(pts_raw), int) if octaves is None else np.asarray(octaves)
        sigma = 1.2 ** np.clip(octaves, 0, 7)
        f = Frame(self.frame_idx, time.perf_counter() if ts is None else ts, image, pts_raw, pts,
                  np.ascontiguousarray(desc, dtype=np.uint8), colors, sigma)
        self.frame_idx += 1
        return f

    def process(self, image_bgr, ts=None) -> Frame:
        return self.process_frame(self.make_frame(image_bgr, ts))

    def process_frame(self, f: Frame) -> Frame:
        t0 = time.perf_counter()
        self._poll_ba()
        self.stats["features"] = len(f.pts)
        if self.state in (NOT_INITIALIZED, INITIALIZING):
            self._initialize(f)
        elif self.state == TRACKING:
            if not self._track(f):
                self.state = LOST
                self.velocity = None
                self.lost_count = 1
        elif self.state == LOST:
            if self._track(f, relocalize=True):
                self.state = TRACKING
                self.lost_count = 0
            else:
                self.lost_count += 1
                if self.cfg.auto_reset_frames and self.lost_count > self.cfg.auto_reset_frames:
                    resets = self.stats["resets"] + 1
                    self.reset()
                    self.stats["resets"] = resets
        if self.state == TRACKING:
            self.trajectory.append(self.camera_center.copy())
            if len(self.trajectory) > 5000:
                self.trajectory = self.trajectory[-5000:]
        self.current = f
        self.stats["proc_ms"] = (time.perf_counter() - t0) * 1000
        return f

    def _knn(self, d1, d2, ratio):
        if len(d1) < 2 or len(d2) < 2:
            return np.zeros(0, int), np.zeros(0, int)
        raw = self.matcher.knnMatch(d1, d2, k=2)
        q, t = [], []
        for m in raw:
            if len(m) == 2 and m[0].distance < ratio * m[1].distance and m[0].distance <= self.cfg.match_th:
                q.append(m[0].queryIdx)
                t.append(m[0].trainIdx)
        q, t = np.array(q, int), np.array(t, int)
        if len(t):
            _, first = np.unique(t, return_index=True)   # one-to-one
            q, t = q[first], t[first]
        return q, t

    def _triangulate(self, R1, t1, R2, t2, p1, p2, max_err=2.0, s1=1.0, s2=1.0):
        P1 = self.K @ np.hstack([R1, t1[:, None]])
        P2 = self.K @ np.hstack([R2, t2[:, None]])
        Xh = cv2.triangulatePoints(P1, P2, p1.T.astype(np.float64), p2.T.astype(np.float64))
        w = Xh[3]
        valid = np.abs(w) > 1e-12
        X = (Xh[:3] / np.where(valid, w, 1.0)).T
        uv1, z1 = project(self.K, R1, t1, X)
        uv2, z2 = project(self.K, R2, t2, X)
        e1 = np.linalg.norm(uv1 - p1, axis=1)
        e2 = np.linalg.norm(uv2 - p2, axis=1)
        C1, C2 = -R1.T @ t1, -R2.T @ t2
        r1, r2 = X - C1, X - C2
        cosp = np.sum(r1 * r2, axis=1) / (np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1) + 1e-12)
        good = valid & (z1 > 0) & (z2 > 0) & (e1 < max_err * s1) & (e2 < max_err * s2) & \
            (cosp < np.cos(np.radians(self.cfg.min_parallax_deg)))
        return X, good

    # -------------------------------------------------------- initialization
    def _initialize(self, f: Frame):
        cfg = self.cfg
        if self.init_ref is None:
            if len(f.pts) >= cfg.init_min_matches:
                self.init_ref = f
                self.state = INITIALIZING
            return
        ref = self.init_ref
        qi, ti = self._knn(ref.desc, f.desc, 0.8)
        self.init_tracks = (ref.pts_raw[qi], f.pts_raw[ti])
        self.stats["matches"] = len(qi)
        if len(qi) < cfg.init_min_matches:
            if len(qi) < cfg.init_min_matches // 2 and len(f.pts) >= cfg.init_min_matches:
                self.init_ref = f                      # scene changed too much: restart
            return
        p1, p2 = ref.pts[qi], f.pts[ti]
        if np.median(np.linalg.norm(p2 - p1, axis=1)) < cfg.init_min_disp_px:
            return
        E, mask = cv2.findEssentialMat(p1, p2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or mask is None:
            return
        if E.shape[0] > 3:
            E = E[:3]
        _, R, t, mpose = cv2.recoverPose(E, p1, p2, self.K, mask=mask.copy())
        inl = mpose.ravel() > 0
        if inl.sum() < cfg.init_min_inliers:
            return
        t = t.ravel()
        R0, t0 = np.eye(3), np.zeros(3)
        X, good = self._triangulate(R0, t0, R, t, p1[inl], p2[inl],
                                    s1=ref.sigma[qi][inl], s2=f.sigma[ti][inl])
        if good.sum() < cfg.init_min_points or good.sum() < 0.5 * inl.sum():
            return
        # require real parallax: small-baseline two-view geometry confuses rotation and translation
        Cf = -R.T @ t
        rays0, rays1 = X[good], X[good] - Cf
        cosp = np.sum(rays0 * rays1, 1) / (np.linalg.norm(rays0, axis=1) * np.linalg.norm(rays1, axis=1))
        if np.degrees(np.median(np.arccos(np.clip(cosp, -1, 1)))) < cfg.init_min_parallax_deg:
            return
        X = X[good]
        q, tt = qi[inl][good], ti[inl][good]
        scale = 1.0 / np.median(X[:, 2])
        X *= scale
        t = t * scale
        ref.R, ref.t = R0, t0
        f.R, f.t = R, t
        kf0 = self.map.add_keyframe(ref)
        kf1 = self.map.add_keyframe(f)
        for k in range(len(X)):
            mp = self.map.add_point(X[k], f.desc[tt[k]], f.colors[tt[k]], kf1.id)
            mp.obs = {kf0.id: int(q[k]), kf1.id: int(tt[k])}
            kf0.point_ids[q[k]] = mp.id
            kf1.point_ids[tt[k]] = mp.id
            f.point_ids[tt[k]] = mp.id
        if self.cfg.use_ba:
            self._bundle_adjust([kf0, kf1], iters=30)
            X = np.array([self.map.points[i].pos for i in kf1.point_ids[kf1.point_ids >= 0]])
            R, t = kf1.R.copy(), kf1.t.copy()
            if len(X) < cfg.init_min_points:
                self.reset()
                return
        self.R, self.t = R, t
        self.velocity = None
        self.state = TRACKING
        self.last_kf = kf1
        self.last_kf_tracked = len(X)
        self.median_depth = float(np.median((X @ R.T + t)[:, 2]))
        self.stats["tracked"] = len(X)
        self.init_ref = None
        self.init_tracks = None
        self.trajectory = [kf0.center.copy()]
        self._fit_planes()

    # -------------------------------------------------------------- tracking
    def _guided_match(self, f, R, t, P, D, radius):
        if len(P) == 0 or len(f.pts) == 0:
            return np.zeros((0, 2), int), np.zeros(0, int)
        uv, z = project(self.K, R, t, P)
        w, h = self.image_size
        inview = (z > 1e-3) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        idx = np.nonzero(inview)[0]
        if len(idx) == 0:
            return np.zeros((0, 2), int), idx
        if HAS_SCIPY:
            k = min(8, len(f.pts))
            dist, nn = f.tree.query(uv[idx], k=k, distance_upper_bound=radius)
            dist, nn = np.asarray(dist).reshape(len(idx), k), np.asarray(nn).reshape(len(idx), k)
            valid = np.isfinite(dist)
            if not valid.any():
                return np.zeros((0, 2), int), idx
            mi = np.broadcast_to(idx[:, None], nn.shape)[valid]
            ki = nn[valid]
        else:
            cands = f.tree.query_ball_point(uv[idx], r=radius)
            lens = np.fromiter((len(c) for c in cands), int, len(cands))
            total = int(lens.sum())
            if total == 0:
                return np.zeros((0, 2), int), idx
            mi = np.repeat(idx, lens)
            ki = np.fromiter(itertools.chain.from_iterable(cands), int, total)
        d = hamming(D[mi], f.desc[ki])
        ok = d <= self.cfg.match_th
        mi, ki, d = mi[ok], ki[ok], d[ok]
        if len(mi) == 0:
            return np.zeros((0, 2), int), idx
        o = np.lexsort((d, mi))
        mi, ki, d = mi[o], ki[o], d[o]
        _, first = np.unique(mi, return_index=True)
        mi, ki, d = mi[first], ki[first], d[first]
        o = np.lexsort((d, ki))
        mi, ki = mi[o], ki[o]
        _, first = np.unique(ki, return_index=True)
        return np.stack([mi[first], ki[first]], axis=1), idx

    def _global_match(self, f, D):
        qi, ti = self._knn(f.desc, D, 0.8)
        return np.stack([ti, qi], axis=1) if len(qi) else np.zeros((0, 2), int)

    def _pnp_ransac(self, f, P, pairs):
        if len(pairs) < 12:
            return None
        obj = P[pairs[:, 0]].astype(np.float64)
        img = f.pts[pairs[:, 1]].astype(np.float64)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, self.K, None, iterationsCount=150,
                                                 reprojectionError=self.cfg.pnp_reproj_px,
                                                 confidence=0.995, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < 12:
            return None
        inl = inl.ravel()
        return cv2.Rodrigues(rvec)[0], tvec.ravel(), pairs[inl]

    def _refine(self, f, P, pairs, R, t, thresholds=(5.0, 3.0), prefilter=None, max_lm=400):
        """Robust pose refinement: LM on (a subsample of) the inliers, re-classify, repeat.
        prefilter: reject correspondences further than this many sigmas from the start pose first."""
        obj = P[pairs[:, 0]].astype(np.float64)
        img = f.pts[pairs[:, 1]].astype(np.float64)
        sig = f.sigma[pairs[:, 1]]
        rvec = cv2.Rodrigues(R)[0]
        tvec = t.reshape(3, 1).copy()
        if prefilter is not None:
            uv, z = project(self.K, R, t, obj)
            keep = (np.linalg.norm(uv - img, axis=1) < prefilter * sig) & (z > 0)
        else:
            keep = np.ones(len(pairs), bool)
        for thr in thresholds:
            idx = np.nonzero(keep)[0]
            if len(idx) < 6:
                return None
            if len(idx) > max_lm:                 # LM cost is linear in points; 400 is plenty for 6 DoF
                idx = idx[np.linspace(0, len(idx) - 1, max_lm).astype(int)]
            rvec, tvec = cv2.solvePnPRefineLM(obj[idx], img[idx], self.K, None, rvec, tvec, LM_CRITERIA)
            Rr = cv2.Rodrigues(rvec)[0]
            uv, z = project(self.K, Rr, tvec.ravel(), obj)
            keep = (np.linalg.norm(uv - img, axis=1) < thr * sig) & (z > 0)
        return cv2.Rodrigues(rvec)[0], tvec.ravel(), pairs[keep]

    def _track(self, f: Frame, relocalize=False) -> bool:
        cfg = self.cfg
        ids, P, D = self.map.arrays()
        if len(ids) < cfg.min_tracked:
            return False
        res = None
        if not relocalize:
            if self.velocity is not None:
                Rv, tv = self.velocity
                Rp, tp = Rv @ self.R, Rv @ self.t + tv
            else:
                Rp, tp = self.R, self.t
            pairs, _ = self._guided_match(f, Rp, tp, P, D, cfg.search_radius)
            if self.velocity is not None and len(pairs) >= 50:
                # good prediction: robust LM from the predicted pose, RANSAC only as a fallback
                quick = self._refine(f, P, pairs, Rp, tp)
                if quick is not None and len(quick[2]) >= max(40, 0.6 * len(pairs)):
                    res = quick
            if res is None and len(pairs) >= 30:
                res = self._pnp_ransac(f, P, pairs)
            if res is None or len(res[2]) < 30:
                pairs, _ = self._guided_match(f, Rp, tp, P, D, cfg.search_radius * 2.5)
                if len(pairs) >= 30:
                    res = self._pnp_ransac(f, P, pairs)
        if res is None or len(res[2]) < 30:
            res = self._pnp_ransac(f, P, self._global_match(f, D))
            if res is None or len(res[2]) < (30 if relocalize else 20):
                return False
        R, t, _ = res
        pairs, view_idx = self._guided_match(f, R, t, P, D, cfg.refine_radius)
        self.stats["matches"] = len(pairs)
        ref = self._refine(f, P, pairs, R, t, thresholds=(3.0,), prefilter=5.0) if len(pairs) >= 12 else None
        if ref is not None and len(ref[2]) >= len(res[2]) * 0.8:
            R, t, pairs = ref
        else:
            pairs = res[2]
        if len(pairs) < cfg.min_tracked:
            return False
        # map point statistics
        for mi in view_idx:
            self.map.points[ids[mi]].visible += 1
        for mi in pairs[:, 0]:
            self.map.points[ids[mi]].found += 1
        f.point_ids[pairs[:, 1]] = ids[pairs[:, 0]]
        if self.R is not None and not relocalize:
            Rv = R @ self.R.T
            self.velocity = (Rv, t - Rv @ self.t)
        else:
            self.velocity = None
        self.R, self.t = R, t
        f.R, f.t = R, t
        self.stats["tracked"] = len(pairs)
        self.last_view_idx = ids[view_idx]
        if self._need_keyframe(f, len(pairs)):
            self._insert_keyframe(f)
        return True

    # --------------------------------------------------------------- mapping
    def _need_keyframe(self, f, tracked):
        cfg = self.cfg
        if f.idx - self.last_kf.frame_idx < cfg.kf_min_frames:
            return False
        baseline = np.linalg.norm(self.camera_center - self.last_kf.center)
        return (tracked < cfg.kf_track_ratio * self.last_kf_tracked or tracked < cfg.kf_min_tracked
                or baseline > cfg.kf_baseline_ratio * self.median_depth)

    def _insert_keyframe(self, f: Frame):
        kf = self.map.add_keyframe(f)
        for kpi in np.nonzero(kf.point_ids >= 0)[0]:
            self.map.points[kf.point_ids[kpi]].obs[kf.id] = int(kpi)
        for nb in reversed(self.map.keyframes[-(self.cfg.tri_neighbors + 1):-1]):
            self._triangulate_new(kf, nb)
        self._cull_points(kf)
        if self.cfg.use_ba and len(self.map.keyframes) >= 3:
            if self.cfg.async_ba:
                self._bundle_adjust(self.map.keyframes[-self.cfg.ba_window:], background=True)
            else:
                t0 = time.perf_counter()
                self._bundle_adjust(self.map.keyframes[-self.cfg.ba_window:])
                self.stats["ba_ms"] = (time.perf_counter() - t0) * 1000
                self.R, self.t = kf.R.copy(), kf.t.copy()
                self.velocity = None
        f.point_ids = kf.point_ids.copy()
        self.last_kf = kf
        self.last_kf_tracked = int((kf.point_ids >= 0).sum())
        ids = kf.point_ids[kf.point_ids >= 0]
        if len(ids):
            X = np.array([self.map.points[i].pos for i in ids])
            self.median_depth = float(np.median((X @ kf.R.T + kf.t)[:, 2]))
        if len(self.map.keyframes) % 3 == 0 or not self.planes:
            self._fit_planes()

    def _triangulate_new(self, kf: Keyframe, nb: Keyframe):
        a = np.nonzero(kf.point_ids < 0)[0]
        b = np.nonzero(nb.point_ids < 0)[0]
        if len(a) < 10 or len(b) < 10:
            return 0
        qa, qb = self._knn(kf.desc[a], nb.desc[b], 0.75)
        if len(qa) < 5:
            return 0
        ia, ib = a[qa], b[qb]
        R_rel = kf.R @ nb.R.T
        t_rel = kf.t - R_rel @ nb.t
        if np.linalg.norm(t_rel) < 1e-6:
            return 0
        F = self.Kinv.T @ skew(t_rel) @ R_rel @ self.Kinv
        x1 = np.hstack([nb.pts[ib], np.ones((len(ib), 1))])
        x2 = np.hstack([kf.pts[ia], np.ones((len(ia), 1))])
        lines = x1 @ F.T
        epi = np.abs(np.sum(x2 * lines, axis=1)) / (np.linalg.norm(lines[:, :2], axis=1) + 1e-12)
        ok = epi < 2.0 * kf.sigma[ia]
        ia, ib = ia[ok], ib[ok]
        if len(ia) == 0:
            return 0
        X, good = self._triangulate(nb.R, nb.t, kf.R, kf.t, nb.pts[ib], kf.pts[ia], max_err=2.0,
                                    s1=nb.sigma[ib], s2=kf.sigma[ia])
        depth = (X @ kf.R.T + kf.t)[:, 2]
        good &= depth < 40 * self.median_depth
        n = 0
        for k in np.nonzero(good)[0]:
            mp = self.map.add_point(X[k], kf.desc[ia[k]], kf.colors[ia[k]], kf.id)
            mp.obs = {nb.id: int(ib[k]), kf.id: int(ia[k])}
            kf.point_ids[ia[k]] = mp.id
            nb.point_ids[ib[k]] = mp.id
            n += 1
        return n

    def _cull_points(self, kf):
        for pid, mp in list(self.map.points.items()):
            if mp.visible >= 8 and mp.found / mp.visible < 0.25:
                self.map.remove_point(pid)
            elif kf.id - mp.created_kf == 3 and len(mp.obs) <= 2:
                self.map.remove_point(pid)

    def _bundle_adjust(self, window, iters=None, background=False):
        """Local BA over `window` keyframes. background=True: solve on the BA thread from a
        snapshot and apply the result at the start of a later frame (no tracking stalls)."""
        if background and self._ba_future is not None:
            return                          # previous BA still running; next keyframe will cover this window
        problem = self._ba_prepare(window)
        if problem is None:
            return
        it = iters or self.cfg.ba_iters
        if background:
            self._ba_future = self._ba_pool.submit(solve_ba, problem, self.K, it)
        else:
            sol = solve_ba(problem, self.K, it)
            if sol is not None:
                self._ba_apply(sol)

    def _poll_ba(self):
        fut = self._ba_future
        if fut is not None and fut.done():
            self._ba_future = None
            try:
                sol = fut.result()
            except Exception:
                sol = None
            if sol is not None and sol["epoch"] == self._epoch:
                self._ba_apply(sol)
                self.stats["ba_ms"] = sol["ms"]

    def _ba_prepare(self, window):
        mp_ids = set()
        for kf in window:
            mp_ids.update(int(p) for p in kf.point_ids[kf.point_ids >= 0] if p in self.map.points)
        if len(mp_ids) < 10:
            return None
        pid_list = sorted(mp_ids)
        pidx = {p: i for i, p in enumerate(pid_list)}
        window_ids = {kf.id for kf in window}
        first_id = self.map.keyframes[0].id
        observers = set()
        for p in pid_list:
            observers.update(self.map.points[p].obs.keys())
        fixed_ids = {k for k in observers if k not in window_ids or k == first_id}
        if not fixed_ids:               # pin the gauge with at least one camera
            fixed_ids.add(min(kf.id for kf in window))
        opt_kfs = [kf for kf in window if kf.id not in fixed_ids]
        if not opt_kfs:
            return None
        cidx = {kf.id: i for i, kf in enumerate(opt_kfs)}
        cam_i, pt_i, uv, fR, ft, sg = [], [], [], [], [], []
        for p in pid_list:
            mp = self.map.points[p]
            for kf_id, kpi in mp.obs.items():
                kf = self.map.kf_by_id.get(kf_id)
                if kf is None:
                    continue
                cam_i.append(cidx.get(kf_id, -1))
                pt_i.append(pidx[p])
                uv.append(kf.pts[kpi])
                sg.append(kf.sigma[kpi])
                fR.append(kf.R)
                ft.append(kf.t)
        return {"epoch": self._epoch, "opt_kf_ids": [kf.id for kf in opt_kfs],
                "R0": np.array([kf.R for kf in opt_kfs]), "t0": np.array([kf.t for kf in opt_kfs]),
                "pid_list": pid_list, "X0": np.array([self.map.points[p].pos for p in pid_list]),
                "cam_i": np.array(cam_i), "pt_i": np.array(pt_i), "uv": np.array(uv),
                "fR": np.array(fR), "ft": np.array(ft), "w": 1.0 / np.array(sg)}

    def _ba_apply(self, sol):
        for kf_id, R, t in zip(sol["opt_kf_ids"], sol["R"], sol["t"]):
            kf = self.map.kf_by_id.get(kf_id)
            if kf is not None:
                kf.R, kf.t = R, t
        for p, X, bad in zip(sol["pid_list"], sol["X"], sol["bad"]):
            mp = self.map.points.get(p)
            if mp is None:
                continue
            if bad:
                self.map.remove_point(p)
            else:
                mp.pos = X
        self.map.version += 1

    # ----------------------------------------------------------- environment
    def _fit_planes(self):
        _, P, _ = self.map.arrays()
        planes = []
        remaining = P.copy()
        rng = np.random.default_rng(0)
        ref_kf = self.map.keyframes[0] if self.map.keyframes else None
        cam_x = ref_kf.R[0] if ref_kf is not None else np.array([1.0, 0, 0])
        C = self.camera_center if self.R is not None else np.zeros(3)
        thr = 0.025 * max(self.median_depth, 1e-6)
        for _ in range(self.cfg.max_planes * 3):
            if len(planes) >= self.cfg.max_planes:
                break
            n_pts = len(remaining)
            if n_pts < 40:
                break
            # vectorized RANSAC: 256 hypotheses scored at once
            tri = remaining[rng.integers(0, n_pts, (256, 3))]
            nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            nn = np.linalg.norm(nrm, axis=1)
            keep = nn > 1e-9
            if not keep.any():
                break
            nrm = nrm[keep] / nn[keep, None]
            offs = np.sum(nrm * tri[keep, 0], axis=1)
            counts = (np.abs(remaining @ nrm.T - offs) < thr).sum(axis=0)
            b = int(np.argmax(counts))
            best_n = int(counts[b])
            best_inl = np.abs(remaining @ nrm[b] - offs[b]) < thr
            if best_inl is None or best_n < max(30, 0.15 * len(P)):
                break
            Q = remaining[best_inl]
            p0 = Q.mean(axis=0)
            _, _, Vt = np.linalg.svd(Q - p0)
            n = Vt[2]
            # a near-duplicate of an existing surface (parallel and close) is absorbed, not added
            dup = any(abs(np.dot(n, pl.n)) > np.cos(np.radians(20)) and
                      abs(np.dot(p0 - pl.p0, pl.n)) < 0.1 * max(self.median_depth, 1e-6) for pl in planes)
            remaining = remaining[~best_inl]
            if dup:
                continue
            if np.dot(n, C - p0) < 0:
                n = -n
            u = cam_x - np.dot(cam_x, n) * n
            if np.linalg.norm(u) < 1e-6:
                u = Vt[0]
            u /= np.linalg.norm(u)
            v = np.cross(n, u)
            q2 = np.stack([(Q - p0) @ u, (Q - p0) @ v], axis=1)
            lo, hi = np.percentile(q2, 3, axis=0), np.percentile(q2, 97, axis=0)
            core = q2[np.all((q2 >= lo) & (q2 <= hi), axis=1)]
            if len(core) < 3:
                core = q2
            hull = cv2.convexHull(core.astype(np.float32)).reshape(-1, 2).astype(np.float64)
            spacing = _nice_step(max(hi - lo) / 8.0)
            ext = (np.floor(lo[0] / spacing) * spacing, np.ceil(hi[0] / spacing) * spacing,
                   np.floor(lo[1] / spacing) * spacing, np.ceil(hi[1] / spacing) * spacing)
            planes.append(Plane(p0, n, u, v, hull, ext, spacing, best_n))
        self.planes = planes

    def refit_environment(self):
        self._fit_planes()


def solve_ba(pb, K, iters):
    """Sparse Levenberg-Marquardt BA (analytic Jacobian, Huber loss, octave-weighted residuals)."""
    t_start = time.perf_counter()
    cam_i, pt_i, uv, fR, ft, w = pb["cam_i"], pb["pt_i"], pb["uv"], pb["fR"], pb["ft"], pb["w"]
    R0 = pb["R0"]
    nc, npt = len(R0), len(pb["pid_list"])
    fixed = cam_i < 0
    x0 = np.hstack([np.hstack([np.zeros((nc, 3)), pb["t0"]]).ravel(), pb["X0"].ravel()])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    mask_opt = ~fixed
    ci = cam_i[mask_opt]
    obs_R = fR.copy()
    obs_R[mask_opt] = R0[ci]
    m = len(uv)
    rows = np.arange(m)

    def camera_coords(x):
        cams = x[: nc * 6].reshape(nc, 6)
        pts = x[nc * 6:].reshape(npt, 3)
        Y = np.einsum("nij,nj->ni", obs_R, pts[pt_i])     # R0 X (or fixed R X)
        Xc = Y + ft
        Xc[mask_opt] = rotate(Y[mask_opt], cams[ci, :3]) + cams[ci, 3:]
        return Xc

    def residuals(x):
        Xc = camera_coords(x)
        z = Xc[:, 2]
        z = np.where(np.abs(z) < 1e-6, 1e-6, z)
        r = np.empty_like(uv)
        r[:, 0] = fx * Xc[:, 0] / z + cx - uv[:, 0]
        r[:, 1] = fy * Xc[:, 1] / z + cy - uv[:, 1]
        return (r * w[:, None]).ravel()

    def jacobian(x):
        cams = x[: nc * 6].reshape(nc, 6)
        Xc = camera_coords(x)
        X, Y, Z = Xc[:, 0], Xc[:, 1], np.where(np.abs(Xc[:, 2]) < 1e-6, 1e-6, Xc[:, 2])
        Jp = np.zeros((m, 2, 3))                          # d(uv)/d(Xc)
        Jp[:, 0, 0] = fx / Z
        Jp[:, 0, 2] = -fx * X / Z ** 2
        Jp[:, 1, 1] = fy / Z
        Jp[:, 1, 2] = -fy * Y / Z ** 2
        Jp *= w[:, None, None]
        # d Xc / d point = R_total ; R_total = exp(delta) R0 for optimized cams
        Rtot = obs_R.copy()
        if nc:
            Rd = np.array([cv2.Rodrigues(c[:3])[0] for c in cams])
            Rtot[mask_opt] = np.einsum("nij,njk->nik", Rd[ci], obs_R[mask_opt])
        Jpt = np.einsum("nij,njk->nik", Jp, Rtot)
        # d Xc / d delta ~= -[Xc - t]x ; d Xc / d t = I
        Pc = Xc - np.where(mask_opt[:, None], cams[np.maximum(cam_i, 0), 3:], 0)
        Sk = np.zeros((m, 3, 3))
        Sk[:, 0, 1], Sk[:, 0, 2] = Pc[:, 2], -Pc[:, 1]
        Sk[:, 1, 0], Sk[:, 1, 2] = -Pc[:, 2], Pc[:, 0]
        Sk[:, 2, 0], Sk[:, 2, 1] = Pc[:, 1], -Pc[:, 0]
        Jrot = np.einsum("nij,njk->nik", Jp, Sk)
        r_idx, c_idx, vals = [], [], []
        for a in range(2):
            for b in range(3):
                r_idx.append(2 * rows + a)
                c_idx.append(nc * 6 + pt_i * 3 + b)
                vals.append(Jpt[:, a, b])
                if nc:
                    r_idx.append(2 * rows[mask_opt] + a)
                    c_idx.append(ci * 6 + b)
                    vals.append(Jrot[mask_opt, a, b])
                    r_idx.append(2 * rows[mask_opt] + a)
                    c_idx.append(ci * 6 + 3 + b)
                    vals.append(Jp[mask_opt, a, b])
        return csr_matrix((np.concatenate(vals), (np.concatenate(r_idx), np.concatenate(c_idx))),
                          shape=(2 * m, nc * 6 + npt * 3))

    try:
        sol = least_squares(residuals, x0, jac=jacobian, loss="huber", f_scale=2.0, method="trf",
                            x_scale="jac", max_nfev=iters, ftol=1e-4, xtol=1e-6)
    except Exception:
        return None
    x = sol.x
    cams = x[: nc * 6].reshape(nc, 6)
    err = np.linalg.norm(residuals(x).reshape(-1, 2), axis=1)   # in sigma units
    bad = np.zeros(npt, bool)
    np.logical_or.at(bad, pt_i, err > 4.0)
    return {"epoch": pb["epoch"], "opt_kf_ids": pb["opt_kf_ids"],
            "R": [cv2.Rodrigues(c[:3])[0] @ Rk for c, Rk in zip(cams, R0)], "t": [c[3:].copy() for c in cams],
            "pid_list": pb["pid_list"], "X": x[nc * 6:].reshape(npt, 3), "bad": bad,
            "ms": (time.perf_counter() - t_start) * 1000}
