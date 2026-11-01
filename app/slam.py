"""C270 Map Lab SLAM service.

Keeps the original public API (CameraCalibration, MonocularSlam, SlamRegistry, load_calibration)
used by app.main and the tests, but runs the full `app.slamkit` engine underneath:

  * ORB features, parallax-checked two-view initialization, motion-model guided tracking
  * keyframes, epipolar-checked triangulation, local bundle adjustment (background thread)
  * relocalization, dominant surface fitting
  * Nav2-style inflated costmap and ORB-SLAM3-style annotated camera frames

Monocular: map scale is arbitrary (the first map's median depth is 1 unit).
"""
from __future__ import annotations

import base64
import json
import math
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.slamkit.config import C270_APPROX_FX_640, SlamConfig
from app.slamkit.core import HAS_SCIPY, INITIALIZING, LOST, NOT_INITIALIZED, TRACKING, MonoSLAM
from app.slamkit.occupancy import MapSnapshot, OccupancyMapper
from app.slamkit.overlay import OverlayRenderer
from app.slamkit.views import draw_orbslam_frame

VIEW_MODES = ("features", "ar", "raw")
MAP_PAYLOAD_INTERVAL = 0.35   # s; full point cloud / keyframes are sent at most ~3x per second
_COSTMAP_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="costmap")


@dataclass
class CameraCalibration:
    width: int = 640
    height: int = 480
    fx: float = C270_APPROX_FX_640
    fy: float = C270_APPROX_FX_640
    cx: float = 320.0
    cy: float = 240.0
    distortion: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_dict(cls, value: dict) -> "CameraCalibration":
        width = max(1, int(value.get("width", 640)))
        height = max(1, int(value.get("height", 480)))
        default_f = C270_APPROX_FX_640 * width / 640
        return cls(
            width=width,
            height=height,
            fx=float(value.get("fx", default_f)),
            fy=float(value.get("fy", default_f)),
            cx=float(value.get("cx", width / 2)),
            cy=float(value.get("cy", height / 2)),
            distortion=tuple(float(x) for x in value.get("distortion", [0, 0, 0, 0, 0])[:8]),
        )

    def matrix_for(self, width: int, height: int) -> np.ndarray:
        sx, sy = width / self.width, height / self.height
        return np.array(
            [[self.fx * sx, 0.0, self.cx * sx], [0.0, self.fy * sy, self.cy * sy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def distortion_array(self) -> np.ndarray:
        return np.asarray(self.distortion, dtype=np.float64)


def _r(values, nd=4):
    return [round(float(v), nd) for v in values]


class MonocularSlam:
    """One mapping session: engine + costmap + frame rendering + export."""

    def __init__(self, calibration: CameraCalibration | None = None, max_points: int = 25_000,
                 config: SlamConfig | None = None):
        self.calibration = calibration or CameraCalibration()
        self.max_points = max_points
        self.config = config
        self.view_mode = "features"
        self.lock = threading.RLock()
        self.reset()

    # ------------------------------------------------------------- lifecycle
    def reset(self) -> None:
        with getattr(self, "lock", threading.RLock()):
            self.engine: MonoSLAM | None = None
            self.renderer: OverlayRenderer | None = None
            self.size: tuple[int, int] | None = None
            self.costmap = OccupancyMapper()
            self.frame_number = 0
            self._points_override = None
            self._colors_override = None
            self._sent_map_version = -1
            self._sent_costmap_version = -1
            self._costmap_map_version = -1
            self._costmap_at = 0.0
            self._costmap_future = None
            self._map_sent_at = 0.0
            self._payload_cache = (None, None)
            self._frame_times: list[float] = []
            self.last_result: dict = {}

    def set_calibration(self, calibration: CameraCalibration) -> None:
        self.calibration = calibration
        self.reset()

    def set_view(self, mode: str) -> None:
        if mode not in VIEW_MODES:
            raise ValueError(f"View must be one of {', '.join(VIEW_MODES)}.")
        self.view_mode = mode

    def resend_map(self) -> None:
        """Next result carries the full map again (e.g. after the browser reconnects)."""
        self._sent_map_version = -1
        self._sent_costmap_version = -1
        self._map_sent_at = 0.0

    def _ensure_engine(self, width: int, height: int) -> None:
        if self.engine is not None and self.size == (width, height):
            return
        K = self.calibration.matrix_for(width, height)
        dist = self.calibration.distortion_array()
        cfg = self.config or SlamConfig()
        self.engine = MonoSLAM(K, dist, (width, height), SlamConfig(**vars(cfg)))
        self.renderer = OverlayRenderer(K, dist, (width, height), label="C270")
        self.size = (width, height)
        self.costmap.reset()

    # ---------------------------------------------------------------- frames
    def process_jpeg(self, payload: bytes, include_preview: bool = False) -> dict:
        encoded = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("The camera frame was not a valid JPEG image.")
        return self.process_frame(frame, include_preview=include_preview)

    def process_frame(self, frame: np.ndarray, include_preview: bool = False) -> dict:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a BGR color frame.")
        height, width = frame.shape[:2]
        if width > 640:
            frame = cv2.resize(frame, (640, round(height * 640 / width)), interpolation=cv2.INTER_AREA)
            height, width = frame.shape[:2]
        with self.lock:
            self._ensure_engine(width, height)
            self.engine.process(frame)
            self.frame_number += 1
            now = time.perf_counter()
            self._frame_times = [t for t in self._frame_times if now - t < 2.0] + [now]
            self._maybe_update_costmap(now)
            result = self._result()
            if include_preview:
                result["preview"] = self.render_preview(frame)
            return result

    def render_preview(self, frame: np.ndarray, quality: int = 72) -> str | None:
        if self.view_mode == "features":
            img = draw_orbslam_frame(frame, self.engine)
        elif self.view_mode == "ar":
            fps = max(len(self._frame_times) - 1, 0) / 2.0
            img = self.renderer.render(frame, self.engine, fps, fps)
        else:
            img = frame
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(jpg).decode("ascii") if ok else None

    def _maybe_update_costmap(self, now: float) -> None:
        """Snapshot the map (cheap, on the SLAM thread) and build the costmap on a worker thread."""
        eng = self.engine
        if eng.map.version == self._costmap_map_version or now - self._costmap_at < 0.8:
            return
        if len(eng.map.keyframes) < 2:
            return
        if self._costmap_future is not None and not self._costmap_future.done():
            return
        self._costmap_at = now
        self._costmap_map_version = eng.map.version
        snapshot = MapSnapshot(eng)
        mapper = self.costmap

        def build():
            try:
                mapper.update(snapshot)
            except Exception:  # never let the 2D map break tracking
                pass

        self._costmap_future = _COSTMAP_POOL.submit(build)

    # --------------------------------------------------------------- results
    def _status(self) -> str:
        eng = self.engine
        st = eng.state
        if st == NOT_INITIALIZED:
            if eng.stats["features"] < eng.cfg.init_min_matches:
                return "Point the camera at a well-lit, textured scene"
            return "Hold on a textured scene, then slide the camera sideways"
        if st == INITIALIZING:
            return f"Initializing — slide the camera sideways slowly ({eng.stats['matches']} matches)"
        if st == LOST:
            return "Tracking lost — move back to a mapped area to relocalize"
        msg = f"Tracking · {len(eng.map.points):,} map points · {len(eng.map.keyframes)} keyframes"
        if eng.stats.get("resets"):
            msg += f" · {eng.stats['resets']} auto-reset(s)"
        return msg

    @property
    def map_points(self) -> list[np.ndarray]:
        if self._points_override is not None:
            return self._points_override
        if self.engine is None:
            return []
        return [mp.pos for mp in self.engine.map.points.values()]

    @map_points.setter
    def map_points(self, value) -> None:
        self._points_override = list(value)

    @property
    def map_colors(self) -> list[tuple[int, int, int]]:
        if self._colors_override is not None:
            return self._colors_override
        if self.engine is None:
            return []
        return [(int(mp.color[2]), int(mp.color[1]), int(mp.color[0])) for mp in self.engine.map.points.values()]

    @map_colors.setter
    def map_colors(self, value) -> None:
        self._colors_override = list(value)

    @property
    def trajectory(self) -> list[np.ndarray]:
        if self.engine is None or not self.engine.trajectory:
            return [np.zeros(3)]
        return self.engine.trajectory

    def _result(self) -> dict:
        eng = self.engine
        fps = max(len(self._frame_times) - 1, 0) / max(self._frame_times[-1] - self._frame_times[0], 1e-6) \
            if len(self._frame_times) > 1 else 0.0
        traj = self.trajectory
        tracking = eng.state == TRACKING
        if eng.R is not None:
            pose_cw = np.eye(4)
            pose_cw[:3, :3], pose_cw[:3, 3] = eng.R, eng.t
        else:
            pose_cw = np.eye(4)
        result = {
            "status": self._status(),
            "state": eng.state,
            "frame": self.frame_number,
            "matches": int(eng.stats.get("matches", 0)),
            "inliers": int(eng.stats.get("tracked", 0)) if tracking else 0,
            "features": int(eng.stats.get("features", 0)),
            "landmarks": len(eng.map.points),
            "keyframe_count": len(eng.map.keyframes),
            "surfaces": len(eng.planes),
            "proc_ms": round(float(eng.stats.get("proc_ms", 0.0)), 1),
            "fps": round(fps, 1),
            "bundle_adjustment": bool(eng.cfg.use_ba),
            "trajectory": [_r(p) for p in traj[:: max(1, len(traj) // 800)]],
            "pose": [round(float(v), 5) for v in pose_cw.reshape(-1)],
            "scale": "arbitrary",
            "map_version": eng.map.version,
        }
        if eng.R is not None:
            result["camera"] = {"center": _r(eng.camera_center), "rotation": _r(eng.R.T.reshape(-1), 5)}
        K = eng.K
        result["intrinsics"] = [round(float(K[0, 0]), 2), round(float(K[1, 1]), 2), round(float(K[0, 2]), 2),
                                round(float(K[1, 2]), 2), self.size[0], self.size[1]]
        # 2D features for the browser-drawn ORB-SLAM3 view; 3D tracked points drawn red in the map
        cur = eng.current
        if tracking and cur is not None:
            sel = cur.point_ids >= 0
            result["features2d"] = np.round(cur.pts_raw[sel]).astype(int).tolist()
            ids = cur.point_ids[sel]
            ids = ids[:: max(1, len(ids) // 700)]
            pos = [eng.map.points[int(i)].pos for i in ids if int(i) in eng.map.points]
            result["tracked"] = np.round(np.array(pos), 4).tolist() if pos else []
        else:
            result["features2d"] = []
            result["tracked"] = []
        if eng.state in (NOT_INITIALIZED, INITIALIZING) and eng.init_tracks is not None:
            a, b = eng.init_tracks
            step = max(1, len(a) // 400)
            result["init_tracks"] = np.round(np.hstack([a[::step], b[::step]])).astype(int).tolist()
        now = time.perf_counter()
        if eng.map.version != self._sent_map_version and (
                self._sent_map_version < 0 or now - self._map_sent_at >= MAP_PAYLOAD_INTERVAL):
            self._sent_map_version = eng.map.version
            self._map_sent_at = now
            result.update(self._map_payload())
        # costmap
        cm = self.costmap
        if cm.meta is not None:
            info = {"version": cm.version, "width": cm.meta["width"], "height": cm.meta["height"],
                    "resolution": round(cm.meta["res"], 5), "floor": bool(cm.meta["floor"])}
            if cm.version != self._sent_costmap_version and cm.png is not None:
                self._sent_costmap_version = cm.version
                info["png"] = base64.b64encode(cm.png).decode("ascii")
            path = cm.to_grid(np.array(traj[:: max(1, len(traj) // 400)]))
            info["path"] = [_r(p, 1) for p in path]
            if eng.R is not None:
                c = cm.to_grid(eng.camera_center)[0]
                info["pose"] = [round(float(c[0]), 1), round(float(c[1]), 1), round(cm.heading(eng.R), 3)]
            result["costmap"] = info
        self.last_result = result
        return result

    def _map_payload(self) -> dict:
        eng = self.engine
        ids, P, _ = eng.map.arrays()
        stride = max(1, math.ceil(len(ids) / 8000))
        sel_ids = ids[::stride]
        colors = [eng.map.points[int(i)].color for i in sel_ids]
        kfs = eng.map.keyframes
        # covisibility graph (ORB-SLAM3's green lines): keyframe pairs sharing >= 30 map points
        recent = kfs[-60:]
        sets = [np.unique(kf.point_ids[kf.point_ids >= 0]) for kf in recent]
        base = len(kfs) - len(recent)
        edges = []
        for a in range(len(recent)):
            for b in range(a + 1, len(recent)):
                if len(np.intersect1d(sets[a], sets[b], assume_unique=True)) >= 30:
                    edges.append([base + a, base + b])
        planes = []
        for pl in eng.planes:
            if len(pl.hull2d) >= 3:
                planes.append({"hull": np.round(pl.to_world(pl.hull2d), 4).tolist(), "normal": _r(pl.n),
                               "center": _r(pl.p0), "u": _r(pl.u), "v": _r(pl.v),
                               "extent": _r(pl.extent), "spacing": round(float(pl.spacing), 5)})
        return {
            "points": np.round(P[::stride], 4).tolist(),
            "colors": [[int(c[2]), int(c[1]), int(c[0])] for c in colors],
            "keyframes": [_r(kf.center) + _r(kf.R.T.reshape(-1), 5) for kf in kfs],
            "graph": edges,
            "spanning_tree": [[i - 1, i] for i in range(1, len(kfs))],
            "planes": planes,
            "median_depth": round(float(eng.median_depth), 4),
        }

    # ---------------------------------------------------------------- export
    def export_ply(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        points, colors = self.map_points, self.map_colors
        with path.open("w", encoding="ascii", newline="\n") as output:
            output.write("ply\nformat ascii 1.0\n")
            output.write("comment Monocular map; coordinates use arbitrary scale\n")
            output.write(f"element vertex {len(points)}\n")
            output.write("property float x\nproperty float y\nproperty float z\n")
            output.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for point, color in zip(points, colors):
                output.write(f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} {color[0]} {color[1]} {color[2]}\n")
        return path

    def export_trajectory_csv(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="ascii", newline="\n") as output:
            output.write("frame,x,y,z\n")
            for index, point in enumerate(self.trajectory):
                output.write(f"{index},{point[0]:.8g},{point[1]:.8g},{point[2]:.8g}\n")
        return path

    def export_costmap_png(self, path: Path) -> Path | None:
        if self.costmap.png is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.costmap.png)
        return path


class SlamRegistry:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._items: dict[str, tuple[MonocularSlam, float]] = {}
        self._lock = threading.Lock()

    def create(self) -> tuple[str, MonocularSlam]:
        session_id = uuid.uuid4().hex
        slam = MonocularSlam()
        with self._lock:
            self._items[session_id] = (slam, time.time())
            self._prune_locked()
        return session_id, slam

    def get(self, session_id: str) -> MonocularSlam | None:
        with self._lock:
            item = self._items.get(session_id)
            if item:
                self._items[session_id] = (item[0], time.time())
                return item[0]
        return None

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._items.pop(session_id, None)

    def export(self, session_id: str) -> tuple[Path, Path] | None:
        slam = self.get(session_id)
        if slam is None:
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = self.output_dir / f"c270-map-{stamp}-{session_id[:6]}"
        with slam.lock:
            ply = slam.export_ply(base.with_suffix(".ply"))
            csv = slam.export_trajectory_csv(base.with_name(base.name + "-trajectory.csv"))
            slam.export_costmap_png(base.with_name(base.name + "-costmap.png"))
        return ply, csv

    def export_costmap(self, session_id: str) -> Path | None:
        slam = self.get(session_id)
        if slam is None:
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        with slam.lock:
            return slam.export_costmap_png(self.output_dir / f"c270-costmap-{stamp}-{session_id[:6]}.png")

    def _prune_locked(self) -> None:
        cutoff = time.time() - 4 * 60 * 60
        expired = [key for key, (_, used) in self._items.items() if used < cutoff]
        for key in expired:
            self._items.pop(key, None)


def load_calibration(path: Path) -> CameraCalibration:
    return CameraCalibration.from_dict(json.loads(path.read_text(encoding="utf-8")))


__all__ = ["CameraCalibration", "MonocularSlam", "SlamRegistry", "load_calibration", "HAS_SCIPY", "VIEW_MODES"]
