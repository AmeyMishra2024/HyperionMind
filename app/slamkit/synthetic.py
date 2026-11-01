"""Ray-cast textured room used as a fake camera (demo mode + automated tests)."""
from __future__ import annotations

import cv2
import numpy as np


def _make_texture(size=2048, seed=1):
    rng = np.random.default_rng(seed)
    tex = np.full((size, size, 3), 128, np.uint8)
    for _ in range(3500):
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        x, y = (int(v) for v in rng.integers(0, size, 2))
        r = int(rng.integers(4, 40))
        if rng.random() < 0.5:
            cv2.circle(tex, (x, y), r, color, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(tex, (x, y), (x + r, y + int(rng.integers(4, 40))), color, -1)
    noise = rng.normal(0, 12, tex.shape)
    return np.clip(tex + noise, 0, 255).astype(np.uint8)


class SyntheticRoom:
    """Box room, camera near the centre. World axes: x right, y down, z forward."""

    # (axis, value, texture offset)
    PLANES = [(1, 1.2, 0), (1, -2.0, 500), (2, 6.0, 1000), (0, -4.0, 1500), (0, 4.0, 250), (2, -3.0, 750)]

    def __init__(self, K, size=(640, 480), fps=30.0, seed=1, noise=2.0):
        self.K = np.asarray(K, np.float64)
        self.w, self.h = size
        self.fps = fps
        self.noise = noise
        self.tex = _make_texture(seed=seed)
        self.tex_scale = 160.0  # texture pixels per world unit
        u, v = np.meshgrid(np.arange(self.w, dtype=np.float64), np.arange(self.h, dtype=np.float64))
        pix = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3)
        self.rays_cam = pix @ np.linalg.inv(self.K).T
        self.rng = np.random.default_rng(seed + 7)

    @staticmethod
    def pose_at(time_s):
        """Ground-truth world->camera pose. Starts with a sideways move (good for init)."""
        t = time_s
        C = np.array([1.0 * np.sin(0.45 * t), 0.08 * np.sin(0.7 * t), 0.5 * (1 - np.cos(0.25 * t))])
        yaw = 0.30 * np.sin(0.22 * t)
        pitch = 0.08 * np.sin(0.31 * t)
        Ry = cv2.Rodrigues(np.array([0.0, yaw, 0.0]))[0]
        Rx = cv2.Rodrigues(np.array([pitch, 0.0, 0.0]))[0]
        R_cw = Ry @ Rx                        # camera->world rotation
        R = R_cw.T
        return R, -R @ C

    def render(self, R, t):
        C = -R.T @ t
        d = self.rays_cam @ R                 # world-frame ray directions (R^T d_cam)
        best = np.full(len(d), np.inf)
        tu = np.zeros(len(d))
        tv = np.zeros(len(d))
        shade = np.ones(len(d))
        for k, (axis, val, off) in enumerate(self.PLANES):
            with np.errstate(divide="ignore", invalid="ignore"):
                s = (val - C[axis]) / d[:, axis]
            hit = (s > 1e-6) & (s < best)
            if not hit.any():
                continue
            P = C + s[hit, None] * d[hit]
            a, b = [i for i in range(3) if i != axis]
            best[hit] = s[hit]
            tu[hit] = P[:, a] * self.tex_scale + off
            tv[hit] = P[:, b] * self.tex_scale + off * 0.7
            shade[hit] = 0.75 + 0.05 * k
        mapx = np.mod(tu, self.tex.shape[1] - 1).reshape(self.h, self.w).astype(np.float32)
        mapy = np.mod(tv, self.tex.shape[0] - 1).reshape(self.h, self.w).astype(np.float32)
        img = cv2.remap(self.tex, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        img = img.astype(np.float32) * shade.reshape(self.h, self.w, 1)
        if self.noise > 0:
            img += self.rng.normal(0, self.noise, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    def frame(self, i):
        R, t = self.pose_at(i / self.fps)
        return self.render(R, t), (R, t)
