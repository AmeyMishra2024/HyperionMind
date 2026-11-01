// C270 Map Lab v2: live monocular SLAM with an ORB-SLAM3-style 3D viewer and a Nav2-style costmap.
// Smooth pipeline: the camera feed streams at its own rate; SLAM results arrive independently and
// the feature boxes / AR overlay are drawn here, on top of the newest video frame.
const $ = (s) => document.querySelector(s);
const ui = {
  camera: $('#cameraSelect'), cameraHint: $('#cameraHint'), cameraName: $('#cameraName'),
  start: $('#startButton'), reset: $('#resetButton'), export: $('#exportButton'), trajectory: $('#trajectoryButton'),
  costmapBtn: $('#costmapButton'), fit: $('#fitButton'), calibration: $('#calibrationFile'), calibrationLabel: $('#calibrationLabel'),
  video: $('#video'), live: $('#liveCanvas'), videoEmpty: $('#videoEmpty'), capture: $('#captureCanvas'),
  canvas: $('#mapCanvas'), frame: $('#mapFrame'), mapEmpty: $('#mapEmpty'), status: $('#statusText'), badge: $('#connectionBadge'),
  ba: $('#baBadge'), landmarks: $('#landmarks'), keyframes: $('#keyframes'), matches: $('#matches'), inliers: $('#inliers'),
  procMs: $('#procMs'), frames: $('#frames'), videoFps: $('#videoFps'), slamFps: $('#slamFps'), latency: $('#latency'),
  cm: $('#costmapCanvas'), cmEmpty: $('#costmapEmpty'), cmInfo: $('#costmapInfo'), goalInfo: $('#goalInfo'), clearGoal: $('#clearGoalButton'),
};
const state = {
  stream: null, socket: null, sessionId: null, busy: false, running: false, direct: true, cameraIndex: '0',
  calibration: null, viewMode: 'features',
  points: [], colors: [], keyframes: [], graph: [], spanning: [], planes: [], medianDepth: 1,
  tracked: [], trajectory: [], camera: null, intr: null, features2d: [], initTracks: null, slamState: 'WAITING',
  kfCount: 0, mpCount: 0, matchCount: 0, bitmap: null, decoding: false, pendingFrame: null,
};
const rate = { video: [], slam: [] };
function tick(list) { const now = performance.now(); list.push(now); while (list.length && now - list[0] > 2000) list.shift(); }
function fpsOf(list) { return list.length > 1 ? ((list.length - 1) * 1000) / (list[list.length - 1] - list[0]) : 0; }
const view = { yaw: -0.72, pitch: -0.38, distance: 3, target: [0, 0, 0], dragging: false, panning: false, lastX: 0, lastY: 0, manual: false };
const opts = { follow: true, points: true, keyframes: true, graph: true, trajectory: false, surfaces: true, grid: true, theme: 'light' };
const THEMES = {
  light: { bg: '#ffffff', point: [0, 0, 0], tracked: '#e0161b', kf: '#1f3bff', cam: '#00b300', graph: 'rgba(0,170,0,0.55)',
    tree: 'rgba(0,120,0,0.8)', traj: '#1f3bff', grid: 'rgba(0,0,0,0.10)', surface: 'rgba(0,140,255,0.10)', surfaceEdge: 'rgba(0,120,220,0.55)' },
  dark: { bg: null, point: null, tracked: '#ff4d4f', kf: '#6d8cff', cam: '#82f2b8', graph: 'rgba(130,242,184,0.45)',
    tree: 'rgba(130,242,184,0.7)', traj: '#67e7ff', grid: 'rgba(80,120,160,0.28)', surface: 'rgba(101,230,255,0.10)', surfaceEdge: 'rgba(101,230,255,0.5)' },
};

function setStatus(text, error = false) { ui.status.textContent = text; ui.status.classList.toggle('error', error); }

// ---------------------------------------------------------------- cameras
async function loadCameras() {
  const selected = ui.camera.value;
  const options = [];
  let names = [];
  try { const r = await fetch('/api/slam/cameras'); if (r.ok) { const j = await r.json(); names = j.cameras || []; setBa(j.bundle_adjustment); } } catch (_) {}
  if (names.length) {
    for (const c of names) options.push([`native:${c.index}`, `Direct USB webcam ${c.index} — ${c.name}`]);
  } else {
    for (let i = 0; i < 3; i++) options.push([`native:${i}`, `Direct USB webcam ${i}`]);
  }
  options.push(['native:synthetic', 'Simulated room (no camera)'], ['', 'Browser camera']);
  try {
    const devices = (await navigator.mediaDevices?.enumerateDevices?.()) || [];
    for (const d of devices.filter((d) => d.kind === 'videoinput' && d.deviceId)) options.push([d.deviceId, `Browser: ${d.label || 'Camera'}`]);
  } catch (_) {}
  ui.camera.innerHTML = '';
  for (const [value, label] of options) { const o = document.createElement('option'); o.value = value; o.textContent = label; ui.camera.append(o); }
  const c270 = names.find((c) => /c270|logitech hd webcam/i.test(c.name));
  if ([...ui.camera.options].some((o) => o.value === selected) && selected !== 'native:0') ui.camera.value = selected;
  else if (c270) ui.camera.value = `native:${c270.index}`;
  ui.cameraHint.textContent = c270 ? `Found ${c270.name} on index ${c270.index}.`
    : names.length ? 'No C270 found by name. Pick it from the list, or plug it in and reload.'
    : 'Direct mode lets Python read the webcam; no browser permission needed.';
}
function setBa(on) { ui.ba.classList.toggle('on', !!on); ui.ba.classList.toggle('off', !on); ui.ba.title = on ? 'Local bundle adjustment is active' : 'SciPy missing: bundle adjustment disabled (pip install scipy)'; }

async function acquireCamera() {
  const base = { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30, max: 30 } };
  return navigator.mediaDevices.getUserMedia({ video: ui.camera.value ? { ...base, deviceId: { exact: ui.camera.value } } : base, audio: false });
}

// ------------------------------------------------------------------ socket
function onVideoFrame(buffer) {
  // decode newest JPEG off the main thread; if one is still decoding, keep only the latest
  if (state.decoding) { state.pendingFrame = buffer; return; }
  state.decoding = true;
  createImageBitmap(new Blob([buffer], { type: 'image/jpeg' })).then((bmp) => {
    state.bitmap?.close?.(); state.bitmap = bmp; tick(rate.video); requestLive();
  }).catch(() => {}).finally(() => {
    state.decoding = false;
    if (state.pendingFrame) { const next = state.pendingFrame; state.pendingFrame = null; onVideoFrame(next); }
  });
}

function connectSocket() {
  return new Promise((resolve, reject) => {
    let ready = false;
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const path = state.direct ? `/ws/slam/webcam?camera=${encodeURIComponent(state.cameraIndex)}` : '/ws/slam';
    const socket = new WebSocket(`${protocol}//${location.host}${path}`);
    state.socket = socket;
    socket.binaryType = 'arraybuffer';
    const timeout = setTimeout(() => { reject(new Error('Webcam connection timed out. Check that the Python server is running.')); socket.close(); }, 15000);
    socket.onerror = () => { clearTimeout(timeout); reject(new Error('Could not connect. Run Start-Webcam-SLAM.cmd and use the page it opens.')); };
    socket.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) { onVideoFrame(event.data); return; }
      const m = JSON.parse(event.data);
      if (m.type === 'ready') {
        ready = true; clearTimeout(timeout); state.sessionId = m.session_id;
        if (state.calibration) socket.send(JSON.stringify(state.calibration));
        if (state.direct) socket.send(JSON.stringify({ type: 'stream' }));
        ui.badge.classList.add('online'); ui.badge.querySelector('span').textContent = 'Mapping';
        resolve(); return;
      }
      if (m.type === 'map') {
        state.busy = false;
        updateMap(m);
        if (!state.direct) setTimeout(captureFrame, 0);
        return;
      }
      if (m.type === 'error') {
        state.busy = false;
        if (!ready || m.fatal) { clearTimeout(timeout); reject(new Error(m.error)); socket.close(); }
        setStatus(m.error, true);
        if (ready && !m.fatal && !state.direct) setTimeout(captureFrame, 250);
        return;
      }
      if (m.type === 'reset') clearMap();
    };
    socket.onclose = () => {
      clearTimeout(timeout); state.busy = false;
      if (!ready) reject(new Error('Camera connection failed. Close other camera apps and retry.'));
      if (state.running) { stopMapping(); setStatus('SLAM connection closed — restart mapping', true); }
    };
  });
}

async function startMapping() {
  ui.start.disabled = true; ui.camera.disabled = true; clearMap();
  state.direct = ui.camera.value.startsWith('native:');
  state.cameraIndex = state.direct ? ui.camera.value.split(':')[1] : '';
  ui.cameraName.textContent = ui.camera.selectedOptions[0]?.textContent.replace(/^Direct USB webcam \d+ — /, '').replace(/^Browser: /, '') || 'Camera';
  setStatus('Opening your webcam…');
  try {
    if (!state.direct) {
      state.stream = await acquireCamera();
      ui.video.srcObject = state.stream; await ui.video.play();
    }
    await connectSocket();
    state.running = true;
    ui.live.hidden = false; ui.videoEmpty.style.display = 'none';
    ui.start.disabled = false; ui.start.textContent = 'Stop mapping'; ui.reset.disabled = false;
    setStatus('Slide the camera sideways to initialize');
    if (!state.direct) { captureFrame(); browserVideoLoop(); }
  } catch (error) { stopMapping(); setStatus(error.message || 'Camera permission was denied', true); }
}

function stopMapping() {
  state.running = false;
  state.stream?.getTracks().forEach((t) => t.stop()); state.stream = null;
  state.socket?.close(); state.socket = null; state.busy = false;
  ui.video.srcObject = null; ui.live.hidden = true; ui.videoEmpty.style.display = 'grid';
  state.bitmap?.close?.(); state.bitmap = null;
  ui.start.textContent = 'Start mapping'; ui.start.disabled = false; ui.camera.disabled = false; ui.reset.disabled = true;
  ui.badge.classList.remove('online'); ui.badge.querySelector('span').textContent = 'Offline';
  setStatus(state.points.length ? 'Map paused' : 'Ready to map');
}

// browser-camera mode: the <video> element is the live feed; frames go to SLAM as fast as it keeps up
function browserVideoLoop() {
  if (!state.running || state.direct) return;
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    ui.video.requestVideoFrameCallback(() => { tick(rate.video); requestLive(); browserVideoLoop(); });
  } else { requestAnimationFrame(() => { tick(rate.video); requestLive(); browserVideoLoop(); }); }
}
function captureFrame() {
  if (!state.running || state.direct || state.busy || state.socket?.readyState !== WebSocket.OPEN) return;
  if (ui.video.readyState < 2) { setTimeout(captureFrame, 100); return; }
  const ctx = ui.capture.getContext('2d');
  ui.capture.width = 640; ui.capture.height = Math.round((640 * ui.video.videoHeight) / ui.video.videoWidth);
  ctx.drawImage(ui.video, 0, 0, ui.capture.width, ui.capture.height);
  state.busy = true;
  ui.capture.toBlob(async (blob) => {
    if (!blob || state.socket?.readyState !== WebSocket.OPEN) { state.busy = false; return; }
    state.socket.send(await blob.arrayBuffer());
  }, 'image/jpeg', 0.8);
}

// --------------------------------------------------------------- live view
let livePending = false;
function requestLive() { if (!livePending) { livePending = true; requestAnimationFrame(() => { livePending = false; renderLive(); }); } }
const PLANE_RGB = ['rgba(40,200,255,', 'rgba(255,170,60,', 'rgba(255,90,200,'];
function renderLive() {
  const src = state.direct ? state.bitmap : (ui.video.readyState >= 2 ? ui.video : null);
  if (!src) return;
  const w = state.direct ? src.width : ui.video.videoWidth, h = state.direct ? src.height : ui.video.videoHeight;
  const W = 640, H = Math.round((640 * h) / w);
  if (ui.live.width !== W || ui.live.height !== H) { ui.live.width = W; ui.live.height = H; }
  const ctx = ui.live.getContext('2d');
  ctx.drawImage(src, 0, 0, W, H);
  if (state.viewMode === 'features') drawFeatures(ctx, W, H);
  else if (state.viewMode === 'ar') drawAR(ctx, W, H);
}
function drawFeatures(ctx, W, H) {
  ctx.lineWidth = 1; ctx.strokeStyle = '#00ff00'; ctx.fillStyle = '#00ff00';
  if (state.slamState === 'TRACKING') {
    ctx.beginPath();
    for (const [x, y] of state.features2d) { ctx.rect(x - 5 + 0.5, y - 5 + 0.5, 10, 10); }
    ctx.stroke();
    for (const [x, y] of state.features2d) ctx.fillRect(x - 1, y - 1, 3, 3);
  } else if (state.initTracks) {
    ctx.beginPath(); for (const [ax, ay, bx, by] of state.initTracks) { ctx.moveTo(ax, ay); ctx.lineTo(bx, by); } ctx.stroke();
  }
  const text = state.slamState === 'TRACKING' ? `SLAM MODE |  KFs: ${state.kfCount}, MPs: ${state.mpCount}, Matches: ${state.matchCount}`
    : state.slamState === 'INITIALIZING' ? 'TRYING TO INITIALIZE' : state.slamState === 'LOST' ? 'TRACK LOST. TRYING TO RELOCALIZE' : 'WAITING FOR IMAGES';
  ctx.fillStyle = 'rgba(0,0,0,0.85)'; ctx.fillRect(0, H - 22, W, 22);
  ctx.fillStyle = '#fff'; ctx.font = '13px monospace'; ctx.fillText(text, 6, H - 7);
}
function projector() {
  const c = state.camera, K = state.intr;
  if (!c || !K) return null;
  const [fx, fy, cx, cy, iw] = K, s = 640 / iw, R = c.rotation, C = c.center;
  return (X) => {
    const dx = X[0] - C[0], dy = X[1] - C[1], dz = X[2] - C[2];
    const xc = R[0] * dx + R[3] * dy + R[6] * dz, yc = R[1] * dx + R[4] * dy + R[7] * dz, zc = R[2] * dx + R[5] * dy + R[8] * dz;
    return [(fx * xc / zc + cx) * s, (fy * yc / zc + cy) * s, zc, xc / zc, yc / zc];
  };
}
function arPolyline(ctx, P, pts, segments = 16) {
  const near = 0.02 * (state.medianDepth || 1);
  let started = false;
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i], b = pts[i + 1];
    for (let k = 0; k <= segments; k++) {
      if (k === segments && i < pts.length - 2) continue;
      const t = k / segments, q = P([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]);
      if (q[2] < near || Math.abs(q[3]) > 1.5 || Math.abs(q[4]) > 1.3) { started = false; continue; }
      if (!started) { ctx.moveTo(q[0], q[1]); started = true; } else ctx.lineTo(q[0], q[1]);
    }
  }
}
function drawAR(ctx, W, H) {
  const P = projector();
  if (!P || state.slamState !== 'TRACKING') { drawFeatures(ctx, W, H); return; }
  const md = state.medianDepth || 1;
  state.planes.forEach((pl, k) => {                                // translucent surfaces + outline
    const q = pl.hull.map(P);
    if (q.every((p) => p[2] > 0.02 * md)) {
      ctx.beginPath(); q.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.closePath();
      ctx.fillStyle = PLANE_RGB[k % 3] + '0.22)'; ctx.fill();
    }
    ctx.beginPath(); arPolyline(ctx, P, [...pl.hull, pl.hull[0]], 6); ctx.strokeStyle = PLANE_RGB[k % 3] + '0.95)'; ctx.lineWidth = 2; ctx.stroke();
  });
  const pl = state.planes[0];
  if (pl && pl.u && opts.grid) {                                    // grid on the dominant surface
    const [u0, u1, v0, v1] = pl.extent, s = pl.spacing, o = pl.center;
    const at = (a, b) => [o[0] + a * pl.u[0] + b * pl.v[0], o[1] + a * pl.u[1] + b * pl.v[1], o[2] + a * pl.u[2] + b * pl.v[2]];
    ctx.beginPath();
    for (let a = u0; a <= u1 + 1e-9; a += s) arPolyline(ctx, P, [at(a, v0), at(a, v1)]);
    for (let b = v0; b <= v1 + 1e-9; b += s) arPolyline(ctx, P, [at(u0, b), at(u1, b)]);
    ctx.strokeStyle = 'rgba(20,20,20,0.6)'; ctx.lineWidth = 2.5; ctx.stroke(); ctx.strokeStyle = PLANE_RGB[0] + '0.9)'; ctx.lineWidth = 1; ctx.stroke();
  }
  if (opts.keyframes) {                                             // keyframe frusta
    ctx.beginPath(); const sz = 0.08 * md;
    for (const kf of state.keyframes.slice(-40)) {
      const C = kf.slice(0, 3), R = kf.slice(3);
      const cw = (x, y, z) => [C[0] + R[0] * x + R[1] * y + R[2] * z, C[1] + R[3] * x + R[4] * y + R[5] * z, C[2] + R[6] * x + R[7] * y + R[8] * z];
      const c = [cw(-sz, -0.75 * sz, sz), cw(sz, -0.75 * sz, sz), cw(sz, 0.75 * sz, sz), cw(-sz, 0.75 * sz, sz)];
      for (const q of c) arPolyline(ctx, P, [C, q], 4);
      arPolyline(ctx, P, [...c, c[0]], 4);
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.8)'; ctx.lineWidth = 1; ctx.stroke();
  }
  if (state.trajectory.length > 1) { ctx.beginPath(); arPolyline(ctx, P, state.trajectory.slice(-600), 1); ctx.strokeStyle = '#ffd400'; ctx.lineWidth = 2; ctx.stroke(); }
  if (opts.points) {                                                // map points coloured by depth
    for (const X of state.points) {
      const q = P(X); if (q[2] <= 0 || q[0] < 0 || q[1] < 0 || q[0] >= W || q[1] >= H) continue;
      const t = Math.max(0, Math.min(1, Math.log(q[2] / (0.3 * md)) / Math.log(8)));
      ctx.fillStyle = `hsl(${Math.round(t * 230)},100%,55%)`; ctx.fillRect(q[0] - 1.5, q[1] - 1.5, 3, 3);
    }
  }
  ctx.fillStyle = 'rgba(0,0,0,0.6)'; ctx.fillRect(0, 0, 250, 20); ctx.fillStyle = '#7CFC00'; ctx.font = '12px monospace';
  ctx.fillText(`AR · ${state.mpCount} pts · ${state.planes.length} surfaces`, 6, 14);
}

// --------------------------------------------------------------- map state
function updateMap(m) {
  tick(rate.slam);
  if (m.points) {
    state.points = m.points; state.colors = m.colors || []; state.keyframes = m.keyframes || [];
    state.graph = m.graph || []; state.spanning = m.spanning_tree || []; state.planes = m.planes || [];
    state.medianDepth = m.median_depth || state.medianDepth;
    if (!opts.follow && !view.manual && state.points.length > 15) fitView(false);
  }
  state.tracked = m.tracked || []; state.trajectory = m.trajectory || []; state.camera = m.camera || null;
  state.features2d = m.features2d || []; state.initTracks = m.init_tracks || null; state.slamState = m.state;
  state.kfCount = m.keyframe_count ?? 0; state.mpCount = m.landmarks ?? 0; state.matchCount = m.inliers ?? 0;
  if (m.intrinsics) state.intr = m.intrinsics;
  if (m.costmap) updateCostmap(m.costmap);
  ui.landmarks.textContent = (m.landmarks ?? 0).toLocaleString();
  ui.keyframes.textContent = m.keyframe_count ?? 0;
  ui.matches.textContent = m.matches ?? 0; ui.inliers.textContent = m.inliers ?? 0;
  ui.procMs.textContent = `${Math.round(m.proc_ms || 0)} ms`; ui.frames.textContent = m.frame ?? 0;
  if ('bundle_adjustment' in m) setBa(m.bundle_adjustment);
  setStatus(m.status, m.state === 'LOST');
  ui.mapEmpty.classList.toggle('hidden', (m.landmarks || 0) > 0);
  ui.export.disabled = !m.landmarks; ui.trajectory.disabled = (m.trajectory || []).length < 2; ui.costmapBtn.disabled = !cm.image;
  if (opts.follow && state.camera) followCamera();
  requestDraw();
  if (!state.direct) requestLive();
}
setInterval(() => {
  ui.videoFps.textContent = `video ${fpsOf(rate.video).toFixed(0)} fps`;
  ui.slamFps.textContent = `SLAM ${fpsOf(rate.slam).toFixed(1)} fps`;
  ui.latency.textContent = state.running ? `${ui.procMs.textContent} / frame` : '—';
}, 500);

function clearMap() {
  Object.assign(state, { points: [], colors: [], keyframes: [], graph: [], spanning: [], planes: [], tracked: [], trajectory: [], camera: null,
    features2d: [], initTracks: null, slamState: 'WAITING' });
  for (const k of ['landmarks', 'keyframes', 'matches', 'inliers', 'frames']) ui[k].textContent = '0';
  ui.mapEmpty.classList.remove('hidden'); ui.export.disabled = true; ui.trajectory.disabled = true; ui.costmapBtn.disabled = true;
  view.manual = false; cm.image = null; cm.cost = null; cm.goal = null; cm.plan = null; ui.clearGoal.disabled = true;
  ui.cmEmpty.style.display = 'grid'; ui.cmInfo.textContent = '—'; drawCostmap(); requestDraw();
}

// ------------------------------------------------------------- 3D viewer
function bounds(list) {
  const xs = list.map((p) => p[0]), ys = list.map((p) => -p[1]), zs = list.map((p) => p[2]);
  const pct = (a, q) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.max(0, Math.floor(q * (s.length - 1))))]; };
  return [[pct(xs, 0.03), pct(xs, 0.97)], [pct(ys, 0.03), pct(ys, 0.97)], [pct(zs, 0.03), pct(zs, 0.97)]];
}
function fitView(draw = true) {
  const all = state.points.length ? state.points : state.trajectory;
  if (!all.length) return;
  const b = bounds(all);
  view.target = b.map(([lo, hi]) => (lo + hi) / 2);
  view.distance = Math.max(b[0][1] - b[0][0], b[1][1] - b[1][0], b[2][1] - b[2][0], 0.15) * 2.0;
  if (draw) requestDraw();
}
function followCamera() {
  // look at a point in front of the camera so the camera and what it sees are both in view
  const c = state.camera.center, r = state.camera.rotation, md = state.medianDepth || 1;
  const ahead = [c[0] + r[2] * md * 0.6, c[1] + r[5] * md * 0.6, c[2] + r[8] * md * 0.6];
  const k = 0.25; // smooth follow
  view.target = [view.target[0] + (ahead[0] - view.target[0]) * k, view.target[1] + (-ahead[1] - view.target[1]) * k, view.target[2] + (ahead[2] - view.target[2]) * k];
  if (!view.manual) view.distance = Math.max(0.5, md * 1.7);
}
function transform(p, w, h) {
  const x = p[0] - view.target[0], y = -p[1] - view.target[1], z = p[2] - view.target[2];
  const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw), cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
  const x1 = cy * x - sy * z, z1 = sy * x + cy * z, y1 = cp * y - sp * z1, z2 = sp * y + cp * z1 + view.distance;
  if (z2 < 0.02) return null;
  const f = Math.min(w, h) * 0.95;
  return [w / 2 + (x1 * f) / z2, h / 2 - (y1 * f) / z2, z2];
}
function seg(ctx, a, b, w, h) { const pa = transform(a, w, h), pb = transform(b, w, h); if (!pa || !pb) return; ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); }
function frustum(ctx, center, rot, size, w, h) {
  // rot: camera->world rotation, row-major 3x3. Camera looks along +z.
  const hw = size, hh = size * 0.75, d = size * 0.8;
  const cw = (x, y, z) => [center[0] + rot[0] * x + rot[1] * y + rot[2] * z, center[1] + rot[3] * x + rot[4] * y + rot[5] * z, center[2] + rot[6] * x + rot[7] * y + rot[8] * z];
  const c = [cw(-hw, -hh, d), cw(hw, -hh, d), cw(hw, hh, d), cw(-hw, hh, d)];
  for (const q of c) seg(ctx, center, q, w, h);
  for (let i = 0; i < 4; i++) seg(ctx, c[i], c[(i + 1) % 4], w, h);
}
let drawPending = false;
function requestDraw() { if (!drawPending) { drawPending = true; requestAnimationFrame(() => { drawPending = false; draw(); }); } }
function draw() {
  const rect = ui.canvas.getBoundingClientRect(), ratio = devicePixelRatio || 1;
  const w = Math.max(1, Math.round(rect.width)), h = Math.max(1, Math.round(rect.height));
  if (ui.canvas.width !== Math.round(w * ratio) || ui.canvas.height !== Math.round(h * ratio)) { ui.canvas.width = Math.round(w * ratio); ui.canvas.height = Math.round(h * ratio); }
  const ctx = ui.canvas.getContext('2d'), T = THEMES[opts.theme];
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, w, h);
  if (T.bg) { ctx.fillStyle = T.bg; ctx.fillRect(0, 0, w, h); }
  const scale = state.medianDepth || 1;
  // ground grid (world y = floor-ish below first camera)
  if (opts.grid) {
    const span = Math.max(0.5, view.distance * 0.7), step = span / 6, gy = scale * 0.5;
    ctx.strokeStyle = T.grid; ctx.lineWidth = 0.8; ctx.beginPath();
    for (let i = -6; i <= 6; i++) {
      seg(ctx, [view.target[0] + i * step, gy, view.target[2] - span], [view.target[0] + i * step, gy, view.target[2] + span], w, h);
      seg(ctx, [view.target[0] - span, gy, view.target[2] + i * step], [view.target[0] + span, gy, view.target[2] + i * step], w, h);
    }
    ctx.stroke();
  }
  // surfaces
  if (opts.surfaces) {
    for (const pl of state.planes) {
      const pr = pl.hull.map((p) => transform(p, w, h));
      if (pr.some((p) => !p)) continue;
      ctx.beginPath(); pr.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.closePath();
      ctx.fillStyle = T.surface; ctx.fill(); ctx.strokeStyle = T.surfaceEdge; ctx.lineWidth = 1; ctx.stroke();
    }
  }
  // map points
  if (opts.points) {
    for (let i = 0; i < state.points.length; i++) {
      const p = transform(state.points[i], w, h); if (!p) continue;
      const s = Math.max(1.2, Math.min(3, (view.distance / p[2]) * 1.8));
      if (T.point) { ctx.fillStyle = 'rgb(0,0,0)'; }
      else { const c = state.colors[i] || [164, 215, 235], fade = Math.max(0.35, Math.min(1, view.distance / p[2])); ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${fade})`; }
      ctx.fillRect(p[0] - s / 2, p[1] - s / 2, s, s);
    }
    ctx.fillStyle = T.tracked;
    for (const q of state.tracked) { const p = transform(q, w, h); if (!p) continue; const s = Math.max(1.8, Math.min(3.5, (view.distance / p[2]) * 2.2)); ctx.fillRect(p[0] - s / 2, p[1] - s / 2, s, s); }
  }
  const kfSize = 0.09 * scale;
  if (opts.graph && state.keyframes.length > 1) {
    const kc = (i) => state.keyframes[i].slice(0, 3);
    ctx.lineWidth = 1; ctx.strokeStyle = T.graph; ctx.beginPath();
    for (const [a, b] of state.graph) if (state.keyframes[a] && state.keyframes[b]) seg(ctx, kc(a), kc(b), w, h);
    ctx.stroke();
    ctx.strokeStyle = T.tree; ctx.lineWidth = 1.4; ctx.beginPath();
    for (const [a, b] of state.spanning) if (state.keyframes[a] && state.keyframes[b]) seg(ctx, kc(a), kc(b), w, h);
    ctx.stroke();
  }
  if (opts.keyframes) {
    ctx.strokeStyle = T.kf; ctx.lineWidth = 1; ctx.beginPath();
    for (const k of state.keyframes) frustum(ctx, k.slice(0, 3), k.slice(3), kfSize, w, h);
    ctx.stroke();
  }
  if (opts.trajectory && state.trajectory.length > 1) {
    ctx.strokeStyle = T.traj; ctx.lineWidth = 1.5; ctx.beginPath(); let started = false;
    for (const q of state.trajectory) { const p = transform(q, w, h); if (!p) { started = false; continue; } if (!started) { ctx.moveTo(p[0], p[1]); started = true; } else ctx.lineTo(p[0], p[1]); }
    ctx.stroke();
  }
  if (state.camera) {
    ctx.strokeStyle = T.cam; ctx.lineWidth = 2; ctx.beginPath(); frustum(ctx, state.camera.center, state.camera.rotation, kfSize * 1.3, w, h); ctx.stroke();
  }
}

// ---------------------------------------------------------------- costmap
const cm = { image: null, cost: null, w: 0, h: 0, version: -1, path: [], pose: null, goal: null, plan: null, planAt: 0, info: null };
const LUT = (() => {
  const lut = new Uint8ClampedArray(256 * 4);
  const set = (i, r, g, b, a = 255) => { lut.set([r, g, b, a], i * 4); };
  set(0, 205, 208, 208);                                        // free
  for (let c = 1; c <= 252; c++) {                              // inflation: cyan (low) -> magenta (high)
    const t = Math.pow(c / 252, 0.8);
    set(c, Math.round(255 * t), Math.round(255 * (1 - t)), 255);
  }
  set(253, 255, 0, 255); set(254, 0, 0, 0); set(255, 77, 106, 102); // inscribed magenta, lethal black, unknown
  return lut;
})();
function updateCostmap(info) {
  cm.path = info.path || []; cm.pose = info.pose || null; cm.info = info;
  ui.cmInfo.textContent = `${info.width}×${info.height} · ${info.floor ? 'floor found' : 'no floor yet'}`;
  if (info.png && info.version !== cm.version) {
    cm.version = info.version;
    const img = new Image();
    img.onload = () => {
      const off = document.createElement('canvas'); off.width = img.width; off.height = img.height;
      const octx = off.getContext('2d'); octx.drawImage(img, 0, 0);
      const src = octx.getImageData(0, 0, img.width, img.height).data;
      cm.w = img.width; cm.h = img.height; cm.cost = new Uint8Array(cm.w * cm.h);
      const out = octx.createImageData(cm.w, cm.h);
      for (let i = 0; i < cm.cost.length; i++) { const c = src[i * 4]; cm.cost[i] = c; out.data.set(LUT.subarray(c * 4, c * 4 + 4), i * 4); }
      octx.putImageData(out, 0, 0); cm.image = off; ui.cmEmpty.style.display = 'none'; ui.costmapBtn.disabled = false;
      cm.planAt = 0; drawCostmap();
    };
    img.src = `data:image/png;base64,${info.png}`;
  } else drawCostmap();
}
function cmLayout() {
  const rect = ui.cm.getBoundingClientRect(), ratio = devicePixelRatio || 1;
  const W = Math.max(1, Math.round(rect.width)), H = Math.max(1, Math.round(rect.height));
  if (ui.cm.width !== Math.round(W * ratio) || ui.cm.height !== Math.round(H * ratio)) { ui.cm.width = Math.round(W * ratio); ui.cm.height = Math.round(H * ratio); }
  const s = cm.w ? Math.min(W / cm.w, H / cm.h) : 1;
  return { W, H, ratio, s, ox: (W - cm.w * s) / 2, oy: (H - cm.h * s) / 2 };
}
function drawCostmap() {
  const L = cmLayout(), ctx = ui.cm.getContext('2d');
  ctx.setTransform(L.ratio, 0, 0, L.ratio, 0, 0); ctx.fillStyle = '#4d6a66'; ctx.fillRect(0, 0, L.W, L.H);
  if (!cm.image) return;
  ctx.imageSmoothingEnabled = false; ctx.drawImage(cm.image, L.ox, L.oy, cm.w * L.s, cm.h * L.s);
  const P = (p) => [L.ox + (p[0] + 0.5) * L.s, L.oy + (p[1] + 0.5) * L.s];
  if (cm.path.length > 1) {
    ctx.strokeStyle = 'rgba(40,60,200,0.85)'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]); ctx.beginPath();
    cm.path.forEach((p, i) => { const q = P(p); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); }); ctx.stroke(); ctx.setLineDash([]);
  }
  maybePlan();
  if (cm.plan && cm.plan.length > 1) {
    ctx.strokeStyle = '#00e640'; ctx.lineWidth = 3; ctx.lineJoin = 'round'; ctx.beginPath();
    cm.plan.forEach((p, i) => { const q = P(p); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); }); ctx.stroke();
  }
  if (cm.goal) {
    const g = P(cm.goal); ctx.fillStyle = '#ff2d2d'; ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(g[0], g[1], 6, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  }
  if (cm.pose) {
    const [x, y] = P(cm.pose), a = cm.pose[2], r = Math.max(6, 2.2 * L.s);
    ctx.fillStyle = 'rgba(255,255,255,0.92)'; ctx.strokeStyle = '#111'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.strokeStyle = '#e0161b'; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + Math.cos(a) * r * 1.9, y + Math.sin(a) * r * 1.9); ctx.stroke();
  }
}
function maybePlan() {
  if (!cm.goal || !cm.cost || !cm.pose) { cm.plan = null; return; }
  const now = performance.now(); if (now - cm.planAt < 400) return; cm.planAt = now;
  cm.plan = astar([Math.round(cm.pose[0]), Math.round(cm.pose[1])], cm.goal);
  ui.goalInfo.textContent = cm.plan ? `Plan: ${cm.plan.length} cells to the goal` : 'No path: the goal is blocked by obstacles';
}
function astar(start, goal) {
  const W = cm.w, H = cm.h, cost = cm.cost;
  const inside = (x, y) => x >= 0 && y >= 0 && x < W && y < H;
  if (!inside(start[0], start[1]) || !inside(goal[0], goal[1])) return null;
  const idx = (x, y) => y * W + x, gi = idx(goal[0], goal[1]), si = idx(start[0], start[1]);
  const passable = (i) => cost[i] < 253 || i === si;           // allow_unknown like Nav2's default
  if (!passable(gi)) return null;
  const step = (i) => 1 + (cost[i] === 255 ? 2.5 : cost[i] / 45);
  const g = new Float32Array(W * H).fill(Infinity), came = new Int32Array(W * H).fill(-1), closed = new Uint8Array(W * H);
  const heap = []; const push = (f, i) => { heap.push([f, i]); let k = heap.length - 1; while (k > 0) { const p = (k - 1) >> 1; if (heap[p][0] <= heap[k][0]) break; [heap[p], heap[k]] = [heap[k], heap[p]]; k = p; } };
  const pop = () => { const top = heap[0], last = heap.pop(); if (heap.length) { heap[0] = last; let k = 0; for (;;) { const l = 2 * k + 1, r = l + 1; let m = k; if (l < heap.length && heap[l][0] < heap[m][0]) m = l; if (r < heap.length && heap[r][0] < heap[m][0]) m = r; if (m === k) break; [heap[m], heap[k]] = [heap[k], heap[m]]; k = m; } } return top; };
  const hfn = (x, y) => Math.hypot(x - goal[0], y - goal[1]);
  g[si] = 0; push(hfn(start[0], start[1]), si);
  const N = [[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1], [1, 1, Math.SQRT2], [1, -1, Math.SQRT2], [-1, 1, Math.SQRT2], [-1, -1, Math.SQRT2]];
  let guard = 0;
  while (heap.length && guard++ < 400000) {
    const [, i] = pop(); if (closed[i]) continue; closed[i] = 1;
    if (i === gi) { const out = []; for (let k = gi; k !== -1; k = came[k]) out.push([k % W, Math.floor(k / W)]); return out.reverse(); }
    const x = i % W, y = Math.floor(i / W);
    for (const [dx, dy, d] of N) {
      const nx = x + dx, ny = y + dy; if (!inside(nx, ny)) continue; const j = idx(nx, ny);
      if (closed[j] || !passable(j)) continue;
      const ng = g[i] + d * step(j); if (ng < g[j]) { g[j] = ng; came[j] = i; push(ng + hfn(nx, ny), j); }
    }
  }
  return null;
}
ui.cm.addEventListener('click', (e) => {
  if (!cm.image) return;
  const L = cmLayout(), r = ui.cm.getBoundingClientRect();
  const x = Math.floor((e.clientX - r.left - L.ox) / L.s), y = Math.floor((e.clientY - r.top - L.oy) / L.s);
  if (x < 0 || y < 0 || x >= cm.w || y >= cm.h) return;
  cm.goal = [x, y]; cm.planAt = 0; ui.clearGoal.disabled = false; drawCostmap();
});
ui.clearGoal.addEventListener('click', () => { cm.goal = null; cm.plan = null; ui.clearGoal.disabled = true; ui.goalInfo.textContent = 'Click the map to set a goal. A* plans around the inflated obstacles.'; drawCostmap(); });

// ---------------------------------------------------------------- controls
ui.start.addEventListener('click', () => (state.running ? stopMapping() : startMapping()));
ui.reset.addEventListener('click', () => { if (state.socket?.readyState === WebSocket.OPEN) state.socket.send(JSON.stringify({ type: 'reset' })); clearMap(); setStatus('Slide the camera sideways to initialize'); });
ui.fit.addEventListener('click', () => { view.manual = false; setOpt('follow', false); fitView(); });
ui.canvas.addEventListener('dblclick', () => { view.manual = false; setOpt('follow', false); fitView(); });
const download = (kind) => { if (state.sessionId) location.href = `/api/slam/${encodeURIComponent(state.sessionId)}/${kind}`; };
ui.export.addEventListener('click', () => download('export'));
ui.trajectory.addEventListener('click', () => download('trajectory'));
ui.costmapBtn.addEventListener('click', () => download('costmap'));
ui.calibration.addEventListener('change', async () => {
  const file = ui.calibration.files[0]; if (!file) return;
  try {
    const value = JSON.parse(await file.text());
    for (const key of ['width', 'height', 'fx', 'fy', 'cx', 'cy']) if (!Number.isFinite(Number(value[key]))) throw new Error(`Missing numeric ${key}`);
    state.calibration = { type: 'calibration', ...value };
    ui.calibrationLabel.textContent = `Loaded ${file.name} (${value.width} × ${value.height}, fx ${Number(value.fx).toFixed(0)})`;
    ui.calibrationLabel.classList.remove('error');
    if (state.socket?.readyState === WebSocket.OPEN) { state.socket.send(JSON.stringify(state.calibration)); clearMap(); }
  } catch (error) { ui.calibrationLabel.textContent = `Invalid calibration: ${error.message}`; ui.calibrationLabel.classList.add('error'); }
});
function setOpt(name, value) { opts[name] = value; const box = document.querySelector(`[data-opt="${name}"]`); if (box) box.checked = value; requestDraw(); }
document.querySelectorAll('[data-opt]').forEach((box) => box.addEventListener('change', () => { opts[box.dataset.opt] = box.checked; if (box.dataset.opt === 'follow' && box.checked) view.manual = false; requestDraw(); }));
$('#topViewButton').addEventListener('click', () => { view.pitch = -1.45; view.yaw = 0; setOpt('follow', false); fitView(); });
$('#sideViewButton').addEventListener('click', () => { view.pitch = -0.38; view.yaw = -0.72; view.manual = false; setOpt('follow', true); });
document.querySelectorAll('#themeToggle button').forEach((b) => b.addEventListener('click', () => {
  opts.theme = b.dataset.theme; document.querySelectorAll('#themeToggle button').forEach((x) => x.classList.toggle('on', x === b));
  ui.frame.classList.toggle('light', opts.theme === 'light'); requestDraw();
}));
document.querySelectorAll('#viewToggle button').forEach((b) => b.addEventListener('click', () => {
  state.viewMode = b.dataset.view; document.querySelectorAll('#viewToggle button').forEach((x) => x.classList.toggle('on', x === b));
  requestLive();
}));
ui.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
ui.canvas.addEventListener('pointerdown', (e) => { view.dragging = true; view.panning = e.button === 2 || e.shiftKey; view.lastX = e.clientX; view.lastY = e.clientY; ui.canvas.setPointerCapture(e.pointerId); });
ui.canvas.addEventListener('pointermove', (e) => {
  if (!view.dragging) return;
  const dx = e.clientX - view.lastX, dy = e.clientY - view.lastY; view.lastX = e.clientX; view.lastY = e.clientY;
  if (view.panning) {
    const k = view.distance / Math.min(ui.canvas.clientWidth, ui.canvas.clientHeight), cy = Math.cos(view.yaw), sy = Math.sin(view.yaw);
    view.target = [view.target[0] - dx * k * cy, view.target[1] + dy * k, view.target[2] + dx * k * sy]; setOpt('follow', false);
  } else { view.yaw += dx * 0.008; view.pitch = Math.max(-1.5, Math.min(1.5, view.pitch + dy * 0.008)); }
  view.manual = true; requestDraw();
});
ui.canvas.addEventListener('pointerup', () => { view.dragging = false; });
ui.canvas.addEventListener('wheel', (e) => { e.preventDefault(); view.distance = Math.max(0.05, view.distance * Math.exp(e.deltaY * 0.001)); view.manual = true; requestDraw(); }, { passive: false });
window.addEventListener('resize', () => { requestDraw(); drawCostmap(); });
window.addEventListener('beforeunload', () => state.stream?.getTracks().forEach((t) => t.stop()));
navigator.mediaDevices?.addEventListener?.('devicechange', loadCameras);
loadCameras(); requestDraw(); drawCostmap();
