# C270 visual SLAM + HyperionMind

This repository includes a fully local monocular visual-SLAM workspace for the Logitech C270 (**C270 Map Lab**) and HyperionMind, a local-first social vision website.

## C270 Map Lab

On this Windows computer, double-click **Start-Webcam-SLAM.cmd**, or in VS Code run **Terminal → Run Task → Start Webcam SLAM**. It starts a local server on port 8765 and opens <http://127.0.0.1:8765/slam>. Pick your C270 in **Video source**; the page lists webcams by name and selects the C270 automatically. Then click **Start mapping** and **slide the camera sideways** 10–20 cm so the map can initialize. **Simulated room (no camera)** lets you try everything without hardware.

The page has three live views:

- **Live 3D map (ORB-SLAM3 view).** Map points are black, and points tracked in the current frame are red. Keyframes are blue frusta, the current camera is green, the covisibility graph is drawn as green lines, and fitted surfaces are shaded. The Pangolin-style menu toggles Follow Camera, Points, KeyFrames, Graph, Trajectory, Surfaces and Grid, plus a Top View. There are Light and Dark backgrounds. Drag to orbit, right-drag to pan, use the wheel to zoom, and double-click to fit.
- **Camera input.** *Features* is the ORB-SLAM3 frame view: green boxes on matched map points and the `SLAM MODE | KFs, MPs, Matches` bar. *AR overlay* draws the reconstructed surfaces, grid, map points and path on top of the live feed. *Raw feed* shows the camera image untouched.
- **2D costmap (Nav2 style).** A top-down occupancy grid built from the map:
  - Unknown areas are dark teal, free space is grey and obstacles are black.
  - Obstacles have a magenta-to-cyan inflation halo.
  - The camera's path is a dashed blue line, and its current pose is a white disc with a heading line.
  - **Click anywhere to set a goal.** A* then plans a green path around the inflated obstacles.

Exports: the point cloud (`.ply`), the trajectory (`.csv`) and the costmap (`.png`, with ROS costmap values: 0 free, 1–252 inflated, 253 inscribed, 254 lethal, 255 unknown). Copies are also written to `slam_output/`.

How the engine works (`app/slamkit/`):

1. **Features:** ORB features spread over an image grid.
2. **Initialization:** two-view initialization that is only accepted with ≥2° of parallax. With the C270's narrow field of view, smaller baselines confuse rotation and translation.
3. **Tracking:** motion-model guided matching, then EPnP RANSAC and Levenberg–Marquardt refinement. If tracking is lost, the engine relocalizes.
4. **Mapping:** keyframes are added as the camera moves. New points are triangulated with epipolar and reprojection checks against neighbouring keyframes.
5. **Bundle adjustment:** local bundle adjustment runs on a background thread and needs SciPy. If SciPy is missing, the badge in the header dims.
6. **Surfaces and floor:** dominant surfaces are fitted with RANSAC. The floor that the costmap uses comes from a fitted surface, or from a dense low layer of points.

On a simulated room the engine tracks about 97 % of frames, with trajectory error around 2 % of the distance travelled. Real results depend on light and texture.

Manual setup:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python tools\run_webcam_slam.py
```

For the best map:

- Use a well-lit area with textured, non-reflective objects.
- Begin pointed at a textured scene, then move the camera sideways by roughly 10–20 cm.
- Move slowly and smoothly. Pure rotation cannot triangulate new 3D points.
- Keep already mapped objects visible as you move into a new part of the scene.
- Avoid people, screens, and other moving objects while mapping.

### Calibrate the C270

The included default intrinsics are an approximation. A one-time calibration substantially improves the geometry because focal length and distortion vary between cameras and focus settings.

Print a checkerboard with 9 × 6 **inner corners**, measure one square, and run:

```powershell
python tools\calibrate_c270.py --camera 0 --columns 9 --rows 6 --square-mm 25
```

Show the board at different positions and angles. Press Space to accept each good view, `R` to undo, and `Q` to stop. After at least eight views, the script writes `c270-calibration.json`. Load that file from the SLAM page (section 02) before mapping. If the C270 is not camera index 0, try `--camera 1` or `--camera 2`.

### What one camera can and cannot do

This is a lightweight feature SLAM implementation: ORB feature matching bootstraps relative pose, landmarks are triangulated into a persistent map, and later frames use 2D/3D PnP pose estimation. It is intended for local experiments and robotics prototyping, not safety-critical navigation.

A monocular camera has no independent way to determine absolute size. The exported shape and trajectory are therefore in **arbitrary scale**, not meters. Metric scale requires a known-size visual target, wheel/IMU data, a depth camera, or a stereo pair. Textureless walls, motion blur, low light, repeated patterns, and moving scenes can also cause tracking loss.

## Automated tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

## HyperionMind

HyperionMind is a local-first social vision website. It counts people in a webcam view, detects a raised hand, assigns an anonymous on-screen identity, and keeps a separate voice conversation for each person.

The lightweight pose model warms up in the background at server startup, so the first camera frame is ready sooner. Wait for the startup warm-up to finish before opening the camera for the fastest first detection.

## Features

- Processes browser camera frames with OpenCV.
- Uses a lightweight YOLO pose model to find people and compare wrist/shoulder positions.
- Stabilizes anonymous identities (`Person 1`, `Person 2`, and so on) using position, size, and upper/lower clothing appearance.
- Automatically greets the longest-waiting raised hand.
- Mirrors the camera like a natural front-facing mirror while keeping labels readable.
- Uses local voice-activity detection, OpenAI transcription, the Responses API, and OpenAI speech generation.
- Locks microphone and typed input while a greeting or response is being processed or spoken.
- Keeps API keys and per-person memory in server RAM only.
- Offers typed messages when the room is noisy.

## Requirements

- Python 3.11 or newer
- A webcam and microphone
- Chrome or Edge recommended
- An OpenAI API key with access to the configured models
- Internet access on the first run so Ultralytics can download `yolo11n-pose.pt`

## Run HyperionMind locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>, enter the API key, choose **Connect**, and then choose **Start camera**. Stand far enough back for shoulders and wrists to be visible. Raise one hand for several frames to begin.

To test without an API key, choose **Try demo without an API key**. Detection, raised-hand activation, speaker switching, per-person memory, typed conversation, and spoken conversation remain available. Demo replies are generated locally, and voice input/output uses browser speech features when supported. Chrome or Edge provides the best demo voice support.

If the pose model cannot load, the app falls back to OpenCV's HOG person detector. The fallback can count standing people but cannot detect raised hands.

## Configuration

The website always asks for the API key at runtime. Other defaults can be overridden with environment variables shown in `.env.example`:

- Conversation: `gpt-5.6-luna`
- Transcription: `gpt-4o-mini-transcribe`
- Speech: `tts-1` (optimized for response speed)
- Pose: `yolo11n-pose.pt`

## Privacy and deployment

This prototype does not save camera frames, microphone recordings, transcripts, face images, or API keys. Conversations and keys disappear when the server process or four-hour session ends. Identities are anonymous visual tracks, not verified identities or face recognition.

Do not expose the development server directly to the internet. For deployment, put it behind HTTPS, restrict allowed origins, add authentication and rate limits, and isolate session storage. Browsers normally permit camera and microphone access only on HTTPS or `localhost`.

## Current limitations

- Tracking is designed for a small room. It can recover an identity for up to two minutes, including across a camera restart in the same session. Similar clothing, major clothing changes, long occlusion, or poor lighting can still confuse anonymous re-identification.
- The first fraction of a word can be clipped because recording begins after voice activity crosses the threshold.
- Demo mode has a deliberately small set of local conversational responses; use an API key for open-ended conversation.
- Echo cancellation depends on the browser and audio hardware. Headphones give the cleanest results.
- A raised hand must be visible above its matching shoulder for three processed frames.
