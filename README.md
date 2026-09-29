# MedPal 2026

Assistive medication-delivery robot for the Raspberry Pi 5 + Coral Edge TPU.

Follows a registered person, avoids obstacles, dispenses medication on
touch or RFID, and provides a browser-based control panel for enrollment,
training, and manual operation.

---

## Table of contents

1. [Hardware](#hardware)
2. [Software stack](#software-stack)
3. [Directory layout](#directory-layout)
4. [Installation](#installation)
5. [First-time setup](#first-time-setup)
6. [Running the robot](#running-the-robot)
7. [Web control panel](#web-control-panel)
8. [RFID enrollment](#rfid-enrollment)
9. [Recognition pipeline](#recognition-pipeline)
10. [Identity lock behavior](#identity-lock-behavior)
11. [Clothing signature](#clothing-signature)
12. [Path planning](#path-planning)
13. [Troubleshooting](#troubleshooting)
14. [Change log](#change-log)

---

## Hardware

### Raspberry Pi 5 (8 GB)

- Ubuntu 24.04 aarch64
- Python 3.11 (system) + `coral_venv` virtualenv
- USB 3.0 port for the Astra Pro (blue connector)

### Coral USB Accelerator

Runs two TFLite models simultaneously:

- Face detection: `ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite`
- Body detection: `ssd_mobilenet_v2_edgetpu.tflite` (COCO person class)

### Orbbec Astra Pro

RGB-D camera. RGB through OpenCV (`/dev/video0`), depth through the
Orbbec SDK.

**Requires USB 3.0.** On USB 2.0 the depth stream falls back to 160×120
and the occupancy grid becomes too coarse for path planning.

### ESP32 (MedPal firmware)

- BTS7960 × 2 motor drivers (4 motors, 2 per side)
- Servo (medication dispenser hatch)
- TTP223 capacitive touch sensor
- MFRC522 RFID reader

Communicates with the Pi over USB serial at 115200 baud.

### Power

- 12 V LiFePO4 battery
- Dedicated 5 V 5 A buck converter for the Pi
- Separate 5 V 3 A buck for the ESP32 and servo (avoids motor noise
  resetting the Pi)

---

## Software stack

| Layer | Component |
|---|---|
| Detection (Coral) | SSD MobileNet V2 Face + SSD MobileNet V2 COCO |
| Landmarks + embedding (CPU) | SCRFD 2.5G + ArcFace w600k_mbf (ONNX) |
| Tracking | Custom SimpleTracker with sticky identity + lock |
| Appearance re-ID | Colour histogram + HOG clothing signature |
| Depth / planning | Astra Pro + custom occupancy grid + A* |
| Web | Flask + MJPEG + REST |
| Hardware control | ESP32 firmware + `serial_motors.py` |

---

## Directory layout

```
/home/medpal/2026medpal/
├── tracker_geminiV3_1.py       Main tracker (Coral + ArcFace + web)
├── web_server.py               Flask control panel (daemon thread)
├── shared.py                   Cross-thread state and queues
├── serial_motors.py            Pi-side ESP32 driver
├── clothing_signature.py       Colour + HOG appearance signature
├── scrfd_detector.py           SCRFD face detector with landmarks
├── arcface_embedder.py         ArcFace ONNX embedder
├── path_planner.py             Occupancy grid + A* + Tesla overlay
├── planner_demo.py             Standalone planner visualization
├── face_collect.py             Enrollment capture (aligned crops)
├── train_face_svm.py           Classifier training
├── scan.py                     Standalone scan test
├── config.py                   Central configuration
├── camera_calib.yml            Camera intrinsics
├── templates/
│   └── index.html              Web UI
├── models/
│   └── scrfd_arcface/
│       ├── det_2.5g.onnx       SCRFD detector
│       └── w600k_mbf.onnx      ArcFace embedder
└── esp32_firmware/
    └── medpal_firmware.ino     Motor + servo + touch + RFID firmware
```

Data directories:

```
/home/medpal/tracking_person/
├── face_training/
│   ├── Kirk/                   Aligned 112×112 crops
│   └── Carl/
├── models/
│   ├── arcface_classifier.pkl  Trained centroid + gallery
│   └── reid.onnx               OSNet body ReID (optional)
├── rfid_persons.json           UID → person mapping
└── authorized_uids.txt         Authorized UID list
```

---

## Installation

```bash
cd /home/medpal/2026medpal
source coral_venv/bin/activate

pip install flask pyserial
pip install onnxruntime
pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral

# Optional body ReID
pip install opencv-python numpy scikit-image
```

Create the calibration file once:

```bash
python3 - <<'EOF'
import cv2, math
W, H, hfov_deg = 640, 480, 60.0
fx = (W / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
fs = cv2.FileStorage("camera_calib.yml", cv2.FILE_STORAGE_WRITE)
fs.write("fx", fx); fs.write("fy", fx)
fs.write("cx", W / 2.0); fs.write("cy", H / 2.0)
fs.release()
print("wrote camera_calib.yml")
EOF
```

---

## First-time setup

**1. Enroll a person.**

```bash
python3 face_collect.py --name Kirk --samples 60
```

Stand in front of the camera. Slowly rotate your head through ±30° yaw
and ±20° pitch. Vary distance 0.5–1.5 m. Aim for 60 green frames.

**2. Train the classifier.**

```bash
python3 train_face_svm.py
```

Check that each person's p10 is above 0.55. If it's below 0.40, the
enrollment crops are misaligned or blurry — re-enroll.

**3. Verify recognition.**

```bash
python3 scan.py
```

Your face should get a green box with your name and confidence 0.75+.

**4. Run the robot.**

```bash
python3 tracker_geminiV3_1.py --follow Kirk --no-body-reid
```

Open `http://<pi-ip>:5000` in a browser on the same Wi-Fi.

---

## Running the robot

```bash
python3 tracker_geminiV3_1.py --follow Kirk --no-body-reid --debug-tracks
```

| Flag | Effect |
|---|---|
| `--follow NAME` | Follow the named person |
| `--no-body-reid` | Disable OSNet body ReID (saves CPU) |
| `--debug-tracks` | Print lock, revive, and scan decisions |
| `--no-planner` | Reactive steering instead of A* |
| `--web-port PORT` | Web server port (default 5000) |
| `--no-web` | Don't start the web server |
| `--list-people` | Show enrolled people and exit |
| `--remove-person NAME` | Delete a person's crops and exit |
| `--test-motors` | Drive each wheel briefly and exit |

**Runtime controls** (video window):

| Key | Action |
|---|---|
| `f` | Start following |
| `s` | Stop |
| `1..9` | Switch follow target to the Nth enrolled person |
| `p` | Toggle planner on/off |
| `q` | Quit |

---

## Web control panel

Open `http://<pi-ip>:5000` (find IP with `hostname -I`).

The panel provides:

- **Live MJPEG stream** with the tracker's AR overlay (boxes, labels,
  path)
- **Manual drive** — D-pad for forward/backward/left/right/stop
- **Medication dispenser** — Open / Close / Toggle servo
- **Register new person** — type a name, click Capture Face, live
  progress on the video feed
- **Retrain Now** — spawns `train_face_svm.py` in a subprocess and
  auto-reloads the classifier on success
- **Follow target** — click a person's name to switch targets
- **RFID management** — enroll cards, authorize/deauthorize, remove

Keyboard shortcuts on the page: arrow keys drive, space stops.

The web server runs in a daemon thread inside the tracker process.
All cross-thread communication flows through `shared.py`:

- `command_queue` — web → tracker
- `state` — tracker → web (status, servo, enrollment progress)
- `frame_holder` — tracker → web (MJPEG source)

The tracker never blocks on the web. The web never blocks on the tracker.

---

## RFID enrollment

**1. Enter the person's name in the RFID panel.** Click Enroll Card.

**2. The ESP32 enters enrollment mode** for 20 seconds. Status line
becomes: `Waiting for card scan (up to 20s). Tap the card on the reader.`

**3. Tap a card on the RC522.** ESP32 replies with
`RFID_ENROLLED:<person>:<uid>`. The Pi writes two files:

- `rfid_persons.json` — `{"A1B2C3D4": "Kirk", ...}`
- `authorized_uids.txt` — flat list of authorized UIDs

**4. Next time that card is scanned** (outside enrollment), the ESP32
sends `RFID:<uid>`, the Pi looks up the person, and switches follow
target automatically.

**5. Removing a card.** Click Remove in the table. Deletes from both
files. The card is forgotten.

**6. Deauthorizing.** Uncheck the Auth checkbox. The card is still
mapped to a person, but scanning it won't trigger a follow switch.

---

## Recognition pipeline

Each frame:

```
1. Coral face detector       → face boxes              ~10 ms
2. Coral body detector       → body boxes              ~10 ms
3. For each face not inside a locked track's region:
   a. SCRFD on the face crop → 5 landmarks             ~60 ms
   b. Align to 112×112 reference pose                  <1 ms
   c. ArcFace ONNX embed                             ~30 ms
   d. Match against centroid+gallery classifier       <1 ms
4. Associate faces with bodies (point-in-box)
5. SimpleTracker.update()                             <5 ms
6. TargetSelector.choose()                            <1 ms
```

**Scan gates** (any one blocks the scan):

- Face is farther than `FACE_RECOGNIZE_MAX_MM` (default 1200 mm)
- Face overlaps a locked track's face region (IoU ≥ 0.15)
- Face center falls inside a recently-scanned (but unlocked) track's
  box, within `FACE_SCAN_COOLDOWN_S`
- Follow target is locked and `SKIP_SCANS_WHEN_LOCKED = True`

**SCRFD runs on the face crop, not the full frame.** This prevents
SCRFD from picking a different person's face and producing a garbage
alignment for the current one.

---

## Identity lock behavior

**Lock acquisition:** A scan that returns a name with confidence ≥
`IDENTITY_LOCK_MIN_CONF` (default 0.85) sets the track as locked. The
name sticks until the track is released or dies.

**While locked:**

- No further scans fire for this track
- Body ReID signatures accumulate
- Clothing signature accumulates (see next section)
- The track is preferred by `TargetSelector` over unlocked tracks

**Lock release:** After `IDENTITY_LOCK_RELEASE_S` (default 3 s) of the
track not being matched by the body detector, the lock is released and
the track reverts to `Unknown`. On the next appearance, a fresh scan
runs.

**Lock revival:** If a locked track dies (unseen for `max_lost` frames)
and the person reappears within `REID_MAX_FRAMES` (30 s at 8 FPS), the
track is revived via:

1. Body ReID signature match (cosine ≥ 0.90)
2. Clothing signature match (similarity ≥ 0.80)

Revived tracks come back locked.

**Duplicate lock prevention:** A name can only be held by one locked
track. If a scan returns a name that's already locked to another track,
the assignment is rejected to prevent two "Kirk" tracks from existing.

---

## Clothing signature

For re-identification when the face isn't visible.

**Signature composition:**

- **Colour histogram** (HSV, 12×8×4 = 384 bins) over the torso region
  (top 10%–55% of the body crop)
- **HOG descriptor** (324 floats) over the same region, resized to
  64×64

Both are L2-normalised. Combined vector is 708 floats.

**Cost:** ~3–5 ms per crop on the Pi 5 CPU.

**Comparison:** Weighted similarity — 0.65 × colour + 0.35 × HOG.
When either descriptor is near-zero (solid-colour clothing), colour
alone is used.

**EMA update:** Each frame a locked track has a body crop, the
reference signature is updated with `alpha = 0.10`. This adapts to
lighting changes without losing the identity.

**Revive threshold:** Similarity ≥ 0.80 triggers a track revival.
Same-person clothing under similar lighting typically scores 0.85–0.95;
different people with similar clothing score 0.60–0.75.

On-screen label shows `[sN,C]` where `C` appears once a clothing
signature has been accumulated.

---

## Path planning

Occupancy grid + A* + Tesla-style blue overlay.

**Depth → grid:** Each depth pixel is back-projected to 3D, un-tilted
to the robot-level frame, and classified as obstacle if height above
floor is in `(floor_band, max_height_mm)`. Floor pixels are ignored.

**Inflation:** Obstacle cells are dilated by `ROBOT_RADIUS_MM` (default
200 mm) so A* doesn't plan through gaps narrower than the chassis.

**A* with string-pulling:** 8-connected A* produces a raw path; a
string-pulling pass removes intermediate waypoints when a straight
segment has clear line-of-sight.

**Goal:** Set interactively (click on the video in `planner_demo.py`)
or programmatically (`planner.set_goal_from_bbox()` for the tracker).

**Overlay:** The path is drawn on the ground plane with a three-layer
Tesla-style blue gradient. The floor is shaded grey.

**Dynamic replanning:** The planner checks each frame whether the first
6 cells of the current path are now occupied. If so, replan immediately
instead of waiting for the periodic timer.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'flask'`**
You're not in the venv. `source coral_venv/bin/activate` first.

**`tflite_runtime not installed`**
The venv doesn't have Coral runtime. Reinstall with
`pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral`.

**Depth is 160×120 instead of 320×240**
Astra Pro is on USB 2.0. Move it to the blue USB 3.0 port on the Pi 5
and use a USB 3.0 cable.

**Training via web button crashes with `num_threads` TypeError**
`arcface_embedder.py` is outdated. The `__init__` should accept
`num_threads=4` as a second parameter. Overwrite the file with the
current version.

**Classifier never reloads after training**
The tracker is missing the `elif cmd_type == "reload_classifier":`
branch in the command drain block. Add it.

**`scan blocked: face overlaps a locked track` printed repeatedly**
That's expected — the locked person's face is being held by the lock.
The scan gate is working.

**Name flickers between `Kirk` and `Unknown`**
The scan confidence is right at the lock threshold. Either lower
`IDENTITY_LOCK_MIN_CONF` to 0.80, or improve enrollment (more diverse
crops, better lighting).

**`p10` below 0.40 after training**
The enrollment crops are misaligned or blurry. Delete the person's
folder and re-enroll with `face_collect.py`. Check that the saved
crops are aligned 112×112 faces, not raw rectangles.

**Recognition fails at > 1 m**
The Astra Pro's 640×480 RGB sensor can't produce enough face pixels
beyond ~1 m. Either get the person closer, or replace the RGB camera
with a 1080p USB webcam.

**Servo jitters or ESP32 resets**
Servo is drawing too much current through the ESP32's regulator.
Power it from a separate 5 V supply with a common ground.

**RC522 reports firmware `0x00` or `0xFF`**
No SPI communication. Check VCC is 3.3 V (not 5 V), GND is solid,
and a 47–220 µF capacitor is across the RC522's VCC and GND pins.

---

## Change log

### Recognition pipeline

- **Added** ArcFace (w600k_mbf) face embedding on CPU with SCRFD
  landmark detection and 5-point alignment.
- **Removed** the old MobileNetV1-ImageNet embedder (1024-D) that
  produced ImageNet embeddings instead of face embeddings.
- **Added** a centroid+gallery classifier format with per-person
  percentile statistics (`p10/p50/p90`) for confidence mapping.
- **Fixed** SCRFD output indexing to be robust to ONNX export order
  (shape-based detection instead of positional).
- **Fixed** SCRFD `np.vstack` crash on strides with zero detections.
- **Fixed** landmark coordinate validation — raises on frame coords
  instead of producing silent wrong embeddings.
- **Fixed** `FACE_RECOGNIZE_MAX_MM` default (was 80, inside the Astra
  blind zone; now 1200).
- **Fixed** `SCRFD_INPUT_SIZE` default (was 320; now 640 for better
  small-face recall).

### Identity tracking

- **Added** sticky identity — `t.name` only changes on a scanned
  detection with a confident match.
- **Added** identity lock — a scan at ≥ `IDENTITY_LOCK_MIN_CONF`
  locks the track and blocks further scans.
- **Added** lock release after `IDENTITY_LOCK_RELEASE_S` (3 s) of the
  track being unseen.
- **Added** lock revival via body ReID signatures or clothing
  signature.
- **Added** duplicate-lock prevention — one name, one locked track.
- **Added** `Track.last_face_box` and `SimpleTracker.is_face_locked`
  so locked tracks block scans by face-region overlap, not just by
  body-box containment.
- **Added** lock-preference bonus in association to prevent a
  new/unknown detection from stealing a locked track's pairing.
- **Added** `SKIP_SCANS_WHEN_LOCKED` config — once the follow target
  is locked, no more scans fire until the lock is released.
- **Fixed** scan gate to run SCRFD on the face region instead of the
  full frame (prevents scanning a different person's face).

### Clothing signature

- **Added** `clothing_signature.py` with colour histogram (HSV,
  12×8×4) + HOG (324-D) over the torso region.
- **Added** EMA-based reference signature update.
- **Added** revival via clothing similarity when body ReID is
  unavailable or fails.
- **Fixed** HOG size expectation (324, not 1764).
- **Fixed** zero-vector HOG handling so solid-colour clothing compares
  correctly (colour-only path).

### Path planning

- **Added** `path_planner.py` with occupancy grid from depth, A*
  with string-pulling, robot-radius inflation, and Tesla-style blue
  path overlay on the ground plane.
- **Fixed** ground-height sign error (`y_world = cam_height - y_rot`,
  was `+ y_rot` — the wrong sign was marking floor as obstacles).
- **Fixed** floor band in grid building (pixels with height in
  `(0, floor_band_mm)` are floor, not obstacle).
- **Fixed** `_project_ground_to_pixel` sign flips (path overlay was
  mirrored left-right and flipped vertically).
- **Fixed** `estimate_ground_plane` missing `cos(phi)` term (radial
  vs Z-depth model).
- **Fixed** `build_grid` from a Python double loop to fully vectorized
  numpy (was 150–250 ms per call, now 2–4 ms).
- **Added** self-calibrating ground plane (estimates camera height
  and tilt from the depth image).
- **Added** `is_path_blocked()` for immediate replanning on dynamic
  obstacles.

### Web control panel

- **Added** Flask web server running in a daemon thread with
  `threaded=True`.
- **Added** `shared.py` with `command_queue`, `state`, `frame_lock`,
  and `frame_holder` for safe cross-thread communication.
- **Added** MJPEG video streaming from `frame_holder`.
- **Added** REST endpoints for manual drive, servo, enrollment,
  training, follow target, and RFID management.
- **Added** enrollment progress in `/api/status` (`enroll_name`,
  `enroll_captured`, `enroll_target`).
- **Fixed** the `train()` JS to POST instead of GET (405 error).
- **Fixed** training subprocess to use `sys.executable` so the venv
  is inherited.
- **Added** stdout/stderr pump so training output streams into the
  tracker console with `[train]` prefix.
- **Added** automatic classifier reload after a successful training
  run.
- **Fixed** enrollment to publish annotated frames to `frame_holder`
  so the browser shows a live box + progress overlay during capture.

### Hardware (ESP32)

- **Added** servo control via raw LEDC PWM (50 Hz, 16-bit).
- **Added** TTP223 touch sensor polling with debounce.
- **Added** MFRC522 RFID reader on HSPI with non-blocking reads.
- **Added** RFID enrollment mode (`RFID_ENROLL:<person>` and
  `RFID_ENROLLED:<person>:<uid>` events).
- **Added** safety watchdog (500 ms) that stops motors on Pi
  disconnect. Servo is not affected.
- **Fixed** serial read to be non-blocking (accumulator, not
  `readStringUntil`).
- **Fixed** RFID polling to be non-blocking (`PICC_IsNewCardPresent`,
  not blocking `PICC_ReadCardSerial`).
- **Avoided** ESP32 strapping pins (0, 2, 5, 12, 15) for outputs.

### Pi-side ESP32 driver

- **Added** `on_touch`, `on_rfid`, `on_rfid_enrolled`, `on_rfid_timeout`
  callbacks.
- **Added** background reader thread dispatching events.
- **Added** `servo_open`, `servo_close`, `servo_angle` methods.
- **Added** `rfid_enroll(person)` and `rfid_enroll_cancel()` methods.
- **Fixed** `cleanup()` to send STOP + servo CLOSE before closing the
  port (was a no-op if the methods didn't exist).
- **Added** `test_motors()` method.

### Configuration

- **Added** `IDENTITY_LOCK_MIN_CONF`, `IDENTITY_LOCK_RELEASE_S`,
  `ASSUMED_FPS`, `SKIP_SCANS_WHEN_LOCKED`, `ROBOT_RADIUS_MM`.
- **Fixed** `FACE_RECOGNIZE_MAX_MM` (80 → 1200).
- **Fixed** `FACE_SCAN_COOLDOWN_S` (0.5 → 2.0).
- **Fixed** `SCRFD_INPUT_SIZE` (320 → 640).
- **Removed** duplicate `SVM_MODEL_PATH` and `REID_PATH` definitions.
- **Removed** the stale `MOBILEFACENET_MODEL` path.

---

## License

Private project. Not for redistribution.