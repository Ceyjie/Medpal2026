Here's a complete README documenting the system. Save it as `README.md` in `~/2026medpal/`.

```markdown
# MedPal Face Tracker

A Raspberry Pi + Coral USB Accelerator robot that detects faces, recognizes
enrolled people, and follows one selected person while avoiding obstacles.

---

## What it does

The system runs a real-time pipeline on a Raspberry Pi 4 with a Coral USB
Accelerator (Edge TPU), an Orbbec Astra depth camera, and an ESP32 motor
controller.

1. **Detect faces** — Coral-compiled SSD MobileNet V2 Face model, ~5 ms per frame
2. **Embed each face** — Coral-compiled FaceNet, produces a 1024-D embedding
3. **Recognize** — nearest-centroid + gallery classifier labels each face
4. **Track** — lightweight IoU tracker smooths detections across frames
5. **Follow** — motor control drives toward one selected person
6. **Learn** — confident recognitions are auto-saved for future retraining
7. **Avoid obstacles** — three-sector depth scan stops the robot if something
   is closer than the target

Every command is driven from a single script, `tracker_geminiV3.py`.

---

## Hardware

| Component | Role |
|---|---|
| Raspberry Pi 4 | Host CPU |
| Coral USB Accelerator | Edge TPU for face detection + embedding |
| Orbbec Astra Pro | RGB + depth camera (USB) |
| ESP32 + motor driver | Wheels, driven over USB serial |
| Display (X11) | Live preview window with overlays |

---

## Setup

### Models

Download the two Coral-compiled models into
`/home/medpal/tracking_person/models/tflite/`:

```bash
cd /home/medpal/tracking_person/models/tflite

# Face detector (SSD MobileNet V2 Face)
wget https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite

# Face embedding (Coral FaceNet, 224x224 -> 1024-D)
wget https://github.com/google-coral/test_data/raw/master/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite

# Label file for the face detector
echo "face" > face_labels.txt
```

### Python environment

```bash
python3 -m venv ~/2026medpal/coral_venv
source ~/2026medpal/coral_venv/bin/activate

pip install numpy opencv-python scikit-learn pyserial

# tflite_runtime + pycoral: install the pre-built wheels from GitHub
# (see the PyCoral release notes for the cp39 aarch64 wheels)
pip install tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl
pip install pycoral-2.0.0-cp39-cp39-linux_aarch64.whl
```

### Config

Edit `config.py` to match your paths and tune thresholds (see
[Tuning](#tuning) below).

---

## Quick start

```bash
cd ~/2026medpal
source coral_venv/bin/activate

# 1. Enroll yourself (30 face crops)
python3 tracker_geminiV3.py --enroll Carl

# 2. Add more crops at a distance for better far-range recognition
python3 tracker_geminiV3.py --enroll Carl --samples 20

# 3. Train the classifier
python3 train_face_svm.py

# 4. Run
python3 tracker_geminiV3.py --follow Carl
# press 'f' in the video window to start following
```

---

## Commands

### Enrollment

```bash
python3 tracker_geminiV3.py --enroll NAME              # 30 crops (default)
python3 tracker_geminiV3.py --enroll NAME --samples 50 # 50 crops
```

Appends to any existing crops. Run 2–3 times per person at different
distances (1 m, 2 m, 3 m) for best coverage.

### Training

```bash
python3 train_face_svm.py
```

Reads every `.jpg` in `face_training/<name>/`, generates 11 augmented
variants per crop (flip, brightness, rotation, zoom, distance simulation,
blur), embeds all of them, and saves a centroid + gallery classifier.

Run after any enrollment or auto-save session.

### Running

```bash
# Explicit follow target
python3 tracker_geminiV3.py --follow Carl

# One person enrolled -> follows automatically
python3 tracker_geminiV3.py

# Two or more enrolled, no --follow -> prompts to pick
python3 tracker_geminiV3.py
```

Add `--auto-train` to retrain automatically on exit if any crops were saved:

```bash
python3 tracker_geminiV3.py --follow Carl --auto-train
```

### Inspection

```bash
python3 tracker_geminiV3.py --list-people
```

Shows enrolled people, crop counts per type (enroll / auto_dir / auto_trk),
and whether each is in the current classifier.

### Cleanup

```bash
python3 tracker_geminiV3.py --clear-auto Carl     # delete only auto-saved crops
python3 tracker_geminiV3.py --remove-person King  # delete everything for King
```

Both commands leave the classifier intact. Re-run `train_face_svm.py` after.

### Motor test

```bash
python3 tracker_geminiV3.py --test-motors
```

Drives each wheel forward, backward, left, right for ~1 s. No camera needed.

---

## Runtime controls

While the video window is focused:

| Key | Action |
|---|---|
| `f` | Start following the selected target |
| `s` | Stop following |
| `1`–`9` | Switch follow target to the Nth enrolled person |
| `q` | Quit |

If `--auto-train` is on, quitting after crops were saved will hand off to
the training script automatically.

---

## Auto-save: how the classifier improves over time

Two kinds of crops get written during a follow session:

| Prefix | Meaning |
|---|---|
| `enroll_*` | Manual enrollment via `--enroll` |
| `auto_dir_*` | Frame was confidently recognized directly (≥ 0.92) |
| `auto_trk_*` | Frame overlapped a recently confirmed track (≥ 0.60) |

The track-confirmation cache remembers recent high-confidence
identifications by location. When a face is confidently identified, the
next ~6 seconds of that same physical face are auto-saved even if the
per-frame confidence dips (head turned, lighting shifted, distance changed).
This is how the classifier learns your appearance under poses the initial
enrollment didn't cover.

Filenames include a timestamp and a per-person session counter, so you can
audit which face was saved when.

The overlay in the top-right shows per-person counters:

```
Carl:32 (d5/t27)
King:18 (d9/t9)
```

`d` = direct saves, `t` = track saves.

### The retraining loop

```
tracker run  ->  auto-save crops  ->  train_face_svm.py  ->  improved classifier
    ^                                                              |
    +--------------------------------------------------------------+
```

Over several sessions the gallery grows and recognition gets progressively
more robust. If you use `--auto-train`, this loop is automatic.

---

## How recognition works

The classifier is a **nearest-centroid + gallery** model:

- **Centroid** — mean of all enrollment embeddings, re-normalized
- **Gallery** — every individual embedding (with augmentation)

At inference, for each enrolled person:

```
score_top  = mean of top-3 cosine similarities to their gallery
score_cent = cosine similarity to their centroid
combined   = 0.7 * score_top + 0.3 * score_cent
```

The highest-scoring person wins, then their confidence is computed by
mapping the combined score through a piecewise ramp anchored at that
person's training percentiles (`p10`, `p50`, `p90`).

**Why not SVM / OneClassSVM?**

Earlier versions used `SVC` and `OneClassSVM`. Both were removed:

- `SVC` cannot train on a single class (fails when one person is enrolled).
- `OneClassSVM` produces decision scores in a narrow range near zero,
  making the sigmoid confidence mapping unusable.

The centroid + gallery model is simpler, has no such failure modes,
and gives directly interpretable cosine similarity numbers.

---

## Tuning

All tunables live in `config.py` and at the top of `tracker_geminiV3.py`.

### Detection

| Setting | Default | Effect |
|---|---|---|
| `CONF_THRES` | 0.35 | Lower detects faces further away; higher reduces false positives |
| `FRAME_W`, `FRAME_H` | 640×480 | Capture resolution |

### Recognition

| Setting | Default | Effect |
|---|---|---|
| `FACE_SVM_CONFIDENCE_THRES` | 0.70 | Recognition floor. Raise to reject strangers, lower to accept the target more often |

### Auto-save

In `AutoSaver` (top of `tracker_geminiV3.py`):

| Setting | Default | Effect |
|---|---|---|
| `MIN_CONF_DIRECT` | 0.92 | Confidence required for a direct save |
| `MIN_CONF_TRACK` | 0.60 | Confidence required for a track-confirmed save |
| `MIN_INTERVAL_S` | 3.0 | Min seconds between saves per person |
| `VARIETY_SIM_THRESH` | 0.90 | Reject crops too similar to last 20 saved |
| `MAX_PER_SESSION` | 50 | Hard cap per person per session |

### Enrollment quality

In `config.py`:

| Setting | Default | Effect |
|---|---|---|
| `ENROLL_MIN_FACE_PX` | 80 | Minimum face size to accept during enrollment |
| `ENROLL_BLUR_THRES` | 40.0 | Reject blurry crops |
| `ENROLL_DIVERSITY_PX` | 25 | Reject frames too similar to the last one |

### Depth / motion

| Setting | Default | Effect |
|---|---|---|
| `REVERSE_DISTANCE_MM` | 60 | Below this, robot backs away |
| `FORWARD_DISTANCE_MM` | 100 | Above this, robot follows forward |
| `OBSTACLE_STOP_MM` | 150 | Center sector stop threshold |
| `FOLLOW_BASE_SPEED` | 60 | Default motor speed (%) |

### Track confirmation

In `TrackConfirmationCache`:

| Setting | Default | Effect |
|---|---|---|
| `HOLD_FRAMES` | 60 | How long a confident ID stays valid (~6 s at 10 FPS) |
| `IOU_THRESH` | 0.4 | Overlap required for a new box to count as the same track |
| `CACHE_MIN_CONF` | 0.85 | Confidence required for the cache to remember an ID |

---

## Project history and design decisions

### What was added

**Face recognition pipeline (replaced body ReID).**
An earlier version used YOLO + a body ReID ONNX model. Body re-identification
is unreliable when people wear similar clothing. The current pipeline uses
face detection + face embedding on the Coral TPU. Face recognition is a
stronger signal for the specific-person problem and rejects unknowns more
reliably.

**Coral-accelerated detection and embedding.**
Both the SSD MobileNet V2 Face detector and the FaceNet embedder run on the
Edge TPU. The detector also has a PyCoral path (preferred) and a raw
`tflite_runtime` fallback for when PyCoral isn't available.

**Centroid + gallery classifier.**
Replaced `OneClassSVM` (score scale was unworkable) and `SVC` (fails with a
single class). The current model uses top-3 gallery similarity blended with
centroid similarity, with a piecewise confidence ramp derived from each
person's training distribution.

**Augmentation.**
Training now generates 11 variants per crop, including distance simulation
(downscale 48×48 then upscale) and Gaussian blur. This makes recognition
tolerant of far-away faces and soft focus without needing to enroll at
every possible distance.

**Better embedder preprocessing.**
Small face crops are padded with a mirrored border before upscaling, and
interpolation switches between cubic (upscaling) and area (downscaling).
This produces cleaner embeddings for distant faces.

**IoU tracker.**
Smoothes detections across frames and prevents flicker when confidence
jitters. Number-key switching changes the follow target live without
restarting.

**Track-confirmation auto-save.**
The classifier improves passively. Confident identifications seed a
location-based cache; subsequent frames of the same face are auto-saved at
a lower confidence floor. This captures poses the initial enrollment
missed.

**Per-person auto-save counters.**
The overlay shows how many crops each person has accumulated this session,
broken down by direct vs track path.

**`--auto-train` with `execv` handoff.**
Retraining runs automatically on exit if crops were saved. The tracker
replaces itself with the training process using `os.execv`, which releases
the Coral USB handle cleanly before the trainer opens it. This eliminates
the `USB transfer error 5` / `Aborted` failures that occur when a parent
and child both hold the Coral.

**Better cleanup on exit.**
Motor `cleanup()` sends a final stop command and closes the serial port.
Depth capture zeroes readings below `MIN_VALID_DEPTH_MM` to avoid treating
sensor-floor noise as a close obstacle.

### What was removed

- **`Motors` (lgpio-based)** — unreachable dead code; `SerialMotors` is used.
- **Body ReID ONNX inference** — replaced by face-based recognition.
- **`target_locked` / `lock_streak` / `loss_streak`** — replaced by track-level
  match votes and the track confirmation cache.
- **`run_auto_train()` subprocess helper** — replaced by `os.execv` direct
  handoff.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No classifier at .../svm_face_classifier.pkl` | Never trained | Run `python3 train_face_svm.py` |
| `classifier is 'svc', expected 'centroid_gallery'` | Old classifier format | Delete the `.pkl` and retrain |
| `USB transfer error 5` after `--auto-train` | Parent holding Coral | Ensure the tracker uses the `os.execv` handoff version |
| `Aborted` at the end of a training run | Coral destructor hits a USB race | Ensure `train_face_svm.py` ends with `os._exit(0)` |
| Motors don't move | ESP32 not enumerated or no PONG | Run `--test-motors`; check `ls /dev/ttyUSB*` |
| Target labelled `Unknown` too often | Threshold too high, or poor enrollment | Lower `FACE_SVM_CONFIDENCE_THRES`, re-enroll |
| Strangers labelled as the target | Threshold too low | Raise `FACE_SVM_CONFIDENCE_THRES`, or enroll a second person |
| Recognition fails when the target is far | Face too small for the embedder | Lower `CONF_THRES`, re-enroll at distance, or accept the 30-40 px limit |
| Overlay counters overlapping | Too many people being saved | Reduce `MAX_PER_SESSION`, or shrink the overlay font |

---

## Files

| Path | Purpose |
|---|---|
| `tracker_geminiV3.py` | Main tracker: detection, recognition, tracking, motors, auto-save, auto-train |
| `train_face_svm.py` | Training: reads crops, augments, embeds, saves centroid + gallery classifier |
| `config.py` | All tunable parameters and file paths |
| `serial_motors.py` | ESP32 serial driver (set_speed, forward, backward, turn_left, turn_right, stop, test_motors, cleanup) |
| `/home/medpal/tracking_person/models/tflite/*.tflite` | Coral-compiled detection and embedding models |
| `/home/medpal/tracking_person/models/svm_face_classifier.pkl` | Trained classifier (centroid + gallery + stats) |
| `/home/medpal/tracking_person/face_training/<name>/` | Enrolled and auto-saved crops per person |

---

## Keyboard reference

While the video window is focused:

```
f   start following
s   stop
1   switch target to enrolled person #1
2   switch target to enrolled person #2
...
9   switch target to enrolled person #9
q   quit
```

Number keys switch the follow target live; you don't need to stop first.
The top-left banner updates from `Following: Carl` to `Following: King`
immediately.
```

## Where to save it

```bash
nano /home/medpal/2026medpal/README.md
# paste the content above
# Ctrl+O, Enter, Ctrl+X
```

Or if you prefer, save it on your development machine and copy it over with `scp`.

## What the README covers

| Section | Purpose |
|---|---|
| **What it does** | High-level pipeline description |
| **Hardware** | Component roles |
| **Setup** | Model downloads, venv, dependencies |
| **Quick start** | Enroll → train → run in four commands |
| **Commands** | Full CLI reference |
| **Runtime controls** | Key bindings |
| **Auto-save** | How learning happens over time |
| **How recognition works** | Centroid + gallery explanation, why SVM was removed |
| **Tuning** | Table of every tunable knob |
| **Project history and design decisions** | What was added and removed, and why |
| **Troubleshooting** | Symptom → cause → fix table |
| **Files** | Directory map |
| **Keyboard reference** | Condensed key bindings |

The "design decisions" section is the most valuable long-term — it documents **why** the SVM was removed, why auto-train uses `execv` instead of `subprocess`, and why cleanup order matters. Anyone coming back to this in six months will understand the reasoning without re-reading every commit.