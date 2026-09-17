#!/usr/bin/env python3
"""
tracker_person.py -- Body-based person recognition and following.

Pipeline:
  1. Coral SSD MobileNet V2 COCO detects persons (whole body).
  2. OSNet ONNX (CPU) embeds each body crop -> 512-D vector.
  3. A centroid+gallery classifier trained on body crops labels each
     embedding as a known person or "Unknown".
  4. IoU tracker smooths boxes across frames.
  5. Motor control follows the selected person.

No face detection. No face embedding. Identity comes from the whole
body appearance. Works when the person turns away, is at distance, or
wears a mask. Fails when people wear similar clothes.

Commands:
    --enroll NAME            Collect body crops for a person
    --samples N              Crops per enrollment (default 60)
    --register-only NAME     Remove all others, then enroll only this person
    --follow NAME            Follow only this person
    --list-people            Show enrolled people and classifier state
    --remove-person NAME     Delete a person's crops
    --clear-auto NAME        Delete only auto-saved crops for a person
    --test-motors            Drive each wheel briefly
    --no-auto-save           Disable auto-saving for this run
    --no-auto-train          Disable auto-retrain on exit

Runtime controls (video window):
    f = start following
    s = stop
    1..9 = switch target to the Nth enrolled person
    q = quit
"""

import os
import sys
import cv2
import numpy as np
import pickle
import argparse
import time
import threading
import shutil

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

sys.path.append('/home/medpal/pyorbbecsdk_v1/build')

try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
    USE_ORBBEC_DEPTH = True
except ImportError:
    USE_ORBBEC_DEPTH = False

CORAL_AVAILABLE = False
CORAL_EDGETPU = False
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    CORAL_AVAILABLE = True
    try:
        _test_delegate = load_delegate('libedgetpu.so.1')
        CORAL_EDGETPU = True
        print("Coral Edge TPU delegate loaded successfully")
    except Exception as e:
        print(f"Coral Edge TPU delegate unavailable: {e}")
except ImportError:
    print("tflite_runtime not installed.")

PYCORAL_AVAILABLE = False
try:
    from pycoral.adapters import common as pycoral_common, detect as pycoral_detect
    from pycoral.utils.edgetpu import make_interpreter as pycoral_make_interpreter
    PYCORAL_AVAILABLE = True
    print("PyCoral available")
except ImportError:
    print("PyCoral not installed (pip install pycoral)")

ONNX_AVAILABLE = False
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    print("onnxruntime available")
except ImportError:
    print("onnxruntime not installed (pip install onnxruntime)")

import config
from serial_motors import SerialMotors


# ============================================================
# Camera
# ============================================================
class Camera:
    def __init__(self):
        self.color_cap = None
        for idx in [1, 0, 2]:
            self.color_cap = cv2.VideoCapture(idx)
            if self.color_cap.isOpened():
                print(f"Opened color camera on /dev/video{idx}")
                break
            self.color_cap.release()
        if self.color_cap is None or not self.color_cap.isOpened():
            raise RuntimeError("Cannot open color camera")
        self.color_cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
        self.color_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
        self.color_frame = None
        self.depth_frame = None
        self.color_running = True
        self.color_thread = threading.Thread(target=self._color_capture, daemon=True)
        self.color_thread.start()
        print("Using cv2.VideoCapture for color")
        if USE_ORBBEC_DEPTH:
            self.depth_pipeline = Pipeline()
            self.depth_config = Config()
            profiles = self.depth_pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            try:
                self.depth_profile = profiles.get_video_stream_profile(320, 240, OBFormat.Y16, 15)
            except:
                self.depth_profile = profiles.get_default_video_stream_profile()
            self.depth_config.enable_stream(self.depth_profile)
            self.depth_pipeline.start(self.depth_config)
            self.depth_running = True
            self.depth_thread = threading.Thread(target=self._depth_capture, daemon=True)
            self.depth_thread.start()
            self.use_orbbec_depth = True
            print(f"Using Astra for depth ({self.depth_profile.get_width()}x{self.depth_profile.get_height()})")
        else:
            self.use_orbbec_depth = False

    def _color_capture(self):
        while self.color_running:
            ret, frame = self.color_cap.read()
            if ret:
                self.color_frame = frame
            time.sleep(0.01)

    def _depth_capture(self):
        while self.depth_running:
            try:
                frames = self.depth_pipeline.wait_for_frames(1000)
                if frames:
                    df = frames.get_depth_frame()
                    if df:
                        w, h = df.get_width(), df.get_height()
                        scale = df.get_depth_scale()
                        data = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(h, w)
                        depth_mm = data.astype(np.float32) * scale
                        depth_mm[depth_mm < config.MIN_VALID_DEPTH_MM] = 0
                        self.depth_frame = depth_mm
            except Exception as e:
                print(f"Depth capture error: {e}")
                time.sleep(1.0)
            time.sleep(0.01)

    def read_color(self):
        return self.color_frame

    def read_depth(self):
        return self.depth_frame

    def stop(self):
        self.color_running = False
        self.color_thread.join()
        self.color_cap.release()
        if self.use_orbbec_depth:
            self.depth_running = False
            self.depth_thread.join()
            self.depth_pipeline.stop()


# ============================================================
# Body Detector (Coral SSD COCO, person class)
# ============================================================
class BodyDetector:
    def __init__(self, model_path, label_path=None):
        self.labels = []
        if label_path and os.path.exists(label_path):
            with open(label_path) as f:
                self.labels = [l.strip() for l in f.readlines()]
        self.PERSON_CLASS = 0
        for i, name in enumerate(self.labels):
            if name.strip().lower() == "person":
                self.PERSON_CLASS = i
                break
        self.conf_thres = getattr(config, "BODY_CONF_THRES", 0.60)
        print(f"BodyDetector: person class = {self.PERSON_CLASS}, "
              f"conf = {self.conf_thres}")

        self.mode = None
        self.engine = None
        self.interpreter = None

        if PYCORAL_AVAILABLE:
            try:
                self.engine = pycoral_make_interpreter(model_path)
                self.engine.allocate_tensors()
                self.mode = "pycoral"
                print("BodyDetector: using PyCoral")
                return
            except Exception as e:
                print(f"BodyDetector: PyCoral failed ({e})")

        if CORAL_EDGETPU:
            delegate = load_delegate('libedgetpu.so.1')
            self.interpreter = Interpreter(model_path, experimental_delegates=[delegate])
        else:
            cpu_path = model_path.replace('_edgetpu', '')
            self.interpreter = Interpreter(cpu_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        self.mode = "tflite_runtime"
        print("BodyDetector: using raw tflite_runtime")

    def infer(self, frame):
        if self.mode == "pycoral":
            return self._infer_pycoral(frame)
        return self._infer_tflite(frame)

    def _infer_pycoral(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp_h, inp_w = pycoral_common.input_size(self.engine)
        resized = cv2.resize(rgb, (inp_w, inp_h))
        pycoral_common.set_input(self.engine, resized)
        self.engine.invoke()
        h, w = frame.shape[:2]
        sx, sy = w / inp_w, h / inp_h
        objs = pycoral_detect.get_objects(self.engine, score_threshold=self.conf_thres)
        results = []
        for obj in objs:
            if obj.id != self.PERSON_CLASS:
                continue
            bbox = obj.bbox
            x = int(bbox.xmin * sx); y = int(bbox.ymin * sy)
            bw = int((bbox.xmax - bbox.xmin) * sx)
            bh = int((bbox.ymax - bbox.ymin) * sy)
            results.append((x, y, bw, bh, float(obj.score)))
        return results

    def _infer_tflite(self, frame):
        h, w = frame.shape[:2]
        target_h, target_w = self.input_shape[1], self.input_shape[2]
        resized = cv2.resize(frame, (target_w, target_h))
        input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        boxes = self.interpreter.tensor(self.output_details[0]['index'])()
        classes = self.interpreter.tensor(self.output_details[1]['index'])()
        scores = self.interpreter.tensor(self.output_details[2]['index'])()
        results = []
        for i in range(int(scores[0].shape[0])):
            score = float(scores[0][i])
            if score < self.conf_thres:
                continue
            cls_id = int(classes[0][i])
            if cls_id != self.PERSON_CLASS:
                continue
            y1, x1, y2, x2 = boxes[0][i]
            x = int(x1 * w); y = int(y1 * h)
            bw = int((x2 - x1) * w); bh = int((y2 - y1) * h)
            results.append((x, y, bw, bh, score))
        return results


# ============================================================
# Body Embedder (OSNet ONNX, CPU)
# ============================================================
class BodyEmbedder:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Body ReID model not found: {model_path}\n"
                f"Run export_reid_onnx.py first."
            )
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime not installed.")

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.input_h = 256
        self.input_w = 128
        if len(shape) == 4:
            if isinstance(shape[2], int) and shape[2] > 0:
                self.input_h = shape[2]
            if isinstance(shape[3], int) and shape[3] > 0:
                self.input_w = shape[3]
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        print(f"BodyEmbedder: input {self.input_w}x{self.input_h} "
              f"(providers: {self.session.get_providers()})")

    def embed(self, crop):
        if crop is None or crop.size == 0:
            return None
        img = cv2.resize(crop, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0).astype(np.float32)
        try:
            out = self.session.run(None, {self.input_name: img})[0]
        except Exception as e:
            print(f"BodyEmbedder error: {e}")
            return None
        vec = out.flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


# ============================================================
# Person Classifier (body centroid + gallery)
# ============================================================
class PersonClassifier:
    def __init__(self, model_path):
        self.names = []
        self.centroids = None
        self.galleries = None
        self.stats_p10 = None
        self.stats_p50 = None
        self.stats_p90 = None
        self.available = False

        if not os.path.exists(model_path):
            print(f"PersonClassifier: no model at {model_path}.")
            return
        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"PersonClassifier: failed to load ({e})")
            return
        if data.get("model_type") != "body_centroid_gallery":
            print(f"PersonClassifier: unexpected type {data.get('model_type')}")
            return

        self.names = list(data["names"])
        self.centroids = data["centroids"]
        self.galleries = [np.asarray(g, dtype=np.float32) for g in data["galleries"]]
        self.stats_p10 = data["stats_p10"]
        self.stats_p50 = data["stats_p50"]
        self.stats_p90 = data["stats_p90"]
        self.available = True
        sizes = [g.shape[0] for g in self.galleries]
        print(f"PersonClassifier: loaded for {self.names} (gallery sizes: {sizes})")

    def predict(self, embedding, top_k=3):
        if not self.available or embedding is None:
            return None, -1.0
        q = embedding.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)

        best_combined = -1.0
        best_idx = -1
        for i, (centroid, gallery) in enumerate(zip(self.centroids, self.galleries)):
            score_cent = float(centroid @ q)
            gallery_sims = gallery @ q
            k = min(top_k, len(gallery_sims))
            top = np.partition(gallery_sims, -k)[-k:]
            score_top = float(top.mean())
            combined = 0.7 * score_top + 0.3 * score_cent
            if combined > best_combined:
                best_combined = combined
                best_idx = i

        if best_idx < 0:
            return None, -1.0

        name = self.names[best_idx]
        p10 = float(self.stats_p10[best_idx])
        p50 = float(self.stats_p50[best_idx])
        p90 = float(self.stats_p90[best_idx])

        if best_combined >= p50:
            span = max(p90 - p50, 1e-6)
            frac = min(1.0, (best_combined - p50) / span)
            confidence = 0.75 + 0.20 * frac
        elif best_combined >= p10:
            span = max(p50 - p10, 1e-6)
            frac = (best_combined - p10) / span
            confidence = 0.55 + 0.20 * frac
        else:
            span = max(p50 - p10, 1e-6)
            overshoot = (p10 - best_combined) / span
            confidence = 0.55 * float(np.exp(-2.0 * overshoot))

        if confidence >= config.PERSON_SVM_CONFIDENCE_THRES:
            return name, confidence
        return None, confidence


# ============================================================
# IoU Tracker (body boxes with sticky identity)
# ============================================================
def iou(boxA, boxB):
    ax1, ay1, aw, ah = boxA
    bx1, by1, bw, bh = boxB
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class Track:
    def __init__(self, tid, box, name, conf, frame_count):
        self.id = tid
        self.box = box
        self.name = name if name is not None else "Unknown"
        self.confidence = conf if name is not None else 0.0
        self.age = 1
        self.lost = 0
        self.vx = 0.0
        self.vy = 0.0
        self.last_class_frame = frame_count
        self.identity_fresh = name is not None


class SimpleTracker:
    MIN_MATCH_SCORE = 0.30
    SMOOTH_ALPHA = 0.55
    VEL_ALPHA = 0.5
    UNKNOWN_MAX_LOST = 20
    IDENTITY_TIMEOUT = getattr(config, "PERSON_IDENTITY_TIMEOUT_FRAMES", 20)

    def __init__(self, max_lost=180):
        self.max_lost = max_lost
        self.tracks = []
        self.next_id = 0

    def _predict(self, t):
        x, y, w, h = t.box
        return (int(x + t.vx), int(y + t.vy), w, h)

    def _match_score(self, boxA, boxB):
        iou_v = iou(boxA, boxB)
        ax, ay, aw, ah = boxA
        bx, by, bw, bh = boxB
        acx, acy = ax + aw / 2, ay + ah / 2
        bcx, bcy = bx + bw / 2, by + bh / 2
        diag = max(1.0, (aw * aw + ah * ah) ** 0.5)
        d = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
        center_score = max(0.0, 1.0 - d / diag)
        return 0.6 * iou_v + 0.4 * center_score

    def update(self, detections, frame_count):
        for t in self.tracks:
            t.lost += 1

        pairs = []
        for ti, track in enumerate(self.tracks):
            pred_box = self._predict(track)
            for di, det in enumerate(detections):
                score = self._match_score(pred_box, det["box"])
                if score >= self.MIN_MATCH_SCORE:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        matched_tracks, matched_dets = set(), set()
        for _, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            t = self.tracks[ti]
            det = detections[di]

            # EMA smoothing
            ox, oy, ow, oh = t.box
            nx, ny, nw, nh = det["box"]
            a = self.SMOOTH_ALPHA
            new_box = (
                int(a * nx + (1 - a) * ox),
                int(a * ny + (1 - a) * oy),
                int(a * nw + (1 - a) * ow),
                int(a * nh + (1 - a) * oh),
            )
            mvx = new_box[0] - t.box[0]
            mvy = new_box[1] - t.box[1]
            t.vx = self.VEL_ALPHA * mvx + (1 - self.VEL_ALPHA) * t.vx
            t.vy = self.VEL_ALPHA * mvy + (1 - self.VEL_ALPHA) * t.vy
            t.box = new_box

            # Identity update
            if det.get("classified", False):
                if det.get("name") is not None and det.get("conf", 0) >= config.PERSON_SVM_CONFIDENCE_THRES:
                    # Confident recognition
                    t.name = det["name"]
                    t.confidence = det["conf"]
                    t.identity_fresh = True
                    t.last_class_frame = frame_count
                else:
                    # Classifier said unknown
                    t.identity_fresh = False
                    # Stick with old name for a while, then revert
                    if t.name != "Unknown":
                        frames_since = frame_count - t.last_class_frame
                        if frames_since > self.IDENTITY_TIMEOUT:
                            t.name = "Unknown"
                            t.confidence = 0.0
            else:
                # No classification this frame (embedding skipped)
                if t.name != "Unknown":
                    t.identity_fresh = False
                    frames_since = frame_count - t.last_class_frame
                    if frames_since > self.IDENTITY_TIMEOUT:
                        t.name = "Unknown"
                        t.confidence = 0.0

            t.lost = 0
            t.age += 1

        # New tracks
        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            name = None
            conf = 0.0
            if det.get("classified", False):
                if det.get("name") is not None and det.get("conf", 0) >= config.PERSON_SVM_CONFIDENCE_THRES:
                    name = det["name"]
                    conf = det["conf"]

            t = Track(self.next_id, det["box"], name, conf, frame_count)
            self.next_id += 1
            self.tracks.append(t)

        alive = []
        for t in self.tracks:
            limit = self.max_lost if t.name != "Unknown" else self.UNKNOWN_MAX_LOST
            if t.lost <= limit:
                alive.append(t)
        self.tracks = alive
        return self.tracks


# ============================================================
# Auto-Saver (body crops)
# ============================================================
class AutoSaver:
    MIN_CONF = 0.85
    MIN_INTERVAL_S = 2.0
    VARIETY_SIM_THRESH = 0.92
    MAX_PER_SESSION = 100

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.recent = {}
        self.last_save_t = {}
        self.saved_count = {}
        self.total_saved = 0

    def maybe_save(self, name, crop, embedding, confidence):
        if not self.enabled or name is None or embedding is None:
            return False
        if confidence < self.MIN_CONF:
            return False
        h, w = crop.shape[:2]
        if w < 60 or h < 120:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_var < 40:
            return False
        now = time.monotonic()
        last = self.last_save_t.get(name, 0.0)
        if now - last < self.MIN_INTERVAL_S:
            return False
        count = self.saved_count.get(name, 0)
        if count >= self.MAX_PER_SESSION:
            return False
        recent = self.recent.get(name, [])
        if recent:
            sims = np.stack(recent) @ embedding.astype(np.float32)
            if float(sims.max()) > self.VARIETY_SIM_THRESH:
                return False

        person_dir = os.path.join(config.PERSON_TRAINING_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        seq = count + 1
        fname = f"auto_{ts}_{seq:04d}.jpg"
        fpath = os.path.join(person_dir, fname)

        if cv2.imwrite(fpath, crop):
            recent.append(embedding.astype(np.float32))
            if len(recent) > 20:
                recent.pop(0)
            self.recent[name] = recent
            self.last_save_t[name] = now
            self.saved_count[name] = count + 1
            self.total_saved += 1
            print(f"AUTO-SAVED: {fname}  (conf {confidence:.2f}, "
                  f"blur {blur_var:.0f}, {count + 1}/{self.MAX_PER_SESSION})")
            return True
        return False

    def summary(self):
        if self.total_saved == 0:
            return
        print(f"\nAuto-saver saved {self.total_saved} body crops:")
        for n, c in self.saved_count.items():
            print(f"  {n}: {c}")


# ============================================================
# Target Selector
# ============================================================
class TargetSelector:
    def __init__(self, follow_name=None):
        self.follow_name = follow_name
        self.locked_name = None
        self.locked_since = 0
        self.SWITCH_MARGIN = 0.08
        self.HOLD_FRAMES = 30

    def set_name(self, name):
        if name and name != self.follow_name:
            print(f"Target switched to: {name}")
            self.follow_name = name
            self.locked_name = None
            self.locked_since = 0

    def choose(self, frame_count, tracks):
        candidates = [t for t in tracks if t.name != "Unknown" and t.lost == 0]
        if not candidates:
            return None
        if self.follow_name is not None:
            wanted = [t for t in candidates
                      if t.name.casefold() == self.follow_name.casefold()]
            if not wanted:
                return None
            candidates = wanted
        best = max(candidates, key=lambda t: t.confidence)
        if self.locked_name is None:
            self.locked_name = best.name
            self.locked_since = frame_count
            return best
        held = [t for t in candidates if t.name == self.locked_name]
        if not held:
            self.locked_name = best.name
            self.locked_since = frame_count
            return best
        held_best = max(held, key=lambda t: t.confidence)
        duration = frame_count - self.locked_since
        if (duration >= self.HOLD_FRAMES
                and best.name != self.locked_name
                and best.confidence - held_best.confidence > self.SWITCH_MARGIN):
            self.locked_name = best.name
            self.locked_since = frame_count
            return best
        return held_best


# ============================================================
# Depth helpers
# ============================================================
def get_sector_distances(depth, person_box, depth_w, depth_h):
    if depth is None:
        return {"left": np.inf, "center": np.inf, "right": np.inf}
    row_start, row_end = int(depth_h * 0.35), int(depth_h * 0.75)
    band = depth[row_start:row_end, :].copy()
    if person_box is not None:
        x, y, w, h = person_box
        scale_x = depth_w / config.FRAME_W
        dx1 = int(x * scale_x)
        dx2 = int((x + w) * scale_x)
        pad = 40
        band[:, max(0, dx1 - pad):min(depth_w, dx2 + pad)] = 0
    third = depth_w // 3
    sectors = {}
    for name, (c1, c2) in [("left", (0, third)),
                            ("center", (third, 2 * third)),
                            ("right", (2 * third, depth_w))]:
        region = band[:, c1:c2]
        valid = region[region > 0]
        sectors[name] = float(np.min(valid)) if valid.size > 0 else np.inf
    return sectors


def calculate_speed(distance_mm):
    if distance_mm is None:
        return config.FOLLOW_BASE_SPEED
    if distance_mm <= config.REVERSE_DISTANCE_MM:
        return config.REVERSE_SPEED
    elif distance_mm < config.FORWARD_DISTANCE_MM:
        return 0
    else:
        extra = ((distance_mm - config.FORWARD_DISTANCE_MM) // 100) * config.SPEED_INCREASE
        return min(config.FOLLOW_BASE_SPEED + extra, config.MAX_SPEED)


# ============================================================
# Enrollment
# ============================================================
def run_enrollment(cam, detector, person_name, target_count):
    person_name = person_name.strip()
    if not person_name:
        print("Enrollment name cannot be empty.")
        return

    person_dir = os.path.join(config.PERSON_TRAINING_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)
    existing = [f for f in os.listdir(person_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    start_index = len(existing)

    print(f"\nEnrolling '{person_name}'")
    print(f"  Output: {person_dir}")
    print(f"  Existing crops: {start_index}")
    print(f"  Target: {target_count}")
    print(f"  Stand full-body in view. Vary pose, distance, angle.")
    print(f"  Press 'q' to stop early.\n")

    captured = 0
    attempts = 0
    max_attempts = target_count * 40
    last_gray = None
    no_person = 0

    cv2.namedWindow("Enrollment", cv2.WINDOW_NORMAL)
    try:
        while captured < target_count and attempts < max_attempts:
            attempts += 1
            frame = cam.read_color()
            if frame is None:
                time.sleep(0.02)
                continue

            persons = detector.infer(frame)
            best = max(persons, key=lambda p: p[2] * p[3]) if persons else None

            status_color = (0, 0, 255)
            status_text = "No person detected"
            saved = False

            if best is not None:
                x, y, w, h, conf = best
                x = max(0, x); y = max(0, y)
                x2 = min(frame.shape[1], x + w)
                y2 = min(frame.shape[0], y + h)
                crop = frame[y:y2, x:x2]

                ok_size = (w >= 60 and h >= 120)
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size > 0 else None
                blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray is not None else 0

                ok_blur = blur_var >= 40
                ok_div = True
                if last_gray is not None and gray is not None:
                    small = cv2.resize(gray, (64, 64))
                    diff = float(np.mean(np.abs(
                        small.astype(np.float32) - last_gray.astype(np.float32)
                    )))
                    ok_div = diff >= 25

                if crop.size > 0 and ok_size and ok_blur and ok_div:
                    fname = os.path.join(person_dir,
                                        f"enroll_{start_index + captured:04d}.jpg")
                    cv2.imwrite(fname, crop)
                    last_gray = cv2.resize(gray, (64, 64))
                    captured += 1
                    saved = True
                    status_color = (0, 255, 0)
                    status_text = f"Captured {captured}/{target_count} (blur {blur_var:.0f})"
                    print(f"  Captured {captured}/{target_count}  "
                          f"[{w}x{h}, blur {blur_var:.0f}]")
                else:
                    reasons = []
                    if not ok_size: reasons.append(f"too small ({w}x{h})")
                    if not ok_blur: reasons.append(f"blurry ({blur_var:.0f})")
                    if not ok_div: reasons.append("too similar")
                    status_color = (0, 200, 255)
                    status_text = "Skipped: " + ", ".join(reasons)

                box_color = (0, 255, 0) if saved else (0, 200, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            else:
                no_person += 1

            cv2.putText(frame, f"Enroll: {person_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, status_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(frame, f"Progress: {captured}/{target_count}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if no_person > 30:
                cv2.putText(frame, "Step back so your whole body is visible",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

            cv2.imshow("Enrollment", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()

    print(f"\nEnrollment finished: {captured} new crops -> {person_dir}")
    print(f"Total: {start_index + captured}")
    print("Next: python3 train_person_reid.py")


# ============================================================
# Listing / Removal
# ============================================================
def run_list_people():
    train_dir = config.PERSON_TRAINING_DIR
    model_path = config.PERSON_CLASSIFIER_PATH
    trained_names = []
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            trained_names = list(data.get("names", []))
            print(f"Trained classifier: {model_path}")
            print(f"  Classes: {trained_names}")
        except Exception as e:
            print(f"Classifier exists but unreadable: {e}")
    else:
        print(f"No trained classifier yet (expected {model_path})")
    print()
    if not os.path.isdir(train_dir):
        print(f"No training directory at {train_dir}")
        return
    people = sorted([d for d in os.listdir(train_dir)
                     if os.path.isdir(os.path.join(train_dir, d))])
    if not people:
        print("No enrolled people yet.")
        return
    print(f"Enrolled people ({train_dir}):")
    for idx, p in enumerate(people, start=1):
        crops = [f for f in os.listdir(os.path.join(train_dir, p))
                 if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        enroll = [f for f in crops if f.startswith("enroll_")]
        auto = [f for f in crops if f.startswith("auto_")]
        marker = "  [trained]" if p in trained_names else ""
        print(f"  [{idx}] {p:20s} {len(crops):4d} crops "
              f"({len(enroll)} enroll, {len(auto)} auto){marker}")
    print("\nUse --follow NAME to select, or press 1..9 at runtime.")


def run_remove_person(name, delete_auto=False):
    name = name.strip()
    person_dir = os.path.join(config.PERSON_TRAINING_DIR, name)
    if not os.path.isdir(person_dir):
        print(f"No enrolled person named '{name}'.")
        return
    crops = [f for f in os.listdir(person_dir)
             if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    auto_crops = [f for f in crops if f.startswith("auto_")]
    enroll_crops = [f for f in crops if not f.startswith("auto_")]
    if delete_auto and enroll_crops:
        for f in auto_crops:
            os.remove(os.path.join(person_dir, f))
        print(f"Removed {len(auto_crops)} auto crops.")
        print(f"Kept {len(enroll_crops)} enrollment crops.")
        return
    print(f"Removing '{name}': {len(crops)} crops")
    shutil.rmtree(person_dir)
    remaining = [d for d in os.listdir(config.PERSON_TRAINING_DIR)
                 if os.path.isdir(os.path.join(config.PERSON_TRAINING_DIR, d))]
    if not remaining:
        if os.path.exists(config.PERSON_CLASSIFIER_PATH):
            os.remove(config.PERSON_CLASSIFIER_PATH)
            print("No people left; removed classifier.")
    else:
        print(f"Remaining: {remaining}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enroll", metavar="NAME")
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--register-only", metavar="NAME")
    parser.add_argument("--follow", metavar="NAME")
    parser.add_argument("--list-people", action="store_true")
    parser.add_argument("--remove-person", metavar="NAME")
    parser.add_argument("--clear-auto", metavar="NAME")
    parser.add_argument("--test-motors", action="store_true")
    parser.add_argument("--no-auto-save", action="store_true")
    parser.add_argument("--no-auto-train", dest="auto_train",
                        action="store_false", default=True)
    args = parser.parse_args()

    if args.list_people:
        run_list_people()
        return
    if args.remove_person:
        run_remove_person(args.remove_person, delete_auto=False)
        return
    if args.clear_auto:
        run_remove_person(args.clear_auto, delete_auto=True)
        return

    if args.register_only:
        train_dir = config.PERSON_TRAINING_DIR
        if os.path.isdir(train_dir):
            for d in os.listdir(train_dir):
                full = os.path.join(train_dir, d)
                if os.path.isdir(full):
                    shutil.rmtree(full)
                    print(f"Removed existing enrollment: {d}")
        if os.path.exists(config.PERSON_CLASSIFIER_PATH):
            os.remove(config.PERSON_CLASSIFIER_PATH)
            print("Removed old classifier.")
        args.enroll = args.register_only

    cam = Camera()
    detector = BodyDetector(config.CORAL_DETECTION_MODEL,
                            config.CORAL_LABELS)

    if args.enroll:
        run_enrollment(cam, detector, args.enroll, args.samples)
        cam.stop()
        return

    if args.test_motors:
        motors = SerialMotors()
        motors.test_motors()
        try: motors.cleanup()
        except Exception: pass
        cam.stop()
        return

    embedder = BodyEmbedder(config.REID_PATH)
    classifier = PersonClassifier(config.PERSON_CLASSIFIER_PATH)

    if not classifier.available or not classifier.names:
        print("\nNo classifier / no enrolled people.")
        print(f"  1. python3 tracker_person.py --enroll NAME --samples 60")
        print(f"  2. python3 train_person_reid.py")
        cam.stop()
        return

    follow_name = args.follow.strip() if args.follow else None
    if follow_name is None:
        if len(classifier.names) == 1:
            follow_name = classifier.names[0]
            print(f"Only one enrolled -> following '{follow_name}'.")
        else:
            print(f"\nEnrolled: {classifier.names}")
            if sys.stdin.isatty():
                while True:
                    pick = input("Enter name to follow (or 'q' to quit): ").strip()
                    if pick.casefold() == 'q':
                        cam.stop()
                        return
                    matches = [n for n in classifier.names
                               if n.casefold() == pick.casefold()]
                    if matches:
                        follow_name = matches[0]
                        print(f"Following '{follow_name}'.")
                        break
                    print(f"'{pick}' not enrolled.")
            else:
                follow_name = classifier.names[0]
                print(f"Non-interactive: defaulting to '{follow_name}'.")
    else:
        matches = [n for n in classifier.names
                   if n.casefold() == follow_name.casefold()]
        if not matches:
            print(f"'{follow_name}' is not enrolled.")
            cam.stop()
            return
        follow_name = matches[0]
        print(f"Following '{follow_name}' (from --follow).")

    selector = TargetSelector(follow_name=follow_name)
    auto_save_enabled = (getattr(config, "AUTO_SAVE_RECOGNIZED_CROPS", True)
                         and not args.no_auto_save)
    auto_saver = AutoSaver(enabled=auto_save_enabled)
    motors = SerialMotors()

    tracker = SimpleTracker(max_lost=180)
    following = False
    last_drive_status = None
    frame_interval = 0.12
    frame_count = 0

    min_body_w = getattr(config, "MIN_BODY_BOX_W", 60)
    min_body_h = getattr(config, "MIN_BODY_BOX_H", 120)
    min_body_area = getattr(config, "MIN_BODY_BOX_AREA", 10000)

    print(f"\nRunning. Following: {selector.follow_name}")
    print("Controls: q=quit, f=follow, s=stop, 1..9=switch target")
    print(f"Auto-save: {'ON' if auto_save_enabled else 'OFF'}")
    print(f"Auto-train: {'ON' if args.auto_train else 'OFF'}")
    cv2.namedWindow("MedPal Person Tracker", cv2.WINDOW_NORMAL)

    try:
        while True:
            frame_start = time.monotonic()
            frame_count += 1
            frame = cam.read_color()
            if frame is None:
                time.sleep(0.01)
                continue
            depth = cam.read_depth()

            # 1. Detect bodies
            raw = detector.infer(frame)
            bodies = []
            for (bx, by, bw, bh, bconf) in raw:
                if (bw >= min_body_w and bh >= min_body_h
                        and bw * bh >= min_body_area):
                    bodies.append((bx, by, bw, bh, bconf))

            # 2. Embed and classify each body
            detections = []
            for (bx, by, bw, bh, bconf) in bodies:
                bx = max(0, bx); by = max(0, by)
                bx2 = min(frame.shape[1], bx + bw)
                by2 = min(frame.shape[0], by + bh)
                body_crop = frame[by:by2, bx:bx2]

                emb = embedder.embed(body_crop) if body_crop.size > 0 else None
                name, conf = (classifier.predict(emb)
                              if emb is not None else (None, -1.0))

                detections.append({
                    "box": (bx, by, bx2 - bx, by2 - by),
                    "name": name,
                    "conf": conf,
                    "classified": True,
                    "embedding": emb,
                    "crop": body_crop,
                })

            # 3. Tracker update
            tracks = tracker.update(detections, frame_count)

            # 4. Auto-save on confident named tracks
            for det in detections:
                if (det.get("classified") and det["name"] is not None
                        and det.get("embedding") is not None):
                    auto_saver.maybe_save(det["name"], det["crop"],
                                          det["embedding"], det["conf"])

            # 5. Target selection
            best_track = selector.choose(frame_count, tracks)
            best_box = best_track.box if best_track else None
            person_cx = (best_box[0] + best_box[2] // 2) if best_box else None
            person_cy = (best_box[1] + best_box[3] // 2) if best_box else None

            # 6. Draw
            for t in tracks:
                x, y, w, h = t.box
                is_selected = (best_track is not None and t.id == best_track.id)
                if t.name != "Unknown":
                    if t.identity_fresh:
                        color = (0, 255, 0)
                        tag = ""
                    else:
                        color = (0, 165, 255)
                        tag = " (mem)"
                    thickness = 3 if is_selected else 2
                    suffix = " *FOLLOW*" if is_selected else ""
                    label = f"#{t.id} {t.name}: {t.confidence:.2f}{tag}{suffix}"
                else:
                    color = (0, 0, 255); thickness = 2
                    label = f"#{t.id} Unknown"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 7. Depth at target
            target_distance_mm = None
            if best_box is not None and depth is not None:
                dh, dw = depth.shape[:2]
                sx = dw / config.FRAME_W
                sy = dh / config.FRAME_H
                dcx = int(person_cx * sx)
                dcy = int(person_cy * sy)
                if 0 <= dcx < dw and 0 <= dcy < dh:
                    y1 = max(0, dcy - 10); y2 = min(dh, dcy + 10)
                    x1 = max(0, dcx - 10); x2 = min(dw, dcx + 10)
                    roi = depth[y1:y2, x1:x2]
                    valid = roi[roi > 0]
                    if valid.size > 0:
                        target_distance_mm = float(np.median(valid))
                    else:
                        target_distance_mm = float(depth[dcy, dcx])
                x, y, w, h = best_box
                if target_distance_mm is not None:
                    color = (0, 255, 0) if target_distance_mm > config.MIN_DISTANCE_MM else (0, 0, 255)
                    cv2.putText(frame, f"{target_distance_mm:.0f}mm",
                                (x, y + h + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if depth is not None:
                _dh, _dw = depth.shape[:2]
                obstacle_sectors = get_sector_distances(depth, best_box, _dw, _dh)
            else:
                obstacle_sectors = {"left": np.inf, "center": np.inf, "right": np.inf}

            frame_cx = config.FRAME_W // 2
            cx = person_cx

            # 8. Motor control
            if following:
                if best_box is None:
                    if last_drive_status != "no_target":
                        print(f"STATUS: {selector.follow_name} not visible.")
                    last_drive_status = "no_target"
                    motors.stop()
                elif obstacle_sectors["center"] < config.OBSTACLE_STOP_MM:
                    if last_drive_status != "obstacle_stop":
                        print(f"STATUS: Obstacle ahead "
                              f"({obstacle_sectors['center']:.0f} mm).")
                    last_drive_status = "obstacle_stop"
                    motors.stop()
                elif target_distance_mm is None:
                    if last_drive_status != "no_depth":
                        print("STATUS: No depth data.")
                    last_drive_status = "no_depth"
                    motors.stop()
                elif target_distance_mm < config.REVERSE_DISTANCE_MM:
                    if last_drive_status != "reversing":
                        print(f"STATUS: Target too close "
                              f"({target_distance_mm:.0f} mm).")
                    last_drive_status = "reversing"
                    motors.set_speed(config.REVERSE_SPEED)
                    motors.backward()
                    if cx is not None:
                        if cx < frame_cx - 80: motors.turn_left()
                        elif cx > frame_cx + 80: motors.turn_right()
                elif target_distance_mm < config.FORWARD_DISTANCE_MM:
                    last_drive_status = "target_close"
                    motors.stop()
                    if cx is not None:
                        if cx < frame_cx - 80:
                            motors.set_speed(config.FOLLOW_BASE_SPEED); motors.turn_left()
                        elif cx > frame_cx + 80:
                            motors.set_speed(config.FOLLOW_BASE_SPEED); motors.turn_right()
                else:
                    last_drive_status = "tracking"
                    speed = calculate_speed(target_distance_mm)
                    if cx is not None:
                        if cx < frame_cx - 80:
                            motors.set_speed(config.FOLLOW_BASE_SPEED); motors.turn_left()
                        elif cx > frame_cx + 80:
                            motors.set_speed(config.FOLLOW_BASE_SPEED); motors.turn_right()
                        else:
                            motors.set_speed(speed); motors.forward()
                    else:
                        motors.set_speed(speed); motors.forward()
            else:
                last_drive_status = "not_following"
                motors.stop()

            # 9. Overlays
            motor_state = "IDLE"
            motor_color = (128, 128, 128)
            if following:
                if best_box is None:
                    motor_state = "NO TARGET"; motor_color = (0, 0, 255)
                elif target_distance_mm is not None:
                    if target_distance_mm < config.REVERSE_DISTANCE_MM:
                        motor_state = "BACKWARD"; motor_color = (0, 0, 255)
                    elif target_distance_mm < config.FORWARD_DISTANCE_MM:
                        motor_state = "STOP"; motor_color = (0, 165, 255)
                    else:
                        speed = calculate_speed(target_distance_mm)
                        motor_state = f"FWD {speed}%"; motor_color = (0, 255, 0)
                else:
                    motor_state = "STOP (no depth)"; motor_color = (0, 0, 255)
            cv2.putText(frame, f"Motor: {motor_state}",
                        (config.FRAME_W - 280, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, motor_color, 2)

            if auto_save_enabled and auto_saver.saved_count:
                y = 55
                for name, total in sorted(auto_saver.saved_count.items()):
                    cv2.putText(frame, f"{name}:{total}",
                                (config.FRAME_W - 200, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    y += 18

            banner_color = (0, 255, 0) if following else (128, 128, 128)
            cv2.putText(frame, f"Following: {selector.follow_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_color, 2)
            cv2.putText(frame, "Mode: BODY recognition",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            elapsed = time.monotonic() - frame_start
            sleep = max(0, frame_interval - elapsed)
            if sleep > 0:
                time.sleep(sleep)

            cv2.imshow("MedPal Person Tracker", frame)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                following = True
                print(f"Started following '{selector.follow_name}'!")
            elif key == ord('s'):
                following = False
                motors.stop()
                print("Stopped following.")
            elif ord('1') <= key <= ord('9'):
                idx = key - ord('1')
                if 0 <= idx < len(classifier.names):
                    selector.set_name(classifier.names[idx])
                else:
                    print(f"No person at index {idx + 1}")

    except KeyboardInterrupt:
        pass
    finally:
        auto_saver.summary()
        try: motors.cleanup()
        except Exception: pass
        try: cam.stop()
        except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass

        if args.auto_train and auto_saver.total_saved > 0:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            train_script = os.path.join(script_dir, "train_person_reid.py")
            if not os.path.exists(train_script):
                print(f"\nAuto-train skipped: {train_script} not found.")
            else:
                print("\n" + "=" * 60)
                print(f"AUTO-TRAIN: {auto_saver.total_saved} new crops")
                print("=" * 60)
                time.sleep(2.0)
                sys.stdout.flush(); sys.stderr.flush()
                try:
                    os.execv(sys.executable, [sys.executable, train_script])
                except Exception as e:
                    print(f"Auto-train exec failed: {e}")
        elif args.auto_train:
            print("\nAuto-train: no crops saved; skipping.")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nFATAL: {e}\n")
        sys.exit(1)