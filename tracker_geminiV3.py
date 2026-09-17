#!/usr/bin/env python3
"""
tracker_geminiV3.py -- Face detection + FaceNet embedding + classifier.

Pipeline:
  1. Coral TPU SSD MobileNet V2 Face detector finds faces.
  2. Coral FaceNet converts each face crop to a 1024-D embedding.
  3. Centroid + gallery classifier labels each embedding.
  4. IoU tracker smooths detections across frames.
  5. Motor control follows ONLY the selected person (--follow NAME).
  6. In-memory online gallery updates.
  7. Auto-saver with track confirmation and per-person counters.
  8. Optional auto-retrain on exit (--auto-train).
     Uses os.execv to REPLACE this process with the trainer, so the Coral
     USB handle is cleanly released by the kernel before the trainer opens it.
     No concurrent access, no USB transfer error 5, no abort.

Commands:
    --enroll NAME            Capture face crops for a person
    --samples N              Crops per enrollment (default 30)
    --follow NAME            Follow only this person (required if >1 enrolled)
    --list-people            Show enrolled people and classifier state
    --remove-person NAME     Delete a person's crops
    --clear-auto NAME        Delete only auto-saved crops for a person
    --test-motors            Drive each wheel briefly
    --no-auto-save           Disable auto-saving for this run
    --auto-train             Retrain classifier on exit if any crops were saved

Runtime controls (video window):
    f = start following
    s = stop
    1..9 = switch target to the Nth enrolled person (live)
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
    print("PyCoral available for face detection")
except ImportError:
    print("PyCoral not installed (pip install pycoral)")

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
# Face Detector
# ============================================================
class FaceDetector:
    def __init__(self, model_path, label_path=None):
        self.labels = []
        if label_path and os.path.exists(label_path):
            with open(label_path) as f:
                self.labels = [l.strip() for l in f.readlines()]

        self.mode = None
        self.engine = None
        self.interpreter = None
        self.last_top_score = 0.0

        if PYCORAL_AVAILABLE:
            try:
                self.engine = pycoral_make_interpreter(model_path)
                self.engine.allocate_tensors()
                self.mode = "pycoral"
                print("FaceDetector: using PyCoral")
                return
            except Exception as e:
                print(f"FaceDetector: PyCoral failed ({e})")

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
        print("FaceDetector: using raw tflite_runtime")

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

        objs = pycoral_detect.get_objects(self.engine, score_threshold=config.CONF_THRES)
        results = []
        self.last_top_score = 0.0
        for obj in objs:
            self.last_top_score = max(self.last_top_score, float(obj.score))
            bbox = obj.bbox
            x = int(bbox.xmin * sx)
            y = int(bbox.ymin * sy)
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
        self.last_top_score = 0.0
        results = []
        for i in range(int(scores[0].shape[0])):
            score = float(scores[0][i])
            if score > self.last_top_score:
                self.last_top_score = score
            if score < config.CONF_THRES:
                continue
            y1, x1, y2, x2 = boxes[0][i]
            x = int(x1 * w)
            y = int(y1 * h)
            bw = int((x2 - x1) * w)
            bh = int((y2 - y1) * h)
            results.append((x, y, bw, bh, score))
        return results

    def top_detection_label(self):
        return f"top face score {self.last_top_score:.3f}"


# ============================================================
# Face Embedder
# ============================================================
class FaceEmbedder:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face embedding model not found: {model_path}\n"
                f"Download it into /home/medpal/tracking_person/models/tflite/."
            )

        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_h = 224
        self.input_w = 224
        self.embedding_dim = 1024
        self.input_dtype = np.uint8

        if CORAL_EDGETPU:
            delegate = load_delegate('libedgetpu.so.1')
            self.interpreter = Interpreter(model_path, experimental_delegates=[delegate])
        else:
            cpu_path = model_path.replace('_edgetpu', '')
            self.interpreter = Interpreter(cpu_path)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        shape = self.input_details[0]['shape']
        if len(shape) == 4:
            h_candidate, w_candidate = int(shape[1]), int(shape[2])
            if h_candidate >= 32 and w_candidate >= 32:
                self.input_h, self.input_w = h_candidate, w_candidate
            elif int(shape[2]) >= 32 and int(shape[3]) >= 32:
                self.input_h, self.input_w = int(shape[2]), int(shape[3])

        out_shape = self.output_details[0]['shape']
        if len(out_shape) > 0:
            self.embedding_dim = int(out_shape[-1])

        self.input_dtype = self.input_details[0]['dtype']

        print(f"FaceEmbedder: input {self.input_w}x{self.input_h} "
              f"dtype={self.input_dtype.__name__}, "
              f"output {self.embedding_dim}-D")

    def embed(self, face_crop):
        h, w = face_crop.shape[:2]
        if h == 0 or w == 0:
            return None

        if w < 96 or h < 96:
            pad_x = max(0, (96 - w) // 2)
            pad_y = max(0, (96 - h) // 2)
            face_crop = cv2.copyMakeBorder(
                face_crop, pad_y, pad_y, pad_x, pad_x,
                cv2.BORDER_REFLECT_101,
            )

        h2, w2 = face_crop.shape[:2]
        if self.input_w >= w2 and self.input_h >= h2:
            interp = cv2.INTER_CUBIC
        else:
            interp = cv2.INTER_AREA

        img = cv2.resize(face_crop, (self.input_w, self.input_h),
                         interpolation=interp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.input_dtype == np.uint8:
            img = img.astype(np.uint8)
        else:
            img = img.astype(np.float32)
            img = (img - 127.5) / 128.0

        img = np.expand_dims(img, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output = self.interpreter.tensor(self.output_details[0]['index'])()

        vec = output.flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


# ============================================================
# Face Recognizer
# ============================================================
class FaceRecognizer:
    MAX_GALLERY = 500
    UPDATE_MIN_CONF = 0.90
    DUP_SIM_THRESH = 0.95

    def __init__(self, model_path):
        self.names = []
        self.centroids = None
        self.galleries = None
        self.stats_p10 = None
        self.stats_p50 = None
        self.stats_p90 = None
        self.available = False

        if not os.path.exists(model_path):
            print(f"FaceRecognizer: no classifier at {model_path}. "
                  f"Run train_face_svm.py first.")
            return

        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"FaceRecognizer: failed to load classifier ({e})")
            return

        if data.get("model_type") != "centroid_gallery":
            print(f"FaceRecognizer: classifier is '{data.get('model_type')}', "
                  f"expected 'centroid_gallery'. Retrain with train_face_svm.py.")
            return

        self.names = list(data["names"])
        self.centroids = data["centroids"]
        self.galleries = [np.asarray(g, dtype=np.float32) for g in data["galleries"]]
        self.stats_p10 = data["stats_p10"]
        self.stats_p50 = data["stats_p50"]
        self.stats_p90 = data["stats_p90"]
        self.available = True

        sizes = [g.shape[0] for g in self.galleries]
        print(f"FaceRecognizer: loaded centroid+gallery classifier "
              f"for {self.names} (gallery sizes: {sizes})")

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

        if confidence >= config.FACE_SVM_CONFIDENCE_THRES:
            return name, confidence
        return None, confidence

    def update_gallery(self, name, embedding, confidence):
        if not self.available or name not in self.names or embedding is None:
            return False
        if confidence < self.UPDATE_MIN_CONF:
            return False

        idx = self.names.index(name)
        g = self.galleries[idx]

        q = embedding.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)

        if len(g) > 0 and float((g @ q).max()) >= self.DUP_SIM_THRESH:
            return False

        g = np.vstack([g, q])
        if g.shape[0] > self.MAX_GALLERY:
            g = g[-self.MAX_GALLERY:]
        self.galleries[idx] = g
        return True


# ============================================================
# Track Confirmation Cache
# ============================================================
class TrackConfirmationCache:
    HOLD_FRAMES = 60
    IOU_THRESH = 0.4
    CACHE_MIN_CONF = 0.85

    def __init__(self):
        self.entries = []

    def update(self, frame_count, box, name, confidence):
        if name != "Unknown" and confidence >= self.CACHE_MIN_CONF:
            self.entries.append((frame_count, box, name))
        self.entries = [e for e in self.entries
                        if frame_count - e[0] <= self.HOLD_FRAMES]

    def confirmed_name_for(self, frame_count, box):
        best_name = None
        best_iou = 0.0
        for (f, b, name) in self.entries:
            if frame_count - f > self.HOLD_FRAMES:
                continue
            v = iou(b, box)
            if v > best_iou:
                best_iou = v
                best_name = name
        if best_name is not None and best_iou >= self.IOU_THRESH:
            return best_name
        return None


# ============================================================
# Auto-Saver (per-person counters)
# ============================================================
class AutoSaver:
    MIN_CONF_DIRECT = 0.92
    MIN_CONF_TRACK = 0.60
    MIN_INTERVAL_S = 3.0
    VARIETY_SIM_THRESH = 0.90
    RECENT_WINDOW = 20
    MAX_PER_SESSION = 50

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.recent = {}
        self.last_save_t = {}
        self.saved_count = {}
        self.saved_direct = {}
        self.saved_track = {}
        self.total_saved = 0
        self.total_direct = 0
        self.total_track = 0

    def maybe_save(self, name, face_crop, embedding, confidence,
                   confirmed_by_track=False):
        if not self.enabled or name is None or embedding is None:
            return False

        min_conf = (self.MIN_CONF_TRACK if confirmed_by_track
                    else self.MIN_CONF_DIRECT)
        if confidence < min_conf:
            return False

        h, w = face_crop.shape[:2]
        if w < config.ENROLL_MIN_FACE_PX or h < config.ENROLL_MIN_FACE_PX:
            return False

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_var < config.ENROLL_BLUR_THRES:
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
            recent_arr = np.stack(recent)
            sims = recent_arr @ embedding.astype(np.float32)
            if float(sims.max()) > self.VARIETY_SIM_THRESH:
                return False

        person_dir = os.path.join(config.FACE_TRAINING_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        seq = count + 1
        tag = "auto_trk" if confirmed_by_track else "auto_dir"
        fname = f"{tag}_{ts}_{seq:03d}.jpg"
        fpath = os.path.join(person_dir, fname)

        if cv2.imwrite(fpath, face_crop):
            recent.append(embedding.astype(np.float32))
            if len(recent) > self.RECENT_WINDOW:
                recent.pop(0)
            self.recent[name] = recent
            self.last_save_t[name] = now
            self.saved_count[name] = count + 1
            self.total_saved += 1
            if confirmed_by_track:
                self.total_track += 1
                self.saved_track[name] = self.saved_track.get(name, 0) + 1
            else:
                self.total_direct += 1
                self.saved_direct[name] = self.saved_direct.get(name, 0) + 1
            print(f"AUTO-SAVED: {fname}  "
                  f"(conf {confidence:.2f}, "
                  f"{'track' if confirmed_by_track else 'direct'}, "
                  f"blur {blur_var:.0f}, "
                  f"session {count + 1}/{self.MAX_PER_SESSION})")
            return True
        return False

    def summary(self):
        if self.total_saved == 0:
            return
        print(f"\nAuto-saver saved {self.total_saved} crops this session "
              f"({self.total_direct} direct, {self.total_track} track):")
        for n in self.saved_count:
            d = self.saved_direct.get(n, 0)
            t = self.saved_track.get(n, 0)
            print(f"  {n}: {self.saved_count[n]} ({d} direct, {t} track)")


# ============================================================
# IoU Tracker
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
    def __init__(self, tid, box, name, conf):
        self.id = tid
        self.box = box
        self.name = name
        self.confidence = conf
        self.age = 1
        self.lost = 0
        self.match_votes = 0


class SimpleTracker:
    def __init__(self, iou_thres=0.3, max_lost=15):
        self.iou_thres = iou_thres
        self.max_lost = max_lost
        self.tracks = []
        self.next_id = 0

    def update(self, detections):
        for t in self.tracks:
            t.lost += 1

        pairs = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                v = iou(track.box, det["box"])
                if v >= self.iou_thres:
                    pairs.append((v, ti, di))
        pairs.sort(reverse=True)

        matched_tracks, matched_dets = set(), set()
        for _, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            t = self.tracks[ti]
            t.box = detections[di]["box"]
            t.name = detections[di]["name"]
            t.confidence = detections[di]["conf"]
            t.lost = 0
            t.age += 1

        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            t = Track(self.next_id, det["box"], det["name"], det["conf"])
            self.next_id += 1
            self.tracks.append(t)

        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]
        return self.tracks


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

        held_duration = frame_count - self.locked_since
        if (held_duration >= self.HOLD_FRAMES
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
def run_enrollment(cam, face_detector, person_name, target_count):
    person_name = person_name.strip()
    if not person_name:
        print("Enrollment name cannot be empty.")
        return

    person_dir = os.path.join(config.FACE_TRAINING_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    existing = [f for f in os.listdir(person_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    start_index = len(existing)

    print(f"\nEnrolling '{person_name}'")
    print(f"  Output directory: {person_dir}")
    print(f"  Existing crops:   {start_index}")
    print(f"  Will capture:     {target_count}")
    print(f"  Look at the camera. Vary your pose and distance slightly.")
    print(f"  For best results, run this 2-3 times at different distances.")
    print(f"  Press 'q' to stop early.\n")

    captured = 0
    attempts = 0
    max_attempts = target_count * 40
    last_saved_gray = None
    no_face_frames = 0

    cv2.namedWindow("Enrollment", cv2.WINDOW_NORMAL)

    try:
        while captured < target_count and attempts < max_attempts:
            attempts += 1
            frame = cam.read_color()
            if frame is None:
                time.sleep(0.02)
                continue

            faces = face_detector.infer(frame)
            best = None
            if faces:
                best = max(faces, key=lambda f: f[2] * f[3])

            status_color = (0, 0, 255)
            status_text = "No face detected"
            saved_this_frame = False

            if best is not None:
                x, y, w, h, conf = best
                x = max(0, x); y = max(0, y)
                x2 = min(frame.shape[1], x + w)
                y2 = min(frame.shape[0], y + h)
                face_crop = frame[y:y2, x:x2]

                if face_crop.size == 0:
                    continue

                ok_size = w >= config.ENROLL_MIN_FACE_PX and h >= config.ENROLL_MIN_FACE_PX

                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                ok_blur = blur_var >= config.ENROLL_BLUR_THRES

                ok_diverse = True
                if last_saved_gray is not None:
                    try:
                        resized = cv2.resize(gray, (64, 64))
                        diff = float(np.mean(np.abs(
                            resized.astype(np.float32) -
                            last_saved_gray.astype(np.float32)
                        )))
                        ok_diverse = diff >= config.ENROLL_DIVERSITY_PX
                    except Exception:
                        ok_diverse = True

                if ok_size and ok_blur and ok_diverse:
                    fname = os.path.join(person_dir,
                                        f"enroll_{start_index + captured:04d}.jpg")
                    cv2.imwrite(fname, face_crop)
                    last_saved_gray = cv2.resize(gray, (64, 64))
                    captured += 1
                    saved_this_frame = True
                    status_color = (0, 255, 0)
                    status_text = (f"Captured {captured}/{target_count} "
                                   f"(blur {blur_var:.0f})")
                    print(f"  Captured {captured}/{target_count}  "
                          f"[size {w}x{h}, blur {blur_var:.0f}]")
                else:
                    reasons = []
                    if not ok_size:
                        reasons.append(f"too small ({w}x{h})")
                    if not ok_blur:
                        reasons.append(f"blurry ({blur_var:.0f})")
                    if not ok_diverse:
                        reasons.append("too similar to last")
                    status_color = (0, 200, 255)
                    status_text = "Skipped: " + ", ".join(reasons)

                box_color = (0, 255, 0) if saved_this_frame else (0, 200, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            else:
                no_face_frames += 1

            cv2.putText(frame, f"Enroll: {person_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, status_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(frame, f"Progress: {captured}/{target_count}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if no_face_frames > 30:
                cv2.putText(frame, "Move closer / improve lighting",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 200, 255), 1)

            cv2.imshow("Enrollment", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()

    print(f"\nEnrollment finished: {captured} new crops saved to {person_dir}")
    print(f"Total crops for '{person_name}': {start_index + captured}")
    print("Next step: python3 train_face_svm.py")


# ============================================================
# Listing
# ============================================================
def run_list_people():
    train_dir = config.FACE_TRAINING_DIR
    svm_path = config.SVM_MODEL_PATH

    trained_names = []
    if os.path.exists(svm_path):
        try:
            with open(svm_path, "rb") as f:
                data = pickle.load(f)
            model_type = data.get("model_type", "unknown")
            trained_names = list(data.get("names", []))
            print(f"Trained classifier: {svm_path}")
            print(f"  Model type: {model_type}")
            print(f"  Classes:    {trained_names}")
        except Exception as e:
            print(f"Trained classifier exists but could not be read: {e}")
    else:
        print(f"No trained classifier yet (expected at {svm_path})")

    print()

    if not os.path.isdir(train_dir):
        print(f"No training directory at {train_dir}")
        return

    people = sorted([d for d in os.listdir(train_dir)
                     if os.path.isdir(os.path.join(train_dir, d))])
    if not people:
        print("No enrolled people yet.")
        print(f"Enroll with:  python3 tracker_geminiV3.py --enroll NAME")
        return

    print(f"Enrolled people ({train_dir}):")
    for idx, p in enumerate(people, start=1):
        all_crops = [f for f in os.listdir(os.path.join(train_dir, p))
                     if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        enroll_crops = [f for f in all_crops if f.startswith("enroll_")]
        auto_direct = [f for f in all_crops if f.startswith("auto_dir_")]
        auto_track = [f for f in all_crops if f.startswith("auto_trk_")]
        other = len(all_crops) - len(enroll_crops) - len(auto_direct) - len(auto_track)
        marker = "  [trained]" if p in trained_names else ""
        print(f"  [{idx}] {p:20s} {len(all_crops):4d} crops "
              f"({len(enroll_crops)} enroll, "
              f"{len(auto_direct)} auto_dir, "
              f"{len(auto_track)} auto_trk"
              + (f", {other} other" if other > 0 else "")
              + f"){marker}")
    print("\nUse --follow NAME to select one, or press 1..9 at runtime.")


# ============================================================
# Removal
# ============================================================
def run_remove_person(name, delete_auto=False):
    name = name.strip()
    person_dir = os.path.join(config.FACE_TRAINING_DIR, name)

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
        print(f"Removed {len(auto_crops)} auto-saved crops for '{name}'.")
        print(f"Kept {len(enroll_crops)} enrollment crops.")
        print("Re-run train_face_svm.py to retrain without the auto crops.")
        return

    print(f"Removing '{name}': {len(crops)} crops from {person_dir}")
    shutil.rmtree(person_dir)

    remaining = [d for d in os.listdir(config.FACE_TRAINING_DIR)
                 if os.path.isdir(os.path.join(config.FACE_TRAINING_DIR, d))]

    if not remaining:
        if os.path.exists(config.SVM_MODEL_PATH):
            os.remove(config.SVM_MODEL_PATH)
            print("No people left; removed the stale classifier as well.")
    else:
        print(f"Remaining people: {remaining}")
        print("Re-run train_face_svm.py to retrain without this person.")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enroll", metavar="NAME",
                        help="Enroll (or add to) a person's face samples")
    parser.add_argument("--samples", type=int, default=config.ENROLL_DEFAULT_COUNT,
                        help=f"Number of crops per enrollment "
                             f"(default: {config.ENROLL_DEFAULT_COUNT})")
    parser.add_argument("--follow", metavar="NAME",
                        help="Only follow this enrolled person")
    parser.add_argument("--list-people", action="store_true",
                        help="List enrolled people and classifier state")
    parser.add_argument("--remove-person", metavar="NAME",
                        help="Delete all face crops for a person")
    parser.add_argument("--clear-auto", metavar="NAME",
                        help="Delete only auto-saved crops for a person")
    parser.add_argument("--test-motors", action="store_true")
    parser.add_argument("--no-auto-save", action="store_true",
                        help="Disable auto-saving of recognized crops")
    parser.add_argument("--auto-train", action="store_true",
                        help="Retrain classifier on exit if any crops were saved")
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

    cam = Camera()
    face_detector = FaceDetector(config.CORAL_FACE_DETECTION_MODEL,
                                  config.CORAL_FACE_LABELS)

    if args.enroll:
        run_enrollment(cam, face_detector, args.enroll, args.samples)
        cam.stop()
        return

    if args.test_motors:
        motors = SerialMotors()
        motors.test_motors()
        try:
            motors.cleanup()
        except Exception:
            pass
        cam.stop()
        return

    face_embedder = FaceEmbedder(config.MOBILEFACENET_MODEL)
    face_recognizer = FaceRecognizer(config.SVM_MODEL_PATH)

    follow_name = args.follow.strip() if args.follow else None

    if not face_recognizer.available or not face_recognizer.names:
        print("\nNo classifier / no enrolled people. Cannot follow.")
        cam.stop()
        return

    if follow_name is None:
        if len(face_recognizer.names) == 1:
            follow_name = face_recognizer.names[0]
            print(f"Only one enrolled person -> following '{follow_name}'.")
        else:
            print(f"\nEnrolled people: {face_recognizer.names}")
            if sys.stdin.isatty():
                while True:
                    pick = input("Enter name to follow (or 'q' to quit): ").strip()
                    if pick.casefold() == 'q':
                        cam.stop()
                        return
                    matches = [n for n in face_recognizer.names
                               if n.casefold() == pick.casefold()]
                    if matches:
                        follow_name = matches[0]
                        print(f"Following '{follow_name}'.")
                        break
                    print(f"'{pick}' is not enrolled. Try again.")
            else:
                follow_name = face_recognizer.names[0]
                print(f"Non-interactive: defaulting to '{follow_name}'.")
    else:
        matches = [n for n in face_recognizer.names
                   if n.casefold() == follow_name.casefold()]
        if not matches:
            print(f"'{follow_name}' is not enrolled. "
                  f"Enrolled people: {face_recognizer.names}")
            cam.stop()
            return
        follow_name = matches[0]
        print(f"Following '{follow_name}' (from --follow).")

    selector = TargetSelector(follow_name=follow_name)

    auto_save_enabled = (
        getattr(config, "AUTO_SAVE_RECOGNIZED_CROPS", True)
        and not args.no_auto_save
    )
    auto_saver = AutoSaver(enabled=auto_save_enabled)

    motors = SerialMotors()

    tracker = SimpleTracker(iou_thres=0.3, max_lost=15)
    following = False
    last_drive_status = None
    frame_interval = 0.12

    frame_count = 0
    track_cache = TrackConfirmationCache()

    print(f"\nRunning. Currently following: {selector.follow_name}")
    print("Controls: q=quit, f=follow, s=stop, 1..9=switch target")
    print(f"Auto-save: {'ON' if auto_save_enabled else 'OFF'} "
          f"(direct >= {AutoSaver.MIN_CONF_DIRECT}, "
          f"track >= {AutoSaver.MIN_CONF_TRACK})")
    if args.auto_train:
        print("Auto-train: ON (will retrain on exit if any crops were saved)")
    cv2.namedWindow("MedPal Face Tracker", cv2.WINDOW_NORMAL)

    try:
        while True:
            frame_start = time.monotonic()
            frame_count += 1
            frame = cam.read_color()
            if frame is None:
                time.sleep(0.01)
                continue
            depth = cam.read_depth()

            faces = face_detector.infer(frame)

            detections = []
            for (x, y, w, h, det_conf) in faces:
                x = max(0, x); y = max(0, y)
                x2 = min(frame.shape[1], x + w)
                y2 = min(frame.shape[0], y + h)
                face_crop = frame[y:y2, x:x2]
                if face_crop.size == 0:
                    continue

                box = (x, y, x2 - x, y2 - y)

                embedding = face_embedder.embed(face_crop)
                name, svm_conf = face_recognizer.predict(embedding)

                track_confirmed_name = track_cache.confirmed_name_for(
                    frame_count, box)

                if name is not None and embedding is not None:
                    face_recognizer.update_gallery(name, embedding, svm_conf)
                    confirmed_by_track = (track_confirmed_name == name)
                    auto_saver.maybe_save(
                        name, face_crop, embedding, svm_conf,
                        confirmed_by_track=confirmed_by_track,
                    )

                track_cache.update(
                    frame_count, box, name if name is not None else "Unknown",
                    svm_conf,
                )

                label = name if name is not None else "Unknown"

                detections.append({
                    "box": box,
                    "name": label,
                    "conf": svm_conf if name else det_conf,
                })

            tracks = tracker.update(detections)

            best_track = selector.choose(frame_count, tracks)

            best_box = best_track.box if best_track else None
            person_cx = (best_box[0] + best_box[2] // 2) if best_box else None
            person_cy = (best_box[1] + best_box[3] // 2) if best_box else None

            for t in tracks:
                x, y, w, h = t.box
                is_selected = (best_track is not None and t.id == best_track.id)

                if t.name != "Unknown":
                    if is_selected:
                        color = (0, 255, 0)
                        thickness = 3
                        label = f"#{t.id} {t.name}: {t.confidence:.2f} *FOLLOW*"
                    else:
                        color = (0, 200, 0)
                        thickness = 2
                        label = f"#{t.id} {t.name}: {t.confidence:.2f}"
                else:
                    color = (0, 0, 255)
                    thickness = 2
                    label = f"#{t.id} Unknown"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            target_distance_mm = None
            if best_box is not None and depth is not None:
                depth_h, depth_w = depth.shape[:2]
                scale_x = depth_w / config.FRAME_W
                scale_y = depth_h / config.FRAME_H
                depth_cx = int(person_cx * scale_x)
                depth_cy = int(person_cy * scale_y)
                if depth_cx < depth.shape[1] and depth_cy < depth.shape[0]:
                    y1 = max(0, depth_cy - 10); y2 = min(depth.shape[0], depth_cy + 10)
                    x1 = max(0, depth_cx - 10); x2 = min(depth.shape[1], depth_cx + 10)
                    roi = depth[y1:y2, x1:x2]
                    valid = roi[roi > 0]
                    if valid.size > 0:
                        target_distance_mm = float(np.median(valid))
                    else:
                        target_distance_mm = float(depth[depth_cy, depth_cx])

                x, y, w, h = best_box
                if target_distance_mm is not None:
                    color = (0, 255, 0) if target_distance_mm > config.MIN_DISTANCE_MM else (0, 0, 255)
                    cv2.putText(frame, f"{target_distance_mm:.0f}mm", (x, y + h + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if depth is not None:
                _dh, _dw = depth.shape[:2]
                obstacle_sectors = get_sector_distances(depth, best_box, _dw, _dh)
            else:
                obstacle_sectors = {"left": np.inf, "center": np.inf, "right": np.inf}

            frame_cx = config.FRAME_W // 2
            cx = person_cx

            if following:
                if best_box is None:
                    if last_drive_status != "no_target":
                        print(f"STATUS: {selector.follow_name} not visible. Robot stopped.")
                    last_drive_status = "no_target"
                    motors.stop()
                elif obstacle_sectors["center"] < config.OBSTACLE_STOP_MM:
                    if last_drive_status != "obstacle_stop":
                        print(f"STATUS: Obstacle ahead ({obstacle_sectors['center']:.0f} mm).")
                    last_drive_status = "obstacle_stop"
                    motors.stop()
                elif target_distance_mm is None:
                    if last_drive_status != "no_depth":
                        print("STATUS: No depth data. Stopped.")
                    last_drive_status = "no_depth"
                    motors.stop()
                elif target_distance_mm < config.REVERSE_DISTANCE_MM:
                    if last_drive_status != "reversing":
                        print(f"STATUS: Target too close ({target_distance_mm:.0f} mm).")
                    last_drive_status = "reversing"
                    motors.set_speed(config.REVERSE_SPEED)
                    motors.backward()
                    if cx is not None:
                        if cx < frame_cx - 80:
                            motors.turn_left()
                        elif cx > frame_cx + 80:
                            motors.turn_right()
                elif target_distance_mm < config.FORWARD_DISTANCE_MM:
                    last_drive_status = "target_close"
                    motors.stop()
                    if cx is not None:
                        if cx < frame_cx - 80:
                            motors.set_speed(config.FOLLOW_BASE_SPEED)
                            motors.turn_left()
                        elif cx > frame_cx + 80:
                            motors.set_speed(config.FOLLOW_BASE_SPEED)
                            motors.turn_right()
                else:
                    last_drive_status = "tracking"
                    speed = calculate_speed(target_distance_mm)
                    if cx is not None:
                        if cx < frame_cx - 80:
                            motors.set_speed(config.FOLLOW_BASE_SPEED)
                            motors.turn_left()
                        elif cx > frame_cx + 80:
                            motors.set_speed(config.FOLLOW_BASE_SPEED)
                            motors.turn_right()
                        else:
                            motors.set_speed(speed)
                            motors.forward()
                    else:
                        motors.set_speed(speed)
                        motors.forward()
            else:
                last_drive_status = "not_following"
                motors.stop()

            motor_state = "IDLE"
            motor_color = (128, 128, 128)
            if following:
                if best_box is None:
                    motor_state = "NO TARGET"
                    motor_color = (0, 0, 255)
                elif target_distance_mm is not None:
                    if target_distance_mm < config.REVERSE_DISTANCE_MM:
                        motor_state = "BACKWARD"
                        motor_color = (0, 0, 255)
                    elif target_distance_mm < config.FORWARD_DISTANCE_MM:
                        motor_state = "STOP"
                        motor_color = (0, 165, 255)
                    else:
                        speed = calculate_speed(target_distance_mm)
                        motor_state = f"FWD {speed}%"
                        motor_color = (0, 255, 0)
                else:
                    motor_state = "STOP (no depth)"
                    motor_color = (0, 0, 255)
            cv2.putText(frame, f"Motor: {motor_state}", (config.FRAME_W - 280, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, motor_color, 2)

            if auto_save_enabled and auto_saver.saved_count:
                items = sorted(auto_saver.saved_count.items())
                y = 55
                for name, total in items:
                    d = auto_saver.saved_direct.get(name, 0)
                    t = auto_saver.saved_track.get(name, 0)
                    text = f"{name}:{total} (d{d}/t{t})"
                    cv2.putText(
                        frame, text,
                        (config.FRAME_W - 260, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 255, 255), 1,
                    )
                    y += 20

            banner_color = (0, 255, 0) if following else (128, 128, 128)
            cv2.putText(frame, f"Following: {selector.follow_name}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_color, 2)

            elapsed = time.monotonic() - frame_start
            sleep = max(0, frame_interval - elapsed)
            if sleep > 0:
                time.sleep(sleep)

            cv2.imshow("MedPal Face Tracker", frame)

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
                if 0 <= idx < len(face_recognizer.names):
                    new_name = face_recognizer.names[idx]
                    selector.set_name(new_name)
                else:
                    print(f"No person at index {idx + 1}")

    except KeyboardInterrupt:
        pass
    finally:
        auto_saver.summary()

        # Release hardware in the parent before any handoff
        try:
            motors.cleanup()
        except Exception:
            pass
        try:
            cam.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        if args.auto_train and auto_saver.total_saved > 0:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            train_script = os.path.join(script_dir, "train_face_svm.py")

            if not os.path.exists(train_script):
                print(f"\nAuto-train skipped: {train_script} not found.")
            else:
                print("\n" + "=" * 60)
                print("AUTO-TRAIN: handing off to train_face_svm.py")
                print("=" * 60)

                # Let the USB subsystem settle after the camera release
                time.sleep(2.0)

                sys.stdout.flush()
                sys.stderr.flush()

                # execv REPLACES this process with the training script.
                # All file descriptors are closed by the kernel on exec,
                # including the Coral USB handle held by our tflite_runtime
                # interpreters. The new process opens the Coral cleanly.
                try:
                    os.execv(sys.executable,
                             [sys.executable, train_script])
                except Exception as e:
                    print(f"Auto-train exec failed: {e}")
                    print("Falling back to subprocess.run...")
                    try:
                        import subprocess as _sp
                        _sp.run([sys.executable, train_script],
                                cwd=script_dir, check=False)
                    except Exception as e2:
                        print(f"Fallback also failed: {e2}")

        elif args.auto_train:
            print("\nAuto-train: no crops saved this session; skipping.")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nFATAL: {e}\n")
        sys.exit(1)
