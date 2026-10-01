#!/usr/bin/env python3
"""
tracker_geminiV3.3.py -- Coral detection + CPU ArcFace + identity lock
                        + clothing signature + Flask web control panel
                        + RFID + touch (gated on lock) + hold-to-move web.

Key behaviors in this version:
  - Touch sensor toggles the dispenser servo ONLY when the follow target
    is recognized and locked. Ghost touches are also gated by Pi-side
    debounce, in addition to the ESP32 firmware debounce.
  - Web manual drive uses hold-to-move: the browser resends the command
    every 200 ms while the button is held; the tracker expires the
    command after 500 ms of silence and stops the motors.
  - Frames published to the web stream include all boxes, labels, and
    overlays that the local window shows.
"""

import os
import sys

os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import numpy as np
import pickle
import argparse
import time
import threading
import shutil
import math
import heapq
import subprocess

sys.path.append('/home/medpal/pyorbbecsdk_v1/build')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
from scrfd_detector import SCRFDDetector
from arcface_embedder import ArcFaceEmbedder
from path_planner import PathPlanner
from clothing_signature import (
    extract_clothing_signature,
    compare_signatures,
    update_reference_signature,
)

from shared import command_queue, state, frame_lock, frame_holder
from web_server import (start_web_server, set_motors_instance,
                        on_rfid_enrolled, on_rfid_timeout,
                        _load_rfid_persons, _load_auth_uids)


# ============================================================
# Module-level touch state
# ============================================================
# Tracks when the last servo toggle happened, so a burst of ghost
# triggers can't fire the servo more than once per debounce window.
_touch_state = {
    "servo_open": False,
    "last_toggle_t": 0.0,
    "debounce_s": 1.0,
}


# ============================================================
# Depth profile chooser
# ============================================================
def _choose_depth_profile(profiles, preferred):
    n = profiles.get_count()
    candidates = []
    for i in range(n):
        pr = profiles.get_stream_profile_by_index(i)
        try:
            vp = pr.as_video_stream_profile()
        except AttributeError:
            continue
        try:
            fmt = vp.get_format()
            if fmt == OBFormat.Y12:
                fmt_rank = 0
            elif fmt == OBFormat.Y11:
                fmt_rank = 1
            else:
                fmt_rank = 2
        except NameError:
            fmt_rank = 2
        candidates.append((vp.get_width(),
                           vp.get_height(),
                           vp.get_fps(),
                           fmt_rank,
                           vp))
    for want_w, want_h, want_fps in preferred:
        matches = [c for c in candidates
                   if c[0] == want_w
                   and c[1] == want_h
                   and c[2] == want_fps]
        if matches:
            matches.sort(key=lambda c: c[3])
            return matches[0][4], (want_w, want_h, want_fps)
    return None, None


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
        self.depth_running = False
        self.use_orbbec_depth = False
        self._depth_bpv_logged = False

        self.color_thread = threading.Thread(
            target=self._color_capture, daemon=True)
        self.color_thread.start()
        print("Using cv2.VideoCapture for color")

        if USE_ORBBEC_DEPTH:
            try:
                self.depth_pipeline = Pipeline()
                self.depth_config = Config()
                profiles = self.depth_pipeline.get_stream_profile_list(
                    OBSensorType.DEPTH_SENSOR)

                self.depth_profile, geom = _choose_depth_profile(
                    profiles,
                    preferred=[
                        (320, 240, 30),
                        (320, 240, 15),
                        (640, 480, 30),
                        (640, 480, 15),
                        (160, 120, 30),
                    ])

                if self.depth_profile is None:
                    default = profiles.get_default_video_stream_profile()
                    try:
                        default = default.as_video_stream_profile()
                    except AttributeError:
                        pass
                    self.depth_profile = default
                    print(f"[camera] no preferred profile; "
                          f"using default "
                          f"{self.depth_profile.get_width()}x"
                          f"{self.depth_profile.get_height()}"
                          f"@{self.depth_profile.get_fps()} "
                          f"fmt={self.depth_profile.get_format()}")
                else:
                    print(f"[camera] depth profile: "
                          f"{geom[0]}x{geom[1]}@{geom[2]}")
                    state["depth_resolution"] = (
                        f"{geom[0]}x{geom[1]}@{geom[2]}")

                self.depth_config.enable_stream(self.depth_profile)
                self.depth_pipeline.start(self.depth_config)
                self.depth_running = True
                self.depth_thread = threading.Thread(
                    target=self._depth_capture, daemon=True)
                self.depth_thread.start()
                self.use_orbbec_depth = True
            except Exception as e:
                print(f"Depth pipeline unavailable: {e}")
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
                        try:
                            scale = df.get_depth_scale()
                        except AttributeError:
                            scale = 1.0

                        raw = df.get_data()

                        if not self._depth_bpv_logged:
                            n_bytes = len(raw)
                            exp16 = h * w * 2
                            exp_y11 = (h * w * 11 + 7) // 8
                            exp_y12 = (h * w * 12 + 7) // 8
                            print(f"[camera] depth bytes/frame: "
                                  f"{n_bytes}  "
                                  f"(uint16 would be {exp16}, "
                                  f"Y11 packed {exp_y11}, "
                                  f"Y12 packed {exp_y12})")
                            self._depth_bpv_logged = True

                        data = np.frombuffer(raw, dtype=np.uint16)
                        if data.size >= h * w:
                            data = data[: h * w].reshape(h, w)
                        else:
                            continue
                        depth_mm = data.astype(np.float32) * scale
                        depth_mm[depth_mm < config.MIN_VALID_DEPTH_MM] = 0
                        self.depth_frame = depth_mm
            except Exception as e:
                print(f"Depth capture error: {e}")
                time.sleep(1.0)

    def read_color(self):
        return self.color_frame

    def read_depth(self):
        return self.depth_frame

    def stop(self):
        self.color_running = False
        try:
            self.color_thread.join(timeout=1.0)
        except Exception:
            pass
        self.color_cap.release()
        if self.use_orbbec_depth:
            self.depth_running = False
            try:
                self.depth_thread.join(timeout=1.0)
            except Exception:
                pass
            try:
                self.depth_pipeline.stop()
            except Exception:
                pass


# ============================================================
# Face Detector
# ============================================================
class FaceDetector:
    def __init__(self, model_path, label_path=None):
        if not PYCORAL_AVAILABLE and not CORAL_AVAILABLE:
            raise RuntimeError(
                "FaceDetector needs pycoral or tflite_runtime.")
        self.labels = []
        if label_path and os.path.exists(label_path):
            with open(label_path) as f:
                self.labels = [l.strip() for l in f.readlines()]
        self.mode = None
        self.engine = None
        self.interpreter = None

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
            self.interpreter = Interpreter(
                model_path, experimental_delegates=[delegate])
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
        objs = pycoral_detect.get_objects(
            self.engine, score_threshold=config.CONF_THRES)
        results = []
        for obj in objs:
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
        self.interpreter.set_tensor(
            self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        boxes = self.interpreter.tensor(
            self.output_details[0]['index'])()
        classes = self.interpreter.tensor(
            self.output_details[1]['index'])()
        scores = self.interpreter.tensor(
            self.output_details[2]['index'])()
        results = []
        for i in range(int(scores[0].shape[0])):
            score = float(scores[0][i])
            if score < config.CONF_THRES:
                continue
            y1, x1, y2, x2 = boxes[0][i]
            x = int(x1 * w); y = int(y1 * h)
            bw = int((x2 - x1) * w); bh = int((y2 - y1) * h)
            results.append((x, y, bw, bh, score))
        return results


# ============================================================
# Body Detector
# ============================================================
class BodyDetector:
    def __init__(self, model_path, label_path=None):
        if not PYCORAL_AVAILABLE and not CORAL_AVAILABLE:
            raise RuntimeError(
                "BodyDetector needs pycoral or tflite_runtime.")
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
        print(f"BodyDetector: person class index = {self.PERSON_CLASS}, "
              f"conf_thres = {self.conf_thres}")

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
            self.interpreter = Interpreter(
                model_path, experimental_delegates=[delegate])
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
        objs = pycoral_detect.get_objects(
            self.engine, score_threshold=self.conf_thres)
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
        self.interpreter.set_tensor(
            self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        boxes = self.interpreter.tensor(
            self.output_details[0]['index'])()
        classes = self.interpreter.tensor(
            self.output_details[1]['index'])()
        scores = self.interpreter.tensor(
            self.output_details[2]['index'])()
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
# Body ReID
# ============================================================
class BodyReID:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Body ReID model not found: {model_path}")
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime not installed.")
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
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
        print(f"BodyReID: input {self.input_w}x{self.input_h}")

    def embed(self, body_crop):
        if body_crop is None or body_crop.size == 0:
            return None
        img = cv2.resize(body_crop, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0).astype(np.float32)
        try:
            out = self.session.run(None, {self.input_name: img})[0]
        except Exception as e:
            print(f"BodyReID inference error: {e}")
            return None
        vec = out.flatten().astype(np.float32)
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
            print(f"FaceRecognizer: no classifier at {model_path}.")
            return
        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"FaceRecognizer: failed to load ({e})")
            return
        if data.get("model_type") != "centroid_gallery":
            print(f"FaceRecognizer: unexpected type "
                  f"{data.get('model_type')}")
            return
        self.names = list(data["names"])
        self.centroids = np.asarray(data["centroids"], dtype=np.float32)
        self.galleries = [np.asarray(g, dtype=np.float32)
                          for g in data["galleries"]]
        self.stats_p10 = np.asarray(data["stats_p10"], dtype=np.float32)
        self.stats_p50 = np.asarray(data["stats_p50"], dtype=np.float32)
        self.stats_p90 = np.asarray(data["stats_p90"], dtype=np.float32)
        self.available = True
        sizes = [g.shape[0] for g in self.galleries]
        print(f"FaceRecognizer: loaded for {self.names} "
              f"(galleries: {sizes})")

    def predict(self, embedding, top_k=3):
        if not self.available or embedding is None:
            return None, -1.0
        q = embedding.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        best_combined = -1.0
        best_idx = -1
        for i, (centroid, gallery) in enumerate(
                zip(self.centroids, self.galleries)):
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
        if not self.available or name not in self.names or \
                embedding is None:
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
# Helpers
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
# Auto-Saver
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
        if w < config.ENROLL_MIN_FACE_PX or \
                h < config.ENROLL_MIN_FACE_PX:
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
                self.saved_track[name] = \
                    self.saved_track.get(name, 0) + 1
            else:
                self.total_direct += 1
                self.saved_direct[name] = \
                    self.saved_direct.get(name, 0) + 1
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
              f"({self.total_direct} direct, "
              f"{self.total_track} track):")
        for n in self.saved_count:
            d = self.saved_direct.get(n, 0)
            t = self.saved_track.get(n, 0)
            print(f"  {n}: {self.saved_count[n]} "
                  f"({d} direct, {t} track)")


# ============================================================
# Track
# ============================================================
class Track:
    MAX_SIG = 10

    def __init__(self, tid, box, name, conf, has_face, frame_count):
        self.id = tid
        self.box = box
        self.name = name if name is not None else "Unknown"
        self.confidence = conf
        self.age = 1
        self.lost = 0
        self.has_face = has_face
        self.last_face_frame = frame_count if has_face else 0
        self.identity_set_frame = frame_count if has_face else 0
        self.identity_fresh = has_face and (name is not None)
        self.body_signatures = []
        self.vx = 0.0
        self.vy = 0.0
        self.identity_source = "face" if has_face else "none"
        self.last_scan_frame = -999

        self.identity_locked = False
        self.lock_frame = -1
        self.lock_source = "none"

        self.last_seen_frame = frame_count
        self.last_face_box = None

        self.clothing_sig = None
        self.clothing_matches = 0

    def add_signature(self, embedding):
        if embedding is None:
            return False
        if self.body_signatures:
            q = embedding.astype(np.float32)
            q = q / (np.linalg.norm(q) + 1e-9)
            stack = np.stack(self.body_signatures)
            if float((stack @ q).max()) >= 0.98:
                return False
        self.body_signatures.append(embedding.astype(np.float32))
        if len(self.body_signatures) > self.MAX_SIG:
            self.body_signatures.pop(0)
        return True


# ============================================================
# SimpleTracker
# ============================================================
class SimpleTracker:
    MIN_MATCH_SCORE = 0.30
    SMOOTH_ALPHA = 0.55
    VEL_ALPHA = 0.5
    REID_MAX_FRAMES = 240
    REID_MAX_DIST = 150
    BODY_REVIVE_THRESH = 0.90
    SIG_ACCUM_MIN_CONF = 0.85
    SIG_ACCUM_INTERVAL = 5
    UNKNOWN_MAX_LOST = 20
    CLOTHING_REVIVE_THRESH = 0.80

    def __init__(self, max_lost=600, debug=False):
        self.max_lost = max_lost
        self.debug = debug
        self.tracks = []
        self.next_id = 0
        self.recently_lost = []
        self._last_sig_frame = {}

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

    def is_face_locked(self, face_box, iou_thresh=0.15):
        fx, fy, fw, fh = face_box
        for t in self.tracks:
            if not getattr(t, "identity_locked", False):
                continue
            fb = getattr(t, "last_face_box", None)
            if fb is None:
                tx, ty, tw, th = t.box
                fb = (tx, ty, tw, th)
            bx, by, bw, bh = fb
            ix1 = max(fx, bx)
            iy1 = max(fy, by)
            ix2 = min(fx + fw, bx + bw)
            iy2 = min(fy + fh, by + bh)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                continue
            union = fw * fh + bw * bh - inter
            iou_v = inter / union if union > 0 else 0.0
            if iou_v >= iou_thresh:
                return True
        return False

    def update(self, detections, frame_count):
        for t in self.tracks:
            t.lost += 1
            t.has_face = False

        pairs = []
        for ti, track in enumerate(self.tracks):
            pred_box = self._predict(track)
            lock_bonus = (0.15
                          if getattr(track, "identity_locked", False)
                          else 0.0)
            for di, det in enumerate(detections):
                score = self._match_score(pred_box, det["box"])
                if score >= self.MIN_MATCH_SCORE:
                    pairs.append((score + lock_bonus, ti, di))
        pairs.sort(reverse=True)

        matched_tracks, matched_dets = set(), set()
        for _, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            t = self.tracks[ti]
            det = detections[di]

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

            if det.get("face_box") is not None:
                t.last_face_box = det["face_box"]

            if det.get("has_face"):
                t.has_face = True
                t.last_face_frame = frame_count

            if det.get("scanned", False):
                t.last_scan_frame = frame_count
                if det.get("name") is not None:
                    candidate = det["name"]
                    conflict = any(
                        (other.id != t.id
                         and other.name.casefold()
                             == candidate.casefold()
                         and getattr(other, "identity_locked", False))
                        for other in self.tracks
                    )
                    if conflict:
                        if self.debug:
                            print(f"[track #{t.id}] scan said "
                                  f"'{candidate}' but that identity "
                                  f"is already locked by another "
                                  f"track -- ignoring")
                    else:
                        t.name = candidate
                        t.confidence = det.get("conf", 0.0)
                        t.identity_fresh = True
                        t.identity_source = "face"
                        lock_min = getattr(
                            config, "IDENTITY_LOCK_MIN_CONF", 0.80)
                        if t.confidence >= lock_min and \
                                not t.identity_locked:
                            t.identity_locked = True
                            t.lock_frame = frame_count
                            t.lock_source = "face"
                            if self.debug:
                                print(f"[track #{t.id}] LOCKED as "
                                      f"'{t.name}' (conf "
                                      f"{t.confidence:.2f})")
                else:
                    if not t.identity_locked:
                        t.name = "Unknown"
                        t.confidence = 0.0
                        t.identity_fresh = False
                        t.identity_source = "none"
                        t.body_signatures = []
                t.identity_set_frame = frame_count
            else:
                t.identity_fresh = False
                if t.identity_locked:
                    t.identity_source = "face"
                elif t.name != "Unknown":
                    t.identity_source = "memory"
                else:
                    t.identity_source = "none"

            if (t.name != "Unknown"
                    and t.confidence >= self.SIG_ACCUM_MIN_CONF
                    and det.get("body_embedding") is not None):
                last_f = self._last_sig_frame.get(t.id, -999)
                if frame_count - last_f >= self.SIG_ACCUM_INTERVAL:
                    if t.add_signature(det["body_embedding"]):
                        self._last_sig_frame[t.id] = frame_count

            if (getattr(t, "identity_locked", False)
                    and t.name != "Unknown"
                    and det.get("body_crop") is not None):
                sig = extract_clothing_signature(det["body_crop"])
                if sig is not None:
                    t.clothing_sig = update_reference_signature(
                        t.clothing_sig, sig, alpha=0.10)
                    t.clothing_matches += 1

            t.lost = 0
            t.last_seen_frame = frame_count
            t.age += 1

        for di, det in enumerate(detections):
            if di in matched_dets:
                continue

            if (not det.get("scanned", False)
                    and not det.get("face_only_proxy", False)):
                bx, by, bw, bh = det["box"]
                if (bw < getattr(config, "MIN_BODY_BOX_W", 60)
                        or bh < getattr(config, "MIN_BODY_BOX_H", 120)
                        or bw * bh < getattr(config,
                                             "MIN_BODY_BOX_AREA",
                                             10000)):
                    continue

            revived_name = None
            revived_conf = 0.0
            revived_sigs = []
            revived_clothing = None

            if (not det.get("scanned", False)
                    and not det.get("face_only_proxy", False)):
                dcx = det["box"][0] + det["box"][2] // 2
                dcy = det["box"][1] + det["box"][3] // 2
                best_score = self.BODY_REVIVE_THRESH

                new_clothing = None
                if det.get("body_crop") is not None:
                    new_clothing = extract_clothing_signature(
                        det["body_crop"])

                for entry in self.recently_lost:
                    fc, lb, nm, cf, sigs, cloth_sig = entry
                    if frame_count - fc > self.REID_MAX_FRAMES:
                        continue
                    lcx = lb[0] + lb[2] // 2
                    lcy = lb[1] + lb[3] // 2
                    d = ((dcx - lcx) ** 2 + (dcy - lcy) ** 2) ** 0.5
                    if d > self.REID_MAX_DIST:
                        continue

                    if det.get("body_embedding") is not None:
                        q = det["body_embedding"].astype(np.float32)
                        q = q / (np.linalg.norm(q) + 1e-9)
                        for sig in sigs:
                            v = float(sig @ q)
                            if v > best_score:
                                best_score = v
                                revived_name = nm
                                revived_conf = max(0.5, min(0.85, v))
                                revived_sigs = sigs
                                revived_clothing = cloth_sig

                    if (cloth_sig is not None
                            and new_clothing is not None):
                        cloth_score = compare_signatures(
                            cloth_sig, new_clothing)
                        if (cloth_score
                                > self.CLOTHING_REVIVE_THRESH):
                            cloth_conf = min(
                                0.85,
                                0.55 + 0.30 * cloth_score)
                            if cloth_conf > revived_conf:
                                revived_name = nm
                                revived_conf = cloth_conf
                                revived_sigs = sigs
                                revived_clothing = cloth_sig

            new_name = None
            new_conf = 0.0
            if det.get("scanned", False):
                candidate = det.get("name")
                if candidate is not None:
                    conflict = any(
                        (other.name.casefold()
                             == candidate.casefold()
                         and getattr(other, "identity_locked", False))
                        for other in self.tracks
                    )
                    if conflict:
                        pass
                    else:
                        new_name = candidate
                        new_conf = det.get("conf", 0.0)
            elif revived_name is not None:
                new_name = revived_name
                new_conf = revived_conf

            t = Track(
                self.next_id,
                det["box"],
                new_name,
                new_conf,
                det.get("has_face", False),
                frame_count,
            )
            if det.get("scanned", False):
                t.identity_source = "face" if new_name else "none"
                t.last_scan_frame = frame_count
                lock_min = getattr(config,
                                   "IDENTITY_LOCK_MIN_CONF", 0.80)
                if new_name is not None and new_conf >= lock_min:
                    t.identity_locked = True
                    t.lock_frame = frame_count
                    t.lock_source = "face"
            elif new_name is not None:
                t.identity_fresh = False
                t.body_signatures = list(revived_sigs)
                t.identity_source = "body"
                t.identity_locked = True
                t.lock_frame = frame_count
                t.lock_source = "body"
                t.clothing_sig = revived_clothing

            if (t.name != "Unknown"
                    and det.get("body_embedding") is not None):
                if t.add_signature(det["body_embedding"]):
                    self._last_sig_frame[t.id] = frame_count

            self.next_id += 1
            self.tracks.append(t)

        lock_release_frames = int(
            getattr(config, "IDENTITY_LOCK_RELEASE_S", 3.0) *
            getattr(config, "ASSUMED_FPS", 8.0))
        for t in self.tracks:
            if (t.identity_locked
                    and t.lost > lock_release_frames):
                t.identity_locked = False
                t.lock_source = "none"
                t.name = "Unknown"
                t.confidence = 0.0
                t.identity_source = "none"
                t.identity_fresh = False
                t.body_signatures = []

        alive = []
        for t in self.tracks:
            limit = (self.max_lost if t.name != "Unknown"
                     else self.UNKNOWN_MAX_LOST)
            if t.lost <= limit:
                alive.append(t)
            else:
                if (t.name != "Unknown"
                        and (t.body_signatures or
                             t.clothing_sig is not None)):
                    self.recently_lost.append(
                        (frame_count, t.box, t.name, t.confidence,
                         list(t.body_signatures),
                         t.clothing_sig)
                    )
                self._last_sig_frame.pop(t.id, None)
        self.tracks = alive

        self.recently_lost = [
            e for e in self.recently_lost
            if frame_count - e[0] <= self.REID_MAX_FRAMES
        ]

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
        if name != self.follow_name:
            print(f"Target switched to: {name}")
            self.follow_name = name
            self.locked_name = None
            self.locked_since = 0

    def choose(self, frame_count, tracks):
        candidates = [t for t in tracks
                      if t.name != "Unknown" and t.lost == 0]
        if not candidates:
            return None
        if self.follow_name is not None:
            wanted = [t for t in candidates
                      if t.name.casefold() ==
                      self.follow_name.casefold()]
            if not wanted:
                return None
            candidates = wanted
        locked = [t for t in candidates
                  if getattr(t, "identity_locked", False)]
        if locked:
            candidates = locked
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
                and best.confidence - held_best.confidence >
                self.SWITCH_MARGIN):
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
        sectors[name] = (float(np.min(valid))
                         if valid.size > 0 else np.inf)
    return sectors


def calculate_speed(distance_mm):
    if distance_mm is None:
        return config.FOLLOW_BASE_SPEED
    if distance_mm <= config.REVERSE_DISTANCE_MM:
        return config.REVERSE_SPEED
    elif distance_mm < config.FOLLOW_MIN_MM:
        return 0
    else:
        span = max(1.0, config.FOLLOW_MAX_MM - config.FOLLOW_MIN_MM)
        frac = min(1.0, (distance_mm - config.FOLLOW_MIN_MM) / span)
        return int(config.MIN_FOLLOW_SPEED +
                   frac * (config.MAX_FOLLOW_SPEED -
                           config.MIN_FOLLOW_SPEED))


# ============================================================
# Touch handler
# ============================================================
def on_touch(pressed):
    """
    Toggle dispenser servo on rising edge, but only if:
      1. It's a press (not a release)
      2. The follow target is currently recognized and locked
      3. The last toggle was longer than the debounce window ago
    """
    if not pressed:
        return

    now = time.monotonic()
    if now - _touch_state["last_toggle_t"] < _touch_state["debounce_s"]:
        print("[touch] ignored (debounce)")
        return

    if not state.get("person_locked", False):
        target = state.get("follow_name") or "the registered person"
        print(f"[touch] ignored (waiting for {target} to be recognized)")
        return

    _touch_state["last_toggle_t"] = now
    motors = None
    try:
        from web_server import get_motors
        motors = get_motors()
    except Exception:
        pass
    if motors is None:
        print("[touch] ignored (no motors instance)")
        return

    if _touch_state["servo_open"]:
        motors.servo_close()
        _touch_state["servo_open"] = False
        state["servo_open"] = False
        print("[touch] servo CLOSED")
    else:
        motors.servo_open()
        _touch_state["servo_open"] = True
        state["servo_open"] = True
        print("[touch] servo OPENED")


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
    state["enroll_name"] = person_name
    state["enroll_captured"] = 0
    state["enroll_target"] = target_count

    captured = 0
    attempts = 0
    max_attempts = target_count * 40
    last_saved_gray = None

    local_window = True
    try:
        cv2.namedWindow("Enrollment", cv2.WINDOW_NORMAL)
    except Exception:
        local_window = False

    try:
        while captured < target_count and attempts < max_attempts:
            attempts += 1
            frame = cam.read_color()
            if frame is None:
                time.sleep(0.02)
                continue

            faces = face_detector.infer(frame)
            best = max(faces, key=lambda f: f[2] * f[3]) \
                if faces else None

            display = frame.copy()
            status_text = "No face detected"
            status_color = (0, 0, 255)
            box_color = (128, 128, 128)
            blur_var = 0.0

            if best is not None:
                x, y, w, h, conf = best
                x = max(0, x); y = max(0, y)
                x2 = min(frame.shape[1], x + w)
                y2 = min(frame.shape[0], y + h)
                face_crop = frame[y:y2, x:x2]

                if face_crop.size > 0:
                    gray = cv2.cvtColor(face_crop,
                                        cv2.COLOR_BGR2GRAY)
                    blur_var = float(
                        cv2.Laplacian(gray, cv2.CV_64F).var())

                    ok_size = (w >= config.ENROLL_MIN_FACE_PX
                               and h >= config.ENROLL_MIN_FACE_PX)
                    ok_blur = blur_var >= config.ENROLL_BLUR_THRES
                    ok_diverse = True
                    if last_saved_gray is not None:
                        small = cv2.resize(gray, (64, 64))
                        diff = float(np.mean(np.abs(
                            small.astype(np.float32)
                            - last_saved_gray.astype(np.float32))))
                        ok_diverse = diff >= config.ENROLL_DIVERSITY_PX

                    if ok_size and ok_blur and ok_diverse:
                        fname = os.path.join(
                            person_dir,
                            f"enroll_{start_index + captured:04d}.jpg")
                        if cv2.imwrite(fname, face_crop):
                            last_saved_gray = cv2.resize(gray, (64, 64))
                            captured += 1
                            state["enroll_captured"] = captured
                            box_color = (0, 255, 0)
                            status_text = (f"SAVED {captured}/"
                                           f"{target_count}")
                            status_color = (0, 255, 0)
                    else:
                        box_color = (0, 200, 255)
                        status_text = "Skip: " + " / ".join(
                            r for r in [
                                None if ok_size else "too small",
                                None if ok_blur else "blurry",
                                None if ok_diverse else "too similar",
                            ] if r)
                        status_color = (0, 200, 255)

                    cv2.rectangle(display, (x, y), (x2, y2),
                                  box_color, 2)

            cv2.putText(display,
                        f"ENROLL: {person_name}   "
                        f"{captured}/{target_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            cv2.putText(display, status_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        status_color, 2)

            with frame_lock:
                frame_holder["frame"] = display

            if local_window:
                try:
                    cv2.imshow("Enrollment", display)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                except Exception:
                    local_window = False

    finally:
        if local_window:
            try:
                cv2.destroyWindow("Enrollment")
            except Exception:
                pass
        state["enroll_name"] = None
        state["enroll_captured"] = 0
        state["enroll_target"] = 0

    print(f"\nEnrollment finished: {captured} crops saved to "
          f"{person_dir}")


# ============================================================
# Listing / Removal
# ============================================================
def run_list_people():
    train_dir = config.FACE_TRAINING_DIR
    svm_path = config.SVM_MODEL_PATH
    trained_names = []
    if os.path.exists(svm_path):
        try:
            with open(svm_path, "rb") as f:
                data = pickle.load(f)
            trained_names = list(data.get("names", []))
            print(f"Trained classifier: {svm_path}")
            print(f"  Classes: {trained_names}")
        except Exception as e:
            print(f"Could not read classifier: {e}")
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
        all_crops = [f for f in os.listdir(os.path.join(train_dir, p))
                     if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        marker = "  [trained]" if p in trained_names else ""
        print(f"  [{idx}] {p:20s} {len(all_crops):4d} crops{marker}")


def run_remove_person(name, delete_auto=False):
    name = name.strip()
    person_dir = os.path.join(config.FACE_TRAINING_DIR, name)
    if not os.path.isdir(person_dir):
        print(f"No enrolled person named '{name}'.")
        return
    crops = [f for f in os.listdir(person_dir)
             if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if delete_auto:
        auto_crops = [f for f in crops
                      if f.startswith(("auto_", "live_"))]
        for f in auto_crops:
            os.remove(os.path.join(person_dir, f))
        print(f"Removed {len(auto_crops)} auto-saved crops for "
              f"'{name}'.")
        return
    print(f"Removing '{name}': {len(crops)} crops")
    shutil.rmtree(person_dir)
    remaining = [d for d in os.listdir(config.FACE_TRAINING_DIR)
                 if os.path.isdir(
                     os.path.join(config.FACE_TRAINING_DIR, d))]
    if not remaining and os.path.exists(config.SVM_MODEL_PATH):
        os.remove(config.SVM_MODEL_PATH)
        print("Removed stale classifier (no people left).")


# ============================================================
# RFID handling
# ============================================================
def handle_rfid_uid(uid):
    uid = uid.strip().upper()
    authorized = _load_auth_uids()
    if uid not in authorized:
        print(f"[rfid] UID {uid} not authorized")
        return
    mapping = _load_rfid_persons()
    person = mapping.get(uid)
    if person:
        print(f"[rfid] switching follow target to {person}")
        command_queue.put(("follow", person))
    else:
        print(f"[rfid] UID {uid} authorized but has no person mapping")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enroll", metavar="NAME")
    parser.add_argument("--samples", type=int,
                        default=config.ENROLL_DEFAULT_COUNT)
    parser.add_argument("--register-only", metavar="NAME")
    parser.add_argument("--follow", metavar="NAME")
    parser.add_argument("--list-people", action="store_true")
    parser.add_argument("--remove-person", metavar="NAME")
    parser.add_argument("--clear-auto", metavar="NAME")
    parser.add_argument("--test-motors", action="store_true")
    parser.add_argument("--no-auto-save", action="store_true")
    parser.add_argument("--auto-train", action="store_true")
    parser.add_argument("--no-body-reid", action="store_true")
    parser.add_argument("--debug-tracks", action="store_true")
    parser.add_argument("--no-planner", action="store_true")
    parser.add_argument("--web-port", type=int, default=5000)
    parser.add_argument("--no-web", action="store_true")
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
        train_dir = config.FACE_TRAINING_DIR
        if os.path.isdir(train_dir):
            for d in os.listdir(train_dir):
                full = os.path.join(train_dir, d)
                if os.path.isdir(full):
                    shutil.rmtree(full)
        if os.path.exists(config.SVM_MODEL_PATH):
            os.remove(config.SVM_MODEL_PATH)
        args.enroll = args.register_only

    planner_on = not args.no_planner

    cam = Camera()
    face_detector = FaceDetector(config.CORAL_FACE_DETECTION_MODEL,
                                 config.CORAL_FACE_LABELS)
    body_detector = BodyDetector(config.CORAL_DETECTION_MODEL,
                                 config.CORAL_LABELS)

    if args.enroll:
        run_enrollment(cam, face_detector, args.enroll, args.samples)
        cam.stop()
        return
    if args.test_motors:
        motors = SerialMotors()
        try:
            motors.test_motors()
        except AttributeError:
            pass
        try:
            motors.cleanup()
        except Exception:
            pass
        cam.stop()
        return

    arcface = ArcFaceEmbedder(config.ARCFACE_MODEL_PATH)
    scrfd = SCRFDDetector(
        config.SCRFD_MODEL_PATH,
        input_size=config.SCRFD_INPUT_SIZE,
        conf_thres=config.SCRFD_CONF_THRES,
        iou_thres=config.SCRFD_IOU_THRES,
    )
    face_recognizer = FaceRecognizer(config.SVM_MODEL_PATH)

    body_reid = None
    if not args.no_body_reid:
        try:
            body_reid = BodyReID(config.REID_PATH)
        except Exception as e:
            print(f"Body ReID disabled: {e}")
            body_reid = None
    print(f"Body ReID: {'ENABLED' if body_reid else 'DISABLED'}")

    motors = SerialMotors(
        on_rfid_enrolled=on_rfid_enrolled,
        on_rfid_timeout=on_rfid_timeout,
        on_rfid=handle_rfid_uid,
        on_touch=on_touch,
    )
    set_motors_instance(motors)
    state["motors_available"] = motors.available

    planner = PathPlanner(
        cam_height_mm=getattr(config, "CAMERA_HEIGHT_MM", 200),
        cam_tilt_deg=getattr(config, "CAMERA_TILT_DEG", 15),
        cam_hfov_deg=getattr(config, "CAMERA_HFOV_DEG", 60),
        grid_size=getattr(config, "OCCUPANCY_GRID_SIZE", 60),
        res_mm=getattr(config, "OCCUPANCY_GRID_RES_MM", 50),
        robot_radius_mm=getattr(config, "ROBOT_RADIUS_MM", 200),
    )
    try:
        planner.load_calibration(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "camera_calib.yml"))
    except Exception as e:
        print(f"[planner] calibration load failed: {e}")

    if not args.no_web:
        start_web_server(host="0.0.0.0", port=args.web_port)

    follow_name = args.follow.strip() if args.follow else None
    if not face_recognizer.available or not face_recognizer.names:
        print("\nNo classifier / no enrolled people. Cannot follow.")
    if follow_name is None and face_recognizer.names:
        if len(face_recognizer.names) == 1:
            follow_name = face_recognizer.names[0]
            print(f"Only one enrolled person -> following "
                  f"'{follow_name}'.")
        else:
            if sys.stdin.isatty():
                while True:
                    pick = input("Enter name to follow "
                                 "(or 'q' to quit): ").strip()
                    if pick.casefold() == 'q':
                        cam.stop()
                        return
                    matches = [n for n in face_recognizer.names
                               if n.casefold() == pick.casefold()]
                    if matches:
                        follow_name = matches[0]
                        break
                    print(f"'{pick}' is not enrolled. Try again.")
            else:
                follow_name = face_recognizer.names[0]
    elif follow_name is not None and face_recognizer.names:
        matches = [n for n in face_recognizer.names
                   if n.casefold() == follow_name.casefold()]
        if not matches:
            print(f"'{follow_name}' is not enrolled.")
            follow_name = None
        else:
            follow_name = matches[0]

    selector = TargetSelector(follow_name=follow_name)
    state["follow_name"] = follow_name
    auto_save_enabled = (
        getattr(config, "AUTO_SAVE_RECOGNIZED_CROPS", True)
        and not args.no_auto_save)
    auto_saver = AutoSaver(enabled=auto_save_enabled)

    tracker = SimpleTracker(max_lost=600, debug=args.debug_tracks)
    following = False
    last_drive_status = None
    frame_interval = 0.12
    frame_count = 0
    track_cache = TrackConfirmationCache()

    cached_path = []
    cached_grid = None
    cached_origin = None
    plan_every = 3

    face_max_mm = getattr(config, "FACE_RECOGNIZE_MAX_MM", 1200)
    scan_cooldown_s = getattr(config, "FACE_SCAN_COOLDOWN_S", 2.0)
    scan_cooldown_frames = int(scan_cooldown_s * 8)
    lock_min_conf = getattr(config, "IDENTITY_LOCK_MIN_CONF", 0.80)
    lock_release_s = getattr(config, "IDENTITY_LOCK_RELEASE_S", 3.0)
    min_body_w = getattr(config, "MIN_BODY_BOX_W", 60)
    min_body_h = getattr(config, "MIN_BODY_BOX_H", 120)
    min_body_area = getattr(config, "MIN_BODY_BOX_AREA", 10000)

    state["manual_cmd"] = None
    state["manual_cmd_t"] = 0.0
    state["person_locked"] = False
    state["person_locked_name"] = None

    print(f"\nRunning. Following: {selector.follow_name}")
    print("Controls: q=quit, f=follow, s=stop, 1..9=switch, p=planner")
    print(f"Planner: {'ON' if planner_on else 'OFF'}")
    print(f"Face scan gate: {face_max_mm} mm, "
          f"cooldown {scan_cooldown_s:.1f} s")
    print(f"Identity lock: min conf {lock_min_conf:.2f}, "
          f"release after {lock_release_s:.1f} s unseen")
    print(f"Skip scans when locked: "
          f"{getattr(config, 'SKIP_SCANS_WHEN_LOCKED', True)}")
    print(f"Touch handler: gated on person_locked")

    try:
        while True:
            frame_start = time.monotonic()
            frame_count += 1
            frame = cam.read_color()
            if frame is None:
                time.sleep(0.01)
                continue

            # ---- Drain command queue ----
            while not command_queue.empty():
                try:
                    cmd_type, payload = command_queue.get_nowait()
                except Exception:
                    break

                if cmd_type == "motor":
                    if payload == "stop":
                        state["manual_cmd"] = None
                        state["following"] = False
                        following = False
                        state["mode"] = "idle"
                        motors.stop()
                    else:
                        state["manual_cmd"] = payload
                        state["manual_cmd_t"] = time.monotonic()
                        if following:
                            following = False
                            state["following"] = False
                        state["mode"] = "manual"

                elif cmd_type == "servo":
                    if payload == "open":
                        motors.servo_open()
                        state["servo_open"] = True
                        _touch_state["servo_open"] = True
                    elif payload == "close":
                        motors.servo_close()
                        state["servo_open"] = False
                        _touch_state["servo_open"] = False
                    elif payload == "toggle":
                        if state["servo_open"]:
                            motors.servo_close()
                            state["servo_open"] = False
                            _touch_state["servo_open"] = False
                        else:
                            motors.servo_open()
                            state["servo_open"] = True
                            _touch_state["servo_open"] = True

                elif cmd_type == "follow":
                    if payload is None:
                        following = False
                        state["following"] = False
                        state["follow_name"] = None
                        selector.set_name(None)
                        motors.stop()
                    else:
                        selector.set_name(payload)
                        state["follow_name"] = payload
                        state["following"] = True
                        following = True

                elif cmd_type == "enroll":
                    state["mode"] = f"enroll:{payload['name']}"
                    run_enrollment(cam, face_detector,
                                   payload["name"],
                                   payload["samples"])
                    state["mode"] = "idle"

                elif cmd_type == "train":
                    state["mode"] = "training"
                    script_dir = os.path.dirname(
                        os.path.abspath(__file__))
                    train_script = os.path.join(
                        script_dir, "train_face_svm.py")
                    if not os.path.exists(train_script):
                        state["mode"] = "idle"
                        continue
                    try:
                        proc = subprocess.Popen(
                            [sys.executable, train_script],
                            cwd=script_dir,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1)

                        def _pump(p):
                            try:
                                for line in p.stdout:
                                    print(f"[train] {line.rstrip()}")
                                p.wait()
                                if p.returncode == 0:
                                    command_queue.put(
                                        ("reload_classifier", None))
                            except Exception:
                                pass

                        threading.Thread(
                            target=_pump, args=(proc,),
                            daemon=True).start()
                    except Exception:
                        pass
                    state["mode"] = "idle"

                elif cmd_type == "reload_classifier":
                    try:
                        face_recognizer = FaceRecognizer(
                            config.SVM_MODEL_PATH)
                        print(f"[reload] classifier reloaded: "
                              f"{face_recognizer.names}")
                    except Exception as e:
                        print(f"[reload] failed: {e}")

            depth = cam.read_depth()

            locked_target_name = None
            if selector.follow_name is not None:
                for _t in tracker.tracks:
                    if (_t.name.casefold()
                            == selector.follow_name.casefold()
                            and getattr(_t, "identity_locked", False)):
                        locked_target_name = _t.name
                        break
            else:
                for _t in tracker.tracks:
                    if getattr(_t, "identity_locked", False):
                        locked_target_name = _t.name
                        break

            skip_all_scans = (
                locked_target_name is not None
                and getattr(config, "SKIP_SCANS_WHEN_LOCKED", True))

            faces = face_detector.infer(frame)
            bodies_raw = body_detector.infer(frame)

            bodies = []
            for (bx, by, bw, bh, bconf) in bodies_raw:
                if (bw >= min_body_w and bh >= min_body_h
                        and bw * bh >= min_body_area):
                    bodies.append((bx, by, bw, bh, bconf))

            face_recs = []
            for (fx, fy, fw, fh, fconf) in faces:
                fx = max(0, fx); fy = max(0, fy)
                fx2 = min(frame.shape[1], fx + fw)
                fy2 = min(frame.shape[0], fy + fh)
                face_crop = frame[fy:fy2, fx:fx2]
                if face_crop.size == 0:
                    continue

                if skip_all_scans:
                    face_recs.append({
                        "box": (fx, fy, fx2 - fx, fy2 - fy),
                        "name": None, "conf": 0.0, "embedding": None,
                        "crop": face_crop, "depth_mm": None,
                        "skipped": True, "scanned": False,
                        "in_cooldown": True,
                    })
                    continue

                face_depth_mm = None
                if depth is not None:
                    dh, dw = depth.shape[:2]
                    sx = dw / config.FRAME_W
                    sy = dh / config.FRAME_H
                    dcx = int((fx + fw // 2) * sx)
                    dcy = int((fy + fh // 2) * sy)
                    if 0 <= dcx < dw and 0 <= dcy < dh:
                        y1 = max(0, dcy - 5); y2 = min(dh, dcy + 5)
                        x1 = max(0, dcx - 5); x2 = min(dw, dcx + 5)
                        roi = depth[y1:y2, x1:x2]
                        valid = roi[roi > 0]
                        if valid.size > 0:
                            face_depth_mm = float(np.median(valid))

                too_far = (face_depth_mm is not None
                           and face_depth_mm > face_max_mm)
                if too_far:
                    face_recs.append({
                        "box": (fx, fy, fx2 - fx, fy2 - fy),
                        "name": None, "conf": 0.0, "embedding": None,
                        "crop": face_crop, "depth_mm": face_depth_mm,
                        "skipped": True, "scanned": False,
                        "in_cooldown": False,
                    })
                    continue

                in_cooldown = False
                if tracker.is_face_locked((fx, fy, fw, fh)):
                    in_cooldown = True
                else:
                    face_cx = fx + fw // 2
                    face_cy = fy + fh // 2
                    margin_x = max(20, fw // 2)
                    margin_y = max(20, fh // 2)
                    for tr in tracker.tracks:
                        tx, ty, tw, th = tr.box
                        if (tx - margin_x <= face_cx
                                <= tx + tw + margin_x
                                and ty - margin_y <= face_cy
                                <= ty + th + margin_y):
                            if getattr(tr, "identity_locked", False):
                                in_cooldown = True
                                break
                            frames_since = frame_count - getattr(
                                tr, "last_scan_frame", -999)
                            if frames_since < scan_cooldown_frames:
                                in_cooldown = True
                                break

                if in_cooldown:
                    face_recs.append({
                        "box": (fx, fy, fx2 - fx, fy2 - fy),
                        "name": None, "conf": 0.0, "embedding": None,
                        "crop": face_crop, "depth_mm": face_depth_mm,
                        "skipped": True, "scanned": False,
                        "in_cooldown": True,
                    })
                    continue

                embedding = None
                name = None
                svm_conf = 0.0

                pad = int(0.25 * max(fw, fh))
                rx1 = max(0, fx - pad)
                ry1 = max(0, fy - pad)
                rx2 = min(frame.shape[1], fx + fw + pad)
                ry2 = min(frame.shape[0], fy + fh + pad)
                face_region = frame[ry1:ry2, rx1:rx2]

                dets = (scrfd.infer(face_region)
                        if face_region.size > 0 else [])

                if dets:
                    rc_cx = (rx2 - rx1) / 2.0
                    rc_cy = (ry2 - ry1) / 2.0

                    def _dist_to_center(d):
                        dx, dy, dw, dh, _, _ = d
                        return ((dx + dw / 2 - rc_cx) ** 2 +
                                (dy + dh / 2 - rc_cy) ** 2)

                    _, _, _, _, _, lm_region = min(
                        dets, key=_dist_to_center)

                    lm_crop = lm_region.copy().astype(np.float32)
                    lm_crop[:, 0] += rx1 - fx
                    lm_crop[:, 1] += ry1 - fy

                    try:
                        embedding = arcface.embed(
                            face_crop, landmarks=lm_crop)
                    except ValueError:
                        embedding = None

                    if embedding is not None:
                        name, svm_conf = face_recognizer.predict(
                            embedding)

                face_recs.append({
                    "box": (fx, fy, fx2 - fx, fy2 - fy),
                    "name": name, "conf": svm_conf,
                    "embedding": embedding,
                    "crop": face_crop, "depth_mm": face_depth_mm,
                    "skipped": False, "scanned": True,
                    "in_cooldown": False,
                })

            body_embs = []
            if body_reid is not None:
                for (bx, by, bw, bh, bconf) in bodies:
                    bx = max(0, bx); by = max(0, by)
                    bx2 = min(frame.shape[1], bx + bw)
                    by2 = min(frame.shape[0], by + bh)
                    body_crop = frame[by:by2, bx:bx2]
                    emb = (body_reid.embed(body_crop)
                           if body_crop.size > 0 else None)
                    body_embs.append(emb)
            else:
                body_embs = [None] * len(bodies)

            detections = []
            used_faces = set()
            for i, (bx, by, bw, bh, bconf) in enumerate(bodies):
                bx = max(0, bx); by = max(0, by)
                bx2 = min(frame.shape[1], bx + bw)
                by2 = min(frame.shape[0], by + bh)
                body_box = (bx, by, bx2 - bx, by2 - by)
                body_crop = (frame[by:by2, bx:bx2]
                             if by2 > by and bx2 > bx else None)

                best_face = None
                best_conf = -1.0
                for fi, fr in enumerate(face_recs):
                    if fi in used_faces:
                        continue
                    fx, fy, fw, fh = fr["box"]
                    fcx = fx + fw // 2
                    fcy = fy + fh // 2
                    if (bx <= fcx <= bx + bw
                            and by <= fcy <= by + bh):
                        face_conf = (fr["conf"]
                                     if fr["name"] is not None else 0.0)
                        if face_conf > best_conf:
                            best_conf = face_conf
                            best_face = (fi, fr)

                if best_face is not None:
                    used_faces.add(best_face[0])
                    fr = best_face[1]
                    detections.append({
                        "box": body_box,
                        "name": fr["name"], "conf": fr["conf"],
                        "has_face": True,
                        "scanned": fr.get("scanned", False),
                        "face_box": fr["box"],
                        "face_crop": fr["crop"],
                        "embedding": fr["embedding"],
                        "body_embedding": body_embs[i],
                        "body_crop": body_crop,
                        "face_only_proxy": False,
                    })
                else:
                    detections.append({
                        "box": body_box,
                        "name": None, "conf": 0.0,
                        "has_face": False, "scanned": False,
                        "face_box": None, "face_crop": None,
                        "embedding": None,
                        "body_embedding": body_embs[i],
                        "body_crop": body_crop,
                        "face_only_proxy": False,
                    })

            for fi, fr in enumerate(face_recs):
                if fi in used_faces:
                    continue
                if not fr.get("scanned", False):
                    continue
                if fr["name"] is None:
                    continue
                fx, fy, fw, fh = fr["box"]
                proxy_h = min(frame.shape[0] - fy, int(fh * 4))
                proxy_box = (fx, fy, fw, proxy_h)
                detections.append({
                    "box": proxy_box,
                    "name": fr["name"], "conf": fr["conf"],
                    "has_face": True, "scanned": True,
                    "face_box": fr["box"],
                    "face_crop": fr["crop"],
                    "embedding": fr["embedding"],
                    "body_embedding": None,
                    "body_crop": None,
                    "face_only_proxy": True,
                })

            tracks = tracker.update(detections, frame_count)

            for det in detections:
                if (det["has_face"] and det.get("scanned", False)
                        and det["name"] is not None
                        and det["embedding"] is not None):
                    face_recognizer.update_gallery(
                        det["name"], det["embedding"], det["conf"])
                    confirmed_by_track = (
                        track_cache.confirmed_name_for(
                            frame_count, det["box"]) == det["name"]
                    )
                    auto_saver.maybe_save(
                        det["name"], det["face_crop"],
                        det["embedding"], det["conf"],
                        confirmed_by_track=confirmed_by_track,
                    )
                    track_cache.update(frame_count, det["box"],
                                       det["name"], det["conf"])

            best_track = selector.choose(frame_count, tracks)
            state["follow_name"] = selector.follow_name
            state["following"] = following

            person_locked = (
                best_track is not None
                and getattr(best_track, "identity_locked", False)
                and best_track.name != "Unknown"
            )
            state["person_locked"] = person_locked
            state["person_locked_name"] = (
                best_track.name if person_locked else None)

            best_box = best_track.box if best_track else None
            person_cx = ((best_box[0] + best_box[2] // 2)
                         if best_box else None)
            person_cy = ((best_box[1] + best_box[3] // 2)
                         if best_box else None)

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

            state["target_distance_mm"] = target_distance_mm
            state["locked_count"] = sum(
                1 for t in tracks
                if getattr(t, "identity_locked", False))
            state["track_count"] = len(tracks)

            if depth is not None:
                _dh, _dw = depth.shape[:2]
                obstacle_sectors = get_sector_distances(
                    depth, best_box, _dw, _dh)
            else:
                obstacle_sectors = {"left": np.inf,
                                    "center": np.inf,
                                    "right": np.inf}

            if (planner_on and depth is not None
                    and frame_count % plan_every == 0):
                grid, origin = planner.build_grid(depth)
                if grid is not None:
                    grid = planner.inflate_obstacles(grid)
                    cached_grid = grid
                    cached_origin = origin
                    if best_box is not None and \
                            target_distance_mm is not None:
                        planner.set_goal_world(target_distance_mm, 0)
                        cached_path = planner.plan(grid, origin)
                    else:
                        cached_path = []

            frame_cx = config.FRAME_W // 2
            cx = person_cx

            # ---- Motor control: E-stop > manual > follow > idle ----
            MANUAL_TIMEOUT_S = 0.5

            if state.get("mode") == "stopped":
                motors.stop()
                last_drive_status = "stopped"

            elif state.get("manual_cmd"):
                age = time.monotonic() - state.get("manual_cmd_t", 0)
                if age > MANUAL_TIMEOUT_S:
                    state["manual_cmd"] = None
                    state["mode"] = "idle"
                    motors.stop()
                    last_drive_status = "manual_timeout"
                else:
                    cmd = state["manual_cmd"]
                    motors.set_speed(config.FOLLOW_BASE_SPEED)
                    if cmd == "forward":
                        motors.forward()
                        last_drive_status = "manual_fwd"
                    elif cmd == "backward":
                        motors.backward()
                        last_drive_status = "manual_back"
                    elif cmd == "left":
                        motors.turn_left()
                        last_drive_status = "manual_left"
                    elif cmd == "right":
                        motors.turn_right()
                        last_drive_status = "manual_right"

            elif following:
                if best_box is None:
                    motors.stop()
                    last_drive_status = "no_target"
                elif obstacle_sectors["center"] < config.OBSTACLE_STOP_MM:
                    motors.stop()
                    last_drive_status = (
                        f"obstacle_stop "
                        f"({obstacle_sectors['center']:.0f}mm)")
                elif target_distance_mm is None:
                    motors.stop()
                    last_drive_status = "no_depth"
                elif target_distance_mm < config.REVERSE_DISTANCE_MM:
                    motors.set_speed(config.REVERSE_SPEED)
                    motors.backward()
                    if cx is not None:
                        if cx < frame_cx - config.DEADBAND_PX:
                            motors.turn_right()
                        elif cx > frame_cx + config.DEADBAND_PX:
                            motors.turn_left()
                    last_drive_status = (
                        f"reverse ({target_distance_mm:.0f}mm)")
                elif target_distance_mm < config.FOLLOW_MIN_MM:
                    if cx is None:
                        motors.stop()
                        last_drive_status = (
                            f"hold ({target_distance_mm:.0f}mm)")
                    elif cx < frame_cx - config.DEADBAND_PX:
                        motors.set_speed(config.TURN_ONLY_SPEED)
                        motors.turn_left()
                        last_drive_status = (
                            f"hold+turn_L ({target_distance_mm:.0f}mm)")
                    elif cx > frame_cx + config.DEADBAND_PX:
                        motors.set_speed(config.TURN_ONLY_SPEED)
                        motors.turn_right()
                        last_drive_status = (
                            f"hold+turn_R ({target_distance_mm:.0f}mm)")
                    else:
                        motors.stop()
                        last_drive_status = (
                            f"hold ({target_distance_mm:.0f}mm)")
                else:
                    span = max(1.0,
                               config.FOLLOW_MAX_MM - config.FOLLOW_MIN_MM)
                    frac = min(1.0,
                               (target_distance_mm - config.FOLLOW_MIN_MM)
                               / span)
                    speed = int(
                        config.MIN_FOLLOW_SPEED
                        + frac * (config.MAX_FOLLOW_SPEED
                                  - config.MIN_FOLLOW_SPEED))
                    if cx is None:
                        motors.set_speed(speed)
                        motors.forward()
                        last_drive_status = (
                            f"fwd {speed}% ({target_distance_mm:.0f}mm)")
                    else:
                        err = cx - frame_cx
                        if abs(err) < config.DEADBAND_PX:
                            motors.set_speed(speed)
                            motors.forward()
                            last_drive_status = (
                                f"fwd {speed}% "
                                f"({target_distance_mm:.0f}mm)")
                        elif err < 0:
                            motors.set_speed(config.TURN_ONLY_SPEED)
                            motors.turn_left()
                            last_drive_status = (
                                f"turn_L ({target_distance_mm:.0f}mm)")
                        else:
                            motors.set_speed(config.TURN_ONLY_SPEED)
                            motors.turn_right()
                            last_drive_status = (
                                f"turn_R ({target_distance_mm:.0f}mm)")

            else:
                motors.stop()
                last_drive_status = "idle"

            # ---- Draw tracks ----
            for t in tracks:
                x, y, w, h = t.box
                is_selected = (best_track is not None
                               and t.id == best_track.id)
                if t.name != "Unknown":
                    src = getattr(t, "identity_source", "memory")
                    is_locked = getattr(t, "identity_locked", False)
                    if t.identity_fresh:
                        color = (0, 255, 0); tag = ""
                    elif is_locked and t.lock_source == "body":
                        color = (255, 200, 0); tag = " LOCK(body)"
                    elif is_locked:
                        color = (0, 200, 255); tag = " LOCK"
                    else:
                        color = (0, 165, 255); tag = " (mem)"
                    thickness = 3 if is_selected else 2
                    suffix = " *FOLLOW*" if is_selected else ""
                    n_sig = len(t.body_signatures)
                    cloth = "C" if t.clothing_sig is not None else "-"
                    label = (f"#{t.id} {t.name}: {t.confidence:.2f}"
                             f"{tag} [s{n_sig},{cloth}]{suffix}")
                else:
                    color = (0, 0, 255); thickness = 2
                    label = f"#{t.id} Unknown"
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              color, thickness)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            for fr in face_recs:
                fx, fy, fw, fh = fr["box"]
                if not fr.get("scanned", False):
                    if fr.get("in_cooldown"):
                        cv2.rectangle(frame, (fx, fy),
                                      (fx + fw, fy + fh),
                                      (200, 120, 0), 1)
                    else:
                        cv2.rectangle(frame, (fx, fy),
                                      (fx + fw, fy + fh),
                                      (128, 128, 128), 1)
                else:
                    cv2.rectangle(frame, (fx, fy),
                                  (fx + fw, fy + fh),
                                  (255, 0, 255), 1)

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
                    elif target_distance_mm < config.FOLLOW_MIN_MM:
                        motor_state = "STOP"
                        motor_color = (0, 165, 255)
                    else:
                        speed = calculate_speed(target_distance_mm)
                        motor_state = f"FWD {speed}%"
                        motor_color = (0, 255, 0)
                else:
                    motor_state = "STOP (no depth)"
                    motor_color = (0, 0, 255)
            cv2.putText(frame, f"Motor: {motor_state}",
                        (config.FRAME_W - 280, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        motor_color, 2)

            banner_color = (0, 255, 0) if following else (128, 128, 128)
            cv2.putText(frame, f"Following: {selector.follow_name}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, banner_color, 2)

            planner_label = "ON" if planner_on else "OFF"
            n_locked = sum(1 for t in tracks
                           if getattr(t, "identity_locked", False))
            skip_label = "skip" if skip_all_scans else "scan"
            touch_label = "ready" if person_locked else "wait"
            cv2.putText(
                frame,
                f"Planner:{planner_label}  locked={n_locked}  "
                f"[{skip_label}]  touch:{touch_label}  "
                f"scan<{face_max_mm}mm",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1)
            if last_drive_status:
                cv2.putText(frame, f"Status: {last_drive_status}",
                            (10, config.FRAME_H - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (180, 180, 180), 1)

            elapsed = time.monotonic() - frame_start
            sleep = max(0, frame_interval - elapsed)
            if sleep > 0:
                time.sleep(sleep)

            # Publish the fully-drawn frame for the web stream
            with frame_lock:
                frame_holder["frame"] = frame.copy()

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
            elif key == ord('p'):
                planner_on = not planner_on
                print(f"Planner {'ON' if planner_on else 'OFF'}")
            elif ord('1') <= key <= ord('9'):
                idx = key - ord('1')
                if 0 <= idx < len(face_recognizer.names):
                    selector.set_name(face_recognizer.names[idx])
                else:
                    print(f"No person at index {idx + 1}")

    except KeyboardInterrupt:
        pass
    finally:
        auto_saver.summary()
        try:
            motors.stop()
        except Exception:
            pass
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


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nFATAL: {e}\n")
        sys.exit(1)
