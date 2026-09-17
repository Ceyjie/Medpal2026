#!/usr/bin/env python3
"""
tracker_geminiV3.py -- Face identity + body ReID + body tracking.

Identity rule:
  - Body track is the unit of identity.
  - t.name only changes when det["scanned"] is True (face < FACE_RECOGNIZE_MAX_MM).
  - A scanned face sets t.name to the recognized name, or "Unknown".
  - If no face is scanned this frame, t.name is left untouched.
  - Body signatures accumulate on ANY frame the track has a name and a body
    embedding is available.
  - Per-track face scan cooldown: after a track is scanned, the same track
    will not be scanned again for FACE_SCAN_COOLDOWN_S seconds.

False-positive suppression:
  - Body detector uses a stricter threshold (BODY_CONF_THRES).
  - Body boxes below MIN_BODY_BOX_* are dropped before tracking.
  - Unknown tracks die after UNKNOWN_MAX_LOST frames.
  - New tracks require a scanned face or a large body box.
  - Face-only proxies only fire for RECOGNIZED faces.

Commands:
    --enroll NAME            Capture face crops for a person
    --samples N              Crops per enrollment (default 30)
    --register-only NAME     Remove all others, then enroll only this person
    --follow NAME            Follow only this person
    --list-people            Show enrolled people and classifier state
    --remove-person NAME     Delete a person's crops
    --clear-auto NAME        Delete only auto-saved crops for a person
    --test-motors            Drive each wheel briefly
    --no-auto-save           Disable auto-saving for this run
    --auto-train             Retrain classifier on exit if any crops were saved
    --no-body-reid           Disable OSNet body ReID (saves CPU)
    --debug-tracks           Print track signature/revival events

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
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        boxes = self.interpreter.tensor(self.output_details[0]['index'])()
        classes = self.interpreter.tensor(self.output_details[1]['index'])()
        scores = self.interpreter.tensor(self.output_details[2]['index'])()
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
# Body Detector (stricter threshold to suppress false positives)
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
# Body ReID (OSNet ONNX, CPU)
# ============================================================
class BodyReID:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Body ReID model not found: {model_path}\n"
                f"Run export_reid_onnx.py to create it."
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
        print(f"BodyReID: input {self.input_w}x{self.input_h} "
              f"(providers: {self.session.get_providers()})")

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
# Face Embedder
# ============================================================
class FaceEmbedder:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face embedding model not found: {model_path}"
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
              f"dtype={self.input_dtype.__name__}, output {self.embedding_dim}-D")

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
        interp = cv2.INTER_CUBIC if (self.input_w >= w2 and self.input_h >= h2) else cv2.INTER_AREA
        img = cv2.resize(face_crop, (self.input_w, self.input_h), interpolation=interp)
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
            print(f"FaceRecognizer: no classifier at {model_path}.")
            return
        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"FaceRecognizer: failed to load ({e})")
            return
        if data.get("model_type") != "centroid_gallery":
            print(f"FaceRecognizer: unexpected type {data.get('model_type')}")
            return
        self.names = list(data["names"])
        self.centroids = data["centroids"]
        self.galleries = [np.asarray(g, dtype=np.float32) for g in data["galleries"]]
        self.stats_p10 = data["stats_p10"]
        self.stats_p50 = data["stats_p50"]
        self.stats_p90 = data["stats_p90"]
        self.available = True
        sizes = [g.shape[0] for g in self.galleries]
        print(f"FaceRecognizer: loaded for {self.names} (gallery sizes: {sizes})")

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
        min_conf = (self.MIN_CONF_TRACK if confirmed_by_track else self.MIN_CONF_DIRECT)
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
            print(f"AUTO-SAVED: {fname}  (conf {confidence:.2f}, "
                  f"{'track' if confirmed_by_track else 'direct'}, "
                  f"blur {blur_var:.0f}, session {count + 1}/{self.MAX_PER_SESSION})")
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
# Track + Tracker
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
        self.last_scan_frame = -999   # last frame this track was scanned

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


class SimpleTracker:
    """
    Sticky identity:
      - t.name only changes when det["scanned"] is True.
      - A scanned face sets t.name to the recognized name, or "Unknown".
      - If no scanned face this frame, t.name is left untouched.
      - Body signatures accumulate on ANY frame the track has a name and
        a body embedding is available (rate-limited, deduplicated).
      - Unknown tracks die quickly (UNKNOWN_MAX_LOST).
      - New tracks require a scanned face or a large body box.
    """
    MIN_MATCH_SCORE = 0.30
    SMOOTH_ALPHA = 0.55
    VEL_ALPHA = 0.5
    REID_MAX_FRAMES = 60
    REID_MAX_DIST = 150
    BODY_REVIVE_THRESH = 0.90
    SIG_ACCUM_MIN_CONF = 0.85
    SIG_ACCUM_INTERVAL = 5
    UNKNOWN_MAX_LOST = 20

    def __init__(self, max_lost=180, debug=False):
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

    def update(self, detections, frame_count):
        for t in self.tracks:
            t.lost += 1
            t.has_face = False

        # ---- Association ----
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

            # EMA smoothing + velocity
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

            if det.get("has_face"):
                t.has_face = True
                t.last_face_frame = frame_count

            # Identity: only a scanned face changes the name
            if det.get("scanned", False):
                t.last_scan_frame = frame_count   # cooldown timer reset
                if det.get("name") is not None:
                    t.name = det["name"]
                    t.confidence = det.get("conf", 0.0)
                    t.identity_fresh = True
                    t.identity_source = "face"
                else:
                    t.name = "Unknown"
                    t.confidence = 0.0
                    t.identity_fresh = False
                    t.identity_source = "none"
                    t.body_signatures = []
                t.identity_set_frame = frame_count
            else:
                t.identity_fresh = False
                if t.name != "Unknown":
                    t.identity_source = "memory"
                else:
                    t.identity_source = "none"

            # Signature accumulation
            if (t.name != "Unknown"
                    and t.confidence >= self.SIG_ACCUM_MIN_CONF
                    and det.get("body_embedding") is not None):
                last_f = self._last_sig_frame.get(t.id, -999)
                if frame_count - last_f >= self.SIG_ACCUM_INTERVAL:
                    if t.add_signature(det["body_embedding"]):
                        self._last_sig_frame[t.id] = frame_count
                        if self.debug:
                            print(f"[track #{t.id}] stored body signature "
                                  f"({len(t.body_signatures)} total, "
                                  f"name={t.name}, conf={t.confidence:.2f})")

            t.lost = 0
            t.age += 1

        # ---- New detections ----
        for di, det in enumerate(detections):
            if di in matched_dets:
                continue

            # GUARD: never start a track from a weak unscanned body box
            if not det.get("scanned", False) and not det.get("face_only_proxy", False):
                bx, by, bw, bh = det["box"]
                if (bw < getattr(config, "MIN_BODY_BOX_W", 60)
                        or bh < getattr(config, "MIN_BODY_BOX_H", 120)
                        or bw * bh < getattr(config, "MIN_BODY_BOX_AREA", 10000)):
                    continue

            revived_name = None
            revived_conf = 0.0
            revived_sigs = []

            if (not det.get("scanned", False)
                    and det.get("body_embedding") is not None
                    and not det.get("face_only_proxy", False)):
                dcx = det["box"][0] + det["box"][2] // 2
                dcy = det["box"][1] + det["box"][3] // 2
                best_score = self.BODY_REVIVE_THRESH
                for (fc, lb, nm, cf, sigs) in self.recently_lost:
                    if frame_count - fc > self.REID_MAX_FRAMES:
                        continue
                    lcx = lb[0] + lb[2] // 2
                    lcy = lb[1] + lb[3] // 2
                    d = ((dcx - lcx) ** 2 + (dcy - lcy) ** 2) ** 0.5
                    if d > self.REID_MAX_DIST:
                        continue
                    q = det["body_embedding"].astype(np.float32)
                    q = q / (np.linalg.norm(q) + 1e-9)
                    for sig in sigs:
                        v = float(sig @ q)
                        if v > best_score:
                            best_score = v
                            revived_name = nm
                            revived_conf = max(0.5, min(0.85, v))
                            revived_sigs = sigs

            new_name = None
            new_conf = 0.0
            if det.get("scanned", False):
                new_name = det.get("name")
                new_conf = det.get("conf", 0.0) if new_name else 0.0
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
            elif new_name is not None:
                t.identity_fresh = False
                t.body_signatures = list(revived_sigs)
                t.identity_source = "body"
                if self.debug:
                    print(f"[new track #{t.id}] revived as '{new_name}' "
                          f"via body ReID ({len(revived_sigs)} sigs)")

            if (t.name != "Unknown"
                    and det.get("body_embedding") is not None):
                if t.add_signature(det["body_embedding"]):
                    self._last_sig_frame[t.id] = frame_count

            self.next_id += 1
            self.tracks.append(t)

        # ---- Retire dead tracks (Unknown dies faster) ----
        alive = []
        for t in self.tracks:
            limit = self.max_lost if t.name != "Unknown" else self.UNKNOWN_MAX_LOST
            if t.lost <= limit:
                alive.append(t)
            else:
                if t.name != "Unknown" and t.body_signatures:
                    self.recently_lost.append(
                        (frame_count, t.box, t.name, t.confidence,
                         list(t.body_signatures))
                    )
                    if self.debug:
                        print(f"[track #{t.id}] died with "
                              f"{len(t.body_signatures)} body sigs -> cached")
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
            best = max(faces, key=lambda f: f[2] * f[3]) if faces else None
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
                            resized.astype(np.float32) - last_saved_gray.astype(np.float32)
                        )))
                        ok_diverse = diff >= config.ENROLL_DIVERSITY_PX
                    except Exception:
                        ok_diverse = True
                if ok_size and ok_blur and ok_diverse:
                    fname = os.path.join(person_dir, f"enroll_{start_index + captured:04d}.jpg")
                    cv2.imwrite(fname, face_crop)
                    last_saved_gray = cv2.resize(gray, (64, 64))
                    captured += 1
                    saved_this_frame = True
                    status_color = (0, 255, 0)
                    status_text = f"Captured {captured}/{target_count} (blur {blur_var:.0f})"
                    print(f"  Captured {captured}/{target_count}  [size {w}x{h}, blur {blur_var:.0f}]")
                else:
                    reasons = []
                    if not ok_size: reasons.append(f"too small ({w}x{h})")
                    if not ok_blur: reasons.append(f"blurry ({blur_var:.0f})")
                    if not ok_diverse: reasons.append("too similar to last")
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
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            cv2.imshow("Enrollment", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
    print(f"\nEnrollment finished: {captured} new crops saved to {person_dir}")
    print(f"Total crops for '{person_name}': {start_index + captured}")


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
              f"({len(enroll_crops)} enroll, {len(auto_direct)} auto_dir, "
              f"{len(auto_track)} auto_trk"
              + (f", {other} other" if other > 0 else "") + f"){marker}")
    print("\nUse --follow NAME to select one, or press 1..9 at runtime.")


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


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enroll", metavar="NAME")
    parser.add_argument("--samples", type=int, default=config.ENROLL_DEFAULT_COUNT)
    parser.add_argument("--register-only", metavar="NAME",
                        help="Remove all others, then enroll only this person")
    parser.add_argument("--follow", metavar="NAME")
    parser.add_argument("--list-people", action="store_true")
    parser.add_argument("--remove-person", metavar="NAME")
    parser.add_argument("--clear-auto", metavar="NAME")
    parser.add_argument("--test-motors", action="store_true")
    parser.add_argument("--no-auto-save", action="store_true")
    parser.add_argument("--auto-train", action="store_true")
    parser.add_argument("--no-body-reid", action="store_true",
                        help="Disable OSNet body ReID (saves CPU)")
    parser.add_argument("--debug-tracks", action="store_true",
                        help="Print track signature/revival events")
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
                    print(f"Removed existing enrollment: {d}")
        if os.path.exists(config.SVM_MODEL_PATH):
            os.remove(config.SVM_MODEL_PATH)
            print("Removed old classifier.")
        args.enroll = args.register_only

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
        motors.test_motors()
        try: motors.cleanup()
        except Exception: pass
        cam.stop()
        return

    face_embedder = FaceEmbedder(config.MOBILEFACENET_MODEL)
    face_recognizer = FaceRecognizer(config.SVM_MODEL_PATH)

    body_reid = None
    if not args.no_body_reid:
        try:
            body_reid = BodyReID(config.REID_PATH)
        except Exception as e:
            print(f"Body ReID disabled: {e}")
            body_reid = None
    if body_reid is None:
        print("Body ReID: DISABLED")
    else:
        print("Body ReID: ENABLED")

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

    tracker = SimpleTracker(max_lost=180, debug=args.debug_tracks)
    following = False
    last_drive_status = None
    frame_interval = 0.12
    frame_count = 0
    track_cache = TrackConfirmationCache()

    face_max_mm = getattr(config, "FACE_RECOGNIZE_MAX_MM", 80)
    scan_cooldown_s = getattr(config, "FACE_SCAN_COOLDOWN_S", 5.0)
    scan_cooldown_frames = int(scan_cooldown_s * 8)   # ~8 FPS
    min_body_w = getattr(config, "MIN_BODY_BOX_W", 60)
    min_body_h = getattr(config, "MIN_BODY_BOX_H", 120)
    min_body_area = getattr(config, "MIN_BODY_BOX_AREA", 10000)

    print(f"\nRunning. Currently following: {selector.follow_name}")
    print("Controls: q=quit, f=follow, s=stop, 1..9=switch target")
    print(f"Auto-save: {'ON' if auto_save_enabled else 'OFF'}")
    print(f"Face scan gate: only faces closer than {face_max_mm} mm")
    print(f"Face scan cooldown: {scan_cooldown_s:.1f} s per track")
    print(f"Body min box: {min_body_w}x{min_body_h}, area >= {min_body_area}")
    if args.auto_train:
        print("Auto-train: ON")
    if args.debug_tracks:
        print("Track debug: ON")
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

            # 1. Detect faces and bodies
            faces = face_detector.infer(frame)
            bodies_raw = body_detector.infer(frame)

            bodies = []
            for (bx, by, bw, bh, bconf) in bodies_raw:
                if (bw >= min_body_w and bh >= min_body_h
                        and bw * bh >= min_body_area):
                    bodies.append((bx, by, bw, bh, bconf))

            # 2. Recognize each face -- with distance gate AND cooldown gate
            face_recs = []
            for (fx, fy, fw, fh, fconf) in faces:
                fx = max(0, fx); fy = max(0, fy)
                fx2 = min(frame.shape[1], fx + fw)
                fy2 = min(frame.shape[0], fy + fh)
                face_crop = frame[fy:fy2, fx:fx2]
                if face_crop.size == 0:
                    continue

                # Depth at face center
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
                        "name": None,
                        "conf": 0.0,
                        "embedding": None,
                        "crop": face_crop,
                        "depth_mm": face_depth_mm,
                        "skipped": True,
                        "scanned": False,
                        "in_cooldown": False,
                    })
                    continue

                # Cooldown gate: if this face is inside a recently-scanned
                # track, skip the scan this frame.
                face_cx = fx + fw // 2
                face_cy = fy + fh // 2
                in_cooldown = False
                for tr in tracker.tracks:
                    tx, ty, tw, th = tr.box
                    if tx <= face_cx <= tx + tw and ty <= face_cy <= ty + th:
                        frames_since = frame_count - getattr(tr, "last_scan_frame", -999)
                        if frames_since < scan_cooldown_frames:
                            in_cooldown = True
                            break

                if in_cooldown:
                    face_recs.append({
                        "box": (fx, fy, fx2 - fx, fy2 - fy),
                        "name": None,
                        "conf": 0.0,
                        "embedding": None,
                        "crop": face_crop,
                        "depth_mm": face_depth_mm,
                        "skipped": True,
                        "scanned": False,
                        "in_cooldown": True,
                    })
                    continue

                # Passed both gates -> scan
                embedding = face_embedder.embed(face_crop)
                name, svm_conf = face_recognizer.predict(embedding)
                face_recs.append({
                    "box": (fx, fy, fx2 - fx, fy2 - fy),
                    "name": name,
                    "conf": svm_conf,
                    "embedding": embedding,
                    "crop": face_crop,
                    "depth_mm": face_depth_mm,
                    "skipped": False,
                    "scanned": True,
                    "in_cooldown": False,
                })

            # 3. Body ReID embeddings
            body_embs = []
            if body_reid is not None:
                for (bx, by, bw, bh, bconf) in bodies:
                    bx = max(0, bx); by = max(0, by)
                    bx2 = min(frame.shape[1], bx + bw)
                    by2 = min(frame.shape[0], by + bh)
                    body_crop = frame[by:by2, bx:bx2]
                    emb = body_reid.embed(body_crop) if body_crop.size > 0 else None
                    body_embs.append(emb)
            else:
                body_embs = [None] * len(bodies)

            # 4. Associate faces with bodies
            detections = []
            used_faces = set()
            for i, (bx, by, bw, bh, bconf) in enumerate(bodies):
                bx = max(0, bx); by = max(0, by)
                bx2 = min(frame.shape[1], bx + bw)
                by2 = min(frame.shape[0], by + bh)
                body_box = (bx, by, bx2 - bx, by2 - by)

                best_face = None
                best_conf = -1.0
                for fi, fr in enumerate(face_recs):
                    if fi in used_faces:
                        continue
                    fx, fy, fw, fh = fr["box"]
                    fcx = fx + fw // 2
                    fcy = fy + fh // 2
                    if bx <= fcx <= bx + bw and by <= fcy <= by + bh:
                        face_conf = fr["conf"] if fr["name"] is not None else 0.0
                        if face_conf > best_conf:
                            best_conf = face_conf
                            best_face = (fi, fr)

                if best_face is not None:
                    used_faces.add(best_face[0])
                    fr = best_face[1]
                    detections.append({
                        "box": body_box,
                        "name": fr["name"],
                        "conf": fr["conf"],
                        "has_face": True,
                        "scanned": fr.get("scanned", False),
                        "face_box": fr["box"],
                        "face_crop": fr["crop"],
                        "embedding": fr["embedding"],
                        "body_embedding": body_embs[i],
                        "face_only_proxy": False,
                    })
                else:
                    detections.append({
                        "box": body_box,
                        "name": None,
                        "conf": 0.0,
                        "has_face": False,
                        "scanned": False,
                        "face_box": None,
                        "face_crop": None,
                        "embedding": None,
                        "body_embedding": body_embs[i],
                        "face_only_proxy": False,
                    })

            # 4b. Face-only fallback -- only for RECOGNIZED faces
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
                    "name": fr["name"],
                    "conf": fr["conf"],
                    "has_face": True,
                    "scanned": True,
                    "face_box": fr["box"],
                    "face_crop": fr["crop"],
                    "embedding": fr["embedding"],
                    "body_embedding": None,
                    "face_only_proxy": True,
                })

            # 5. Tracker update
            tracks = tracker.update(detections, frame_count)

            # 6. Auto-save + online gallery update
            for det in detections:
                if (det["has_face"] and det.get("scanned", False)
                        and det["name"] is not None
                        and det["embedding"] is not None):
                    face_recognizer.update_gallery(det["name"], det["embedding"], det["conf"])
                    confirmed_by_track = (
                        track_cache.confirmed_name_for(frame_count, det["box"]) == det["name"]
                    )
                    auto_saver.maybe_save(
                        det["name"], det["face_crop"], det["embedding"], det["conf"],
                        confirmed_by_track=confirmed_by_track,
                    )
                    track_cache.update(frame_count, det["box"], det["name"], det["conf"])

            # 7. Choose follow target
            best_track = selector.choose(frame_count, tracks)
            best_box = best_track.box if best_track else None
            person_cx = (best_box[0] + best_box[2] // 2) if best_box else None
            person_cy = (best_box[1] + best_box[3] // 2) if best_box else None

            # 8. Draw tracks
            for t in tracks:
                x, y, w, h = t.box
                is_selected = (best_track is not None and t.id == best_track.id)

                if t.name != "Unknown":
                    src = getattr(t, "identity_source", "memory")
                    if t.identity_fresh:
                        color = (0, 255, 0)
                        tag = ""
                    elif src == "body":
                        color = (0, 255, 255)
                        tag = " (body)"
                    else:
                        color = (0, 165, 255)
                        tag = " (mem)"
                    thickness = 3 if is_selected else 2
                    suffix = " *FOLLOW*" if is_selected else ""
                    n_sig = len(t.body_signatures)
                    label = (f"#{t.id} {t.name}: {t.confidence:.2f}"
                             f"{tag} [s{n_sig}]{suffix}")
                else:
                    color = (0, 0, 255); thickness = 2
                    label = f"#{t.id} Unknown"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Face boxes: gray (too far), blue (cooldown), magenta (scanned)
            for fr in face_recs:
                fx, fy, fw, fh = fr["box"]
                if not fr.get("scanned", False):
                    if fr.get("in_cooldown"):
                        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh),
                                      (200, 120, 0), 1)
                        cv2.putText(frame, "cooldown", (fx, fy - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 120, 0), 1)
                    else:
                        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh),
                                      (128, 128, 128), 1)
                        d = fr.get("depth_mm")
                        txt = f"{d:.0f}mm (no scan)" if d else "no scan"
                        cv2.putText(frame, txt, (fx, fy - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                else:
                    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh),
                                  (255, 0, 255), 1)

            # 9. Depth at target
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
                    cv2.putText(frame, f"{target_distance_mm:.0f}mm", (x, y + h + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if depth is not None:
                _dh, _dw = depth.shape[:2]
                obstacle_sectors = get_sector_distances(depth, best_box, _dw, _dh)
            else:
                obstacle_sectors = {"left": np.inf, "center": np.inf, "right": np.inf}

            frame_cx = config.FRAME_W // 2
            cx = person_cx

            # 10. Motor control
            if following:
                if best_box is None:
                    if last_drive_status != "no_target":
                        print(f"STATUS: {selector.follow_name} not visible.")
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

            # 11. Overlays
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
            cv2.putText(frame, f"Motor: {motor_state}", (config.FRAME_W - 280, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, motor_color, 2)

            if auto_save_enabled and auto_saver.saved_count:
                y = 55
                for name, total in sorted(auto_saver.saved_count.items()):
                    d = auto_saver.saved_direct.get(name, 0)
                    t = auto_saver.saved_track.get(name, 0)
                    cv2.putText(frame, f"{name}:{total} (d{d}/t{t})",
                                (config.FRAME_W - 260, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
                    y += 20

            banner_color = (0, 255, 0) if following else (128, 128, 128)
            cv2.putText(frame, f"Following: {selector.follow_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_color, 2)
            cv2.putText(frame, f"Face scan: <{face_max_mm}mm, cooldown {scan_cooldown_s:.1f}s",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

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
                    selector.set_name(face_recognizer.names[idx])
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
            train_script = os.path.join(script_dir, "train_face_svm.py")
            if not os.path.exists(train_script):
                print(f"\nAuto-train skipped: {train_script} not found.")
            else:
                print("\n" + "=" * 60)
                print("AUTO-TRAIN: handing off to train_face_svm.py")
                print("=" * 60)
                time.sleep(2.0)
                sys.stdout.flush(); sys.stderr.flush()
                try:
                    os.execv(sys.executable, [sys.executable, train_script])
                except Exception as e:
                    print(f"Auto-train exec failed: {e}")
        elif args.auto_train:
            print("\nAuto-train: no crops saved this session; skipping.")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nFATAL: {e}\n")
        sys.exit(1)
