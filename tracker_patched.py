#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import pickle
import argparse
import time
import threading
import re
from math import sqrt

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

sys.path.append('/home/medpal/pyorbbecsdk_v1/build')

try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
    USE_ORBBEC_DEPTH = True
except ImportError:
    USE_ORBBEC_DEPTH = False

import onnxruntime as ort

# Coral TPU support (optional)
CORAL_AVAILABLE = False
CORAL_EDGETPU = False
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    CORAL_AVAILABLE = True
    try:
        delegate = load_delegate('libedgetpu.so.1')
        CORAL_EDGETPU = True
        print("Coral Edge TPU delegate loaded successfully")
    except Exception as e:
        print(f"Coral Edge TPU delegate unavailable: {e}")
        print("Tip: Check USB power (use powered hub) or try a different USB port.")
except ImportError:
    print("tflite_runtime not installed. Coral TPU not available.")

import config
from serial_motors import SerialMotors

LEFT_RPWM = config.LEFT_RPWM
LEFT_LPWM = config.LEFT_LPWM
LEFT_REN  = config.LEFT_REN
LEFT_LEN  = config.LEFT_LEN
RIGHT_RPWM = config.RIGHT_RPWM
RIGHT_LPWM = config.RIGHT_LPWM
RIGHT_REN  = config.RIGHT_REN
RIGHT_LEN  = config.RIGHT_LEN

FRAME_W, FRAME_H = config.FRAME_W, config.FRAME_H
CONF_THRES = config.CONF_THRES
COSINE_THRES = config.COSINE_THRES

YOLO_PATH = config.YOLO_PATH
REID_PATH = config.REID_PATH
TARGET_PATH = config.TARGET_PATH
TARGETS_DIR = config.TARGETS_DIR

# Speed config (percentages)
FOLLOW_BASE_SPEED = config.FOLLOW_BASE_SPEED
REVERSE_SPEED = config.REVERSE_SPEED
SPEED_INCREASE = config.SPEED_INCREASE
MAX_SPEED = config.MAX_SPEED

# Distance thresholds (mm)
MIN_DISTANCE_MM = config.MIN_DISTANCE_MM
REVERSE_DISTANCE_MM = config.REVERSE_DISTANCE_MM
DISTANCE_THRESHOLD_MM = config.DISTANCE_THRESHOLD_MM

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
        self.color_cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.color_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.color_frame = None
        self.depth_frame = None  # always set, even if depth never starts -- prevents AttributeError in read_depth()
        self.color_running = True
        self.color_thread = threading.Thread(target=self._color_capture, daemon=True)
        self.color_thread.start()
        print("Using cv2.VideoCapture for color")
        if USE_ORBBEC_DEPTH:
            self.depth_pipeline = Pipeline()
            self.depth_config = Config()
            profiles = self.depth_pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            # Try to use low-bandwidth profile for USB2.0 compatibility
            try:
                # 320x240 @ 15fps to reduce USB2.0 bandwidth
                self.depth_profile = profiles.get_video_stream_profile(320, 240, OBFormat.Y16, 15)
            except:
                # Fallback to default if specific profile not available
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
                        scale = df.get_depth_scale()  # 1.0 for Astra (already in mm)
                        data = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(h, w)
                        # Store as float32 in mm
                        self.depth_frame = data.astype(np.float32) * scale
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


class YOLO:
    def __init__(self, path):
        self.session = ort.InferenceSession(path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        if isinstance(self.input_shape[2], str) or self.input_shape[2] is None:
            self.h, self.w = 640, 640
        else:
            self.h, self.w = self.input_shape[2], self.input_shape[3]

    def infer(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True)
        outputs = self.session.run(None, {self.input_name: blob})[0]
        return self._postprocess(outputs, frame.shape)

    def _postprocess(self, outputs, shape):
        boxes = []
        h, w = shape[:2]
        out = outputs[0].T  # (8400, 84)
        for det in out:
            conf = det[4:].max()
            if conf < CONF_THRES:
                continue
            cls_id = det[4:].argmax()
            if cls_id != 0:
                continue
            cx, cy, bw, bh = det[:4]
            x = int((cx - bw/2))
            y = int((cy - bh/2))
            bw = int(bw)
            bh = int(bh)
            boxes.append((x, y, bw, bh, float(conf)))
        return boxes


class ReID:
    def __init__(self, path):
        self.session = ort.InferenceSession(path)
        self.input_name = self.session.get_inputs()[0].name

    def embed(self, crop):
        img = cv2.resize(crop, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        out = self.session.run(None, {self.input_name: img})[0]
        vec = out.flatten()
        return vec / np.linalg.norm(vec)


class CoralDetector:
    """Person detection using Coral TPU (SSD MobileNet V2)."""

    def __init__(self, model_path, cpu_model_path=None, label_path=None):
        self.labels = []
        if label_path and os.path.exists(label_path):
            with open(label_path) as f:
                self.labels = [l.strip() for l in f.readlines()]

        # Derive the real "person" class index from the labels file instead of
        # trusting the hardcoded COCO convention -- that hardcoded value (1)
        # was wrong for this specific labels file and silently rejected every
        # real person detection despite high confidence scores.
        self.PERSON_CLASS = 1
        for i, name in enumerate(self.labels):
            if name.strip().lower() == "person":
                self.PERSON_CLASS = i
                break
        print(f"CoralDetector: using class index {self.PERSON_CLASS} for 'person' "
              f"(derived from labels file, not hardcoded)")

        if CORAL_EDGETPU:
            delegate = load_delegate('libedgetpu.so.1')
            self.interpreter = Interpreter(model_path,
                                           experimental_delegates=[delegate])
        else:
            cpu_path = cpu_model_path or model_path.replace('_edgetpu', '')
            self.interpreter = Interpreter(cpu_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']

    def infer(self, frame):
        h, w = frame.shape[:2]
        # Resize to model input size (300x300 for SSD MobileNet)
        target_h, target_w = self.input_shape[1], self.input_shape[2]
        resized = cv2.resize(frame, (target_w, target_h))
        input_data = np.expand_dims(resized, axis=0).astype(np.uint8)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # SSD MobileNet outputs: detection_boxes, detection_classes, detection_scores, num_detections
        boxes = self.interpreter.tensor(self.output_details[0]['index'])()
        classes = self.interpreter.tensor(self.output_details[1]['index'])()
        scores = self.interpreter.tensor(self.output_details[2]['index'])()

        # DIAGNOSTIC: track the single best raw detection this call, any class,
        # ignoring both the confidence threshold and the person-class filter.
        # This tells us whether the model sees NOTHING (raw scores near 0 for
        # everything) vs sees SOMETHING but not classified as person above
        # threshold (a real, different problem).
        self.last_top_score = 0.0
        self.last_top_class = -1

        results = []
        for i in range(int(scores[0].shape[0])):
            score = float(scores[0][i])
            cls = int(classes[0][i])
            if score > self.last_top_score:
                self.last_top_score = score
                self.last_top_class = cls
            if score < CONF_THRES or cls != self.PERSON_CLASS:
                continue
            y1, x1, y2, x2 = boxes[0][i]
            x = int(x1 * w)
            y = int(y1 * h)
            bw = int((x2 - x1) * w)
            bh = int((y2 - y1) * h)
            results.append((x, y, bw, bh, score))
        return results

    def top_detection_label(self):
        if self.last_top_class < 0:
            return f"nothing at all (top raw score {self.last_top_score:.3f})"
        name = self.labels[self.last_top_class] if self.last_top_class < len(self.labels) else str(self.last_top_class)
        return f"'{name}' at {self.last_top_score:.3f} (need >={CONF_THRES} AND class==person to count)"


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def normalise_embedding(embedding):
    """Return one float32, unit-length ReID embedding."""
    embedding = np.asarray(embedding, dtype=np.float32).flatten()
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


def load_target_gallery(data):
    """Read both new gallery targets and older single-embedding target files."""
    if isinstance(data, dict):
        saved = data.get("embeddings", data.get("embedding"))
        name = data.get("name", "Target")
    else:
        saved = data
        name = "Target"

    if saved is None:
        return name, []
    saved = np.asarray(saved)
    if saved.ndim == 1:
        saved = saved[np.newaxis, :]
    return name, [normalise_embedding(embedding) for embedding in saved]


def profile_key(name):
    """Create a stable, filesystem-safe key while preserving the display name."""
    key = re.sub(r"[^a-z0-9]+", "_", name.strip().casefold()).strip("_")
    if not key:
        raise ValueError("Target name must contain at least one letter or number.")
    return key


def profile_path(name):
    return os.path.join(TARGETS_DIR, f"{profile_key(name)}.pkl")


def list_profiles():
    """Return saved named profiles as (display_name, path, sample_count)."""
    profiles = []
    if os.path.isdir(TARGETS_DIR):
        for filename in sorted(os.listdir(TARGETS_DIR)):
            if not filename.endswith(".pkl"):
                continue
            path = os.path.join(TARGETS_DIR, filename)
            try:
                with open(path, "rb") as f:
                    name, gallery = load_target_gallery(pickle.load(f))
                profiles.append((name, path, len(gallery)))
            except Exception as error:
                print(f"Skipping unreadable target profile '{filename}': {error}")

    # The original single target remains visible and usable until it is
    # deliberately re-registered under a named profile.
    if os.path.exists(TARGET_PATH):
        try:
            with open(TARGET_PATH, "rb") as f:
                name, gallery = load_target_gallery(pickle.load(f))
            if not any(saved_name.casefold() == name.casefold() for saved_name, _, _ in profiles):
                profiles.append((f"{name} (legacy default)", TARGET_PATH, len(gallery)))
        except Exception as error:
            print(f"Skipping unreadable legacy target: {error}")
    return profiles


def load_named_profile(name):
    """Load one named target, falling back to a matching legacy target."""
    path = profile_path(name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return load_target_gallery(pickle.load(f))

    if os.path.exists(TARGET_PATH):
        with open(TARGET_PATH, "rb") as f:
            saved_name, gallery = load_target_gallery(pickle.load(f))
        if saved_name.casefold() == name.strip().casefold():
            return saved_name, gallery
    raise FileNotFoundError(f"No target profile named '{name}'. Use --list-targets to see available names.")


def save_named_profile(name, gallery):
    """Save one profile only; same name replaces only that profile."""
    display_name = name.strip()
    path = profile_path(display_name)
    os.makedirs(TARGETS_DIR, exist_ok=True)
    replaced = os.path.exists(path)
    with open(path, "wb") as f:
        pickle.dump({"name": display_name, "embeddings": np.stack(gallery)}, f)
    return replaced


def remove_named_profile(name):
    """Remove exactly one named profile, never the entire profile directory."""
    path = profile_path(name)
    if os.path.exists(path):
        os.remove(path)
        return True
    if os.path.exists(TARGET_PATH):
        with open(TARGET_PATH, "rb") as f:
            saved_name, _ = load_target_gallery(pickle.load(f))
        if saved_name.casefold() == name.strip().casefold():
            os.remove(TARGET_PATH)
            return True
    return False


def best_gallery_score(embedding, gallery):
    """Score an embedding against the closest enrolled appearance sample."""
    if not gallery:
        return -1.0
    return max(cosine(embedding, enrolled) for enrolled in gallery)


def capture_gallery(cam, detector, reid, sample_count=10):
    """Capture separate frames for enrollment, keeping every useful sample."""
    embeddings = []
    for _ in range(sample_count):
        time.sleep(0.12)  # Let the camera thread provide a fresh frame.
        frame = cam.read_color()
        if frame is None:
            continue
        boxes = detector.infer(frame)
        if not boxes:
            continue
        x, y, w, h, _ = max(boxes, key=lambda box: box[4])
        crop = frame[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
        if crop.size > 0 and w > 20 and h > 40:
            embeddings.append(reid.embed(crop))
            print(f"  Sample {len(embeddings)}/{sample_count}")
    return [normalise_embedding(embedding) for embedding in embeddings]


class Motors:
    def __init__(self):
        try:
            import lgpio
            self.chip = lgpio.gpiochip_open(0)
            self.lgpio = lgpio
            self.pins = [LEFT_RPWM, LEFT_LPWM, LEFT_REN, LEFT_LEN,
                         RIGHT_RPWM, RIGHT_LPWM, RIGHT_REN, RIGHT_LEN]
            for pin in self.pins:
                try:
                    lgpio.gpio_free(self.chip, pin)
                except:
                    pass
                lgpio.gpio_claim_output(self.chip, pin, 0)
            # Enable motors
            for pin in [LEFT_REN, LEFT_LEN, RIGHT_REN, RIGHT_LEN]:
                lgpio.gpio_write(self.chip, pin, 1)
            self.available = True
            self.active_pins = []
            self.current_speed = config.FOLLOW_BASE_SPEED / 100  # Convert % to 0-1
            print("Motors initialized (lgpio)")
        except Exception as e:
            self.available = False
            print(f"Motors unavailable: {e}")
    
    def _pwm(self, pin, duty):
        self.lgpio.tx_pwm(self.chip, pin, 1000, duty * 100)
    
    def set_speed(self, speed):
        """Set speed (0-100%) and apply to active motors."""
        self.current_speed = max(0.0, min(1.0, speed / 100))
        if self.active_pins:
            for pin in self.active_pins:
                self._pwm(pin, self.current_speed)
        print(f"Speed: {speed}%")
        return self.current_speed
    
    def stop(self):
        if not self.available:
            return
        self.active_pins = []
        for pin in [LEFT_RPWM, LEFT_LPWM, RIGHT_RPWM, RIGHT_LPWM]:
            self._pwm(pin, 0)
    
    def forward(self):
        if not self.available:
            return
        self.stop()
        self.active_pins = [LEFT_LPWM, RIGHT_LPWM]
        self._pwm(LEFT_RPWM, 0)
        self._pwm(RIGHT_RPWM, 0)
        self._pwm(LEFT_LPWM, self.current_speed)
        self._pwm(RIGHT_LPWM, self.current_speed)

    def backward(self):
        if not self.available:
            return
        self.stop()
        self.active_pins = [LEFT_RPWM, RIGHT_RPWM]
        self._pwm(LEFT_LPWM, 0)
        self._pwm(RIGHT_LPWM, 0)
        self._pwm(LEFT_RPWM, self.current_speed)
        self._pwm(RIGHT_RPWM, self.current_speed)
    
    def turn_left(self):
        """Tank turn LEFT: left wheel forward, right wheel reverse."""
        if not self.available:
            return
        self.stop()
        self.active_pins = [LEFT_LPWM, RIGHT_RPWM]
        self._pwm(LEFT_RPWM, 0)
        self._pwm(RIGHT_LPWM, 0)
        self._pwm(LEFT_LPWM, self.current_speed)
        self._pwm(RIGHT_RPWM, self.current_speed)

    def turn_right(self):
        """Tank turn RIGHT: left wheel reverse, right wheel forward."""
        if not self.available:
            return
        self.stop()
        self.active_pins = [LEFT_RPWM, RIGHT_LPWM]
        self._pwm(LEFT_LPWM, 0)
        self._pwm(RIGHT_RPWM, 0)
        self._pwm(LEFT_RPWM, self.current_speed)
        self._pwm(RIGHT_LPWM, self.current_speed)
    
    def test_motors(self):
        """Test all motors: forward, stop, reverse, stop, turn left, turn right."""
        if not self.available:
            print("Motors not available!")
            return
        print("Testing motors...")
        # Forward at 60%
        print("  Forward...")
        self.set_speed(60)
        self.forward()
        time.sleep(2)
        self.stop()
        time.sleep(0.5)
        # Reverse at 60%
        print("  Reverse...")
        self.set_speed(60)
        self.backward()
        time.sleep(2)
        self.stop()
        time.sleep(0.5)
        # Turn left at 60%
        print("  Turn left...")
        self.turn_left()
        time.sleep(2)
        self.stop()
        time.sleep(0.5)
        # Turn right at 60%
        print("  Turn right...")
        self.turn_right()
        time.sleep(2)
        self.stop()
        print("Motor test complete!")
    
    def cleanup(self):
        """Release GPIO resources."""
        if self.available:
            self.stop()
            for pin in [LEFT_REN, LEFT_LEN, RIGHT_REN, RIGHT_LEN]:
                self.lgpio.gpio_write(self.chip, pin, 0)
            for pin in self.pins:
                try:
                    self.lgpio.gpio_free(self.chip, pin)
                except:
                    pass
            try:
                self.lgpio.gpiochip_close(self.chip)
            except:
                pass
            print("GPIO cleaned up")


def compute_steering(bbox, frame_w, frame_h, speed=None, distance_mm=None):
    # Stop if too close
    if distance_mm is not None and distance_mm < config.MIN_DISTANCE_MM:
        return 0, 0
    
    x, y, w, h = bbox
    cx = x + w // 2
    frame_cx = frame_w // 2
    
    # Use provided speed or default
    if speed is not None:
        base = speed
    else:
        base = config.FOLLOW_BASE_SPEED
    
    # If centered (within 50px), move forward
    if abs(cx - frame_cx) <= 50:
        return base, base
    
    # Not centered - turn toward person
    turn_speed = config.FOLLOW_BASE_SPEED // 2
    if cx < frame_cx:  # Person is left, turn left
        return -turn_speed, turn_speed
    else:  # Person is right, turn right
        return turn_speed, -turn_speed


def calculate_speed(distance_mm):
    """Calculate motor speed: 60% base at 100mm + 10% per 100mm above."""
    if distance_mm is None:
        return config.FOLLOW_BASE_SPEED
    if distance_mm <= config.REVERSE_DISTANCE_MM:  # 0-75mm
        return config.REVERSE_SPEED  # 40% for backward
    elif distance_mm < config.FORWARD_DISTANCE_MM:  # 75-99mm
        return 0  # Stop zone
    else:  # ≥100mm
        # 60% base at 100mm, +10% per 100mm above 100mm
        extra = ((distance_mm - config.FORWARD_DISTANCE_MM) // 100) * config.SPEED_INCREASE
        return min(config.FOLLOW_BASE_SPEED + extra, config.MAX_SPEED)


def get_sector_distances(depth, person_box, depth_w, depth_h):
    """
    Split the depth frame into left/center/right sectors and return the
    minimum valid distance (mm) in each, excluding the tracked person's own
    columns so the robot doesn't treat the person it's following as an
    obstacle. Uses a row band (35%-75% of height) to avoid floor/ceiling.
    Returns dict with 'left', 'center', 'right' -- np.inf if no valid reading.
    """
    if depth is None:
        return {"left": np.inf, "center": np.inf, "right": np.inf}

    row_start, row_end = int(depth_h * 0.35), int(depth_h * 0.75)
    band = depth[row_start:row_end, :].copy()

    if person_box is not None:
        x, y, w, h = person_box
        # person_box is in color-frame coords; scale to depth-frame coords
        scale_x = depth_w / config.FRAME_W
        scale_y = depth_h / config.FRAME_H
        dx1 = int(x * scale_x)
        dx2 = int((x + w) * scale_x)
        band[:, max(0, dx1):min(depth_w, dx2)] = 0  # 0 = treated as invalid below

    third = depth_w // 3
    sectors = {}
    for name, (c1, c2) in [("left", (0, third)), ("center", (third, 2 * third)), ("right", (2 * third, depth_w))]:
        region = band[:, c1:c2]
        valid = region[region > 0]
        sectors[name] = float(np.min(valid)) if valid.size > 0 else np.inf
    return sectors


def apply_obstacle_avoidance(target_speed, cx, frame_cx, sectors):
    """
    Takes the speed/steering the follow logic already decided on and reduces
    or overrides it based on obstacle sectors. Returns (speed, extra_turn),
    where extra_turn is an additional bias applied to steering (+ = steer
    right-away-from-left-obstacle, - = steer left-away-from-right-obstacle).
    Does NOT touch person-distance stop/reverse logic -- this only guards
    against things beside/ahead of the robot that the person-tracking logic
    doesn't know about.
    """
    center, left, right = sectors["center"], sectors["left"], sectors["right"]
    extra_turn = 0

    if center < config.OBSTACLE_STOP_MM:
        return 0, 0  # hard stop, no forward motion regardless of person distance

    if center < config.OBSTACLE_SLOW_MM:
        # linear taper between STOP and SLOW distance
        frac = (center - config.OBSTACLE_STOP_MM) / (config.OBSTACLE_SLOW_MM - config.OBSTACLE_STOP_MM)
        target_speed = int(target_speed * max(0.0, min(1.0, frac)))

    if left < config.OBSTACLE_SLOW_MM or right < config.OBSTACLE_SLOW_MM:
        extra_turn = config.OBSTACLE_SIDE_BIAS if left < right else -config.OBSTACLE_SIDE_BIAS

    return target_speed, extra_turn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", nargs="?", const="", metavar="NAME",
                        help="Register or replace a named target profile")
    parser.add_argument("--target", metavar="NAME", help="Use this named target profile")
    parser.add_argument("--list-targets", action="store_true", help="Show saved target profiles")
    parser.add_argument("--remove-target", metavar="NAME", help="Remove one named target profile")
    parser.add_argument("--test-motors", action="store_true", help="Test motor movements")
    args = parser.parse_args()

    if args.list_targets:
        profiles = list_profiles()
        if not profiles:
            print("No saved target profiles.")
        else:
            print("Saved target profiles:")
            for name, _, samples in profiles:
                print(f"  - {name}: {samples} gallery sample(s)")
        return

    if args.remove_target:
        try:
            removed = remove_named_profile(args.remove_target)
        except ValueError as error:
            parser.error(str(error))
        if removed:
            print(f"Removed target profile '{args.remove_target}'.")
        else:
            print(f"No target profile named '{args.remove_target}'.")
        return

    if args.register is not None and args.target and args.register and \
            args.register.casefold() != args.target.casefold():
        parser.error("Use one name: --register NAME or --target NAME --register.")

    cam = Camera()

    # Try Coral TPU detection first, fall back to YOLO (CPU)
    detector = None
    detector_name = "CPU (YOLO)"
    try:
        if CORAL_EDGETPU:
            detector = CoralDetector(config.CORAL_DETECTION_MODEL,
                                     config.CORAL_DETECTION_MODEL_CPU,
                                     config.CORAL_LABELS)
            detector_name = "Coral TPU (SSD MobileNet)"
            print("Using Coral TPU for detection")
        else:
            raise Exception("No Edge TPU delegate")
    except Exception:
        detector = YOLO(YOLO_PATH)
        detector_name = "CPU (YOLO)"
        print("Using YOLO (CPU) for detection")

    reid = ReID(REID_PATH)
    motors = SerialMotors()

    if args.test_motors:
        motors.test_motors()
        return

    target_gallery = []
    target_name = "Target"
    following = False
    target_locked = False
    lock_streak = 0
    loss_streak = 0
    last_drive_status = None
    registration_done = False
    reversing = False  # Kept for backward compatibility, not used now
    if args.target:
        try:
            target_name, target_gallery = load_named_profile(args.target)
        except (FileNotFoundError, ValueError) as error:
            print(error)
            cam.stop()
            return
        following = True
        print(f"Loaded target: {target_name} ({len(target_gallery)} gallery sample(s))")
    elif os.path.exists(TARGET_PATH) and args.register is None:
        with open(TARGET_PATH, "rb") as f:
            target_name, target_gallery = load_target_gallery(pickle.load(f))
        following = True  # Auto-start following when legacy target loaded
        print(f"Loaded target: {target_name} ({len(target_gallery)} gallery sample(s))")

    try:
        os.nice(10)
    except:
        pass
    print("Running. Focus the 'MedPal Tracker' video window for controls: q=quit, r=register, s=stop, f=follow.")
    frame_count = 0
    frame_interval = 0.12
    cv2.namedWindow("MedPal Tracker", cv2.WINDOW_NORMAL)
    try:
        while True:
            frame_start = time.monotonic()
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"DEBUG: Frame {frame_count}, gallery samples: {len(target_gallery)}, locked: {target_locked}")
            frame = cam.read_color()
            if frame is None:
                time.sleep(0.01)
                continue

            depth = cam.read_depth()

            boxes = detector.infer(frame)
            if not boxes:
                motors.stop()
                if last_drive_status != "no_person":
                    print("STATUS: No person detected. Robot stopped.")
                last_drive_status = "no_person"
                cv2.putText(frame, f"No person detected ({detector_name})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if frame_count % 30 == 0 and hasattr(detector, "top_detection_label"):
                    print(f"DEBUG: model's best raw guess this frame: {detector.top_detection_label()}")
                if frame_count % 60 == 0:
                    debug_path = f"/tmp/no_detect_frame_{frame_count}.jpg"
                    cv2.imwrite(debug_path, frame)
                    print(f"DEBUG: saved no-detection snapshot to {debug_path}")
                cv2.imshow("MedPal Tracker", frame)
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    following = False
                    motors.stop()
                    print("Stopped following.")
                continue

            if args.register is not None and not registration_done:
                print("Collecting up to 10 target samples. Stay visible and vary your pose slightly...")
                captured_gallery = capture_gallery(cam, detector, reid)
                if captured_gallery:
                    requested_name = args.register.strip() or args.target or \
                        (input("Enter name: ").strip() if sys.stdin.isatty() else "Target")
                    try:
                        replaced = save_named_profile(requested_name, captured_gallery)
                    except ValueError as error:
                        print(f"Registration failed: {error}")
                    else:
                        target_name = requested_name
                        target_gallery = captured_gallery
                        print(f"Target '{target_name}' {'replaced' if replaced else 'registered'} "
                              f"with {len(target_gallery)} gallery samples!")
                else:
                    print("Failed to capture any target samples!")
                registration_done = True

            # Initialize
            target_distance_mm = None
            person_cx = None
            person_cy = None
            
            # Get best box
            best_score = -1.0
            if not target_gallery:
                best_box = max(boxes, key=lambda b: b[4])[:4] if boxes else None
            else:
                best_box = None
                highest_seen = -1.0
                candidate_box = None
                for (x, y, w, h, conf) in boxes:
                    crop = frame[y:y+h, x:x+w]
                    if crop.size == 0:
                        continue
                    emb = reid.embed(crop)
                    score = best_gallery_score(emb, target_gallery)
                    highest_seen = max(highest_seen, score)
                    if score > best_score:
                        best_score = score
                        candidate_box = (x, y, w, h)

                # Lock only after repeated good matches. Once locked, retain
                # the last valid target for a few bad frames, but never drive
                # from an unconfirmed new candidate.
                if best_score >= config.COSINE_THRES:
                    lock_streak = min(lock_streak + 1, config.REID_LOCK_FRAMES)
                    loss_streak = 0
                    if not target_locked and lock_streak >= config.REID_LOCK_FRAMES:
                        target_locked = True
                        print(f"ReID lock acquired after {lock_streak} matching frames.")
                else:
                    lock_streak = 0
                    if target_locked:
                        loss_streak += 1
                        if loss_streak >= config.REID_LOSS_FRAMES:
                            target_locked = False
                            print(f"ReID lock lost after {loss_streak} weak frames.")

                # A lock may survive a short dip, but motion must use a
                # currently confident box.  We deliberately stop during the
                # dip instead of steering toward a possibly wrong person.
                best_box = candidate_box if target_locked and best_score >= config.COSINE_THRES else None
                if frame_count % 15 == 0:
                    lock_status = "LOCKED" if target_locked else \
                        f"acquiring {lock_streak}/{config.REID_LOCK_FRAMES}"
                    print(f"ReID: score {highest_seen:.3f} (need >= {config.COSINE_THRES}), "
                          f"{lock_status}; boxes={len(boxes)}")
            
            # Get person position
            if best_box is not None:
                x, y, w, h = best_box
                person_cx = x + w // 2
                person_cy = y + h // 2

            # Obstacle sectors (independent of person tracking, excludes person's own bbox)
            if depth is not None:
                _dh, _dw = depth.shape[:2]
                obstacle_sectors = get_sector_distances(depth, best_box, _dw, _dh)
            else:
                obstacle_sectors = {"left": np.inf, "center": np.inf, "right": np.inf}

            # Sample depth at person center
            if depth is not None and person_cx is not None:
                depth_h, depth_w = depth.shape[:2]
                scale_x = depth_w / config.FRAME_W
                scale_y = depth_h / config.FRAME_H
                depth_cx = int(person_cx * scale_x)
                depth_cy = int(person_cy * scale_y)
                if depth_cx < depth.shape[1] and depth_cy < depth.shape[0]:
                    y1 = max(0, depth_cy-10)
                    y2 = min(depth.shape[0], depth_cy+10)
                    x1 = max(0, depth_cx-10)
                    x2 = min(depth.shape[1], depth_cx+10)
                    roi = depth[y1:y2, x1:x2]
                    valid = roi[roi > 0]
                    if valid.size > 0:
                        target_distance_mm = float(np.median(valid))
                    else:
                        target_distance_mm = float(depth[depth_cy, depth_cx])
            
            # Show visualization for registered person
            if best_box and target_gallery and target_locked:
                x, y, w, h = best_box
                cx = x + w // 2
                cy = y + h // 2
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 255), 1)
                label = f"{target_name}: {best_score:.2f}"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if target_distance_mm is not None:
                    if target_distance_mm > config.MIN_DISTANCE_MM:
                        color = (0, 255, 0)
                    elif target_distance_mm > config.REVERSE_DISTANCE_MM:
                        color = (0, 255, 255)
                    else:
                        color = (0, 0, 255)
                    cv2.putText(frame, f"{target_distance_mm:.0f}mm", (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else:
                    cv2.putText(frame, "No depth", (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Motor control - ONLY when following
            frame_cx = config.FRAME_W // 2
            cx = person_cx  # Use person position for motor control
            
            if following:
                if obstacle_sectors["center"] < config.OBSTACLE_STOP_MM:
                    # Hard obstacle override: something is directly ahead, closer
                    # than we ever want to be regardless of where the person is.
                    if last_drive_status != "obstacle_stop":
                        print(f"STATUS: Obstacle ahead ({obstacle_sectors['center']:.0f} mm). Robot stopped.")
                    last_drive_status = "obstacle_stop"
                    motors.stop()
                elif target_distance_mm is None:
                    # No depth data at all -- STOP, don't guess. Reversing on
                    # missing data (rather than a real close-distance reading)
                    # is unsafe: it means a camera glitch or SDK failure drives
                    # the robot backward instead of doing nothing.
                    if last_drive_status != "no_depth":
                        print("STATUS: No depth data. Robot stopped safely; it will not reverse.")
                    last_drive_status = "no_depth"
                    motors.stop()
                elif target_distance_mm < config.REVERSE_DISTANCE_MM:
                    # Genuinely measured too close -> Reverse at 30%
                    if last_drive_status != "reversing":
                        print(f"STATUS: Target too close ({target_distance_mm:.0f} mm). Reversing.")
                    last_drive_status = "reversing"
                    motors.set_speed(config.REVERSE_SPEED)
                    motors.backward()
                    if cx is not None:
                        if cx < frame_cx - 80:
                            motors.turn_left()
                        elif cx > frame_cx + 80:
                            motors.turn_right()
                elif target_distance_mm < config.FORWARD_DISTANCE_MM:
                    # 60-99mm: Stop forward/backward, but still turn
                    last_drive_status = "target_close"
                    motors.stop()
                    if cx is not None:
                        if cx < frame_cx - 80:
                            motors.set_speed(config.FOLLOW_BASE_SPEED)
                            motors.turn_left()
                        elif cx > frame_cx + 80:
                            motors.set_speed(config.FOLLOW_BASE_SPEED)
                            motors.turn_right()
                else:  # >=100mm
                    # Forward with steering
                    last_drive_status = "tracking"
                    speed = calculate_speed(target_distance_mm)
                    speed, obstacle_turn_bias = apply_obstacle_avoidance(
                        speed, cx, frame_cx, obstacle_sectors)
                    if obstacle_turn_bias != 0:
                        print(f"DEBUG: Obstacle steer bias={obstacle_turn_bias} "
                              f"L={obstacle_sectors['left']:.0f} R={obstacle_sectors['right']:.0f}")
                    if cx is not None:
                        if cx < frame_cx - 80:
                            motors.set_speed(config.FOLLOW_BASE_SPEED)
                            motors.turn_left()
                        elif cx > frame_cx + 80:
                            motors.set_speed(config.FOLLOW_BASE_SPEED)
                            motors.turn_right()
                        elif obstacle_turn_bias > 0:
                            motors.set_speed(min(config.MAX_SPEED, config.FOLLOW_BASE_SPEED + obstacle_turn_bias))
                            motors.turn_right()  # steer away from blocked left side
                        elif obstacle_turn_bias < 0:
                            motors.set_speed(min(config.MAX_SPEED, config.FOLLOW_BASE_SPEED - obstacle_turn_bias))
                            motors.turn_left()  # steer away from blocked right side
                        else:
                            motors.set_speed(speed)
                            motors.forward()
                    else:
                        motors.set_speed(speed)
                        motors.forward()
            else:
                # Not following
                last_drive_status = "not_following"
                motors.stop()
            
            # Show depth on top-left
            if target_distance_mm is not None:
                if target_distance_mm > config.MIN_DISTANCE_MM:
                    depth_color = (0, 255, 0)
                elif target_distance_mm > config.REVERSE_DISTANCE_MM:
                    depth_color = (0, 255, 255)
                else:
                    depth_color = (0, 0, 255)
                cv2.putText(frame, f"Depth: {target_distance_mm:.0f}mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, depth_color, 2)
            
            # Motor state on top-right
            motor_state = "IDLE"
            motor_color = (128, 128, 128)
            if following:
                if target_distance_mm is not None:
                    if target_distance_mm < config.REVERSE_DISTANCE_MM:
                        motor_state = "BACKWARD"
                        motor_color = (0, 0, 255)
                    elif target_distance_mm < config.FORWARD_DISTANCE_MM:
                        if cx is not None:
                            if cx < frame_cx - 80:
                                motor_state = "TURN L"
                                motor_color = (0, 255, 255)
                            elif cx > frame_cx + 80:
                                motor_state = "TURN R"
                                motor_color = (0, 255, 255)
                            else:
                                motor_state = "STOP"
                                motor_color = (0, 165, 255)
                        else:
                            motor_state = "STOP"
                            motor_color = (0, 165, 255)
                    else:  # >=100mm
                        speed = calculate_speed(target_distance_mm)
                        if cx is not None:
                            if cx < frame_cx - 80:
                                motor_state = "TURN L"
                                motor_color = (0, 255, 255)
                            elif cx > frame_cx + 80:
                                motor_state = "TURN R"
                                motor_color = (0, 255, 255)
                            else:
                                motor_state = f"FWD {speed}%"
                                motor_color = (0, 255, 0)
                        else:
                            motor_state = f"FWD {speed}%"
                            motor_color = (0, 255, 0)
                else:
                    motor_state = "WAITING FOR TARGET" if best_box is None else "STOP (no depth)"
                    motor_color = (0, 0, 255)
            cv2.putText(frame, f"Motor: {motor_state}", (config.FRAME_W-250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, motor_color, 2)

            # Obstacle sector overlay -- watch these while testing to sanity-check
            # OBSTACLE_STOP_MM / OBSTACLE_SLOW_MM against real readings.
            def _fmt(d):
                return "inf" if d == np.inf else f"{d:.0f}"
            sec_y = 55
            for i, name in enumerate(["left", "center", "right"]):
                val = obstacle_sectors[name]
                col = (0, 0, 255) if val < config.OBSTACLE_STOP_MM else \
                      (0, 255, 255) if val < config.OBSTACLE_SLOW_MM else (0, 255, 0)
                cv2.putText(frame, f"{name[0].upper()}:{_fmt(val)}mm",
                            (config.FRAME_W - 250 + i * 85, sec_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

            elapsed = time.monotonic() - frame_start
            sleep = max(0, frame_interval - elapsed)
            if sleep > 0:
                time.sleep(sleep)

            cv2.imshow("MedPal Tracker", frame)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                if target_gallery:
                    following = True
                    print("Started following!")
                else:
                    print("No target registered! Press 'r' first.")
            elif key == ord('s'):
                following = False
                reversing = False
                motors.stop()
                print("Stopped following.")
            elif key == ord('r'):
                if boxes:
                    print("Collecting up to 10 replacement target samples...")
                    replacement_gallery = capture_gallery(cam, detector, reid)
                    if replacement_gallery:
                        entered_name = input(f"Enter name [{target_name}]: ").strip() if sys.stdin.isatty() else target_name
                        saved_name = entered_name or target_name
                        try:
                            replaced = save_named_profile(saved_name, replacement_gallery)
                        except ValueError as error:
                            print(f"Registration failed: {error}")
                        else:
                            target_gallery = replacement_gallery
                            target_name = saved_name
                            target_locked = False
                            lock_streak = 0
                            loss_streak = 0
                            action = "updated" if replaced else "registered"
                            print(f"Target '{target_name}' {action} with {len(target_gallery)} gallery samples!")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            motors.cleanup()
        except:
            pass
        try:
            cam.stop()
        except:
            pass
        try:
            cv2.destroyAllWindows()
        except:
            pass


if __name__ == "__main__":
    main()
