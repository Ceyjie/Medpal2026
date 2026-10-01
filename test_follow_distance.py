#!/usr/bin/env python3
"""
test_follow_distance.py -- verify follow-distance thresholds.

Watches the camera, detects faces (Coral) and bodies (Coral), computes
the depth at the target's center, classifies the reading into a follow
zone, and prints what motor command the tracker *would* send.

No motors are driven. No classifier is needed. Just depth + detection.

Usage:
    python3 test_follow_distance.py
    python3 test_follow_distance.py --use-body    # use body center instead of face
    python3 test_follow_distance.py --depth-only  # ignore detection, sample frame center

Controls:
    q = quit
    p = print a snapshot of the current reading to console
"""

import os
import sys
import time
import argparse
import threading

import cv2
import numpy as np

sys.path.append('/home/medpal/pyorbbecsdk_v1/build')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
    USE_ORBBEC_DEPTH = True
except ImportError:
    USE_ORBBEC_DEPTH = False
    print("[init] pyorbbecsdk not available -- depth disabled")

PYCORAL_AVAILABLE = False
try:
    from pycoral.adapters import common as pycoral_common, detect as pycoral_detect
    from pycoral.utils.edgetpu import make_interpreter as pycoral_make_interpreter
    PYCORAL_AVAILABLE = True
    print("[init] PyCoral available")
except ImportError:
    print("[init] PyCoral not installed")

CORAL_EDGETPU = False
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    try:
        load_delegate('libedgetpu.so.1')
        CORAL_EDGETPU = True
        print("[init] Coral Edge TPU delegate available")
    except Exception as e:
        print(f"[init] Coral delegate unavailable: {e}")
except ImportError:
    print("[init] tflite_runtime not installed")

import config


# ============================================================
# Detection zones -- mirror the tracker's logic
# ============================================================
def classify_zone(distance_mm):
    """
    Return (zone_name, description, would_command).
    Mirrors the motor-control block in tracker_geminiV3_1.py.
    """
    if distance_mm is None:
        return ("NO_DEPTH", "no usable depth at target",
                "STOP")
    if distance_mm < config.REVERSE_DISTANCE_MM:
        return ("REVERSE",
                f"too close ({distance_mm:.0f} < "
                f"{config.REVERSE_DISTANCE_MM})",
                "BACKWARD + steer")
    if distance_mm < config.FOLLOW_MIN_MM:
        return ("HOLD",
                f"in stop zone ({distance_mm:.0f} in "
                f"[{config.REVERSE_DISTANCE_MM}, "
                f"{config.FOLLOW_MIN_MM}])",
                "STOP (turn if off-axis)")
    if distance_mm < config.FOLLOW_MAX_MM:
        # Speed ramp
        span = max(1.0, config.FOLLOW_MAX_MM - config.FOLLOW_MIN_MM)
        frac = (distance_mm - config.FOLLOW_MIN_MM) / span
        speed = int(config.MIN_FOLLOW_SPEED +
                    frac * (config.MAX_FOLLOW_SPEED -
                            config.MIN_FOLLOW_SPEED))
        return ("FOLLOW",
                f"ramp ({distance_mm:.0f} -> {speed}%)",
                f"FORWARD {speed}% (steer if off-axis)")
    return ("FAR",
            f"far ({distance_mm:.0f} >= {config.FOLLOW_MAX_MM})",
            f"FORWARD {config.MAX_FOLLOW_SPEED}% (capped)")


# ============================================================
# Camera
# ============================================================
class Camera:
    def __init__(self):
        self.color_cap = None
        for idx in [1, 0, 2]:
            self.color_cap = cv2.VideoCapture(idx)
            if self.color_cap.isOpened():
                print(f"[camera] color on /dev/video{idx}")
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

        self.color_thread = threading.Thread(
            target=self._color_capture, daemon=True)
        self.color_thread.start()

        if USE_ORBBEC_DEPTH:
            try:
                self.depth_pipeline = Pipeline()
                self.depth_config = Config()
                profiles = self.depth_pipeline.get_stream_profile_list(
                    OBSensorType.DEPTH_SENSOR)
                self.depth_profile = None
                for w, h, fps in [(320, 240, 15), (320, 240, 30),
                                  (640, 480, 15)]:
                    try:
                        self.depth_profile = \
                            profiles.get_video_stream_profile(
                                w, h, OBFormat.Y16, fps)
                        print(f"[camera] depth {w}x{h}@{fps}")
                        break
                    except Exception:
                        continue
                if self.depth_profile is None:
                    self.depth_profile = \
                        profiles.get_default_video_stream_profile()
                    print(f"[camera] depth default "
                          f"({self.depth_profile.get_width()}x"
                          f"{self.depth_profile.get_height()})")
                self.depth_config.enable_stream(self.depth_profile)
                self.depth_pipeline.start(self.depth_config)
                self.depth_running = True
                self.depth_thread = threading.Thread(
                    target=self._depth_capture, daemon=True)
                self.depth_thread.start()
                self.use_orbbec_depth = True
            except Exception as e:
                print(f"[camera] depth unavailable: {e}")

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
                        data = np.frombuffer(
                            df.get_data(),
                            dtype=np.uint16).reshape(h, w)
                        depth_mm = data.astype(np.float32) * scale
                        depth_mm[depth_mm < config.MIN_VALID_DEPTH_MM] = 0
                        self.depth_frame = depth_mm
            except Exception as e:
                print(f"[camera] depth error: {e}")
                time.sleep(1.0)
            time.sleep(0.01)

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
# Face detector (Coral)
# ============================================================
class FaceDetector:
    def __init__(self, model_path):
        self.mode = None
        self.engine = None
        self.interpreter = None

        if PYCORAL_AVAILABLE:
            try:
                self.engine = pycoral_make_interpreter(model_path)
                self.engine.allocate_tensors()
                self.mode = "pycoral"
                print("[face] using PyCoral")
                return
            except Exception as e:
                print(f"[face] PyCoral failed: {e}")

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
        return [(int(o.bbox.xmin * sx), int(o.bbox.ymin * sy),
                 int((o.bbox.xmax - o.bbox.xmin) * sx),
                 int((o.bbox.ymax - o.bbox.ymin) * sy),
                 float(o.score)) for o in objs]

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
# Depth sampling
# ============================================================
def sample_depth(depth_mm, box, frame_w, frame_h,
                 roi_size=10):
    """
    Return median depth in mm at the center of `box` (x, y, w, h),
    or None if no valid readings.
    """
    if depth_mm is None or box is None:
        return None
    x, y, w, h = box
    cx = x + w // 2
    cy = y + h // 2

    dh, dw = depth_mm.shape[:2]
    sx = dw / frame_w
    sy = dh / frame_h

    dcx = int(cx * sx)
    dcy = int(cy * sy)

    if not (0 <= dcx < dw and 0 <= dcy < dh):
        return None

    y1 = max(0, dcy - roi_size)
    y2 = min(dh, dcy + roi_size + 1)
    x1 = max(0, dcx - roi_size)
    x2 = min(dw, dcx + roi_size + 1)

    roi = depth_mm[y1:y2, x1:x2]
    valid = roi[roi > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid))


# ============================================================
# Drawing
# ============================================================
ZONE_COLORS = {
    "REVERSE": (0, 0, 255),      # red
    "HOLD":    (0, 165, 255),    # orange
    "FOLLOW":  (0, 255, 0),      # green
    "FAR":     (255, 200, 0),    # cyan-blue
    "NO_DEPTH": (128, 128, 128), # grey
    "NO_TARGET": (128, 128, 128),
}


def draw_zone_bar(frame, distance_mm):
    """
    Draw a horizontal bar at the top showing the distance zones and
    a marker for where the current reading falls.
    """
    h, w = frame.shape[:2]
    bar_h = 26
    bar_y = 30
    bar_x1 = 10
    bar_x2 = w - 10
    bar_w = bar_x2 - bar_x1

    # Ranges: reverse (0-500), hold (500-500), follow ramp (500-900),
    # far (900-3000). But since HOLD is empty when MIN==REVERSE, we
    # draw zones as: red reverse [0, REVERSE), orange hold
    # [REVERSE, FOLLOW_MIN), green follow [FOLLOW_MIN, FOLLOW_MAX),
    # cyan far [FOLLOW_MAX, 2000].
    max_mm = 2000.0
    def mm_to_x(mm):
        return int(bar_x1 + bar_w * min(1.0, mm / max_mm))

    # Zone fills
    zones = [
        (0, config.REVERSE_DISTANCE_MM,
         (0, 0, 180)),                                       # reverse
        (config.REVERSE_DISTANCE_MM, config.FOLLOW_MIN_MM,
         (0, 120, 200)),                                     # hold
        (config.FOLLOW_MIN_MM, config.FOLLOW_MAX_MM,
         (0, 140, 0)),                                       # follow
        (config.FOLLOW_MAX_MM, max_mm,
         (140, 90, 0)),                                      # far
    ]
    for lo, hi, color in zones:
        x1 = mm_to_x(lo)
        x2 = mm_to_x(hi)
        cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_h),
                      color, -1)

    # Border
    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h),
                  (200, 200, 200), 1)

    # Zone labels
    labels = [
        (config.REVERSE_DISTANCE_MM // 2, "REV"),
        ((config.REVERSE_DISTANCE_MM + config.FOLLOW_MIN_MM) // 2, "HOLD"),
        ((config.FOLLOW_MIN_MM + config.FOLLOW_MAX_MM) // 2, "FOLLOW"),
        ((config.FOLLOW_MAX_MM + int(max_mm)) // 2, "FAR"),
    ]
    for mm, label in labels:
        x = mm_to_x(mm) - 18
        if x < bar_x1:
            x = bar_x1 + 2
        cv2.putText(frame, label, (x, bar_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)

    # Marker for current reading
    if distance_mm is not None:
        mx = mm_to_x(distance_mm)
        cv2.line(frame, (mx, bar_y - 4), (mx, bar_y + bar_h + 4),
                 (0, 255, 255), 2)
        cv2.putText(frame, f"{distance_mm:.0f}",
                    (mx - 20, bar_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-body", action="store_true",
                        help="Use body box center instead of face")
    parser.add_argument("--depth-only", action="store_true",
                        help="Ignore detection; sample frame center only")
    parser.add_argument("--use-max", action="store_true",
                        help="Use median of largest box (default: "
                             "median of largest face)")
    args = parser.parse_args()

    print("=" * 60)
    print("test_follow_distance.py")
    print("=" * 60)
    print(f"  face detector   : Coral"
          if PYCORAL_AVAILABLE or CORAL_EDGETPU
          else f"  face detector   : unavailable")
    print(f"  zones           :")
    print(f"    REVERSE  < {config.REVERSE_DISTANCE_MM} mm")
    print(f"    HOLD     {config.REVERSE_DISTANCE_MM} - "
          f"{config.FOLLOW_MIN_MM} mm")
    print(f"    FOLLOW   {config.FOLLOW_MIN_MM} - "
          f"{config.FOLLOW_MAX_MM} mm  "
          f"(speed {config.MIN_FOLLOW_SPEED}-"
          f"{config.MAX_FOLLOW_SPEED}%)")
    print(f"    FAR      >= {config.FOLLOW_MAX_MM} mm")
    print("=" * 60)
    print()

    cam = Camera()
    for _ in range(60):
        if cam.read_color() is not None:
            break
        time.sleep(0.05)

    face_detector = None
    if not args.depth_only:
        if PYCORAL_AVAILABLE or CORAL_EDGETPU:
            face_detector = FaceDetector(
                config.CORAL_FACE_DETECTION_MODEL)
        else:
            print("[main] no Coral runtime -- forcing depth-only mode")
            args.depth_only = True

    cv2.namedWindow("Follow Distance Test", cv2.WINDOW_NORMAL)

    frame_count = 0
    last_zone_print_t = 0.0
    last_zone = None

    try:
        while True:
            frame = cam.read_color()
            if frame is None:
                time.sleep(0.01)
                continue
            frame_count += 1
            depth = cam.read_depth()
            disp = frame.copy()

            distance_mm = None
            target_box = None

            if args.depth_only:
                # Just sample the frame center
                target_box = (config.FRAME_W // 2 - 30,
                              config.FRAME_H // 2 - 30,
                              60, 60)
                distance_mm = sample_depth(
                    depth, target_box,
                    config.FRAME_W, config.FRAME_H)
                cv2.rectangle(disp,
                              (target_box[0], target_box[1]),
                              (target_box[0] + target_box[2],
                               target_box[1] + target_box[3]),
                              (200, 200, 200), 1)
                cv2.putText(disp, "frame center",
                            (target_box[0], target_box[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (200, 200, 200), 1)
            else:
                faces = face_detector.infer(frame)
                if faces:
                    best = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h, conf = best
                    target_box = (x, y, w, h)
                    distance_mm = sample_depth(
                        depth, target_box,
                        config.FRAME_W, config.FRAME_H)
                    cv2.rectangle(disp, (x, y), (x + w, y + h),
                                  (0, 255, 255), 2)
                    cv2.putText(disp, f"face {conf:.2f}",
                                (x, y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 255), 1)

            # Classify and show
            if distance_mm is None:
                zone, desc, cmd = classify_zone(None)
            else:
                zone, desc, cmd = classify_zone(distance_mm)

            color = ZONE_COLORS.get(zone, (200, 200, 200))

            # Big zone label
            cv2.putText(disp, f"{zone}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(disp, desc, (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            cv2.putText(disp, f"would: {cmd}", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

            # Zone bar at top
            draw_zone_bar(disp, distance_mm)

            # Console print on zone change or every 1 s
            now = time.monotonic()
            if zone != last_zone or now - last_zone_print_t > 1.0:
                print(f"[{frame_count:5d}] "
                      f"dist={'-' if distance_mm is None else f'{distance_mm:.0f}':>5} mm  "
                      f"zone={zone:9s}  "
                      f"would={cmd}")
                last_zone = zone
                last_zone_print_t = now

            cv2.imshow("Follow Distance Test", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                print()
                print(f"--- snapshot at frame {frame_count} ---")
                print(f"  distance_mm : {distance_mm}")
                print(f"  zone        : {zone}")
                print(f"  description : {desc}")
                print(f"  would_command: {cmd}")
                if target_box is not None:
                    print(f"  target_box  : {target_box}")
                if depth is not None:
                    print(f"  depth shape : {depth.shape}")
                    print(f"  depth range : "
                          f"{depth.min():.0f} - {depth.max():.0f} mm")
                print()

    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print(f"Frames processed: {frame_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()