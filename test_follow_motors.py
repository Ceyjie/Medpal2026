#!/usr/bin/env python3
"""
test_follow_motors.py -- follow motor test.

Depth reading matches test_camera.py exactly:
    - Ask for Y16 @ 320x240 @ 15fps first, same call test_camera.py makes
    - Raw uint16 buffer, no filtering, no floor, no zeroing
    - Sample a 10x10 ROI at the body box center, median of valid pixels

If no Y16 profile is exposed by the SDK (USB2 link, firmware, etc.)
this script prints every depth profile it can see and falls back to
the SDK default with a loud warning rather than silently refusing.

Zones (from config):
    dist < REVERSE_DISTANCE_MM   -> backward
    REVERSE .. FOLLOW_MIN_MM     -> hold
    FOLLOW_MIN .. FOLLOW_MAX_MM  -> forward, ramped speed
    >= FOLLOW_MAX_MM             -> forward, max speed

Steering: pixel deadband from config.DEADBAND_PX.

Usage:
    python3 test_follow_motors.py
    python3 test_follow_motors.py --click
    python3 test_follow_motors.py --face
    python3 test_follow_motors.py --no-motors
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

PYCORAL_AVAILABLE = False
try:
    from pycoral.adapters import common as pycoral_common, detect as pycoral_detect
    from pycoral.utils.edgetpu import make_interpreter as pycoral_make_interpreter
    PYCORAL_AVAILABLE = True
except ImportError:
    pass

CORAL_EDGETPU = False
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    try:
        load_delegate('libedgetpu.so.1')
        CORAL_EDGETPU = True
    except Exception:
        pass
except ImportError:
    pass

import config
from serial_motors import SerialMotors


# ============================================================
# Depth profile chooser -- match test_camera.py, always return something
# ============================================================
def _choose_depth_profile(profiles):
    """
    Return (video_profile, (w, h, fps), fmt) or (None, None, None).

    Strategy:
      1. Ask for exactly what test_camera.py asks for:
             get_video_stream_profile(320, 240, OBFormat.Y16, 15)
         and a few common Y16 variations.
      2. Otherwise enumerate every profile the SDK exposes, print it
         for diagnostics, and prefer Y16 if present.
      3. Otherwise fall back to get_default_video_stream_profile()
         with a loud warning -- better a wrong-scale reading than no
         reading at all, but we make the problem visible.
    """
    # 0) Diagnostic -- prove OBFormat.Y16 is resolvable
    y16_const = None
    try:
        y16_const = OBFormat.Y16
        print(f"[camera] OBFormat.Y16 = {y16_const!r}")
    except Exception as e:
        print(f"[camera] OBFormat.Y16 lookup failed: {e}")

    # 1) Direct request, exactly like test_camera.py
    if y16_const is not None:
        for w, h, fps in [(320, 240, 15), (320, 240, 30),
                          (640, 480, 15), (640, 480, 30),
                          (160, 120, 30), (160, 120, 15)]:
            try:
                vp = profiles.get_video_stream_profile(
                    w, h, y16_const, fps)
                print(f"[camera] direct Y16 hit: {w}x{h}@{fps}")
                return vp, (w, h, fps), y16_const
            except Exception:
                continue

    # 2) Enumerate what the SDK actually exposes
    try:
        n = profiles.get_count()
    except Exception as e:
        print(f"[camera] cannot enumerate profiles: {e}")
        n = 0

    print(f"[camera] no direct Y16 match; enumerating {n} "
          f"depth profile(s):")
    y16_candidate = None
    any_candidate = None
    for i in range(n):
        pr = None
        vp = None
        try:
            pr = profiles.get_stream_profile_by_index(i)
        except Exception as e:
            print(f"[camera]   #{i}: get_stream_profile_by_index "
                  f"failed: {e}")
            continue
        try:
            vp = pr.as_video_stream_profile()
        except Exception as e:
            print(f"[camera]   #{i}: not a video profile ({e})")
            continue

        fmt = None
        try:
            fmt = vp.get_format()
        except Exception:
            try:
                fmt = pr.get_format()
            except Exception:
                fmt = None

        try:
            w, h = vp.get_width(), vp.get_height()
            fps = vp.get_fps()
        except Exception:
            w, h, fps = 0, 0, 0

        print(f"[camera]   #{i}: {w}x{h}@{fps} fmt={fmt}")

        is_y16 = False
        if y16_const is not None and fmt is not None:
            try:
                is_y16 = (fmt == y16_const)
            except Exception:
                is_y16 = False

        if is_y16 and y16_candidate is None:
            y16_candidate = (vp, (w, h, fps), fmt)
        if any_candidate is None:
            any_candidate = (vp, (w, h, fps), fmt)

    if y16_candidate is not None:
        print(f"[camera] picked Y16 from enumeration: "
              f"{y16_candidate[1]}")
        return y16_candidate

    if any_candidate is not None:
        print(f"[camera] WARNING: no Y16 profile available; falling "
              f"back to {any_candidate[1]} fmt={any_candidate[2]} -- "
              f"if distances look wrong, this is why")
        return any_candidate

    # 3) SDK default
    try:
        vp = profiles.get_default_video_stream_profile()
        try:
            vp = vp.as_video_stream_profile()
        except Exception:
            pass
        try:
            fmt = vp.get_format()
        except Exception:
            fmt = None
        w, h = vp.get_width(), vp.get_height()
        fps = vp.get_fps()
        print(f"[camera] WARNING: using SDK default depth profile "
              f"{w}x{h}@{fps} fmt={fmt}")
        return vp, (w, h, fps), fmt
    except Exception as e:
        print(f"[camera] no default depth profile: {e}")
        return None, None, None


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

        self.color_thread = threading.Thread(
            target=self._color_capture, daemon=True)
        self.color_thread.start()

        self.use_orbbec_depth = False
        self.depth_format = None
        if USE_ORBBEC_DEPTH:
            try:
                self.depth_pipeline = Pipeline()
                self.depth_config = Config()
                profiles = self.depth_pipeline.get_stream_profile_list(
                    OBSensorType.DEPTH_SENSOR)

                self.depth_profile, geom, fmt = _choose_depth_profile(
                    profiles)

                if self.depth_profile is None:
                    raise RuntimeError(
                        "no usable depth profile from SDK")

                self.depth_format = fmt
                print(f"[camera] depth: {geom[0]}x{geom[1]}"
                      f"@{geom[2]} fmt={fmt}")
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
        # Same as test_camera.py: read the SDK buffer as uint16 mm.
        # If the profile isn't Y16 the scale will be wrong; warn once.
        warned_fmt = False
        while self.depth_running:
            try:
                frames = self.depth_pipeline.wait_for_frames(100)
                if not frames:
                    continue
                df = frames.get_depth_frame()
                if not df:
                    continue

                if not warned_fmt:
                    try:
                        f = df.get_format()
                    except Exception:
                        f = None
                    try:
                        is_y16 = (f == OBFormat.Y16)
                    except Exception:
                        is_y16 = False
                    if f is not None and not is_y16:
                        print(f"[camera] WARNING: depth frame format "
                              f"is {f}, not Y16 -- distances may be "
                              f"scaled wrong")
                    warned_fmt = True

                w, h = df.get_width(), df.get_height()
                data = np.frombuffer(df.get_data(), dtype=np.uint16)
                if data.size >= h * w:
                    self.depth_frame = data[: h * w].reshape(
                        h, w).astype(np.float32)
            except Exception:
                time.sleep(0.05)

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
# Coral detectors
# ============================================================
class _CoralDetectorBase:
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
                return
            except Exception as e:
                print(f"[detector] PyCoral failed: {e}")

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
            self.engine, score_threshold=self.conf_thres)
        results = []
        for obj in objs:
            if not self._accept_class(obj.id):
                continue
            bbox = obj.bbox
            results.append((
                int(bbox.xmin * sx), int(bbox.ymin * sy),
                int((bbox.xmax - bbox.xmin) * sx),
                int((bbox.ymax - bbox.ymin) * sy),
                float(obj.score)))
        return results

    def _infer_tflite(self, frame):
        h, w = frame.shape[:2]
        th, tw = self.input_shape[1], self.input_shape[2]
        resized = cv2.resize(frame, (tw, th))
        data = np.expand_dims(resized, axis=0).astype(np.uint8)
        self.interpreter.set_tensor(self.input_details[0]['index'], data)
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
            if not self._accept_class(cls_id):
                continue
            y1, x1, y2, x2 = boxes[0][i]
            results.append((
                int(x1 * w), int(y1 * h),
                int((x2 - x1) * w), int((y2 - y1) * h),
                score))
        return results

    def _accept_class(self, cls_id):
        return True


class FaceDetector(_CoralDetectorBase):
    def __init__(self, model_path, label_path=None):
        self.conf_thres = config.CONF_THRES
        super().__init__(model_path, label_path)


class BodyDetector(_CoralDetectorBase):
    def __init__(self, model_path, label_path=None):
        self.conf_thres = getattr(config, "BODY_CONF_THRES", 0.60)
        super().__init__(model_path, label_path)
        self.PERSON_CLASS = 0
        for i, name in enumerate(self.labels):
            if name.strip().lower() == "person":
                self.PERSON_CLASS = i
                break

    def _accept_class(self, cls_id):
        return int(cls_id) == self.PERSON_CLASS


# ============================================================
# Depth sampling -- same 10x10 ROI as test_camera.py
# ============================================================
def sample_depth(depth_mm, box, frame_w, frame_h):
    """
    Sample depth at the body box center.

    Uses the same 10x10 ROI and same median-of-valid approach as
    test_camera.py. If the ROI has no valid pixels, falls back to
    the raw center pixel value (no filtering).
    """
    if depth_mm is None or box is None:
        return None
    x, y, w, h = box
    dh, dw = depth_mm.shape[:2]
    sx = dw / frame_w
    sy = dh / frame_h
    cx = x + w // 2
    cy = y + h // 2
    dcx = int(cx * sx)
    dcy = int(cy * sy)
    if not (0 <= dcx < dw and 0 <= dcy < dh):
        return None
    y1 = max(0, dcy - 5)
    y2 = min(dh, dcy + 5)
    x1 = max(0, dcx - 5)
    x2 = min(dw, dcx + 5)
    roi = depth_mm[y1:y2, x1:x2]
    valid = roi[roi > 0]
    if valid.size > 0:
        return float(np.median(valid))
    return float(depth_mm[dcy, dcx])


def depth_debug(depth_mm, box, frame_w, frame_h):
    if depth_mm is None:
        return "depth frame: NONE"
    if box is None:
        v = depth_mm[depth_mm > 0]
        return (f"frame valid={v.size}/{depth_mm.size}"
                + (f" min={v.min():.0f} max={v.max():.0f}"
                   if v.size else ""))
    x, y, w, h = box
    dh, dw = depth_mm.shape[:2]
    sx, sy = dw / frame_w, dh / frame_h
    dcx = int((x + w // 2) * sx)
    dcy = int((y + h // 2) * sy)
    if not (0 <= dcx < dw and 0 <= dcy < dh):
        return f"center ({dcx},{dcy}) outside depth frame"
    r = depth_mm[max(0, dcy - 5):dcy + 5, max(0, dcx - 5):dcx + 5]
    v = r[r > 0]
    return (f"center ({dcx},{dcy}) ROI valid={v.size}/{r.size}"
            + (f" min={v.min():.0f} max={v.max():.0f}"
               if v.size else
               f" raw_center={depth_mm[dcy, dcx]:.0f}"))


# ============================================================
# Follow decision
# ============================================================
def follow_decision(person_cx, frame_w, target_distance_mm):
    reverse_mm    = getattr(config, "REVERSE_DISTANCE_MM", 65)
    follow_min_mm = getattr(config, "FOLLOW_MIN_MM", 95)
    follow_max_mm = getattr(config, "FOLLOW_MAX_MM", 900)
    reverse_speed = getattr(config, "REVERSE_SPEED", 30)
    follow_base   = getattr(config, "FOLLOW_BASE_SPEED", 40)
    max_speed     = getattr(config, "MAX_SPEED", 100)
    speed_inc     = getattr(config, "SPEED_INCREASE", 10)
    deadband_px   = getattr(config, "DEADBAND_PX", 40)
    turn_speed    = getattr(config, "TURN_ONLY_SPEED", 40)

    frame_cx = frame_w // 2

    if target_distance_mm is None:
        if person_cx is None:
            return ("stop", 0, "no_target")
        if person_cx < frame_cx - deadband_px:
            return ("turn_left", turn_speed, "turn_L (no depth)")
        if person_cx > frame_cx + deadband_px:
            return ("turn_right", turn_speed, "turn_R (no depth)")
        return ("stop", 0, "no_depth")

    if target_distance_mm < reverse_mm:
        if person_cx is None:
            return ("backward", reverse_speed,
                    f"reverse ({target_distance_mm:.0f}mm)")
        if person_cx < frame_cx - deadband_px:
            return ("turn_left", turn_speed,
                    f"reverse+turn_L ({target_distance_mm:.0f}mm)")
        if person_cx > frame_cx + deadband_px:
            return ("turn_right", turn_speed,
                    f"reverse+turn_R ({target_distance_mm:.0f}mm)")
        return ("backward", reverse_speed,
                f"reverse ({target_distance_mm:.0f}mm)")

    if target_distance_mm < follow_min_mm:
        if person_cx is None:
            return ("stop", 0, f"hold ({target_distance_mm:.0f}mm)")
        if person_cx < frame_cx - deadband_px:
            return ("turn_left", turn_speed,
                    f"hold+turn_L ({target_distance_mm:.0f}mm)")
        if person_cx > frame_cx + deadband_px:
            return ("turn_right", turn_speed,
                    f"hold+turn_R ({target_distance_mm:.0f}mm)")
        return ("stop", 0, f"hold ({target_distance_mm:.0f}mm)")

    extra = int((target_distance_mm - follow_min_mm) // 100) * speed_inc
    speed = min(follow_base + extra, max_speed)

    if person_cx is None:
        return ("forward", speed,
                f"fwd {speed}% ({target_distance_mm:.0f}mm)")
    if person_cx < frame_cx - deadband_px:
        return ("turn_left", turn_speed,
                f"turn_L ({target_distance_mm:.0f}mm)")
    if person_cx > frame_cx + deadband_px:
        return ("turn_right", turn_speed,
                f"turn_R ({target_distance_mm:.0f}mm)")
    return ("forward", speed,
            f"fwd {speed}% ({target_distance_mm:.0f}mm)")


def send_action(motors, action, speed_pct):
    if motors is None or not motors.available:
        return
    if action == "stop":
        motors.stop()
    elif action == "forward":
        motors.set_speed(speed_pct); motors.forward()
    elif action == "backward":
        motors.set_speed(speed_pct); motors.backward()
    elif action == "turn_left":
        motors.set_speed(speed_pct); motors.turn_left()
    elif action == "turn_right":
        motors.set_speed(speed_pct); motors.turn_right()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--click", action="store_true")
    parser.add_argument("--face", action="store_true")
    parser.add_argument("--no-motors", action="store_true")
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("test_follow_motors.py -- test_camera.py depth reading")
    print("=" * 60)
    print(f"  reverse < {config.REVERSE_DISTANCE_MM}mm @ "
          f"{config.REVERSE_SPEED}%")
    print(f"  hold    {config.REVERSE_DISTANCE_MM} - "
          f"{config.FOLLOW_MIN_MM}mm")
    print(f"  follow  {config.FOLLOW_MIN_MM} - "
          f"{config.FOLLOW_MAX_MM}mm")
    print(f"  far     >= {config.FOLLOW_MAX_MM}mm")
    print(f"  deadband: ±{config.DEADBAND_PX}px  "
          f"turn speed: {config.TURN_ONLY_SPEED}%")
    print("  depth: raw uint16, 10x10 ROI at target center, "
          "no filtering")
    print()

    cam = Camera()
    for _ in range(60):
        if cam.read_color() is not None:
            break
        time.sleep(0.05)

    detector = None
    detector_kind = "none"
    if not args.click:
        if not (PYCORAL_AVAILABLE or CORAL_EDGETPU):
            print("[init] no Coral runtime -- forcing click mode")
            args.click = True
        elif args.face:
            detector = FaceDetector(config.CORAL_FACE_DETECTION_MODEL,
                                     config.CORAL_FACE_LABELS)
            detector_kind = "face"
            print("[detector] face model loaded")
        else:
            detector = BodyDetector(config.CORAL_DETECTION_MODEL,
                                     config.CORAL_LABELS)
            detector_kind = "body"
            print("[detector] body model loaded")

    motors = None
    if not args.no_motors:
        motors = SerialMotors()
        if not motors.available:
            print("[motors] ESP32 not available -- DRY")
            motors = None

    click_state = {"px": None, "py": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_state["px"] = x
            click_state["py"] = y
            print(f"[click] target at ({x},{y})")

    win = "follow test"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if args.click:
        cv2.setMouseCallback(win, on_mouse)

    print("Controls: q=quit, space=toggle dry-run, c=clear click")

    start = time.monotonic()
    frame_count = 0
    last_print = 0.0
    dry = args.no_motors

    try:
        while True:
            if args.duration and time.monotonic() - start >= args.duration:
                break

            frame = cam.read_color()
            if frame is None:
                time.sleep(0.01)
                continue
            frame_count += 1
            depth = cam.read_depth()
            disp = frame.copy()

            target_box = None
            src = "none"
            if args.click:
                if click_state["px"] is not None:
                    px, py = click_state["px"], click_state["py"]
                    bw, bh = 40, 60
                    target_box = (max(0, px - bw // 2),
                                  max(0, py - bh // 2),
                                  bw, bh)
                    src = "click"
            else:
                dets = detector.infer(frame)
                if detector_kind == "body":
                    dets = [d for d in dets if d[2] >= 40 and d[3] >= 80]
                if dets:
                    best = max(dets, key=lambda d: d[2] * d[3])
                    target_box = (best[0], best[1], best[2], best[3])
                    src = f"{detector_kind} {best[4]:.2f}"

            person_cx = None
            target_distance_mm = None
            if target_box is not None:
                x, y, w, h = target_box
                person_cx = x + w // 2
                target_distance_mm = sample_depth(
                    depth, target_box, config.FRAME_W, config.FRAME_H)

            action, speed_pct, note = follow_decision(
                person_cx, config.FRAME_W, target_distance_mm)

            if not dry:
                send_action(motors, action, speed_pct)

            # Overlay
            frame_cx = config.FRAME_W // 2
            cv2.line(disp, (frame_cx, 0), (frame_cx, config.FRAME_H),
                     (100, 100, 100), 1)
            cv2.line(disp, (frame_cx - config.DEADBAND_PX, 0),
                     (frame_cx - config.DEADBAND_PX, config.FRAME_H),
                     (80, 80, 80), 1)
            cv2.line(disp, (frame_cx + config.DEADBAND_PX, 0),
                     (frame_cx + config.DEADBAND_PX, config.FRAME_H),
                     (80, 80, 80), 1)

            if target_box is not None:
                x, y, w, h = target_box
                c = (0, 255, 0) if target_distance_mm is not None \
                    else (0, 165, 255)
                cv2.rectangle(disp, (x, y), (x + w, y + h), c, 2)
                cv2.putText(disp, src, (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

                # Crosshair at the exact sample point (body center)
                if depth is not None:
                    dh, dw = depth.shape[:2]
                    sx = dw / config.FRAME_W
                    sy = dh / config.FRAME_H
                    dcx = int((x + w // 2) * sx)
                    dcy = int((y + h // 2) * sy)
                    ccx = int(dcx / sx)
                    ccy = int(dcy / sy)
                    cv2.drawMarker(disp, (ccx, ccy),
                                   (0, 255, 255),
                                   cv2.MARKER_CROSS, 14, 2)
                    cv2.circle(disp, (ccx, ccy), 12,
                               (0, 255, 255), 1)

            dist_str = (f"{target_distance_mm:.0f} mm"
                        if target_distance_mm is not None else "no depth")
            cv2.putText(disp, f"dist: {dist_str}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            action_color = {
                "forward": (0, 255, 0),
                "backward": (0, 0, 255),
                "turn_left": (0, 200, 255),
                "turn_right": (0, 200, 255),
                "stop": (128, 128, 128),
            }.get(action, (200, 200, 200))
            dry_tag = " [DRY]" if dry else ""
            cv2.putText(disp,
                        f"{action} @ {speed_pct}%  ({note}){dry_tag}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, action_color, 2)

            cv2.putText(disp,
                        f"motors: "
                        f"{'connected' if (motors and motors.available) else 'DRY'}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 200, 200), 1)

            now = time.monotonic()
            if now - last_print >= 0.5:
                cx_str = f"{person_cx:>4}" \
                    if person_cx is not None else " -  "
                d_str = f"{target_distance_mm:>5.0f}" \
                    if target_distance_mm is not None else "    -"
                print(f"[{frame_count:5d}] cx={cx_str}  dist={d_str}  "
                      f"-> {action:11s} {speed_pct:>3d}%  ({note})"
                      f"{' [DRY]' if dry else ''}")
                last_print = now

            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                dry = not dry
                print(f"[mode] {'DRY' if dry else 'live'}")
                if dry and motors:
                    motors.stop()
            elif key == ord('c'):
                click_state["px"] = None
                click_state["py"] = None
                print("[click] target cleared")
                if motors:
                    motors.stop()

    except KeyboardInterrupt:
        pass
    finally:
        if motors is not None:
            try:
                motors.stop()
            except Exception:
                pass
            try:
                motors.cleanup()
            except Exception:
                pass
        cam.stop()
        cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print(f"frames: {frame_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
