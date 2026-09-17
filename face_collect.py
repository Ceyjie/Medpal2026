#!/usr/bin/env python3
"""
face_collect.py -- Standalone face detection + auto-crop for training.

Detects faces with the Coral TPU, crops them with quality filters, and
saves to FACE_TRAINING_DIR/<name>/. No tracker, no body ReID, no motors --
just clean face collection.

Usage:
    # Continuous collect, 100 crops, saved under "Carl"
    python3 face_collect.py --name Carl --samples 100

    # One-shot: collect for 10 seconds then exit
    python3 face_collect.py --name Carl --duration 10

    # Show detected faces without saving (preview mode)
    python3 face_collect.py --preview

    # Save into a custom directory (not under face_training)
    python3 face_collect.py --name Carl --output /tmp/carl_faces

Runtime controls:
    s = start / stop collection
    q = quit

Quality filters (all must pass to save):
  - Minimum face size  : ENROLL_MIN_FACE_PX
  - Minimum sharpness  : ENROLL_BLUR_THRES (Laplacian variance)
  - Diversity          : ENROLL_DIVERSITY_PX vs the last saved crop
  - Optional distance  : FACE_RECOGNIZE_MAX_MM (set 0 to disable)
  - Save rate limit    : MIN_SAVE_INTERVAL_S (default 0.15 s)
"""

import os
import sys
import cv2
import numpy as np
import argparse
import time
import threading

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

import config


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
            self.depth_frame = None
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
            except Exception:
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
# Face Detector (Coral SSD Face)
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
# Quality helpers
# ============================================================
def evaluate_face(face_crop, last_saved_gray,
                  min_px, blur_thres, diversity_px):
    """
    Return (ok, reasons, blur_var, gray_small).
      ok:              True if the crop passes all filters
      reasons:         list of strings describing why it failed
      blur_var:        Laplacian variance (sharpness score)
      gray_small:      64x64 grayscale for the diversity check
    """
    reasons = []
    h, w = face_crop.shape[:2]

    if w < min_px or h < min_px:
        reasons.append(f"too small ({w}x{h})")

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_var < blur_thres:
        reasons.append(f"blurry ({blur_var:.0f})")

    gray_small = cv2.resize(gray, (64, 64))
    if last_saved_gray is not None:
        try:
            diff = float(np.mean(np.abs(
                gray_small.astype(np.float32) - last_saved_gray.astype(np.float32)
            )))
            if diff < diversity_px:
                reasons.append("too similar to last")
        except Exception:
            pass

    return (len(reasons) == 0, reasons, blur_var, gray_small)


def face_depth_mm(depth, fx, fy, fw, fh):
    """Return the median depth (mm) at the face's center, or None."""
    if depth is None:
        return None
    dh, dw = depth.shape[:2]
    sx = dw / config.FRAME_W
    sy = dh / config.FRAME_H
    dcx = int((fx + fw // 2) * sx)
    dcy = int((fy + fh // 2) * sy)
    if not (0 <= dcx < dw and 0 <= dcy < dh):
        return None
    y1 = max(0, dcy - 5); y2 = min(dh, dcy + 5)
    x1 = max(0, dcx - 5); x2 = min(dw, dcx + 5)
    roi = depth[y1:y2, x1:x2]
    valid = roi[roi > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid))


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Person name (folder for saved crops)")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of crops to collect (default 100)")
    parser.add_argument("--output", default=None,
                        help="Custom output dir (default: FACE_TRAINING_DIR/<name>)")
    parser.add_argument("--preview", action="store_true",
                        help="Only show detected faces, save nothing")
    parser.add_argument("--duration", type=float, default=None,
                        help="Auto-stop after N seconds (overrides --samples)")
    parser.add_argument("--max-distance", type=float, default=None,
                        help="Only save faces closer than this (mm). "
                             "0 disables the distance check.")
    parser.add_argument("--save-interval", type=float, default=0.15,
                        help="Minimum seconds between saves (default 0.15)")
    args = parser.parse_args()

    if not args.preview and not args.name:
        print("ERROR: --name is required unless --preview is used.")
        return

    # Resolve output directory
    if args.preview:
        out_dir = None
    elif args.output:
        out_dir = args.output
    else:
        out_dir = os.path.join(config.FACE_TRAINING_DIR, args.name)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Distance filter
    if args.max_distance is None:
        max_dist = getattr(config, "FACE_RECOGNIZE_MAX_MM", 0)
    else:
        max_dist = args.max_distance
    if max_dist <= 0:
        max_dist = None

    min_px = getattr(config, "ENROLL_MIN_FACE_PX", 50)
    blur_thres = getattr(config, "ENROLL_BLUR_THRES", 40.0)
    diversity_px = getattr(config, "ENROLL_DIVERSITY_PX", 25)

    print()
    print("=" * 60)
    print("face_collect.py -- face detection + auto-crop for training")
    print("=" * 60)
    if args.preview:
        print("MODE:   PREVIEW (nothing saved)")
    else:
        print(f"NAME:   {args.name}")
        print(f"OUTPUT: {out_dir}")
        print(f"TARGET: {args.samples} crops"
              + (f", or {args.duration:.1f}s" if args.duration else ""))
    print(f"FILTERS:")
    print(f"  min size  : {min_px} px")
    print(f"  min blur  : {blur_thres:.0f} (Laplacian var)")
    print(f"  diversity : {diversity_px} vs last crop")
    print(f"  max dist  : {max_dist if max_dist else 'none'}")
    print(f"  save gap  : {args.save_interval:.2f}s")
    print("=" * 60)
    print()

    # ---- Init ----
    cam = Camera()
    detector = FaceDetector(config.CORAL_FACE_DETECTION_MODEL)

    cv2.namedWindow("Face Collect", cv2.WINDOW_NORMAL)

    collected = 0
    last_saved_gray = None
    last_save_t = 0.0
    start_time = time.monotonic()
    collecting = True       # auto-start
    last_status = ""
    last_status_color = (255, 255, 255)
    skip_streak = 0
    no_face_frames = 0

    print("Auto-started. Press 's' to pause / resume, 'q' to quit.")

    try:
        while True:
            frame_start = time.monotonic()
            frame = cam.read_color()
            if frame is None:
                time.sleep(0.01)
                continue
            depth = cam.read_depth()

            faces = detector.infer(frame)

            now = time.monotonic()

            # Stop conditions
            if args.duration is not None:
                if now - start_time >= args.duration:
                    print(f"\nReached --duration ({args.duration:.1f}s). Stopping.")
                    break
            if not args.preview and collected >= args.samples:
                print(f"\nCollected {collected} crops. Stopping.")
                break

            display = frame.copy()

            if not faces:
                no_face_frames += 1
                status_text = "No face detected"
                status_color = (0, 0, 255)
                if no_face_frames > 30:
                    status_text = "No face detected - move closer / fix lighting"
            else:
                no_face_frames = 0

                # Sort by area, biggest face first
                faces_sorted = sorted(faces, key=lambda f: f[2] * f[3],
                                      reverse=True)

                # Only consider the biggest face for saving
                best = faces_sorted[0]
                bx, by, bw, bh, bconf = best
                bx = max(0, bx); by = max(0, by)
                bx2 = min(frame.shape[1], bx + bw)
                by2 = min(frame.shape[0], by + bh)
                face_crop = frame[by:by2, bx:bx2]

                if face_crop.size == 0:
                    status_text = "Empty crop"
                    status_color = (0, 0, 255)
                else:
                    # Distance check
                    fd = face_depth_mm(depth, bx, by, bx2 - bx, by2 - by)
                    dist_ok = (max_dist is None) or (fd is not None and fd <= max_dist)

                    ok, reasons, blur_var, gray_small = evaluate_face(
                        face_crop, last_saved_gray,
                        min_px=min_px,
                        blur_thres=blur_thres,
                        diversity_px=diversity_px,
                    )

                    if not dist_ok:
                        d_str = f"{fd:.0f}mm" if fd is not None else "no depth"
                        reasons.insert(0, f"too far ({d_str})")

                    can_save = ok and dist_ok

                    # Draw the box in a color that reflects the state
                    if can_save:
                        box_color = (0, 255, 0)     # green: ready
                        status_color = (0, 255, 0)
                        status_text = f"Ready ({blur_var:.0f})"
                    else:
                        box_color = (0, 200, 255)   # amber: skipped
                        status_color = (0, 200, 255)
                        status_text = "Skipped: " + ", ".join(reasons) if reasons else "Skipped"

                    cv2.rectangle(display, (bx, by), (bx2, by2), box_color, 2)
                    if fd is not None:
                        cv2.putText(display, f"{fd:.0f}mm",
                                    (bx, by2 + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

                    # Draw secondary faces dim
                    for (sx, sy_, sw, sh, sc) in faces_sorted[1:]:
                        cv2.rectangle(display, (sx, sy_), (sx + sw, sy_ + sh),
                                      (128, 128, 128), 1)

                    # Save
                    save_interval_ok = (now - last_save_t) >= args.save_interval
                    if (not args.preview
                            and collecting
                            and can_save
                            and save_interval_ok):
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        ms = int(now * 1000) % 1000
                        fname = f"capture_{ts}_{ms:03d}_{collected + 1:04d}.jpg"
                        fpath = os.path.join(out_dir, fname)
                        if cv2.imwrite(fpath, face_crop):
                            collected += 1
                            last_saved_gray = gray_small
                            last_save_t = now
                            skip_streak = 0
                            print(f"  Saved {collected}/{args.samples}  "
                                  f"[{fname}  size {bw}x{bh}  "
                                  f"blur {blur_var:.0f}"
                                  + (f"  dist {fd:.0f}mm" if fd is not None else "")
                                  + "]")
                        else:
                            print(f"  Failed to write {fpath}")
                    elif not args.preview and collecting and not can_save:
                        skip_streak += 1

            # ---- Overlay ----
            y = 30
            cv2.putText(display, f"Name: {args.name or '(preview)'}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            y += 25
            if args.preview:
                cv2.putText(display, "PREVIEW - nothing saved",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 255), 1)
            else:
                cv2.putText(display, f"Collected: {collected}/{args.samples}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0) if collected > 0 else (255, 255, 255), 2)
            y += 25
            state = "COLLECTING" if collecting else "PAUSED"
            state_color = (0, 255, 0) if collecting else (0, 165, 255)
            cv2.putText(display, f"State: {state}  (press 's' to toggle)",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        state_color, 1)
            y += 25
            cv2.putText(display, status_text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

            cv2.imshow("Face Collect", display)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                collecting = not collecting
                print(f"Collection {'started' if collecting else 'paused'}.")

    except KeyboardInterrupt:
        pass
    finally:
        try: cam.stop()
        except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass

    print()
    print("=" * 60)
    print(f"Done. Collected {collected} crops.")
    if not args.preview:
        print(f"Saved to: {out_dir}")
        total = len([f for f in os.listdir(out_dir)
                     if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        print(f"Total crops in folder: {total}")
        print()
        print("Next step: python3 train_face_svm.py")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nFATAL: {e}\n")
        sys.exit(1)
