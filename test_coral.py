#!/usr/bin/env python3
"""
PHASE 0 TEST -- CORAL ONLY

Proves the Coral USB Accelerator loads and runs your actual detection
model, independent of the Astra camera and everything else. Uses your
webcam (plain cv2, not the Astra) just to get *some* image in front of
the model -- the point here is testing Coral, not the depth camera.

What "pass" looks like:
  - Console prints "Coral Edge TPU delegate loaded successfully"
  - Console prints a real inference time in milliseconds, repeatedly,
    without errors
  - When a person is in frame, you see at least one detection with a
    reasonable confidence score printed

If the delegate fails to load: check
  1. USB power -- try a powered hub, Coral is power-hungry
  2. `python3 -c "import tflite_runtime"` works in THIS environment
     (must be the same pinned Python version you set up for Coral,
     not your default Pi Python)
  3. `lsusb` shows the Coral device at all

Controls: 'q' to quit.
"""
import os
import sys
import time

import cv2
import numpy as np

# Match your real project's config exactly
MODEL_PATH = "/home/medpal/tracking_person/models/tflite/ssd_mobilenet_v2_edgetpu.tflite"
MODEL_PATH_CPU = "/home/medpal/tracking_person/models/tflite/ssd_mobilenet_v2.tflite"
LABELS_PATH = "/home/medpal/tracking_person/models/tflite/coco_labels.txt"
CONF_THRES = 0.5
PERSON_CLASS = 1  # COCO person class id

FRAME_W, FRAME_H = 640, 480


def load_labels(path):
    if os.path.exists(path):
        with open(path) as f:
            return [l.strip() for l in f.readlines()]
    print(f"[coral] WARNING: labels file not found at {path}")
    return []


def load_interpreter():
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
    except ImportError as e:
        print(f"[coral] FAILED to import tflite_runtime: {e}")
        print("[coral] This Python environment doesn't have tflite_runtime installed.")
        print("[coral] Make sure you're running this inside your pinned Coral venv, not system Python.")
        return None, False

    edgetpu = False
    interpreter = None
    try:
        delegate = load_delegate('libedgetpu.so.1')
        if not os.path.exists(MODEL_PATH):
            print(f"[coral] FAILED: edgetpu model not found at {MODEL_PATH}")
            return None, False
        interpreter = Interpreter(MODEL_PATH, experimental_delegates=[delegate])
        edgetpu = True
        print("[coral] Coral Edge TPU delegate loaded successfully")
    except Exception as e:
        print(f"[coral] Edge TPU delegate unavailable: {e}")
        print("[coral] Tip: check USB power (use a powered hub) or try a different USB port.")
        if os.path.exists(MODEL_PATH_CPU):
            print("[coral] Falling back to CPU model so you can at least verify the pipeline logic.")
            interpreter = Interpreter(MODEL_PATH_CPU)
        else:
            print(f"[coral] CPU fallback model also not found at {MODEL_PATH_CPU}. Stopping.")
            return None, False

    interpreter.allocate_tensors()
    return interpreter, edgetpu


def infer(interpreter, frame, input_details, output_details, input_shape):
    target_h, target_w = input_shape[1], input_shape[2]
    resized = cv2.resize(frame, (target_w, target_h))
    input_data = np.expand_dims(resized, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    t0 = time.perf_counter()
    interpreter.invoke()
    infer_ms = (time.perf_counter() - t0) * 1000

    boxes = interpreter.tensor(output_details[0]['index'])()
    classes = interpreter.tensor(output_details[1]['index'])()
    scores = interpreter.tensor(output_details[2]['index'])()

    h, w = frame.shape[:2]
    results = []
    for i in range(int(scores[0].shape[0])):
        score = float(scores[0][i])
        cls = int(classes[0][i])
        if score < CONF_THRES:
            continue
        y1, x1, y2, x2 = boxes[0][i]
        x, y = int(x1 * w), int(y1 * h)
        bw, bh = int((x2 - x1) * w), int((y2 - y1) * h)
        results.append((x, y, bw, bh, score, cls))
    return results, infer_ms


def main():
    print("=" * 60)
    print("PHASE 0: CORAL-ONLY TEST (no Astra depth, no motors, no ReID)")
    print("=" * 60)

    labels = load_labels(LABELS_PATH)
    interpreter, edgetpu = load_interpreter()
    if interpreter is None:
        print("[coral] Stopping -- fix the delegate/model issue above before moving on.")
        return

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    print(f"[coral] Model input shape: {input_shape}, running on {'EDGE TPU' if edgetpu else 'CPU (fallback)'}")

    cap = None
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            print(f"[coral] Using webcam at index {idx} just to feed the model something")
            break
        cap.release()
        cap = None

    if cap is None:
        print("[coral] No webcam found -- running on a single blank test frame instead.")
        test_frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        results, ms = infer(interpreter, test_frame, input_details, output_details, input_shape)
        print(f"[coral] Inference ran in {ms:.1f} ms. {len(results)} detections (expected 0 on a blank frame).")
        print("[coral] If this ran without errors, the TPU/model pipeline itself works.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        results, ms = infer(interpreter, frame, input_details, output_details, input_shape)

        for (x, y, w, h, score, cls) in results:
            label = labels[cls] if cls < len(labels) else str(cls)
            tag = f"PERSON {score:.2f}" if cls == PERSON_CLASS else f"{label} {score:.2f}"
            color = (0, 255, 0) if cls == PERSON_CLASS else (128, 128, 128)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, tag, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(frame, f"{ms:.1f} ms/inference ({'TPU' if edgetpu else 'CPU'})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Coral Test", frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"[coral] frame {frame_count}: {ms:.1f} ms/inference, {len(results)} detection(s)")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[coral] Test ended after {frame_count} frames.")


if __name__ == "__main__":
    main()
