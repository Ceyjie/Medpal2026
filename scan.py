#!/usr/bin/env python3
"""
scan.py -- End-to-end face scan: detect -> align -> embed -> match.

Minimal working version of the recognition loop. No motors, no body
tracking, no Coral. Just:

    frame -> SCRFD (5 landmarks) -> align 112x112 -> ArcFace 512-D
          -> match against gallery -> name on screen

Usage:
    python3 scan.py
    python3 scan.py --duration 60
    python3 scan.py --no-classifier
    python3 scan.py --dump-embeddings
"""

import os
# Silence ONNX Runtime's harmless GPU probe warning before importing it
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")

import sys
import time
import argparse
import pickle

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from scrfd_detector import SCRFDDetector
from arcface_embedder import ArcFaceEmbedder


# ============================================================
# Classifier (same format as train_face_svm.py writes)
# ============================================================
class Classifier:
    def __init__(self, path):
        self.available = False
        self.names = []
        self.centroids = None
        self.galleries = None
        self.p10 = None
        self.p50 = None
        self.p90 = None

        if not os.path.exists(path):
            print(f"[classifier] no file at {path}")
            return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"[classifier] load failed: {e}")
            return
        if data.get("model_type") != "centroid_gallery":
            print(f"[classifier] unexpected type: {data.get('model_type')}")
            return

        self.names = list(data["names"])
        self.centroids = np.asarray(data["centroids"], dtype=np.float32)
        self.galleries = [np.asarray(g, dtype=np.float32)
                          for g in data["galleries"]]
        self.p10 = np.asarray(data["stats_p10"], dtype=np.float32)
        self.p50 = np.asarray(data["stats_p50"], dtype=np.float32)
        self.p90 = np.asarray(data["stats_p90"], dtype=np.float32)
        self.available = True

        sizes = [g.shape[0] for g in self.galleries]
        print(f"[classifier] loaded {self.names} (galleries: {sizes})")

    def predict(self, q, top_k=3):
        """
        Match q (512-D unit vector) against all enrolled people.
        Returns (name, confidence, raw_score). name is None if no
        candidate passes the threshold.
        """
        if not self.available or q is None:
            return None, 0.0, 0.0

        q = q.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)

        best_score = -1.0
        best_idx = -1
        for i, (centroid, gallery) in enumerate(zip(self.centroids,
                                                    self.galleries)):
            score_cent = float(centroid @ q)
            sims = gallery @ q
            k = min(top_k, len(sims))
            top = np.partition(sims, -k)[-k:]
            score_top = float(top.mean())
            combined = 0.7 * score_top + 0.3 * score_cent
            if combined > best_score:
                best_score = combined
                best_idx = i

        if best_idx < 0:
            return None, 0.0, 0.0

        name = self.names[best_idx]
        p10 = float(self.p10[best_idx])
        p50 = float(self.p50[best_idx])
        p90 = float(self.p90[best_idx])

        # Map raw cosine to a confidence using the person's own
        # within-class similarity distribution.
        if best_score >= p50:
            span = max(p90 - p50, 1e-6)
            frac = min(1.0, (best_score - p50) / span)
            confidence = 0.75 + 0.20 * frac
        elif best_score >= p10:
            span = max(p50 - p10, 1e-6)
            frac = (best_score - p10) / span
            confidence = 0.55 + 0.20 * frac
        else:
            span = max(p50 - p10, 1e-6)
            overshoot = (p10 - best_score) / span
            confidence = 0.55 * float(np.exp(-2.0 * overshoot))

        threshold = getattr(config, "FACE_SVM_CONFIDENCE_THRES", 0.75)
        if confidence >= threshold:
            return name, confidence, best_score
        return None, confidence, best_score


# ============================================================
# Helpers
# ============================================================
def cosine(a, b):
    return float(np.dot(a, b) /
                 (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def open_camera(device=None):
    probe = [device] if device is not None else [0, 1, 2]
    for idx in probe:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            for _ in range(10):
                cap.read()
                time.sleep(0.03)
            print(f"[camera] opened /dev/video{idx}")
            return cap
        cap.release()
    return None


def draw_landmarks(img, lm, color=(0, 255, 255), radius=2):
    for (px, py) in lm:
        cv2.circle(img, (int(px), int(py)), radius, color, -1)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after N seconds (default: run until 'q')")
    parser.add_argument("--device", type=int, default=None,
                        help="Camera index (default: probe 0, 1, 2)")
    parser.add_argument("--no-classifier", action="store_true",
                        help="Ignore any trained classifier")
    parser.add_argument("--dump-embeddings", action="store_true",
                        help="Print embedding norm + first 4 values per scan")
    parser.add_argument("--min-face-px", type=int, default=40,
                        help="Skip detections smaller than this")
    args = parser.parse_args()

    print("=" * 60)
    print("scan.py -- SCRFD + ArcFace end-to-end")
    print("=" * 60)

    # ---- Load models ----
    detector = SCRFDDetector(
        config.SCRFD_MODEL_PATH,
        input_size=config.SCRFD_INPUT_SIZE,
        conf_thres=config.SCRFD_CONF_THRES,
        iou_thres=config.SCRFD_IOU_THRES,
    )
    embedder = ArcFaceEmbedder(config.ARCFACE_MODEL_PATH)

    # ---- Load classifier ----
    classifier = None if args.no_classifier else Classifier(
        getattr(config, "SVM_MODEL_PATH", "")
    )
    if classifier and classifier.available:
        print(f"[classifier] recognition ENABLED for {classifier.names}")
    else:
        print("[classifier] recognition DISABLED (embeddings only)")

    # ---- Open camera ----
    cam = open_camera(args.device)
    if cam is None:
        print("[camera] no camera found")
        return

    print()
    print("Controls:  q = quit")
    print()

    # ---- Loop ----
    start = time.monotonic()
    frame_count = 0
    det_count = 0
    detect_ms = []
    embed_ms = []

    try:
        while True:
            if args.duration and time.monotonic() - start >= args.duration:
                break

            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            frame_count += 1

            # ---- Detect ----
            t0 = time.monotonic()
            detections = detector.infer(frame)
            dt = (time.monotonic() - t0) * 1000.0
            detect_ms.append(dt)

            detections.sort(key=lambda d: d[2] * d[3], reverse=True)

            # ---- For each detection: align, embed, match ----
            for det in detections:
                x, y, w, h, conf, lm_frame = det
                if w < args.min_face_px or h < args.min_face_px:
                    continue
                det_count += 1

                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(frame.shape[1], x + w)
                y2 = min(frame.shape[0], y + h)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Landmarks from SCRFD are in full-frame coords.
                # embed() expects landmarks in the crop's coord system.
                lm_crop = lm_frame.copy().astype(np.float32)
                lm_crop[:, 0] -= x1
                lm_crop[:, 1] -= y1

                # ---- Embed ----
                t1 = time.monotonic()
                try:
                    emb = embedder.embed(crop, landmarks=lm_crop)
                except ValueError as e:
                    print(f"  embed skipped: {e}")
                    emb = None
                et = (time.monotonic() - t1) * 1000.0
                embed_ms.append(et)

                if emb is None:
                    continue

                if args.dump_embeddings:
                    norm = float(np.linalg.norm(emb))
                    print(f"  scan: {w}x{h} conf={conf:.2f}  "
                          f"norm={norm:.4f}  "
                          f"head={emb[:4].round(3).tolist()}")

                # ---- Match ----
                name, match_conf, raw_score = None, 0.0, 0.0
                if classifier and classifier.available:
                    name, match_conf, raw_score = classifier.predict(emb)

                # ---- Draw ----
                if name is not None:
                    box_color = (0, 255, 0)
                    label = f"{name}  {match_conf:.2f}"
                elif classifier and classifier.available:
                    box_color = (0, 165, 255)
                    label = f"Unknown?  {match_conf:.2f}"
                else:
                    box_color = (200, 200, 200)
                    label = "no classifier"

                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              box_color, 2)
                cv2.putText(frame, label, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            box_color, 2)
                draw_landmarks(frame, lm_frame)

            # ---- Overlay ----
            cv2.putText(frame, f"frames={frame_count}  faces={det_count}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            if detect_ms:
                d_avg = sum(detect_ms) / len(detect_ms)
                cv2.putText(frame, f"detect {d_avg:.0f} ms", (10, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1)
            if embed_ms:
                e_avg = sum(embed_ms) / len(embed_ms)
                cv2.putText(frame, f"embed  {e_avg:.0f} ms", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1)

            cv2.imshow("scan.py", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        cam.release()
        cv2.destroyAllWindows()

    # ---- Summary ----
    print()
    print("=" * 60)
    print(f"Frames:  {frame_count}")
    print(f"Faces:   {det_count}")
    if detect_ms:
        detect_ms.sort()
        n = len(detect_ms)
        print(f"Detect:  avg {sum(detect_ms)/n:.0f}  "
              f"p50 {detect_ms[n//2]:.0f}  "
              f"p95 {detect_ms[int(n*0.95)]:.0f} ms")
    if embed_ms:
        embed_ms.sort()
        n = len(embed_ms)
        print(f"Embed:   avg {sum(embed_ms)/n:.0f}  "
              f"p50 {embed_ms[n//2]:.0f}  "
              f"p95 {embed_ms[int(n*0.95)]:.0f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
