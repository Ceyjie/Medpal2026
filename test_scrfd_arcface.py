#!/usr/bin/env python3
"""
test_scrfd_arcface.py -- Headless SCRFD + ArcFace test.

What this proves:
  1. SCRFD detects faces and returns 5 landmarks in the correct coord system.
  2. ArcFace alignment produces a correctly-oriented 112x112 crop.
  3. Embeddings from the same person cluster (cosine > 0.55), different
     people separate (cosine < 0.35).

Saves to /tmp/scrfd_test/:
  frame_NNNN.jpg         -- annotated full frames
  aligned/frame_NNNN.jpg -- aligned 112x112 crops
  aligned/side_NNNN.jpg  -- [original crop | aligned] side by side
  aligned/land_NNNN.jpg  -- landmarks drawn on the crop
  snapshot_X.npy         -- the first embedding of each distinct person

Run:
  cd /home/medpal/2026medpal
  python3 test_scrfd_arcface.py --duration 60 --save-every 3
"""
import os
import sys
import time
import argparse
import cv2
import numpy as np

sys.path.append('/home/medpal/pyorbbecsdk_v1/build')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from scrfd_detector import SCRFDDetector
from arcface_embedder import ArcFaceEmbedder, align_face

OUT_DIR = "/tmp/scrfd_test"


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=60,
                        help="Total seconds to run")
    parser.add_argument("--save-every", type=float, default=3.0,
                        help="Save an annotated full frame every N seconds")
    parser.add_argument("--aligned-every", type=float, default=1.0,
                        help="Save aligned crops at most once per N seconds")
    parser.add_argument("--min-face-px", type=int, default=40,
                        help="Ignore face boxes smaller than this (px)")
    parser.add_argument("--device", type=int, default=None,
                        help="Camera index. If unset, probes 1, 0, 2.")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/aligned", exist_ok=True)
    print(f"Saving to {OUT_DIR}/ and {OUT_DIR}/aligned/")

    # --- Load models ---
    detector = SCRFDDetector(config.SCRFD_MODEL_PATH,
                             input_size=config.SCRFD_INPUT_SIZE,
                             conf_thres=config.SCRFD_CONF_THRES,
                             iou_thres=config.SCRFD_IOU_THRES)
    embedder = ArcFaceEmbedder(config.ARCFACE_MODEL_PATH)

    # --- Open camera ---
    cam = None
    probe = [args.device] if args.device is not None else [1, 0, 2]
    for idx in probe:
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            print(f"Opened camera on /dev/video{idx}")
            cam = c
            break
        c.release()
    if cam is None:
        print("Cannot open any camera")
        return
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Warm up: discard the first few frames
    for _ in range(10):
        cam.read()
        time.sleep(0.05)

    # --- Loop state ---
    start = time.monotonic()
    last_frame_save = 0.0
    last_aligned_save = 0.0
    frames = 0
    saved_frames = 0
    saved_aligned = 0
    total_faces = 0
    detect_ms = []
    embed_ms = []
    snapshots = {}          # name -> embedding
    pair_log = []           # (frame, nameA, nameB, cosine)
    printed_first_lm = False

    print(f"\nRunning for {args.duration}s.")
    print("Auto-snapshotting up to 3 distinct faces.")
    print("Aligned crops saved to aligned/ for inspection.\n")

    try:
        while time.monotonic() - start < args.duration:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue
            frames += 1

            # ---- Detect ----
            t0 = time.monotonic()
            detections = detector.infer(frame)
            t_detect = (time.monotonic() - t0) * 1000.0
            detect_ms.append(t_detect)

            embs_this = []
            boxes_this = []

            for det in detections:
                (x, y, w, h, conf, landmarks) = det

                # Skip tiny detections
                if w < args.min_face_px or h < args.min_face_px:
                    continue

                x1c = max(0, x)
                y1c = max(0, y)
                x2c = min(frame.shape[1], x + w)
                y2c = min(frame.shape[0], y + h)
                crop = frame[y1c:y2c, x1c:x2c]
                if crop.size == 0:
                    continue

                # Landmarks from SCRFD are in FULL-FRAME coords -> translate
                lm = landmarks.copy().astype(np.float32)
                lm[:, 0] -= x1c
                lm[:, 1] -= y1c

                # One-time landmark sanity print
                if not printed_first_lm:
                    printed_first_lm = True
                    print(f"\n[first detection] box=({x1c},{y1c},{x2c-x1c},{y2c-y1c}) "
                          f"conf={conf:.3f} crop={crop.shape[1]}x{crop.shape[0]}")
                    for i, (px, py) in enumerate(lm):
                        print(f"  landmark {i}: ({px:6.1f}, {py:6.1f})")
                    print()

                # ---- Align (for saving) ----
                now = time.monotonic()
                if now - last_aligned_save >= args.aligned_every:
                    try:
                        aligned, _ = align_face(crop, lm)
                        cv2.imwrite(
                            f"{OUT_DIR}/aligned/frame_{frames:04d}.jpg",
                            aligned)
                        # Side-by-side: original | aligned
                        orig_big = cv2.resize(crop, (224, 224))
                        alg_big = cv2.resize(aligned, (224, 224))
                        side = np.hstack([orig_big, alg_big])
                        cv2.imwrite(
                            f"{OUT_DIR}/aligned/side_{frames:04d}.jpg", side)
                        # Landmark annotation on the original crop
                        crop_dbg = crop.copy()
                        for i, (px, py) in enumerate(lm):
                            cv2.circle(crop_dbg, (int(px), int(py)), 3,
                                       (0, 255, 255), -1)
                            cv2.putText(crop_dbg, str(i),
                                        (int(px) + 4, int(py)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                        (255, 255, 255), 1)
                        cv2.imwrite(
                            f"{OUT_DIR}/aligned/land_{frames:04d}.jpg",
                            crop_dbg)
                        saved_aligned += 1
                        last_aligned_save = now
                    except Exception as e:
                        print(f"align/save failed: {e}")

                # ---- Embed ----
                t1 = time.monotonic()
                emb = embedder.embed(crop, landmarks=lm)
                embed_ms.append((time.monotonic() - t1) * 1000.0)

                if emb is not None:
                    embs_this.append(emb)
                    boxes_this.append((x, y, w, h, conf))

                # Draw on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                for (px, py) in landmarks:
                    cv2.circle(frame, (int(px), int(py)), 2, (0, 255, 255), -1)
                cv2.putText(frame, f"{conf:.2f}",
                            (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 255, 0), 1)

            total_faces += len(embs_this)

            # ---- Auto-snapshot: up to 3 distinct people ----
            for i, emb in enumerate(embs_this):
                if len(snapshots) >= 3:
                    break
                # Consider this face distinct if it's far from every
                # existing snapshot and from every earlier face this frame
                too_close = False
                for existing in snapshots.values():
                    if cosine(existing, emb) >= 0.55:
                        too_close = True
                        break
                if too_close:
                    continue
                for earlier in embs_this[:i]:
                    if cosine(earlier, emb) >= 0.55:
                        too_close = True
                        break
                if too_close:
                    continue

                name = chr(ord('A') + len(snapshots))
                snapshots[name] = emb
                print(f"Auto-snapshot '{name}' captured "
                      f"(frame {frames}, "
                      f"{len(boxes_this[i]) if i < len(boxes_this) else '?'})")
                np.save(f"{OUT_DIR}/snapshot_{name}.npy", emb)

            # ---- Live similarity log (once per second) ----
            if embs_this and len(snapshots) >= 1:
                for i, emb in enumerate(embs_this):
                    sims = {n: cosine(e, emb) for n, e in snapshots.items()}
                    best_name = max(sims, key=sims.get)
                    best_sim = sims[best_name]
                    if len(snapshots) >= 2:
                        # log the pair if it's this frame's first face
                        if i == 0:
                            pair_log.append((frames, best_name, best_sim))

            # ---- Overlay ----
            cv2.putText(frame, f"detect {t_detect:.0f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
            y_off = 56
            for name, e in snapshots.items():
                if embs_this:
                    sim = cosine(e, embs_this[0])
                    cv2.putText(frame, f"vs {name}: {sim:+.3f}",
                                (10, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 0), 2)
                    y_off += 24

            # ---- Save full frame periodically ----
            now = time.monotonic()
            if now - last_frame_save >= args.save_every:
                path = f"{OUT_DIR}/frame_{saved_frames:04d}.jpg"
                cv2.imwrite(path, frame)
                print(f"  saved {path}  "
                      f"(faces={len(embs_this)}, detect={t_detect:.0f}ms)")
                saved_frames += 1
                last_frame_save = now

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        cam.release()

    # ---- Summary ----
    print()
    print("=" * 60)
    print(f"Frames processed: {frames}")
    print(f"Total faces:      {total_faces}")
    if detect_ms:
        detect_ms.sort()
        n = len(detect_ms)
        print(f"SCRFD detect:     avg {sum(detect_ms)/n:6.0f} ms   "
              f"p50 {detect_ms[n//2]:6.0f}   "
              f"p95 {detect_ms[int(n*0.95)]:6.0f}")
    if embed_ms:
        embed_ms.sort()
        n = len(embed_ms)
        print(f"ArcFace embed:    avg {sum(embed_ms)/n:6.0f} ms   "
              f"p50 {embed_ms[n//2]:6.0f}   "
              f"p95 {embed_ms[int(n*0.95)]:6.0f}")
    print(f"Aligned crops:    {saved_aligned}")
    print(f"Auto-snapshots:   {sorted(snapshots.keys())}")

    if len(snapshots) > 1:
        print()
        print("Pairwise cosine between snapshots:")
        names = sorted(snapshots.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sim = cosine(snapshots[names[i]], snapshots[names[j]])
                verdict = ("SAME" if sim > 0.55
                           else "different" if sim < 0.35
                           else "ambiguous")
                print(f"  {names[i]} vs {names[j]}: {sim:+.3f}   [{verdict}]")

    if pair_log:
        sims = [s for (_, _, s) in pair_log]
        sims_sorted = sorted(sims)
        n = len(sims_sorted)
        print()
        print(f"Within-session match distribution ({n} frames):")
        print(f"  min  {sims_sorted[0]:+.3f}")
        print(f"  p10  {sims_sorted[int(n*0.10)]:+.3f}")
        print(f"  p50  {sims_sorted[n//2]:+.3f}")
        print(f"  p90  {sims_sorted[int(n*0.90)]:+.3f}")
        print(f"  max  {sims_sorted[-1]:+.3f}")

    print()
    print("INSPECT THESE FILES:")
    print(f"  {OUT_DIR}/aligned/side_*.jpg   <- original | aligned")
    print(f"  {OUT_DIR}/aligned/land_*.jpg   <- numbered landmark dots")
    print("=" * 60)


if __name__ == "__main__":
    main()
