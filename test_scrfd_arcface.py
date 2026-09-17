#!/usr/bin/env python3
"""Headless SCRFD + ArcFace test with aligned-crop inspection."""
import os
import sys
import time
import argparse
import cv2
import numpy as np

sys.path.append('/home/medpal/pyorbbecsdk_v1/build')
import config
from scrfd_detector import SCRFDDetector
from arcface_embedder import ArcFaceEmbedder, align_face

OUT_DIR = "/tmp/scrfd_test"


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument("--save-every", type=float, default=2.0)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/aligned", exist_ok=True)
    print(f"Saving to {OUT_DIR}/ and {OUT_DIR}/aligned/")

    detector = SCRFDDetector(config.SCRFD_MODEL_PATH,
                             input_size=config.SCRFD_INPUT_SIZE,
                             conf_thres=config.SCRFD_CONF_THRES,
                             iou_thres=config.SCRFD_IOU_THRES)
    embedder = ArcFaceEmbedder(config.ARCFACE_MODEL_PATH)

    cam = None
    for idx in [1, 0, 2]:
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
    for _ in range(10):
        cam.read()
        time.sleep(0.05)

    start = time.monotonic()
    last_save = 0
    frames = 0
    saved = 0
    total_faces = 0
    detect_ms = []
    embed_ms = []
    snapshots = {}

    print(f"\nRunning for {args.duration}s.")
    print("Auto-snapshotting up to 3 distinct faces.")
    print("Aligned 112x112 crops will be saved for inspection.\n")

    try:
        while time.monotonic() - start < args.duration:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue
            frames += 1

            t0 = time.monotonic()
            detections = detector.infer(frame)
            t_detect = (time.monotonic() - t0) * 1000
            detect_ms.append(t_detect)

            embs_this = []
            for det in detections:
                (x, y, w, h, conf, landmarks) = det
                x1c = max(0, x); y1c = max(0, y)
                x2c = min(frame.shape[1], x + w)
                y2c = min(frame.shape[0], y + h)
                crop = frame[y1c:y2c, x1c:x2c]
                if crop.size == 0:
                    continue
                lm = landmarks.copy()
                lm[:, 0] -= x1c
                lm[:, 1] -= y1c

                # --- Save the aligned crop for inspection ---
                try:
                    aligned, _ = align_face(crop, lm)
                    cv2.imwrite(f"{OUT_DIR}/aligned/frame_{frames:04d}.jpg",
                                aligned)
                    # Side-by-side: original | aligned (both upscaled to 224)
                    orig_big = cv2.resize(crop, (224, 224))
                    alg_big = cv2.resize(aligned, (224, 224))
                    side = np.hstack([orig_big, alg_big])
                    cv2.imwrite(f"{OUT_DIR}/aligned/side_{frames:04d}.jpg",
                                side)
                except Exception as e:
                    print(f"align failed: {e}")

                # --- Also save landmarks annotated on the crop ---
                crop_dbg = crop.copy()
                for i, (px, py) in enumerate(lm):
                    cv2.circle(crop_dbg, (int(px), int(py)), 3,
                               (0, 255, 255), -1)
                    cv2.putText(crop_dbg, str(i), (int(px) + 4, int(py)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (255, 255, 255), 1)
                cv2.imwrite(f"{OUT_DIR}/aligned/landmarks_{frames:04d}.jpg",
                            crop_dbg)

                t1 = time.monotonic()
                emb = embedder.embed(crop, landmarks=lm)
                embed_ms.append((time.monotonic() - t1) * 1000)
                if emb is not None:
                    embs_this.append(emb)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                for (px, py) in landmarks:
                    cv2.circle(frame, (int(px), int(py)), 2, (0, 255, 255), -1)

            total_faces += len(embs_this)

            # Auto-snapshot
            if len(snapshots) < 3 and embs_this:
                is_new = all(cosine(e, embs_this[0]) < 0.55
                             for e in snapshots.values())
                if is_new:
                    name = chr(ord('A') + len(snapshots))
                    snapshots[name] = embs_this[0]
                    print(f"Auto-snapshot '{name}' captured")
                    np.save(f"{OUT_DIR}/snapshot_{name}.npy", embs_this[0])

            # Overlay similarity
            y_off = 30
            cv2.putText(frame, f"detect {t_detect:.0f}ms",
                        (10, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
            y_off += 26
            for name, e in snapshots.items():
                if embs_this:
                    sim = cosine(e, embs_this[0])
                    cv2.putText(frame, f"vs {name}: {sim:.3f}",
                                (10, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 0), 2)
                    y_off += 24

            now = time.monotonic()
            if now - last_save >= args.save_every:
                last_save = now
                path = f"{OUT_DIR}/frame_{saved:04d}.jpg"
                cv2.imwrite(path, frame)
                print(f"  saved {path}  "
                      f"(faces={len(embs_this)}, detect={t_detect:.0f}ms)")
                saved += 1

    except KeyboardInterrupt:
        pass

    cam.release()

    print()
    print("=" * 60)
    print(f"Frames processed: {frames}")
    print(f"Total faces:      {total_faces}")
    if detect_ms:
        print(f"SCRFD detect:     avg {sum(detect_ms)/len(detect_ms):.0f} ms")
    if embed_ms:
        print(f"ArcFace embed:    avg {sum(embed_ms)/len(embed_ms):.0f} ms")
    print(f"Auto-snapshots:   {list(snapshots.keys())}")
    if len(snapshots) > 1:
        print()
        names = list(snapshots.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sim = cosine(snapshots[names[i]], snapshots[names[j]])
                print(f"  {names[i]} vs {names[j]}: {sim:.3f}")
    print()
    print(f"INSPECT THESE FILES:")
    print(f"  {OUT_DIR}/aligned/side_0001.jpg  <- original | aligned")
    print(f"  {OUT_DIR}/aligned/landmarks_0001.jpg  <- numbered landmark dots")
    print("=" * 60)


if __name__ == "__main__":
    main()
