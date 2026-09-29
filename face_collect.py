#!/usr/bin/env python3
"""
face_collect.py -- Capture aligned 112x112 face crops for ArcFace training.

Detects faces with SCRFD (CPU), aligns to the ArcFace reference pose,
saves the aligned crop. No Coral, no tracker, no motors.

Usage:
    python3 face_collect.py --name Carl --samples 60
    python3 face_collect.py --preview
    python3 face_collect.py --name Carl --output /tmp/carl_faces
"""
import os
import sys
import time
import argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from scrfd_detector import SCRFDDetector
from arcface_embedder import align_face


def open_camera(device=None):
    for idx in ([device] if device is not None else [0, 1, 2]):
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


def evaluate(gray_small, last_gray, min_px, w, h, blur_var,
             blur_thres, diversity_px):
    reasons = []
    if w < min_px or h < min_px:
        reasons.append(f"too small ({w}x{h})")
    if blur_var < blur_thres:
        reasons.append(f"blurry ({blur_var:.0f})")
    if last_gray is not None:
        diff = float(np.mean(np.abs(
            gray_small.astype(np.float32) - last_gray.astype(np.float32)
        )))
        if diff < diversity_px:
            reasons.append("too similar to last")
    return reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--output", default=None)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--save-interval", type=float, default=0.15)
    args = parser.parse_args()

    if not args.preview and not args.name:
        print("ERROR: --name required unless --preview")
        return

    out_dir = None
    if not args.preview:
        out_dir = args.output or os.path.join(config.FACE_TRAINING_DIR, args.name)
        os.makedirs(out_dir, exist_ok=True)

    min_px = getattr(config, "ENROLL_MIN_FACE_PX", 50)
    blur_thres = getattr(config, "ENROLL_BLUR_THRES", 40.0)
    diversity_px = getattr(config, "ENROLL_DIVERSITY_PX", 25)

    detector = SCRFDDetector(
        config.SCRFD_MODEL_PATH,
        input_size=config.SCRFD_INPUT_SIZE,
        conf_thres=config.SCRFD_CONF_THRES,
        iou_thres=config.SCRFD_IOU_THRES,
    )

    cam = open_camera()
    if cam is None:
        print("Cannot open camera")
        return

    print()
    print("=" * 60)
    print(f"face_collect.py -- aligned 112x112 crops for '{args.name}'")
    print(f"  output: {out_dir}")
    print(f"  target: {args.samples}")
    print(f"  filters: min_px={min_px} blur>={blur_thres} diversity>={diversity_px}")
    print("  Slowly rotate your head: left, right, up, down.")
    print("  Vary distance 0.5-1.5 m. Press 'q' to stop early.")
    print("=" * 60)
    print()

    cv2.namedWindow("face_collect", cv2.WINDOW_NORMAL)

    collected = 0
    last_gray = None
    last_save_t = 0.0
    start = time.monotonic()

    try:
        while True:
            if args.duration and time.monotonic() - start >= args.duration:
                break
            if not args.preview and collected >= args.samples:
                break

            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            dets = detector.infer(frame)
            display = frame.copy()
            status = "no face"
            status_color = (0, 0, 255)
            box_color = (128, 128, 128)

            if dets:
                # Pick the largest face
                x, y, w, h, conf, lm_frame = max(
                    dets, key=lambda d: d[2] * d[3]
                )
                x1 = max(0, x); y1 = max(0, y)
                x2 = min(frame.shape[1], x + w)
                y2 = min(frame.shape[0], y + h)

                try:
                    aligned, _ = align_face(frame, lm_frame, image_size=112)
                except Exception as e:
                    aligned = None
                    status = f"align failed: {e}"

                if aligned is not None:
                    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    gray_small = cv2.resize(gray, (64, 64))
                    reasons = evaluate(
                        gray_small, last_gray, min_px, w, h, blur_var,
                        blur_thres, diversity_px,
                    )

                    if not reasons and not args.preview:
                        now = time.monotonic()
                        if now - last_save_t >= args.save_interval:
                            ts = time.strftime("%Y%m%d_%H%M%S")
                            ms = int(now * 1000) % 1000
                            fname = f"enroll_{ts}_{ms:03d}_{collected+1:04d}.jpg"
                            fpath = os.path.join(out_dir, fname)
                            if cv2.imwrite(fpath, aligned):
                                collected += 1
                                last_gray = gray_small
                                last_save_t = now
                                box_color = (0, 255, 0)
                                status = (f"saved {collected}/{args.samples}  "
                                          f"blur {blur_var:.0f}")
                                status_color = (0, 255, 0)
                                print(f"  saved {fname}  "
                                      f"blur {blur_var:.0f}  size {w}x{h}")
                    elif not reasons and args.preview:
                        box_color = (255, 255, 0)
                        status = f"ready (preview)  blur {blur_var:.0f}"
                        status_color = (255, 255, 0)
                    else:
                        box_color = (0, 200, 255)
                        status = "skip: " + ", ".join(reasons)
                        status_color = (0, 200, 255)

                    cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
                    for (px, py) in lm_frame:
                        cv2.circle(display, (int(px), int(py)),
                                   2, (0, 255, 255), -1)

                    # Show the aligned crop as an inset
                    if aligned is not None:
                        inset = cv2.resize(aligned, (168, 168),
                                           interpolation=cv2.INTER_NEAREST)
                        display[10:178, display.shape[1]-178:display.shape[1]-10] = inset
                        cv2.rectangle(
                            display,
                            (display.shape[1] - 178, 10),
                            (display.shape[1] - 10, 178),
                            (0, 255, 255), 1,
                        )

            cv2.putText(display,
                        f"{args.name or 'preview'}: {collected}/{args.samples}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.putText(display, status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
            cv2.imshow("face_collect", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        cv2.destroyAllWindows()

    print()
    print(f"Done. Collected {collected} aligned crops.")
    if not args.preview:
        total = len([f for f in os.listdir(out_dir)
                     if f.lower().endswith((".jpg", ".png"))])
        print(f"Total in {out_dir}: {total}")


if __name__ == "__main__":
    main()
