#!/usr/bin/env python3
"""
PHASE 0 TEST -- CAMERA ONLY

Proves the Astra Pro delivers usable RGB and depth, independent of
Coral, ReID, and motors. If this doesn't work cleanly, nothing built
on top of it will either -- fix this first.

What "pass" looks like:
  - Two windows open: color feed and a depth colormap.
  - The printed distance at the center crosshair changes sensibly as
    you move something toward/away from the camera.
  - Distance reads roughly correct starting around 40-60cm out (closer
    than that is the sensor's blind zone -- expect 0/invalid there,
    that's normal, not a bug).

Controls: 'q' to quit.
"""
import os
import sys
import time

import cv2
import numpy as np

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# Match your real project's sdk path -- adjust if yours differs.
sys.path.append('/home/medpal/pyorbbecsdk_v1/build')

FRAME_W, FRAME_H = 640, 480
DEPTH_W, DEPTH_H = 320, 240


def open_color_camera():
    """Same probing approach as tracker.py -- tries a few device indices."""
    for idx in [1, 0, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            print(f"[camera] Color device opened at index {idx}")
            return cap
        cap.release()
    return None


def open_depth_pipeline():
    try:
        from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
    except ImportError as e:
        print(f"[camera] FAILED to import pyorbbecsdk: {e}")
        print("[camera] Check the sys.path.append() line above matches your real SDK build path.")
        return None

    try:
        pipeline = Pipeline()
        cfg = Config()
        profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        try:
            profile = profiles.get_video_stream_profile(DEPTH_W, DEPTH_H, OBFormat.Y16, 15)
        except Exception:
            print("[camera] Requested depth profile unavailable, falling back to default.")
            profile = profiles.get_default_video_stream_profile()
        cfg.enable_stream(profile)
        pipeline.start(cfg)
        print("[camera] Depth pipeline started")
        return pipeline
    except Exception as e:
        print(f"[camera] FAILED to start depth pipeline: {e}")
        print("[camera] Check USB connection/power -- Astra Pro needs USB3 and stable power.")
        return None


def read_depth_frame(pipeline):
    """Returns a (H, W) uint16 numpy array in mm, or None."""
    from pyorbbecsdk import OBFormat  # local import, only needed here
    try:
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            return None
        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            return None
        w, h = depth_frame.get_width(), depth_frame.get_height()
        data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        return data.reshape((h, w))
    except Exception:
        return None


def main():
    print("=" * 60)
    print("PHASE 0: CAMERA-ONLY TEST (no Coral, no motors, no ReID)")
    print("=" * 60)

    color_cap = open_color_camera()
    if color_cap is None:
        print("[camera] FAILED: no color camera found. Stop here -- fix this before anything else.")
        return

    depth_pipeline = open_depth_pipeline()
    if depth_pipeline is None:
        print("[camera] Depth failed to start -- color-only mode. Fix depth before moving on.")

    frame_count = 0
    last_print = time.monotonic()

    while True:
        ret, color = color_cap.read()
        if not ret or color is None:
            print("[camera] WARNING: dropped color frame")
            time.sleep(0.01)
            continue

        cx, cy = FRAME_W // 2, FRAME_H // 2
        cv2.drawMarker(color, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.imshow("Color (Astra Pro)", color)

        if depth_pipeline is not None:
            depth = read_depth_frame(depth_pipeline)
            if depth is not None:
                dh, dw = depth.shape
                dcx, dcy = dw // 2, dh // 2
                roi = depth[max(0, dcy - 5):dcy + 5, max(0, dcx - 5):dcx + 5]
                valid = roi[roi > 0]
                center_mm = float(np.median(valid)) if valid.size else 0.0

                # colorize for display
                vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                cv2.drawMarker(vis, (dcx, dcy), (255, 255, 255), cv2.MARKER_CROSS, 15, 2)
                cv2.imshow("Depth (Astra Pro)", vis)

                now = time.monotonic()
                if now - last_print > 0.5:
                    status = "BLIND ZONE / INVALID" if center_mm == 0 else f"{center_mm:.0f} mm"
                    print(f"[camera] center distance: {status}")
                    last_print = now

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    color_cap.release()
    if depth_pipeline is not None:
        depth_pipeline.stop()
    cv2.destroyAllWindows()
    print(f"[camera] Test ended after {frame_count} color frames.")


if __name__ == "__main__":
    main()
