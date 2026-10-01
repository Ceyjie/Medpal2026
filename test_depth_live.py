#!/usr/bin/env python3
"""
test_depth_live.py -- live depth + color view with depth readout.

Shows two windows:
    Color (Astra Pro)   -- RGB feed with crosshair at center
    Depth (Astra Pro)   -- depth colormap, red = close, blue = far

Also prints center-pixel depth to the console every 0.5 s.

Move the camera around: point at close objects, then far walls.
The depth image should visibly change (colors shift) and the
console reading should track the actual distance.

Controls:
    q   quit
    s   save a snapshot of both frames to /tmp/
"""

import sys
sys.path.append('/home/medpal/pyorbbecsdk_v1/build')

import os
import time
import numpy as np
import cv2
from pyorbbecsdk import Pipeline, Config, OBSensorType


def pick_depth_profile(profiles):
    """Return (video_profile, (w,h,fps,format))."""
    n = profiles.get_count()
    for i in range(n):
        pr = profiles.get_stream_profile_by_index(i)
        try:
            vp = pr.as_video_stream_profile()
        except AttributeError:
            continue
        if vp.get_width() == 320 and vp.get_height() == 240:
            return vp, (vp.get_width(), vp.get_height(),
                        vp.get_fps(), vp.get_format())
    default = profiles.get_default_video_stream_profile()
    try:
        vp = default.as_video_stream_profile()
    except AttributeError:
        vp = default
    return vp, (vp.get_width(), vp.get_height(),
                vp.get_fps(), vp.get_format())


def main():
    print("=" * 60)
    print("test_depth_live.py -- live color + depth")
    print("=" * 60)

    # ---- Color camera (cv2.VideoCapture) ----
    color_cap = None
    for idx in [1, 0, 2]:
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            color_cap = c
            print(f"[color] opened /dev/video{idx}")
            break
        c.release()
    if color_cap is None:
        print("[color] no camera found")
        return

    # ---- Depth pipeline (Orbbec SDK) ----
    p = Pipeline()
    cfg = Config()
    profiles = p.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    video_profile, geom = pick_depth_profile(profiles)
    print(f"[depth] profile: {geom[0]}x{geom[1]}@{geom[2]} fmt={geom[3]}")
    cfg.enable_stream(video_profile)
    p.start(cfg)

    # Warm up
    for _ in range(10):
        color_cap.read()
        time.sleep(0.03)

    win_color = "Color (Astra Pro)"
    win_depth = "Depth (Astra Pro)"
    cv2.namedWindow(win_color, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_depth, cv2.WINDOW_NORMAL)

    last_print = 0.0
    frame_count = 0

    print()
    print("Move the camera. Watch both windows and the console line:")
    print("  dist = NNN mm  at  x=?, y=?  (min=.. p50=.. max=..)")
    print()
    print("Controls: q=quit, s=save snapshot")
    print()

    try:
        while True:
            # ---- Color ----
            ret, color = color_cap.read()
            if not ret or color is None:
                time.sleep(0.01)
                continue
            frame_count += 1

            # Crosshair at color center
            ch, cw = color.shape[:2]
            ccx, ccy = cw // 2, ch // 2
            cv2.drawMarker(color, (ccx, ccy),
                           (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

            # ---- Depth ----
            frames = p.wait_for_frames(500)
            depth_view = None
            center_mm = 0
            stats = ""
            if frames is not None:
                df = frames.get_depth_frame()
                if df is not None:
                    dw = df.get_width()
                    dh = df.get_height()
                    raw = df.get_data()
                    arr = np.frombuffer(raw, dtype=np.uint16)
                    expected = dw * dh
                    if arr.size == expected:
                        arr = arr.reshape(dh, dw)
                        valid = arr[arr > 0]

                        # Colormap: red=close, blue=far.
                        # Normalise only across the valid range so the
                        # image is always readable even when the whole
                        # scene is at a single distance (broken sensor
                        # case shows a uniform color).
                        if valid.size > 0:
                            vmin = int(valid.min())
                            vmax = int(valid.max())
                            # Guard against zero-range (stuck sensor)
                            if vmax - vmin < 10:
                                vmax = vmin + 100
                            clipped = np.clip(arr, vmin, vmax)
                            norm = ((clipped - vmin)
                                    / max(1, vmax - vmin) * 255.0
                                    ).astype(np.uint8)
                            norm[arr == 0] = 0
                            depth_view = cv2.applyColorMap(
                                norm, cv2.COLORMAP_JET)

                            # Center value
                            dcx, dcy = dw // 2, dh // 2
                            roi = arr[max(0, dcy - 5):dcy + 6,
                                      max(0, dcx - 5):dcx + 6]
                            rvalid = roi[roi > 0]
                            if rvalid.size > 0:
                                center_mm = float(np.median(rvalid))

                            stats = (f"min={vmin} p50="
                                     f"{int(np.median(valid))} "
                                     f"max={vmax} "
                                     f"valid={100*valid.size/arr.size:.0f}%")

                            cv2.drawMarker(depth_view, (dcx, dcy),
                                           (255, 255, 255),
                                           cv2.MARKER_CROSS, 15, 2)
                            # Big center readout on the depth image
                            txt = (f"{center_mm:.0f} mm"
                                   if center_mm > 0
                                   else "no depth at center")
                            cv2.putText(depth_view, txt, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (255, 255, 255), 2)
                            cv2.putText(depth_view,
                                        f"frame range: {vmin}-{vmax} mm",
                                        (10, 55),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (200, 200, 200), 1)
                        else:
                            depth_view = np.zeros((dh, dw, 3),
                                                  dtype=np.uint8)
                            cv2.putText(depth_view, "all zero",
                                        (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (0, 0, 255), 2)

            # ---- Show ----
            cv2.imshow(win_color, color)
            if depth_view is not None:
                cv2.imshow(win_depth, depth_view)

            # ---- Console print ----
            now = time.monotonic()
            if now - last_print >= 0.5:
                if center_mm > 0:
                    print(f"[{frame_count:5d}] "
                          f"dist={center_mm:5.0f} mm  {stats}")
                else:
                    print(f"[{frame_count:5d}] "
                          f"dist=  --   mm  {stats}")
                last_print = now

            # ---- Keys ----
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = time.strftime("%H%M%S")
                cv2.imwrite(f"/tmp/color_{ts}.jpg", color)
                if depth_view is not None:
                    cv2.imwrite(f"/tmp/depth_{ts}.jpg", depth_view)
                print(f"[save] /tmp/color_{ts}.jpg "
                      f"/tmp/depth_{ts}.jpg")

    except KeyboardInterrupt:
        pass
    finally:
        color_cap.release()
        try:
            p.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print()
        print("Done.")


if __name__ == "__main__":
    main()