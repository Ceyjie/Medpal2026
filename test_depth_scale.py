#!/usr/bin/env python3
"""
test_depth_scale.py -- check whether the Astra Pro depth readings scale
with distance or are stuck at one value.

Point at three clearly different distances in sequence:
    1. ~10 cm   (very close -- a book or your hand)
    2. ~60 cm   (arm's length)
    3. ~3 m     (across the room -- wall or ceiling)

A working sensor returns roughly:
    p50 ~ 100 mm at 10 cm
    p50 ~ 600 mm at 60 cm
    p50 ~ 3000 mm at 3 m

A stuck sensor returns the same value regardless of distance.

Press Ctrl+C to stop.
"""

import sys
sys.path.append('/home/medpal/pyorbbecsdk_v1/build')

import time
import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType


def pick_depth_profile(profiles):
    """Return (video_profile, (w,h,fps,format))."""
    n = profiles.get_count()
    for i in range(n):
        pr = profiles.get_stream_profile_by_index(i)
        try:
            vp = pr.as_video_stream_profile()   # cast to video profile
        except AttributeError:
            continue
        if vp.get_width() == 320 and vp.get_height() == 240:
            return vp, (vp.get_width(), vp.get_height(),
                        vp.get_fps(), vp.get_format())
    # Fall back to default, cast if possible
    default = profiles.get_default_video_stream_profile()
    try:
        vp = default.as_video_stream_profile()
    except AttributeError:
        vp = default
    return vp, (vp.get_width(), vp.get_height(),
                vp.get_fps(), vp.get_format())


def main():
    print("=" * 60)
    print("test_depth_scale.py -- is the Astra depth stuck or scaling?")
    print("=" * 60)

    p = Pipeline()
    cfg = Config()
    profiles = p.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

    video_profile, geom = pick_depth_profile(profiles)
    print(f"Using profile: {geom[0]}x{geom[1]}@{geom[2]} fmt={geom[3]}")
    cfg.enable_stream(video_profile)   # <-- pass video profile
    p.start(cfg)

    print()
    print("Move the camera slowly through three distances:")
    print("  10 cm  ->  book or hand very close to the lens")
    print("  60 cm  ->  arm's length, aimed at a wall")
    print("  3 m    ->  across the room, aimed at a wall or ceiling")
    print()
    print("Press Ctrl+C to stop.")
    print()

    time.sleep(2)

    header_printed = False
    try:
        while True:
            frames = p.wait_for_frames(1000)
            if frames is None:
                print("  no frame")
                continue

            df = frames.get_depth_frame()
            if df is None:
                print("  no depth frame")
                continue

            w, h = df.get_width(), df.get_height()

            try:
                scale = df.get_depth_scale()
            except AttributeError:
                scale = 1.0

            raw = df.get_data()
            arr = np.frombuffer(raw, dtype=np.uint16)
            expected = w * h
            if arr.size != expected:
                arr = arr[:expected]
            arr = arr.reshape(h, w)

            valid = arr[arr > 0]
            if valid.size == 0:
                print(f"  {w}x{h}  all zero (no valid pixels)")
                time.sleep(0.3)
                continue

            p10 = int(np.percentile(valid, 10))
            p50 = int(np.percentile(valid, 50))
            p90 = int(np.percentile(valid, 90))
            vmin = int(valid.min())
            vmax = int(valid.max())
            pct_valid = 100.0 * valid.size / arr.size

            if not header_printed:
                print(f"  {'scale':>7} {'min':>6} {'p10':>6} "
                      f"{'p50':>6} {'p90':>6} {'max':>6} "
                      f"{'valid%':>7}")
                print("  " + "-" * 52)
                header_printed = True

            print(f"  {scale:>7.3f} {vmin:>6} {p10:>6} "
                  f"{p50:>6} {p90:>6} {vmax:>6} "
                  f"{pct_valid:>6.1f}%")

            time.sleep(0.3)

    except KeyboardInterrupt:
        print()
        print("Interrupted.")
    finally:
        p.stop()


if __name__ == "__main__":
    main()
