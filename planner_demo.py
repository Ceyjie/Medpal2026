#!/usr/bin/env python3
"""
planner_demo.py -- standalone video path planner with AR overlay.

Shows gray floor shading + sky-blue A* path, occupancy grid panel, and
a side-view point cloud. Self-calibrates the ground plane. Inflates
obstacles by robot radius.

Controls:
    click    set a goal on the floor
    c        clear goal
    p        pause planning
    r        force re-estimate of ground plane
    m        toggle flat sky-blue / Tesla gradient
    q        quit

Usage:
    python3 planner_demo.py
    python3 planner_demo.py --goal-forward 1500
    python3 planner_demo.py --motors --follow-speed 50
"""

import os
import sys
import time
import argparse
import threading
import math

import cv2
import numpy as np

sys.path.append('/home/medpal/pyorbbecsdk_v1/build')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
    USE_ORBBEC_DEPTH = True
except ImportError:
    USE_ORBBEC_DEPTH = False
    print("pyorbbecsdk not available -- depth disabled")

import config
from path_planner import PathPlanner


CALIB_PATH = "/home/medpal/2026medpal/camera_calib.yml"


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
        self.use_orbbec_depth = False

        self.color_thread = threading.Thread(
            target=self._color_capture, daemon=True)
        self.color_thread.start()

        if USE_ORBBEC_DEPTH:
            try:
                self.depth_pipeline = Pipeline()
                self.depth_config = Config()
                profiles = self.depth_pipeline.get_stream_profile_list(
                    OBSensorType.DEPTH_SENSOR)
                self.depth_profile = None
                for w, h, fps in [(320, 240, 15), (320, 240, 30),
                                  (640, 480, 15)]:
                    try:
                        self.depth_profile = profiles.get_video_stream_profile(
                            w, h, OBFormat.Y16, fps)
                        print(f"[camera] depth {w}x{h}@{fps}")
                        break
                    except Exception:
                        continue
                if self.depth_profile is None:
                    self.depth_profile = \
                        profiles.get_default_video_stream_profile()
                    print(f"[camera] depth default "
                          f"({self.depth_profile.get_width()}x"
                          f"{self.depth_profile.get_height()})")
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
        while self.depth_running:
            try:
                frames = self.depth_pipeline.wait_for_frames(1000)
                if frames:
                    df = frames.get_depth_frame()
                    if df:
                        w, h = df.get_width(), df.get_height()
                        try:
                            scale = df.get_depth_scale()
                        except AttributeError:
                            scale = 1.0
                        data = np.frombuffer(
                            df.get_data(), dtype=np.uint16).reshape(h, w)
                        depth_mm = data.astype(np.float32) * scale
                        depth_mm[depth_mm < config.MIN_VALID_DEPTH_MM] = 0
                        self.depth_frame = depth_mm
            except Exception as e:
                print(f"[camera] depth error: {e}")
                time.sleep(1.0)
            time.sleep(0.01)

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
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motors", action="store_true",
                        help="Actually drive motors along the path")
    parser.add_argument("--follow-speed", type=int, default=50)
    parser.add_argument("--turn-speed", type=int, default=40)
    parser.add_argument("--goal-forward", type=float, default=1200.0)
    parser.add_argument("--goal-lateral", type=float, default=0.0)
    parser.add_argument("--plan-every", type=int, default=3)
    parser.add_argument("--calibrate-every", type=int, default=30)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--no-inflate", action="store_true")
    parser.add_argument("--floor-alpha", type=float, default=0.55,
                        help="Strength of gray floor shading (0..1)")
    parser.add_argument("--tesla", action="store_true",
                        help="Start in Tesla gradient mode instead of sky blue")
    args = parser.parse_args()

    print("=" * 60)
    print("planner_demo.py -- AR overlay + A* planner")
    print("=" * 60)

    cam = Camera()
    for _ in range(60):
        if cam.read_color() is not None:
            break
        time.sleep(0.05)

    planner = PathPlanner(
        cam_height_mm=getattr(config, "CAMERA_HEIGHT_MM", 200),
        cam_tilt_deg=getattr(config, "CAMERA_TILT_DEG", 15),
        cam_hfov_deg=getattr(config, "CAMERA_HFOV_DEG", 60),
        grid_size=getattr(config, "OCCUPANCY_GRID_SIZE", 60),
        res_mm=getattr(config, "OCCUPANCY_GRID_RES_MM", 50),
        robot_radius_mm=getattr(config, "ROBOT_RADIUS_MM", 200),
    )
    planner.load_calibration(CALIB_PATH)
    planner.set_goal_world(args.goal_forward, args.goal_lateral)

    motors = None
    if args.motors:
        try:
            from serial_motors import SerialMotors
            motors = SerialMotors()
            if not motors.available:
                print("[motors] unavailable -- visual-only")
                motors = None
        except Exception as e:
            print(f"[motors] init failed: {e}")
            motors = None

    state = {
        "click": None,
        "paused": False,
        "force_calib": False,
        "force_replan": False,
        "tesla": args.tesla,
    }

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    win = "path planner"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    cached_grid = None
    cached_origin = None
    cached_path = []
    last_status = "idle"
    last_calib_frame = -999

    frame_count = 0
    start = time.monotonic()
    fps_t0 = time.monotonic()
    fps_n = 0
    fps = 0.0

    print()
    print(f"Default goal: forward {args.goal_forward:.0f} mm, "
          f"lateral {args.goal_lateral:+.0f} mm")
    print(f"Robot radius: {planner.robot_radius_mm:.0f} mm "
          f"({'OFF' if args.no_inflate else 'ON'})")
    print(f"Floor alpha:  {args.floor_alpha:.2f}")
    print(f"Render mode:  {'Tesla gradient' if state['tesla'] else 'sky blue'}")
    print("Controls: click=goal  c=clear  p=pause  r=recalibrate  m=mode  q=quit")
    print(f"Motors:   {'ENABLED' if motors else 'visual-only'}")
    print()

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

            # ---- Click -> goal ----
            if state["click"] is not None:
                px, py = state["click"]
                state["click"] = None
                cell = planner.set_goal_pixel(
                    px, py, depth, config.FRAME_W, config.FRAME_H)
                if cell is not None:
                    print(f"[goal] pixel({px},{py}) -> cell {cell}  "
                          f"world(fwd={planner.goal_world[0]:.0f}mm, "
                          f"lat={planner.goal_world[1]:+.0f}mm)")
                    state["force_replan"] = True
                else:
                    print(f"[goal] pixel({px},{py}) has no valid depth")

            # ---- Ground plane self-calibration ----
            do_calib = (state["force_calib"] or
                        frame_count - last_calib_frame >=
                        args.calibrate_every)
            if do_calib and depth is not None:
                state["force_calib"] = False
                last_calib_frame = frame_count
                est = planner.estimate_ground_plane(depth)
                if est is not None:
                    H_new, tilt_new = est
                    planner.cam_height_mm = (0.9 * planner.cam_height_mm +
                                             0.1 * H_new)
                    planner.cam_tilt_deg = (0.9 * planner.cam_tilt_deg +
                                            0.1 * tilt_new)
                    print(f"[calib] H={planner.cam_height_mm:.0f}mm "
                          f"tilt={planner.cam_tilt_deg:.1f}°")

            # ---- Plan ----
            # Rebuild the grid every frame (cheap now that build_grid is
            # vectorized). Replan when:
            #   - the timer fires
            #   - the current path is blocked by a new obstacle
            #   - the user just set a new goal
            #   - we have no path and a goal exists
            if not state["paused"]:
                grid, origin = planner.build_grid(depth)
                if grid is not None:
                    if not args.no_inflate:
                        grid = planner.inflate_obstacles(grid)
                    cached_grid = grid
                    cached_origin = origin

                    timer_fire = (frame_count % args.plan_every == 0)
                    blocked = planner.is_path_blocked(
                        grid, cached_path, lookahead=6)
                    need_plan = (state["force_replan"] or timer_fire or
                                 blocked or
                                 (not cached_path and
                                  planner.goal_world is not None))
                    if need_plan and planner.goal_world is not None:
                        cached_path = planner.plan(grid, origin)
                        state["force_replan"] = False

            # ---- Drive ----
            if motors is not None and cached_path:
                last_status = planner.steer(
                    motors, cached_path,
                    follow_speed=args.follow_speed,
                    turn_speed=args.turn_speed)
            elif motors is not None:
                motors.stop()
                last_status = "no_path"

            # ---- Draw ----
            disp = frame.copy()

            # AR overlay (gray floor + blue path)
            if state["tesla"]:
                disp = planner.draw_path_on_frame(disp, cached_path)
            else:
                disp = planner.render_ar_overlay(
                    disp, depth, cached_path,
                    floor_gray_alpha=args.floor_alpha,
                    sky_blue=(235, 206, 135),
                    path_thickness=14,
                )

            # Depth inset (bottom-left)
            if depth is not None:
                dnorm = cv2.normalize(depth, None, 0, 255,
                                      cv2.NORM_MINMAX).astype(np.uint8)
                dvis = cv2.applyColorMap(dnorm, cv2.COLORMAP_JET)
                dvis = cv2.resize(dvis, (160, 120))
                y0 = disp.shape[0] - 130
                x0 = 10
                disp[y0:y0 + 120, x0:x0 + 160] = dvis
                cv2.rectangle(disp, (x0, y0), (x0 + 160, y0 + 120),
                              (200, 200, 200), 1)
                cv2.putText(disp, "depth", (x0 + 4, y0 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1)

            # Top-down map
            disp = planner.draw_overlay(disp, cached_grid, cached_path)

            # Side-view point cloud
            disp = planner.draw_pointcloud_panel(disp, depth)

            # Goal text
            if planner.goal_world is not None:
                gfwd, glat = planner.goal_world
                cv2.putText(disp,
                            f"goal: fwd {gfwd:.0f}mm  lat {glat:+.0f}mm",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 255), 2)

            # Camera state
            cv2.putText(disp,
                        f"H={planner.cam_height_mm:.0f}mm "
                        f"tilt={planner.cam_tilt_deg:.1f}°",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (180, 180, 180), 1)

            # Status
            state_txt = "PAUSED" if state["paused"] else "PLANNING"
            mode_txt = "tesla" if state["tesla"] else "sky blue"
            cv2.putText(disp,
                        f"{state_txt}  [{mode_txt}]  "
                        f"path={len(cached_path)} cells",
                        (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)
            cv2.putText(disp, f"status: {last_status}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)

            # FPS
            fps_n += 1
            now = time.monotonic()
            if now - fps_t0 >= 1.0:
                fps = fps_n / (now - fps_t0)
                fps_n = 0
                fps_t0 = now
            cv2.putText(disp, f"{fps:.1f} fps",
                        (config.FRAME_W - 90, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1)

            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                planner.goal_world = None
                cached_path = []
                print("[goal] cleared")
            elif key == ord('p'):
                state["paused"] = not state["paused"]
                print(f"[planner] "
                      f"{'paused' if state['paused'] else 'running'}")
            elif key == ord('r'):
                state["force_calib"] = True
                print("[calib] requested")
            elif key == ord('m'):
                state["tesla"] = not state["tesla"]
                print(f"[render] mode = "
                      f"{'tesla gradient' if state['tesla'] else 'sky blue'}")

    except KeyboardInterrupt:
        print("\nInterrupted.")

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
    print(f"Frames: {frame_count}")
    print(f"Final: H={planner.cam_height_mm:.0f}mm "
          f"tilt={planner.cam_tilt_deg:.1f}°")
    print(f"Goal:  {planner.goal_world}")
    print("=" * 60)


if __name__ == "__main__":
    main()
