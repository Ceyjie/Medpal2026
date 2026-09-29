"""
path_planner.py -- depth-driven occupancy grid + A* planner + AR overlay.

Standalone. No face recognition, no Coral, no classifier.

Bugs fixed relative to the previous version:
  1. Ground-height sign: y_world = cam_height_mm - y_rot (was + y_rot).
     This was marking the floor as obstacles in build_grid, coloring
     floor pixels red in the point-cloud panel, and producing a wrong
     floor mask in render_ar_overlay.
  2. build_grid now applies a floor band (0..floor_band_mm) so floor
     pixels are never classified as obstacles.
  3. _project_ground_to_pixel: both x_cam and y_cam sign-corrected.
     Left/right mirror and vertical flip of the AR path overlay fixed.
  4. estimate_ground_plane: added the cos(phi) term to the fit
     (radial vs Z-depth model mismatch).
  5. build_grid is now fully vectorized (was a Python double loop).

Extensions:
  - String-pulling path smoothing (removes unnecessary waypoints
    while preserving line-of-sight).
  - is_path_blocked() helper for immediate replanning on dynamic
    obstacle appearance.

Typical use:
    planner = PathPlanner()
    planner.load_calibration("camera_calib.yml")
    grid, origin = planner.build_grid(depth_mm)
    grid = planner.inflate_obstacles(grid)
    planner.set_goal_world(forward_mm=1500, lateral_mm=0)
    path = planner.plan(grid, origin)

    # Each subsequent frame:
    if planner.is_path_blocked(grid, path):
        path = planner.plan(grid, origin)

    frame = planner.render_ar_overlay(frame, depth_mm, path)
    frame = planner.draw_overlay(frame, grid, path)
    frame = planner.draw_pointcloud_panel(frame, depth_mm)
"""

import os
import math
import heapq
import numpy as np
import cv2

try:
    import config
except ImportError:
    config = None


# Colors (BGR)
TESLA_GLOW = (255, 80, 0)
TESLA_MID = (255, 140, 20)
TESLA_CORE = (255, 220, 120)
TESLA_STAR = (0, 0, 255)
SKY_BLUE = (235, 206, 135)
FLOOR_GRAY = (128, 128, 128)


class PathPlanner:
    def __init__(self,
                 cam_height_mm=200.0,
                 cam_tilt_deg=15.0,
                 cam_hfov_deg=60.0,
                 grid_size=60,
                 res_mm=50.0,
                 min_valid_mm=55.0,
                 max_valid_mm=3000.0,
                 max_height_mm=1500.0,
                 floor_band_mm=80.0,
                 min_forward_mm=150.0,
                 step=2,
                 robot_radius_mm=None):
        self.cam_height_mm = cam_height_mm
        self.cam_tilt_deg = cam_tilt_deg
        self.cam_hfov_deg = cam_hfov_deg
        self.grid_size = grid_size
        self.res_mm = res_mm
        self.min_valid_mm = min_valid_mm
        self.max_valid_mm = max_valid_mm
        self.max_height_mm = max_height_mm
        # Anything below floor_band_mm above the floor is treated as floor,
        # not obstacle. Accounts for sensor noise + small tilt errors.
        self.floor_band_mm = floor_band_mm
        # Ignore depth closer than this (robot's own shadow / near clipping)
        self.min_forward_mm = min_forward_mm
        self.step = step
        if robot_radius_mm is None:
            robot_radius_mm = (getattr(config, "ROBOT_RADIUS_MM", 200)
                               if config else 200)
        self.robot_radius_mm = robot_radius_mm

        self.origin = (grid_size // 2, grid_size // 2)
        self.goal_world = None

        # Provisional intrinsics (overwritten by load_calibration)
        frame_w = getattr(config, "FRAME_W", 640) if config else 640
        frame_h = getattr(config, "FRAME_H", 480) if config else 480
        hfov_rad = math.radians(self.cam_hfov_deg)
        self.fx = (frame_w / 2.0) / math.tan(hfov_rad / 2.0)
        self.fy = self.fx
        self.cx = frame_w / 2.0
        self.cy = frame_h / 2.0

    # ==================================================================
    # Calibration
    # ==================================================================
    def load_calibration(self, path):
        """Load camera intrinsics from a YAML file. Falls back to values
        derived from config HFOV if the file doesn't exist."""
        if not path or not os.path.exists(path):
            frame_w = getattr(config, "FRAME_W", 640) if config else 640
            frame_h = getattr(config, "FRAME_H", 480) if config else 480
            hfov_rad = math.radians(self.cam_hfov_deg)
            self.fx = (frame_w / 2.0) / math.tan(hfov_rad / 2.0)
            self.fy = self.fx
            self.cx = frame_w / 2.0
            self.cy = frame_h / 2.0
            print(f"[planner] intrinsics from config: "
                  f"fx={self.fx:.1f} fy={self.fy:.1f} "
                  f"cx={self.cx:.1f} cy={self.cy:.1f}")
            return

        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        try:
            self.fx = float(fs.getNode("fx").real())
            self.fy = float(fs.getNode("fy").real())
            self.cx = float(fs.getNode("cx").real())
            self.cy = float(fs.getNode("cy").real())
        except Exception as e:
            print(f"[planner] failed to read {path}: {e}; using config")
            fs.release()
            self.load_calibration(None)
            return
        fs.release()
        print(f"[planner] intrinsics from {path}: "
              f"fx={self.fx:.1f} fy={self.fy:.1f} "
              f"cx={self.cx:.1f} cy={self.cy:.1f}")

    # ==================================================================
    # Ground plane estimation (FIX: cos(phi) term)
    # ==================================================================
    def estimate_ground_plane(self, depth_mm):
        """
        Estimate camera height and downward tilt from the depth image.

        Fits z(y) = H * cos(phi) / sin(tilt + phi), where phi is the
        angle of the pixel row below the optical axis. Uses medians of
        the bottom portion of the depth image.

        Returns (height_mm, tilt_deg) or None if the fit fails.
        """
        if depth_mm is None:
            return None
        h, w = depth_mm.shape[:2]
        cy = self.cy
        fy = self.fy if self.fy > 0 else h / 2.0

        rows = []
        depths = []
        for y in range(int(h * 0.6), h, max(1, h // 40)):
            row = depth_mm[y, :]
            valid = row[(row > self.min_valid_mm) &
                        (row < self.max_valid_mm)]
            if valid.size < w * 0.15:
                continue
            rows.append(y)
            depths.append(float(np.median(valid)))

        if len(rows) < 4:
            return None

        rows = np.array(rows, dtype=np.float32)
        depths = np.array(depths, dtype=np.float32)
        phi = np.arctan((rows - cy) / fy)

        # precompute cos(phi) outside the grid search since it doesn't
        # depend on tilt
        cos_phi = np.cos(phi)

        best = (None, None, float("inf"))
        for tilt_deg in np.linspace(3.0, 30.0, 55):
            tilt = math.radians(tilt_deg)
            denom = np.sin(tilt + phi)
            # protect against pathological values near phi = -tilt
            denom = np.where(np.abs(denom) < 1e-3, 1e-3, denom)
            pred = cos_phi / denom
            H_est = float((depths * pred).sum() /
                          (pred * pred).sum())
            err = float(((depths - H_est * pred) ** 2).mean())
            if err < best[2]:
                best = (H_est, tilt_deg, err)

        if best[0] is None:
            return None
        H_est, tilt_deg, _ = best
        if not (50.0 < H_est < 500.0):
            return None
        return (float(H_est), float(tilt_deg))

    # ==================================================================
    # Occupancy grid (FIX: sign, floor cutoff, vectorized)
    # ==================================================================
    def build_grid(self, depth_mm):
        """
        Project a depth frame into a top-down occupancy grid.

        Returns (grid, origin) or (None, None). Grid values:
            0   = free
            1   = occupied
            255 = unknown
        """
        if depth_mm is None:
            return None, None

        dh, dw = depth_mm.shape[:2]

        # Intrinsics at depth resolution
        hfov = math.radians(self.cam_hfov_deg)
        vfov = hfov * (dh / dw)
        fx = (dw / 2.0) / math.tan(hfov / 2.0)
        fy = (dh / 2.0) / math.tan(vfov / 2.0)
        cx = dw / 2.0
        cy = dh / 2.0

        tilt = math.radians(self.cam_tilt_deg)
        c_t = math.cos(tilt)
        s_t = math.sin(tilt)
        H = self.cam_height_mm

        # Vectorized back-projection
        ys, xs = np.mgrid[0:dh, 0:dw].astype(np.float32)
        z = depth_mm.astype(np.float32)
        valid = (z >= self.min_valid_mm) & (z <= self.max_valid_mm)
        z_safe = np.where(valid, z, 1.0)

        x_cam = (xs - cx) * z_safe / fx
        y_cam = (ys - cy) * z_safe / fy
        z_cam = z_safe

        # Un-tilt to robot-level frame.
        #   y_rot =  cos(t) * y_cam + sin(t) * z_cam
        #   z_rot = -sin(t) * y_cam + cos(t) * z_cam
        #   y_world = cam_height_mm - y_rot   (FIX: sign)
        y_rot = c_t * y_cam + s_t * z_cam
        z_rot = -s_t * y_cam + c_t * z_cam

        y_world = H - y_rot
        forward = z_rot
        lateral = x_cam

        # FIX: floor band. Anything within [0, floor_band_mm] above the
        # floor is treated as floor, not obstacle. Anything higher, up
        # to max_height_mm, is an obstacle.
        is_obstacle = (y_world > self.floor_band_mm) & \
                      (y_world <= self.max_height_mm)
        fwd_ok = (forward > self.min_forward_mm) & \
                 (forward <= self.max_valid_mm)
        obstacle = valid & is_obstacle & fwd_ok

        # Convert to grid cell indices
        gs = self.grid_size
        cols = (self.origin[0] + lateral / self.res_mm).astype(np.int32)
        rows = (self.origin[1] - forward / self.res_mm).astype(np.int32)
        in_bounds = ((cols >= 0) & (cols < gs) &
                     (rows >= 0) & (rows < gs))
        mark = obstacle & in_bounds

        grid = np.full((gs, gs), 255, dtype=np.uint8)
        grid[rows[mark], cols[mark]] = 1

        # Free zone around the robot origin
        r0, c0 = self.origin
        grid[max(0, r0 - 2):r0 + 3, max(0, c0 - 2):c0 + 3] = 0

        return grid, self.origin

    def inflate_obstacles(self, grid, robot_radius_mm=None):
        """Inflate occupied cells by the robot radius (configuration
        space). A* then can't plan through gaps narrower than the
        chassis."""
        if grid is None:
            return grid
        radius_mm = robot_radius_mm if robot_radius_mm is not None \
            else self.robot_radius_mm
        radius_cells = max(1, int(round(radius_mm / self.res_mm)))

        diameter = 2 * radius_cells + 1
        yy, xx = np.mgrid[:diameter, :diameter]
        disk = (((yy - radius_cells) ** 2 +
                 (xx - radius_cells) ** 2) <= radius_cells ** 2
                ).astype(np.uint8)

        occupied = (grid == 1).astype(np.uint8)
        inflated = cv2.dilate(occupied, disk, iterations=1).astype(bool)

        out = grid.copy()
        out[inflated] = 1

        # Restore free zone around robot origin
        r0, c0 = self.origin
        pad = radius_cells + 2
        y0 = max(0, r0 - pad); y1 = min(self.grid_size, r0 + pad + 1)
        x0 = max(0, c0 - pad); x1 = min(self.grid_size, c0 + pad + 1)
        out[y0:y1, x0:x1] = 0
        return out

    # ==================================================================
    # Goal setting
    # ==================================================================
    def set_goal_world(self, forward_mm, lateral_mm):
        self.goal_world = (float(forward_mm), float(lateral_mm))
        return self._world_to_grid(self.goal_world)

    def set_goal_pixel(self, px, py, depth_mm, frame_w, frame_h):
        if depth_mm is None:
            return None
        dh, dw = depth_mm.shape[:2]
        sx = dw / frame_w
        sy = dh / frame_h
        dx = int(px * sx)
        dy = int(py * sy)
        if not (0 <= dx < dw and 0 <= dy < dh):
            return None
        z = float(depth_mm[dy, dx])
        if z < self.min_valid_mm or z > self.max_valid_mm:
            return None

        cx_img = dw / 2.0
        cy_img = dh / 2.0
        hfov_rad = math.radians(self.cam_hfov_deg)
        vfov_rad = hfov_rad * (dh / dw)
        fx = (dw / 2.0) / math.tan(hfov_rad / 2.0)
        fy = (dh / 2.0) / math.tan(vfov_rad / 2.0)

        x_cam = (dx - cx_img) * z / fx
        y_cam = (dy - cy_img) * z / fy
        z_cam = z

        tilt_rad = math.radians(self.cam_tilt_deg)
        c_t = math.cos(tilt_rad)
        s_t = math.sin(tilt_rad)
        z_rot = -s_t * y_cam + c_t * z_cam
        forward_mm = z_rot
        lateral_mm = x_cam
        return self.set_goal_world(forward_mm, lateral_mm)

    def set_goal_from_bbox(self, box, depth_mm, frame_w, frame_h):
        if depth_mm is None or box is None:
            return None
        x, y, w, h = box
        px = x + w // 2
        py = y + h
        return self.set_goal_pixel(px, py, depth_mm, frame_w, frame_h)

    def _world_to_grid(self, world):
        forward_mm, lateral_mm = world
        gs = self.grid_size
        col = self.origin[0] + int(lateral_mm / self.res_mm)
        row = self.origin[1] - int(forward_mm / self.res_mm)
        col = max(0, min(gs - 1, col))
        row = max(0, min(gs - 1, row))
        return (row, col)

    # ==================================================================
    # A* with string-pulling
    # ==================================================================
    def plan(self, grid, origin):
        if grid is None or self.goal_world is None:
            return []
        goal = self._world_to_grid(self.goal_world)
        raw = self.astar(grid, origin, goal)
        if len(raw) < 3:
            return raw
        return self._string_pull(grid, raw)

    @staticmethod
    def astar(grid, start, goal):
        H, W = grid.shape
        sr, sc = start
        gr, gc = goal
        if not (0 <= gr < H and 0 <= gc < W):
            return []
        if grid[gr, gc] == 1:
            found = False
            for radius in range(1, 8):
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        r, c = gr + dr, gc + dc
                        if 0 <= r < H and 0 <= c < W and grid[r, c] != 1:
                            gr, gc = r, c
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                return []

        def h(r, c):
            return abs(r - gr) + abs(c - gc)

        open_set = [(h(sr, sc), 0.0, (sr, sc))]
        came_from = {}
        g_score = {(sr, sc): 0.0}
        visited = set()
        while open_set:
            _, g, current = heapq.heappop(open_set)
            if current == (gr, gc):
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            if current in visited:
                continue
            visited.add(current)
            cr, cc = current
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if grid[nr, nc] == 1:
                    continue
                step_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
                tentative = g + step_cost
                if (nr, nc) not in g_score or tentative < g_score[(nr, nc)]:
                    g_score[(nr, nc)] = tentative
                    came_from[(nr, nc)] = current
                    heapq.heappush(
                        open_set,
                        (tentative + h(nr, nc), tentative, (nr, nc)))
        return []

    @staticmethod
    def _los_clear(grid, a, b):
        """Bresenham line-of-sight check: True if the straight segment
        from cell `a` to cell `b` does not pass through any occupied
        cell (grid == 1)."""
        r0, c0 = a
        r1, c1 = b
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc
        r, c = r0, c0
        H, W = grid.shape
        while True:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if grid[r, c] == 1:
                return False
            if (r, c) == (r1, c1):
                return True
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc

    def _string_pull(self, grid, path):
        """
        Remove intermediate waypoints when a straight segment between
        the two endpoints has clear line of sight. Produces a minimal
        polyline that a spline fit can then smooth.
        """
        if not path or len(path) < 3:
            return path
        out = [path[0]]
        i = 0
        n = len(path)
        while i < n - 1:
            j = n - 1
            while j > i + 1:
                if self._los_clear(grid, path[i], path[j]):
                    break
                j -= 1
            out.append(path[j])
            i = j
        return out

    def is_path_blocked(self, grid, path, lookahead=6):
        """True if any of the first `lookahead` cells of `path` is now
        occupied. Use this each frame to trigger immediate replanning
        when a dynamic obstacle appears mid-path."""
        if grid is None or not path:
            return False
        H, W = grid.shape
        n = min(lookahead, len(path))
        for i in range(n):
            r, c = path[i]
            if 0 <= r < H and 0 <= c < W and grid[r, c] == 1:
                return True
        return False

    # ==================================================================
    # Path -> steering
    # ==================================================================
    def lookahead(self, path, cells=6):
        if not path or len(path) < 2:
            return None
        idx = min(cells, len(path) - 1)
        tr, tc = path[idx]
        lateral_mm = (tc - self.origin[0]) * self.res_mm
        forward_mm = (self.origin[1] - tr) * self.res_mm
        return lateral_mm, forward_mm

    def steer(self, motors, path,
              follow_speed=60, turn_speed=40,
              turn_deadband_mm=60, min_fwd_mm=100):
        result = self.lookahead(path)
        if result is None:
            motors.stop()
            return "no_path"
        lat_mm, fwd_mm = result
        if fwd_mm < min_fwd_mm:
            motors.stop()
            return f"arrived ({fwd_mm:.0f}mm)"
        if lat_mm < -turn_deadband_mm:
            motors.set_speed(turn_speed)
            motors.turn_left()
            return f"turn_L ({lat_mm:+.0f}mm)"
        if lat_mm > turn_deadband_mm:
            motors.set_speed(turn_speed)
            motors.turn_right()
            return f"turn_R ({lat_mm:+.0f}mm)"
        motors.set_speed(follow_speed)
        motors.forward()
        return f"fwd ({fwd_mm:.0f}mm)"

    # ==================================================================
    # Projection helpers (FIX: sign flips)
    # ==================================================================
    def _project_ground_to_pixel(self, forward_mm, lateral_mm,
                                 frame_w, frame_h):
        """
        Project a point on the floor (forward_mm ahead of the robot,
        lateral_mm to its right) to pixel coordinates.

        Correct signs: for a point directly ahead on the floor, y_cam
        is negative (above image center), so the ground appears in the
        upper part of the frame at close range and near the horizon at
        far range.
        """
        tilt_rad = math.radians(self.cam_tilt_deg)
        c_t = math.cos(tilt_rad)
        s_t = math.sin(tilt_rad)
        H = self.cam_height_mm

        # World point relative to camera (X = lateral, Y = up, Z = fwd)
        dx = forward_mm   # Z in world
        dy = lateral_mm   # X in world
        dz = -H           # Y in world (below camera)

        # Projection onto camera axes.
        #   x_cam = dy
        #   y_cam = -(dz * c_t + dx * s_t)
        #   z_cam =  dx * c_t - dz * s_t
        z_cam = dx * c_t - dz * s_t
        x_cam = dy
        y_cam = -(dz * c_t + dx * s_t)

        if z_cam <= 50.0:
            return None
        u = self.cx + self.fx * x_cam / z_cam
        v = self.cy + self.fy * y_cam / z_cam
        return (u, v)

    def _smooth_path_ground(self, path, samples=60):
        """
        Take the polyline of grid cells and produce a smooth list of
        (forward_mm, lateral_mm) points suitable for projection onto
        the ground plane. Uses a quadratic fit in forward for length
        >= 3, otherwise returns the raw points.
        """
        if not path or len(path) < 2:
            return []
        points = []
        for (r, c) in path:
            forward_mm = (self.origin[1] - r) * self.res_mm
            lateral_mm = (c - self.origin[0]) * self.res_mm
            if forward_mm < 150:
                continue
            points.append((forward_mm, lateral_mm))
        if len(points) < 2:
            return points

        fwd_arr = np.array([p[0] for p in points], dtype=np.float32)
        lat_arr = np.array([p[1] for p in points], dtype=np.float32)
        if len(points) >= 3:
            try:
                coeffs = np.polyfit(fwd_arr, lat_arr, 2)
                dense_f = np.linspace(fwd_arr.min(), fwd_arr.max(),
                                      samples)
                dense_l = np.polyval(coeffs, dense_f)
                return list(zip(dense_f.tolist(), dense_l.tolist()))
            except np.linalg.LinAlgError:
                pass
        return points

    # ==================================================================
    # Floor mask (FIX: sign)
    # ==================================================================
    def _compute_floor_mask(self, depth_mm, frame_w, frame_h):
        """
        Soft floor mask at frame resolution (float32, 0..1).

        A pixel is "floor" if its un-tilted height is near zero.
        """
        if depth_mm is None or depth_mm.size == 0:
            return None
        dh, dw = depth_mm.shape[:2]
        hfov = math.radians(self.cam_hfov_deg)
        vfov = hfov * (dh / dw)
        fx = (dw / 2.0) / math.tan(hfov / 2.0)
        fy = (dh / 2.0) / math.tan(vfov / 2.0)
        cx = dw / 2.0
        cy = dh / 2.0

        tilt = math.radians(self.cam_tilt_deg)
        c_t = math.cos(tilt)
        s_t = math.sin(tilt)
        H = self.cam_height_mm

        ys, xs = np.mgrid[0:dh, 0:dw].astype(np.float32)
        z = depth_mm.astype(np.float32)
        valid = (z >= self.min_valid_mm) & (z <= self.max_valid_mm)
        z_safe = np.where(valid, z, 1.0)

        x_cam = (xs - cx) * z_safe / fx
        y_cam = (ys - cy) * z_safe / fy
        z_cam = z_safe

        y_rot = c_t * y_cam + s_t * z_cam
        # FIX: correct sign
        y_world = H - y_rot

        floor_mask = valid & (y_world > -self.floor_band_mm) & \
                              (y_world < self.floor_band_mm)

        floor_u8 = floor_mask.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        floor_u8 = cv2.morphologyEx(floor_u8, cv2.MORPH_OPEN, kernel)
        floor_u8 = cv2.morphologyEx(floor_u8, cv2.MORPH_CLOSE, kernel)

        floor_full = cv2.resize(floor_u8, (frame_w, frame_h),
                                interpolation=cv2.INTER_NEAREST)
        floor_full = cv2.GaussianBlur(floor_full, (7, 7), 0)
        return floor_full.astype(np.float32) / 255.0

    # ==================================================================
    # AR overlay
    # ==================================================================
    def render_ar_overlay(self, frame, depth_mm, path,
                          floor_gray_alpha=0.55,
                          floor_tint=FLOOR_GRAY,
                          sky_blue=SKY_BLUE,
                          path_thickness=14,
                          path_glow=True):
        out = frame.copy()
        h, w = frame.shape[:2]

        # Gray floor shading
        floor_alpha = self._compute_floor_mask(depth_mm, w, h)
        if floor_alpha is not None:
            tinted = np.full_like(out, floor_tint, dtype=np.uint8)
            alpha = (floor_alpha * floor_gray_alpha)[..., None]
            out = (out.astype(np.float32) * (1.0 - alpha)
                   + tinted.astype(np.float32) * alpha
                   ).astype(np.uint8)

        # Sky-blue path
        if path and len(path) >= 2:
            ground_pts = self._smooth_path_ground(path, samples=60)
            if len(ground_pts) >= 2:
                pixel_pts = []
                for (fwd, lat) in ground_pts:
                    p = self._project_ground_to_pixel(fwd, lat, w, h)
                    if p is None:
                        continue
                    u, v = p
                    if not (-w <= u <= 2 * w and -h <= v <= 2 * h):
                        continue
                    pixel_pts.append((int(u), int(v)))

                if len(pixel_pts) >= 2:
                    pts = np.array(pixel_pts, dtype=np.int32)

                    if path_glow:
                        overlay = out.copy()
                        cv2.polylines(overlay, [pts], False,
                                      sky_blue,
                                      thickness=path_thickness * 2,
                                      lineType=cv2.LINE_AA)
                        out = cv2.addWeighted(overlay, 0.30, out, 0.70, 0)

                    overlay = out.copy()
                    cv2.polylines(overlay, [pts], False,
                                  sky_blue,
                                  thickness=path_thickness,
                                  lineType=cv2.LINE_AA)
                    out = cv2.addWeighted(overlay, 0.90, out, 0.10, 0)

                    core = tuple(min(255, c + 40) for c in sky_blue)
                    overlay = out.copy()
                    cv2.polylines(overlay, [pts], False, core,
                                  thickness=max(2, path_thickness // 4),
                                  lineType=cv2.LINE_AA)
                    out = cv2.addWeighted(overlay, 0.95, out, 0.05, 0)

        # Goal star
        if self.goal_world is not None:
            p = self._project_ground_to_pixel(
                self.goal_world[0], self.goal_world[1], w, h)
            if p is not None:
                u, v = p
                if 0 <= u < w and 0 <= v < h:
                    cv2.drawMarker(out, (int(u), int(v)), sky_blue,
                                   cv2.MARKER_STAR, 18, 3, cv2.LINE_AA)
                    cv2.circle(out, (int(u), int(v)), 12,
                               sky_blue, 2, cv2.LINE_AA)

        return out

    # ==================================================================
    # Legacy Tesla gradient overlay
    # ==================================================================
    def draw_path_on_frame(self, frame, path):
        if not path or len(path) < 2:
            return frame
        h, w = frame.shape[:2]
        ground_pts = self._smooth_path_ground(path, samples=60)
        if len(ground_pts) < 2:
            return frame

        pixel_pts = []
        for (fwd, lat) in ground_pts:
            p = self._project_ground_to_pixel(fwd, lat, w, h)
            if p is None:
                continue
            u, v = p
            if not (-w <= u <= 2 * w and -h <= v <= 2 * h):
                continue
            pixel_pts.append((int(u), int(v)))
        if len(pixel_pts) < 2:
            return frame

        pts_array = np.array(pixel_pts, dtype=np.int32)

        overlay = frame.copy()
        cv2.polylines(overlay, [pts_array], False,
                      TESLA_GLOW, thickness=22, lineType=cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        overlay = frame.copy()
        cv2.polylines(overlay, [pts_array], False,
                      TESLA_MID, thickness=11, lineType=cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

        overlay = frame.copy()
        cv2.polylines(overlay, [pts_array], False,
                      TESLA_CORE, thickness=3, lineType=cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, 0.95, frame, 0.05, 0)

        if self.goal_world is not None:
            p = self._project_ground_to_pixel(
                self.goal_world[0], self.goal_world[1], w, h)
            if p is not None:
                u, v = p
                if 0 <= u < w and 0 <= v < h:
                    cv2.drawMarker(frame, (int(u), int(v)), TESLA_STAR,
                                   cv2.MARKER_STAR, 18, 3, cv2.LINE_AA)
                    cv2.circle(frame, (int(u), int(v)), 12,
                               TESLA_STAR, 2, cv2.LINE_AA)
        return frame

    # ==================================================================
    # Top-down map panel
    # ==================================================================
    def draw_overlay(self, frame, grid, path,
                     overlay_w=200, overlay_h=200):
        if grid is None:
            return frame
        gs = grid.shape[0]
        cell_px = max(2, overlay_w // gs)

        cell_colors = np.full((gs, gs, 3), (25, 25, 25), dtype=np.uint8)
        cell_colors[grid == 1] = (0, 0, 180)
        cell_colors[grid == 0] = (50, 50, 50)
        r0, c0 = self.origin
        y0 = max(0, r0 - 2); y1 = r0 + 3
        x0 = max(0, c0 - 2); x1 = c0 + 3
        cell_colors[y0:y1, x0:x1] = (50, 50, 50)

        panel = cv2.resize(cell_colors, (overlay_w, overlay_h),
                           interpolation=cv2.INTER_NEAREST)

        ox = self.origin[0] * cell_px + cell_px // 2
        oy = self.origin[1] * cell_px + cell_px // 2
        cv2.circle(panel, (ox, oy), 4, (0, 255, 0), -1)

        if self.goal_world is not None:
            g = self._world_to_grid(self.goal_world)
            gx = g[1] * cell_px + cell_px // 2
            gy = g[0] * cell_px + cell_px // 2
            cv2.drawMarker(panel, (gx, gy), SKY_BLUE,
                           cv2.MARKER_STAR, 12, 2)

        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            p1 = (c1 * cell_px + cell_px // 2,
                  r1 * cell_px + cell_px // 2)
            p2 = (c2 * cell_px + cell_px // 2,
                  r2 * cell_px + cell_px // 2)
            cv2.line(panel, p1, p2, SKY_BLUE, 2, cv2.LINE_AA)

        fh, fw = frame.shape[:2]
        x_off = fw - overlay_w - 10
        y_off = 10
        frame[y_off:y_off + overlay_h, x_off:x_off + overlay_w] = panel
        return frame

    # ==================================================================
    # Side-view point cloud (FIX: sign)
    # ==================================================================
    def draw_pointcloud_panel(self, frame, depth_mm, panel_size=180):
        if depth_mm is None:
            return frame
        h, w = depth_mm.shape[:2]
        cx_img = w / 2.0
        cy_img = h / 2.0
        hfov_rad = math.radians(self.cam_hfov_deg)
        vfov_rad = hfov_rad * (h / w)
        fx = (w / 2.0) / math.tan(hfov_rad / 2.0)
        fy = (h / 2.0) / math.tan(vfov_rad / 2.0)
        tilt_rad = math.radians(self.cam_tilt_deg)
        c_t = math.cos(tilt_rad)
        s_t = math.sin(tilt_rad)
        H = self.cam_height_mm

        panel = np.full((panel_size, panel_size, 3), 20, dtype=np.uint8)
        fwd_max = float(self.max_valid_mm)
        h_min = -H - 200
        h_max = self.max_height_mm + 100

        step = max(1, self.step * 2)
        for y in range(0, h, step):
            for x in range(0, w, step):
                z = float(depth_mm[y, x])
                if z < self.min_valid_mm or z > self.max_valid_mm:
                    continue
                x_cam = (x - cx_img) * z / fx
                y_cam = (y - cy_img) * z / fy
                z_cam = z
                y_rot = c_t * y_cam + s_t * z_cam
                z_rot = -s_t * y_cam + c_t * z_cam
                # FIX: correct sign
                y_world = H - y_rot
                forward = z_rot

                if 0 <= y_world < self.floor_band_mm:
                    color = (0, 200, 0)          # floor: green
                elif (self.floor_band_mm <= y_world < self.max_height_mm):
                    color = (0, 0, 200)          # obstacle: red
                else:
                    color = (80, 80, 80)         # ignored: grey

                px = int(panel_size * forward / fwd_max)
                py = panel_size - 1 - int(
                    panel_size * (y_world - h_min) / (h_max - h_min))
                if 0 <= px < panel_size and 0 <= py < panel_size:
                    panel[py, px] = color

        py0 = panel_size - 1 - int(
            panel_size * (0 - h_min) / (h_max - h_min))
        cv2.circle(panel, (0, py0), 4, (0, 255, 255), -1)

        fh, fw = frame.shape[:2]
        x_off = fw - panel_size - 10
        y_off = fh - panel_size - 10
        if x_off >= 0 and y_off >= 0:
            frame[y_off:y_off + panel_size,
                  x_off:x_off + panel_size] = panel
            cv2.rectangle(frame,
                          (x_off, y_off),
                          (x_off + panel_size, y_off + panel_size),
                          (200, 200, 200), 1)
            cv2.putText(frame, "side view",
                        (x_off + 4, y_off + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1)
        return frame
