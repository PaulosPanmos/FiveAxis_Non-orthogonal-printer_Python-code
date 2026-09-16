# -*- coding: utf-8 -*-
"""
S03_femoral_joint_visualizer.py

PyVista visualizer for the solved femoral-joint benchmark.

The script reads x y z_table B C from the solved IK file, reconstructs the
machine geometry and draws separate traces for the Cura mandrel and dome path.
It is a visual check only and does not modify the solved path.

Machine geometry and file paths are read from S00_femoral_joint_parameters.py.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyvista as pv

import S00_femoral_joint_parameters as P


# ============================================================
# Benchmark files
# ============================================================

BASE_DIR = Path(P.BASE_DIR)
SOLVED_IK_PATH = Path(P.SOLVED_IK_PATH)

# Used to split the trace between mandrel/base and dome sections.
CURA_BASE_CLDATA_PATH = Path(P.CURA_BASE_CLDATA_PATH)

ANIMATION_PATH = BASE_DIR / "03_femoral_joint_visualization.mp4"


# ============================================================
# Animation and playback
# ============================================================

SAVE_ANIMATION = True

ANIMATION_FPS = 30
ANIMATION_FRAME_EVERY = 2

# Playback stride.
VIZ_STRIDE = 10

# Interactive preview frame rate.
VIZ_FPS = 60

# Set True only if the solved IK file stores B/C in degrees.
VIZ_ANGLES_DEG = False


# ============================================================
# Trace display
# ============================================================

VIZ_SHOW_TRACE = True
VIZ_TRACE_EVERY = 1
VIZ_TRACE_AS_TUBE = False
VIZ_TRACE_RADIUS = 0.35

VIZ_DRAW_TRACE_LIVE = True
VIZ_TRACE_REDRAW_EVERY = 5

VIZ_SHOW_DEBUG_FRAMES = False
VIZ_TRIAD_SCALE = 20.0

TRACE_COLOR_CURA = "#D6CFC7"
TRACE_COLOR_SPIRAL = "orange"


# ============================================================
# Visual-only geometry
# ============================================================

# These values affect only the displayed meshes, not the kinematic model.
TOOL_HOLDER_VISUAL_OFFSET_FROM_KINEMATIC_LOCAL = np.array(
    [27.75, 0.0, -15.0],
    dtype=float,
)
BED_VISUAL_OFFSET = np.array([0.0, 0.0, -5.0], dtype=float)
C_ARM_VISUAL_EXTRA_LENGTH = 30.0


# ============================================================
# Coordinate convention
# ============================================================

# Combined CL data are centered around X0 Y0.
XY_ORIGIN = P.XY_ORIGIN

# Must match the IK convention.
ZTABLE_POSITIVE_DOWN = P.ZTABLE_POSITIVE_DOWN


# ============================================================
# Machine geometry
# ============================================================

BASE_SIZE = np.asarray(P.BASE_SIZE, dtype=float)
BED_SIZE = np.asarray(P.BED_SIZE, dtype=float)

FRAME_X = float(P.FRAME_X)
FRAME_Y = float(P.FRAME_Y)
POST_RADIUS = float(P.POST_RADIUS)
BEAM_RADIUS = float(P.BEAM_RADIUS)

GANTRY_Z = float(P.GANTRY_Z)
SPINDLE_HEIGHT = float(P.SPINDLE_HEIGHT)
SPINDLE_RADIUS = float(P.SPINDLE_RADIUS)

CARRIAGE_SIZE = np.asarray(P.CARRIAGE_SIZE, dtype=float)
BED_CENTER_Z0 = float(P.BED_CENTER_Z0)

B_PIVOT_FROM_CARRIAGE_CENTER = np.asarray(
    P.B_PIVOT_FROM_CARRIAGE_CENTER,
    dtype=float,
)

B_BRACKET_ZERO_TILT_DEG = float(P.B_BRACKET_ZERO_TILT_DEG)
TOOL_ZERO_ROT_Y_DEG = float(P.TOOL_ZERO_ROT_Y_DEG)

B_BRACKET_SIZE = np.asarray(P.B_BRACKET_SIZE, dtype=float)
B_BRACKET_LEN = float(P.B_BRACKET_LEN)
B_BRACKET_CENTER_FROM_B_LOCAL = np.asarray(
    P.B_BRACKET_CENTER_FROM_B_LOCAL,
    dtype=float,
)
C_FROM_BRACKET_END_LOCAL = np.asarray(P.C_FROM_BRACKET_END_LOCAL, dtype=float)

C_ARM_SIZE = np.asarray(P.C_ARM_SIZE, dtype=float)
C_ARM_CENTER_FROM_C_LOCAL = np.asarray(P.C_ARM_CENTER_FROM_C_LOCAL, dtype=float)
TOOL_FROM_C_LOCAL = np.asarray(P.TOOL_FROM_C_LOCAL, dtype=float)

TOOL_HOLDER_SIZE = np.asarray(P.TOOL_HOLDER_SIZE, dtype=float)
TOOL_HOLDER_CENTER_FROM_TOOL_LOCAL = np.asarray(
    P.TOOL_HOLDER_CENTER_FROM_TOOL_LOCAL,
    dtype=float,
)
TIP_FROM_TOOL_HOLDER_CENTER_LOCAL = np.asarray(
    P.TIP_FROM_TOOL_HOLDER_CENTER_LOCAL,
    dtype=float,
)

VIZ_AXIS_RADIUS = float(P.VIZ_AXIS_RADIUS)
VIZ_AXIS_B_LEN = float(P.VIZ_AXIS_B_LEN)
VIZ_AXIS_C_LEN = float(P.VIZ_AXIS_C_LEN)
VIZ_TIP_SPHERE_RADIUS = float(P.VIZ_TIP_SPHERE_RADIUS)

# Nozzle cone used in the display.
VIZ_NOZZLE_CONE_HEIGHT = 18.0
VIZ_NOZZLE_CONE_RADIUS = 6.0
VIZ_NOZZLE_CONE_RESOLUTION = 32
VIZ_SHOW_NOZZLE_CONE = True

B_AXIS_SIGN = float(P.B_AXIS_SIGN)
C_AXIS_SIGN = float(P.C_AXIS_SIGN)


# ============================================================
# Camera
# ============================================================

CAMERA_POSITION = P.CAMERA_POSITION


# ============================================================
# Helpers
# ============================================================

def _unit(v):
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n


def Rx(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, ca, -sa],
            [0.0, sa, ca],
        ],
        dtype=float,
    )


def Ry(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array(
        [
            [ca, 0.0, sa],
            [0.0, 1.0, 0.0],
            [-sa, 0.0, ca],
        ],
        dtype=float,
    )


def Rz(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array(
        [
            [ca, -sa, 0.0],
            [sa, ca, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def T_of(R, p):
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(p, dtype=float).reshape(3)
    return T


def load_commands_txt(path: Path) -> np.ndarray:
    rows = []
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Solved IK file not found:\n{path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()

            if not s or s.startswith("#"):
                continue

            parts = s.split()

            if len(parts) >= 5:
                rows.append(
                    [
                        float(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                    ]
                )

    if not rows:
        raise ValueError(f"No numeric rows parsed from {path}")

    return np.asarray(rows, dtype=float)


def count_numeric_rows(path: Path, min_cols: int = 1) -> int:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found:\n{path}")

    count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()

            if not s or s.startswith("#"):
                continue

            parts = s.split()

            if len(parts) >= min_cols:
                count += 1

    return count


def add_axes_triads(plotter, origin, R, scale=VIZ_TRIAD_SCALE, line_width=4):
    origin = np.asarray(origin, dtype=float)
    R = np.asarray(R, dtype=float)

    x_end = origin + scale * R[:, 0]
    y_end = origin + scale * R[:, 1]
    z_end = origin + scale * R[:, 2]

    plotter.add_lines(np.vstack([origin, x_end]), color="red", width=line_width)
    plotter.add_lines(np.vstack([origin, y_end]), color="green", width=line_width)
    plotter.add_lines(np.vstack([origin, z_end]), color="blue", width=line_width)


def make_cylinder_mesh_local_z(radius, length, resolution=24):
    return pv.Cylinder(
        center=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 1.0),
        radius=radius,
        height=length,
        resolution=resolution,
    )


def make_nozzle_cone_mesh_local_tip_at_origin(
    radius=VIZ_NOZZLE_CONE_RADIUS,
    height=VIZ_NOZZLE_CONE_HEIGHT,
    resolution=VIZ_NOZZLE_CONE_RESOLUTION,
):
    """Create a nozzle cone with its tip at the local origin."""
    radius = float(radius)
    height = float(height)
    resolution = int(max(8, resolution))

    theta = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)

    points = [[0.0, 0.0, 0.0]]  # apex at local origin

    for a in theta:
        points.append([
            radius * np.cos(a),
            radius * np.sin(a),
            height,
        ])

    # Optional base center for a closed base.
    base_center_idx = len(points)
    points.append([0.0, 0.0, height])

    faces = []

    # Side triangles.
    for i in range(resolution):
        j = 1 + i
        k = 1 + ((i + 1) % resolution)
        faces.extend([3, 0, j, k])

    # Base triangles.
    for i in range(resolution):
        j = 1 + ((i + 1) % resolution)
        k = 1 + i
        faces.extend([3, base_center_idx, j, k])

    return pv.PolyData(
        np.asarray(points, dtype=float),
        np.asarray(faces, dtype=np.int64),
    )


B_BRACKET_ZERO_ROT = Ry(np.deg2rad(B_BRACKET_ZERO_TILT_DEG))
TOOL_ZERO_ROT_FROM_HOLDER = Ry(np.deg2rad(TOOL_ZERO_ROT_Y_DEG))


# ============================================================
# Visualizer
# ============================================================

class ExplicitFramePrinterViz:
    def __init__(
        self,
        xy_origin=XY_ORIGIN,
        show_trace=VIZ_SHOW_TRACE,
        trace_every=VIZ_TRACE_EVERY,
        trace_as_tube=VIZ_TRACE_AS_TUBE,
        trace_radius=VIZ_TRACE_RADIUS,
        show_debug_frames=VIZ_SHOW_DEBUG_FRAMES,
        draw_trace_live=VIZ_DRAW_TRACE_LIVE,
        trace_redraw_every=VIZ_TRACE_REDRAW_EVERY,
        save_animation=SAVE_ANIMATION,
        animation_path=ANIMATION_PATH,
        animation_fps=ANIMATION_FPS,
        animation_frame_every=ANIMATION_FRAME_EVERY,
    ):
        self.xy_origin = xy_origin
        self.show_trace = bool(show_trace)
        self.trace_every = int(max(1, trace_every))
        self.trace_as_tube = bool(trace_as_tube)
        self.trace_radius = float(trace_radius)
        self.show_debug_frames = bool(show_debug_frames)

        self.draw_trace_live = bool(draw_trace_live)
        self.trace_redraw_every = int(max(1, trace_redraw_every))

        self.save_animation = bool(save_animation)
        self.animation_path = str(animation_path)
        self.animation_fps = int(max(1, animation_fps))
        self.animation_frame_every = int(max(1, animation_frame_every))

        self.base_size = BASE_SIZE.copy()
        self.bed_size = BED_SIZE.copy()
        self.gantry_z = GANTRY_Z
        self.spindle_height = SPINDLE_HEIGHT
        self.carriage_size = CARRIAGE_SIZE.copy()

        self.base_center = np.array(
            [0.0, 0.0, self.base_size[2] / 2.0],
            dtype=float,
        )

        self.z_base_top = float(self.base_size[2])
        self.bed_center_z0 = BED_CENTER_Z0

        self.step_counter = 0

        self.trace_points_bed_cura = []
        self.trace_points_bed_spiral = []

        self.trace_actor_cura = None
        self.trace_actor_spiral = None

        # Real/kinematic bed position.
        # This is used for trace conversion and should not be visually fudged.
        self.bed_pos = np.array([0.0, 0.0, self.bed_center_z0], dtype=float)

        # Visual-only bed position.
        # Only the PyVista bed mesh uses this.
        self.bed_visual_pos = self.bed_pos + BED_VISUAL_OFFSET

        self.plotter = pv.Plotter()
        self.plotter.set_background("white")

        self._build_scene()
        self.plotter.camera_position = CAMERA_POSITION
        self.plotter.show(auto_close=False, interactive=True, interactive_update=True)

        if self.save_animation:
            self.plotter.open_movie(self.animation_path, framerate=self.animation_fps)
            self.plotter.write_frame()

    def _build_scene(self):
        base_mesh = pv.Cube(
            center=tuple(self.base_center),
            x_length=float(self.base_size[0]),
            y_length=float(self.base_size[1]),
            z_length=float(self.base_size[2]),
        )
        self.plotter.add_mesh(base_mesh, color="gray")

        bed_mesh = pv.Cube(
            center=(0.0, 0.0, 0.0),
            x_length=float(self.bed_size[0]),
            y_length=float(self.bed_size[1]),
            z_length=float(self.bed_size[2]),
        )
        self.bed_actor = self.plotter.add_mesh(bed_mesh, color="lightgray")
        self.bed_actor.position = tuple(self.bed_visual_pos)

        spindle_x = 0.0
        spindle_y = +self.base_size[1] / 2.0 - SPINDLE_RADIUS
        spindle_center = np.array(
            [
                spindle_x,
                spindle_y,
                self.z_base_top + self.spindle_height / 2.0,
            ],
            dtype=float,
        )

        spindle_mesh = pv.Cylinder(
            center=tuple(spindle_center),
            direction=(0.0, 0.0, 1.0),
            radius=SPINDLE_RADIUS,
            height=self.spindle_height,
            resolution=24,
        )
        self.plotter.add_mesh(spindle_mesh, color=(0.45, 0.45, 0.45))

        post_h = self.gantry_z - self.z_base_top
        corners = [
            (+FRAME_X / 2, +FRAME_Y / 2),
            (+FRAME_X / 2, -FRAME_Y / 2),
            (-FRAME_X / 2, +FRAME_Y / 2),
            (-FRAME_X / 2, -FRAME_Y / 2),
        ]

        for cx, cy in corners:
            center = (cx, cy, self.z_base_top + post_h / 2.0)
            post = pv.Cylinder(
                center=center,
                direction=(0.0, 0.0, 1.0),
                radius=POST_RADIUS,
                height=post_h,
                resolution=18,
            )
            self.plotter.add_mesh(post, color=(0.35, 0.35, 0.35))

        for sign in (+1, -1):
            y = sign * FRAME_Y / 2
            beam = pv.Cylinder(
                center=(0.0, y, self.gantry_z),
                direction=(1.0, 0.0, 0.0),
                radius=BEAM_RADIUS,
                height=FRAME_X,
                resolution=18,
            )
            self.plotter.add_mesh(beam, color=(0.30, 0.30, 0.30))

        for sign in (+1, -1):
            x = sign * FRAME_X / 2
            beam = pv.Cylinder(
                center=(x, 0.0, self.gantry_z),
                direction=(0.0, 1.0, 0.0),
                radius=BEAM_RADIUS,
                height=FRAME_Y,
                resolution=18,
            )
            self.plotter.add_mesh(beam, color=(0.30, 0.30, 0.30))

        carriage_mesh = pv.Cube(
            center=(0.0, 0.0, 0.0),
            x_length=float(self.carriage_size[0]),
            y_length=float(self.carriage_size[1]),
            z_length=float(self.carriage_size[2]),
        )
        self.carriage_actor = self.plotter.add_mesh(carriage_mesh, color="#F2EFE7")

        b_bracket_mesh = pv.Cube(
            center=(0.0, 0.0, 0.0),
            x_length=float(B_BRACKET_SIZE[0]),
            y_length=float(B_BRACKET_SIZE[1]),
            z_length=float(B_BRACKET_SIZE[2]),
        )
        self.b_bracket_actor = self.plotter.add_mesh(b_bracket_mesh, color="#08648f")

        # Displayed C-arm length.
        c_arm_visual_size = C_ARM_SIZE.copy()
        c_arm_visual_size[0] += float(C_ARM_VISUAL_EXTRA_LENGTH)

        c_arm_mesh = pv.Cube(
            center=(0.0, 0.0, 0.0),
            x_length=float(c_arm_visual_size[0]),
            y_length=float(c_arm_visual_size[1]),
            z_length=float(c_arm_visual_size[2]),
        )
        self.c_arm_actor = self.plotter.add_mesh(c_arm_mesh, color="#36ae43")

        tool_holder_mesh = pv.Cube(
            center=(0.0, 0.0, 0.0),
            x_length=float(TOOL_HOLDER_SIZE[0]),
            y_length=float(TOOL_HOLDER_SIZE[1]),
            z_length=float(TOOL_HOLDER_SIZE[2]),
        )
        self.tool_holder_actor = self.plotter.add_mesh(tool_holder_mesh, color=(0.85, 0.10, 0.10))

        b_axis_mesh = make_cylinder_mesh_local_z(VIZ_AXIS_RADIUS, VIZ_AXIS_B_LEN)
        self.b_axis_actor = self.plotter.add_mesh(b_axis_mesh, color="orange")

        c_axis_mesh = make_cylinder_mesh_local_z(VIZ_AXIS_RADIUS, VIZ_AXIS_C_LEN)
        self.c_axis_actor = self.plotter.add_mesh(c_axis_mesh, color="orange")

        tip_mesh = pv.Sphere(
            radius=VIZ_TIP_SPHERE_RADIUS,
            center=(0.0, 0.0, 0.0),
            theta_resolution=16,
            phi_resolution=16,
        )
        self.tip_actor = self.plotter.add_mesh(tip_mesh, color="black")

        nozzle_cone_mesh = make_nozzle_cone_mesh_local_tip_at_origin()
        self.nozzle_cone_actor = self.plotter.add_mesh(
            nozzle_cone_mesh,
            color="yellow",
        )

        # Set nozzle visibility after adding the mesh.
        self.nozzle_cone_actor.SetVisibility(bool(VIZ_SHOW_NOZZLE_CONE))

        self.plotter.add_axes()

    def _map_xy(self, x, y):
        if self.xy_origin == "corner":
            x -= self.bed_size[0] / 2.0
            y -= self.bed_size[1] / 2.0
        return x, y

    def _world_to_bed(self, p_world):
        return np.asarray(p_world, dtype=float) - self.bed_pos

    def _bed_to_world_points(self, pts_bed):
        pts_bed = np.asarray(pts_bed, dtype=float)
        return pts_bed + self.bed_pos.reshape(1, 3)

    def step(self, x, y, z_table, b_angle, c_angle, is_cura_part: bool = False):
        self.step_counter += 1

        x, y = self._map_xy(float(x), float(y))

        z = float(z_table)

        if not ZTABLE_POSITIVE_DOWN:
            z = -z

        # Kinematic bed position.
        self.bed_pos = np.array(
            [0.0, 0.0, self.bed_center_z0 - z],
            dtype=float,
        )

        # Displayed bed position.
        self.bed_visual_pos = self.bed_pos + BED_VISUAL_OFFSET
        self.bed_actor.position = tuple(self.bed_visual_pos)

        carriage_center = np.array(
            [
                x,
                y,
                self.gantry_z - self.carriage_size[2] / 2.0,
            ],
            dtype=float,
        )
        self.carriage_actor.position = tuple(carriage_center)

        R_car = np.eye(3)
        p_Baxis = carriage_center + R_car @ B_PIVOT_FROM_CARRIAGE_CENTER

        B_rot = B_AXIS_SIGN * float(b_angle)
        C_rot = C_AXIS_SIGN * float(c_angle)

        # B rotation carries the 45-degree B bracket.
        R_B = Rz(B_rot) @ B_BRACKET_ZERO_ROT
        p_b_bracket = p_Baxis + R_B @ B_BRACKET_CENTER_FROM_B_LOCAL
        p_b_bracket_end = p_b_bracket + R_B @ np.array(
            [0.0, 0.0, -0.5 * B_BRACKET_LEN],
            dtype=float,
        )
        p_Caxis = p_b_bracket_end + R_B @ C_FROM_BRACKET_END_LOCAL

        # C rotation carries the C-axis arm and tool holder.
        R_C = R_B @ Rx(C_rot)

        # Shift the displayed C-arm center so the extra length extends toward the tool.
        c_arm_visual_center_from_C_local = (
            C_ARM_CENTER_FROM_C_LOCAL
            + np.array([0.5 * C_ARM_VISUAL_EXTRA_LENGTH, 0.0, 0.0], dtype=float)
        )
        p_c_arm = p_Caxis + R_C @ c_arm_visual_center_from_C_local

        p_tool = p_Caxis + R_C @ TOOL_FROM_C_LOCAL
        R_tool = R_C @ TOOL_ZERO_ROT_FROM_HOLDER

        # Kinematic tool-holder reference point.
        p_tool_holder = p_tool + R_tool @ TOOL_HOLDER_CENTER_FROM_TOOL_LOCAL
        p_tip = p_tool_holder + R_tool @ TIP_FROM_TOOL_HOLDER_CENTER_LOCAL

        # Displayed tool-holder center.
        p_tool_holder_visual = (
            p_tool_holder
            + R_tool @ TOOL_HOLDER_VISUAL_OFFSET_FROM_KINEMATIC_LOCAL
        )

        R_axis_B = np.eye(3)
        p_axis_B = p_Baxis

        c_axis_dir = _unit(R_B[:, 0])
        helper = np.array([0.0, 0.0, 1.0], dtype=float)

        if abs(np.dot(helper, c_axis_dir)) > 0.95:
            helper = np.array([0.0, 1.0, 0.0], dtype=float)

        x_cyl = _unit(np.cross(helper, c_axis_dir))
        y_cyl = _unit(np.cross(c_axis_dir, x_cyl))
        R_axis_C = np.column_stack((x_cyl, y_cyl, c_axis_dir))
        p_axis_C = p_Caxis

        self.b_bracket_actor.user_matrix = T_of(R_B, p_b_bracket)
        self.c_arm_actor.user_matrix = T_of(R_C, p_c_arm)
        self.tool_holder_actor.user_matrix = T_of(R_tool, p_tool_holder_visual)
        self.tip_actor.user_matrix = T_of(np.eye(3), p_tip)
        self.nozzle_cone_actor.user_matrix = T_of(R_tool, p_tip)

        self.b_axis_actor.user_matrix = T_of(R_axis_B, p_axis_B)
        self.c_axis_actor.user_matrix = T_of(R_axis_C, p_axis_C)

        if self.show_debug_frames:
            add_axes_triads(self.plotter, p_Baxis, R_B, scale=VIZ_TRIAD_SCALE, line_width=3)
            add_axes_triads(self.plotter, p_Caxis, R_C, scale=VIZ_TRIAD_SCALE, line_width=3)
            add_axes_triads(self.plotter, p_tool, R_tool, scale=VIZ_TRIAD_SCALE, line_width=3)

        if self.show_trace and (self.step_counter % self.trace_every == 0):
            p_tip_bed = self._world_to_bed(p_tip).copy()

            if is_cura_part:
                self.trace_points_bed_cura.append(p_tip_bed)
                trace_len = len(self.trace_points_bed_cura)
            else:
                self.trace_points_bed_spiral.append(p_tip_bed)
                trace_len = len(self.trace_points_bed_spiral)

            if self.draw_trace_live and (trace_len % self.trace_redraw_every == 0):
                self.draw_trace()

        self.plotter.update()

        if self.save_animation and (self.step_counter % self.animation_frame_every == 0):
            self.plotter.write_frame()

    def _draw_one_trace(self, pts_bed, old_actor, color):
        if len(pts_bed) < 2:
            return old_actor

        pts_world = self._bed_to_world_points(pts_bed)

        if old_actor is not None:
            try:
                self.plotter.remove_actor(old_actor)
            except Exception:
                pass

        poly = pv.lines_from_points(pts_world, close=False)

        if self.trace_as_tube:
            trace_mesh = poly.tube(radius=self.trace_radius)
            new_actor = self.plotter.add_mesh(trace_mesh, color=color)
        else:
            new_actor = self.plotter.add_mesh(poly, color=color, line_width=4)

        return new_actor

    def draw_trace(self):
        if not self.show_trace:
            return

        self.trace_actor_cura = self._draw_one_trace(
            pts_bed=self.trace_points_bed_cura,
            old_actor=self.trace_actor_cura,
            color=TRACE_COLOR_CURA,
        )

        self.trace_actor_spiral = self._draw_one_trace(
            pts_bed=self.trace_points_bed_spiral,
            old_actor=self.trace_actor_spiral,
            color=TRACE_COLOR_SPIRAL,
        )

        self.plotter.update()

        if self.save_animation:
            self.plotter.write_frame()


def main():
    data_full = load_commands_txt(SOLVED_IK_PATH)
    cura_base_row_count = count_numeric_rows(CURA_BASE_CLDATA_PATH, min_cols=6)

    print(
        f"Loaded {len(data_full)} IK rows; "
        f"dome starts at row {cura_base_row_count}"
    )

    stride = max(1, int(VIZ_STRIDE))

    row_indices_full = np.arange(len(data_full), dtype=int)
    data = data_full[::stride, :]
    row_indices = row_indices_full[::stride]

    print(f"Visualizing {len(data)} rows (stride {stride})")
    print()

    viz = ExplicitFramePrinterViz(
        xy_origin=XY_ORIGIN,
        show_trace=VIZ_SHOW_TRACE,
        trace_every=VIZ_TRACE_EVERY,
        trace_as_tube=VIZ_TRACE_AS_TUBE,
        trace_radius=VIZ_TRACE_RADIUS,
        show_debug_frames=VIZ_SHOW_DEBUG_FRAMES,
        draw_trace_live=VIZ_DRAW_TRACE_LIVE,
        trace_redraw_every=VIZ_TRACE_REDRAW_EVERY,
        save_animation=SAVE_ANIMATION,
        animation_path=ANIMATION_PATH,
        animation_fps=ANIMATION_FPS,
        animation_frame_every=ANIMATION_FRAME_EVERY,
    )

    x = data[:, 0]
    y = data[:, 1]
    zt = data[:, 2]
    b_cmd = data[:, 3]
    c_cmd = data[:, 4]

    if VIZ_ANGLES_DEG:
        b_cmd = np.deg2rad(b_cmd)
        c_cmd = np.deg2rad(c_cmd)

    dt_target = (1.0 / VIZ_FPS) if VIZ_FPS > 0 else 0.0
    last = time.perf_counter()

    for row_i in range(data.shape[0]):
        original_row_idx = int(row_indices[row_i])
        is_cura_part = original_row_idx < cura_base_row_count

        viz.step(
            x[row_i],
            y[row_i],
            zt[row_i],
            b_cmd[row_i],
            c_cmd[row_i],
            is_cura_part=is_cura_part,
        )

        if dt_target > 0 and not SAVE_ANIMATION:
            now = time.perf_counter()
            sleep = dt_target - (now - last)

            if sleep > 0:
                time.sleep(sleep)

            last = time.perf_counter()

    viz.draw_trace()

    if viz.save_animation:
        viz.plotter.write_frame()
        viz.plotter.close()
        print(f"Saved animation: {ANIMATION_PATH}")
        return

    print("Visualization complete.")

    while True:
        viz.plotter.update()
        time.sleep(0.02)


if __name__ == "__main__":
    main()