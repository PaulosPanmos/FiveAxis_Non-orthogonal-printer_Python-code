# -*- coding: utf-8 -*-
"""
S01_femoral_joint_cldata_generator.py

Generates CL data for the femoral-joint benchmark.

The Cura mandrel path is converted to x y z i j k CL data and combined with
one or more non-planar hemisphere spiral layers. Safe inter-layer reposition
moves are stored in separate no-extrusion index files. A 3D preview is shown
after generation.

Settings and file paths are read from S00_femoral_joint_parameters.py.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

import S00_femoral_joint_parameters as P


# ============================================================
# Files
# ============================================================

INPUT_GCODE = Path(P.INPUT_GCODE_PATH)

OUTPUT_CURA_CLDATA = Path(P.CURA_BASE_CLDATA_PATH)
OUTPUT_SPIRAL_CLDATA = Path(P.SPIRAL_CLDATA_PATH)
OUTPUT_SPIRAL_NO_EXTRUDE_INDEX_FILE = Path(P.SPIRAL_NO_EXTRUDE_INDICES_PATH)

OUTPUT_COMBINED_CLDATA = Path(P.CLDATA_PATH)
OUTPUT_COMBINED_NO_EXTRUDE_INDEX_FILE = Path(P.COMBINED_NO_EXTRUDE_INDICES_PATH)


# ============================================================
# Cura conversion
# ============================================================

BED_CENTER_X_MM = float(P.CL_CURA_BED_CENTER_X_MM)
BED_CENTER_Y_MM = float(P.CL_CURA_BED_CENTER_Y_MM)

CL_Z_OFFSET_MM = float(P.CL_Z_OFFSET_MM)

TOOL_IJK_BASE = np.asarray(P.CL_BASE_TOOL_IJK, dtype=float).reshape(3)

INCLUDE_TRAVEL_MOVES_FROM_CURA = bool(P.CL_INCLUDE_TRAVEL_MOVES_FROM_CURA)
START_AFTER_LAYER_0 = bool(P.CL_START_AFTER_LAYER_0)
SKIP_TYPES = set(P.CL_SKIP_TYPES)


# ============================================================
# Mandrel and dome geometry
# ============================================================

SPIRAL_SPHERE_CENTER_X_MM = float(P.CL_SPIRAL_SPHERE_CENTER_X_MM)
SPIRAL_SPHERE_CENTER_Y_MM = float(P.CL_SPIRAL_SPHERE_CENTER_Y_MM)
SPIRAL_SPHERE_CENTER_Z_MM = float(P.CL_SPIRAL_SPHERE_CENTER_Z_MM)

SPIRAL_MANDREL_RADIUS_MM = float(P.CL_SPIRAL_MANDREL_RADIUS_MM)

PREVIEW_CYLINDER_RADIUS_MM = float(P.CL_PREVIEW_CYLINDER_RADIUS_MM)
PREVIEW_CYLINDER_HEIGHT_MM = float(P.CL_PREVIEW_CYLINDER_HEIGHT_MM)


SPHERE_CENTER_X_MM = float(P.CL_SPHERE_CENTER_X_MM)
SPHERE_CENTER_Y_MM = float(P.CL_SPHERE_CENTER_Y_MM)
SPHERE_CENTER_Z_MM = float(P.CL_SPHERE_CENTER_Z_MM)




# ============================================================
# Dome spiral
# ============================================================

NUMBER_OF_SPIRAL_LAYERS = int(P.CL_NUMBER_OF_SPIRAL_LAYERS)

FIRST_DOME_CLEARANCE_MM = float(P.CL_FIRST_DOME_CLEARANCE_MM)
LAYER_HEIGHT_MM = float(P.CL_LAYER_HEIGHT_MM)

SPIRAL_LINE_SPACING_MM = float(P.CL_SPIRAL_LINE_SPACING_MM)
POINT_SPACING_MM = float(P.CL_POINT_SPACING_MM)

SURFACE_OFFSET_MM = float(P.CL_SURFACE_OFFSET_MM)

THETA_START_DEG = float(P.CL_THETA_START_DEG)
THETA_END_DEG = float(P.CL_THETA_END_DEG)

PRINT_TOP_TO_BOTTOM = bool(P.CL_PRINT_TOP_TO_BOTTOM)
LAYER_PHASE_SHIFT_DEG = float(P.CL_LAYER_PHASE_SHIFT_DEG)


# ============================================================
# Travel and placement
# ============================================================

ADD_SAFE_REPOSITION_BETWEEN_LAYERS = bool(P.CL_ADD_SAFE_REPOSITION_BETWEEN_LAYERS)
INTER_LAYER_Z_LIFT_MM = float(P.CL_INTER_LAYER_Z_LIFT_MM)
TRAVEL_IJK = np.asarray(P.CL_TRAVEL_IJK, dtype=float).reshape(3)

NORMAL_SIGN = float(P.CL_NORMAL_SIGN)

MAP_LOWEST_POINT_TO_BED_TOP = bool(P.CL_MAP_LOWEST_POINT_TO_BED_TOP)
BED_SIZE_MM = np.asarray(P.BED_SIZE, dtype=float).reshape(3)
Z_LIFT_MM = float(P.CL_Z_LIFT_MM)


# ============================================================
# Preview
# ============================================================

SHOW_PLOT = True
SHOW_BASE_GEOMETRY = True
SHOW_TOOL_VECTORS_SPIRAL = True

VIZ_STEP_SPIRAL = int(P.CL_VIZ_STEP)


# ============================================================
# Helpers
# ============================================================

_WORD_RE = re.compile(r"([A-Za-z])([-+]?\d*\.?\d+)")


def _normalize_rows(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.where(n < eps, 1.0, n)
    return v / n


def _arc_lengths(poly: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.zeros(poly.shape[0], dtype=float)
    s[1:] = np.cumsum(d)
    return s


def _resample_polyline_by_arclength(
    poly: np.ndarray,
    ds: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if ds <= 0:
        raise ValueError("ds must be > 0")

    s = _arc_lengths(poly)
    length = s[-1]

    if length <= 0:
        return poly.copy(), s.copy()

    n = int(np.floor(length / ds)) + 1
    n = max(n, 2)

    s_rs = np.linspace(0.0, length, n, endpoint=True)

    x = np.interp(s_rs, s, poly[:, 0])
    y = np.interp(s_rs, s, poly[:, 1])
    z = np.interp(s_rs, s, poly[:, 2])

    poly_rs = np.column_stack((x, y, z))
    return poly_rs, s_rs


def _set_equal_aspect_3d(ax, xs, ys, zs):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    zs = np.asarray(zs, dtype=float)

    x_mid = 0.5 * (xs.max() + xs.min())
    y_mid = 0.5 * (ys.max() + ys.min())
    z_mid = 0.5 * (zs.max() + zs.min())

    max_range = max(
        xs.max() - xs.min(),
        ys.max() - ys.min(),
        zs.max() - zs.min(),
        1e-9,
    )

    half = 0.5 * max_range

    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)

    ax.set_box_aspect((1, 1, 1))


def strip_comment(line: str) -> tuple[str, str]:
    if ";" in line:
        code, comment = line.split(";", 1)
        return code.strip(), comment.strip()
    return line.strip(), ""


def parse_words(code: str) -> dict[str, float]:
    words = {}

    for letter, value in _WORD_RE.findall(code):
        letter = letter.upper()

        try:
            words[letter] = float(value)
        except ValueError:
            pass

    return words


def is_positive_extrusion_move(
    e_old: float,
    e_new: float,
    absolute_extrusion: bool,
) -> bool:
    if absolute_extrusion:
        return e_new > e_old

    return e_new > 0.0


def write_cldata(path: Path, rows: np.ndarray, header_comment: str = ""):
    path = Path(path)

    with open(path, "w", encoding="utf-8") as f:
        if header_comment:
            for line in header_comment.splitlines():
                f.write(f"# {line}\n")

        f.write("# x y z i j k\n")

        for row in rows:
            f.write(
                f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} "
                f"{row[3]:.6f} {row[4]:.6f} {row[5]:.6f}\n"
            )


def write_index_file(path: Path, indices: list[int] | set[int], header_comment: str = ""):
    path = Path(path)
    indices_sorted = sorted(int(i) for i in indices)

    with open(path, "w", encoding="utf-8") as f:
        if header_comment:
            for line in header_comment.splitlines():
                f.write(f"# {line}\n")

        f.write("# row_idx\n")
        f.write("# Each index marks the destination row of a travel move without extrusion.\n")

        for idx in indices_sorted:
            f.write(f"{idx}\n")


def print_cldata_diagnostics(name: str, rows: np.ndarray):
    xyz = rows[:, :3]

    if len(rows) > 1:
        seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    else:
        seg = np.array([0.0])

    print(
        f"{name}: {len(rows)} rows, "
        f"X[{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}] "
        f"Y[{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}] "
        f"Z[{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}] mm"
    )
    print(
        f"  segment mean/min/max: {seg.mean():.4f} / "
        f"{seg.min():.4f} / {seg.max():.4f} mm"
    )


# ============================================================
# Cura G-code to CLdata
# ============================================================

def convert_cura_gcode_to_cldata_rows(
    input_gcode: Path = INPUT_GCODE,
) -> np.ndarray:
    input_gcode = Path(input_gcode)

    if not input_gcode.exists():
        raise FileNotFoundError(f"G-code file not found:\n{input_gcode}")

    x = None
    y = None
    z = None
    e = 0.0

    absolute_xyz = True
    absolute_extrusion = True

    in_print_region = not START_AFTER_LAYER_0
    current_type = None

    rows = []

    with open(input_gcode, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            code, comment = strip_comment(raw_line)

            if comment.startswith("LAYER:0"):
                in_print_region = True

            if comment.startswith("TYPE:"):
                current_type = comment.replace("TYPE:", "").strip().upper()

            if not in_print_region:
                continue

            if current_type in SKIP_TYPES:
                continue

            if not code:
                continue

            words = parse_words(code)

            if "G" in words:
                g = int(words["G"])

                if g == 90:
                    absolute_xyz = True
                    continue

                if g == 91:
                    absolute_xyz = False
                    continue

                if g == 92:
                    if "X" in words:
                        x = words["X"]
                    if "Y" in words:
                        y = words["Y"]
                    if "Z" in words:
                        z = words["Z"]
                    if "E" in words:
                        e = words["E"]
                    continue

            if "M" in words:
                m = int(words["M"])

                if m == 82:
                    absolute_extrusion = True
                    continue

                if m == 83:
                    absolute_extrusion = False
                    continue

            if "G" not in words:
                continue

            g = int(words["G"])

            if g not in (0, 1):
                continue

            old_x, old_y, old_z, old_e = x, y, z, e

            if absolute_xyz:
                if "X" in words:
                    x = words["X"]
                if "Y" in words:
                    y = words["Y"]
                if "Z" in words:
                    z = words["Z"]
            else:
                if x is None:
                    x = 0.0
                if y is None:
                    y = 0.0
                if z is None:
                    z = 0.0

                if "X" in words:
                    x += words["X"]
                if "Y" in words:
                    y += words["Y"]
                if "Z" in words:
                    z += words["Z"]

            has_e = "E" in words

            if has_e:
                e_word = words["E"]

                if absolute_extrusion:
                    e_new = e_word
                    extruding = is_positive_extrusion_move(old_e, e_new, absolute_extrusion)
                    e = e_new
                else:
                    e_new = e_word
                    extruding = is_positive_extrusion_move(old_e, e_new, absolute_extrusion)
                    e = old_e + e_new
            else:
                extruding = False

            if x is None or y is None or z is None:
                continue

            if INCLUDE_TRAVEL_MOVES_FROM_CURA:
                if old_x is not None and old_y is not None and old_z is not None:
                    moved_xyz = (x != old_x) or (y != old_y) or (z != old_z)
                    export_move = moved_xyz
                else:
                    export_move = False
            else:
                export_move = extruding

            if not export_move:
                continue

            x_out = x - BED_CENTER_X_MM
            y_out = y - BED_CENTER_Y_MM
            z_out = z + CL_Z_OFFSET_MM

            i, j, k = TOOL_IJK_BASE

            rows.append([x_out, y_out, z_out, i, j, k])

    rows = np.asarray(rows, dtype=float)

    if rows.size == 0:
        raise RuntimeError(
            "No CLdata rows were generated from the Cura G-code. "
            "Check P.CL_START_AFTER_LAYER_0, P.CL_INCLUDE_TRAVEL_MOVES_FROM_CURA, and input file."
        )

    return rows


# ============================================================
# Hemisphere spiral generation
# ============================================================

def _sphere_center() -> np.ndarray:
    return np.array([
        SPHERE_CENTER_X_MM,
        SPHERE_CENTER_Y_MM,
        SPHERE_CENTER_Z_MM,
    ], dtype=float)


def _layer_radius(layer_index: int) -> float:
    return (
        SPIRAL_MANDREL_RADIUS_MM
        + FIRST_DOME_CLEARANCE_MM
        + layer_index * LAYER_HEIGHT_MM
    )


def _hemisphere_point_and_normal(
    theta: np.ndarray,
    phi: np.ndarray,
    sphere_radius_mm: float,
):
    radius = float(sphere_radius_mm)

    xc = SPHERE_CENTER_X_MM
    yc = SPHERE_CENTER_Y_MM
    zc = SPHERE_CENTER_Z_MM

    x = xc + radius * np.sin(theta) * np.cos(phi)
    y = yc + radius * np.sin(theta) * np.sin(phi)
    z = zc + radius * np.cos(theta)

    p_surface = np.column_stack((x, y, z))

    n_out = np.column_stack((
        x - xc,
        y - yc,
        z - zc,
    ))
    n_out = _normalize_rows(n_out)

    return p_surface, n_out


def _generate_one_spiral_layer(
    sphere_radius_mm: float,
    layer_index: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    radius = float(sphere_radius_mm)

    if radius <= 0:
        raise ValueError("sphere_radius_mm must be > 0")
    if SPIRAL_LINE_SPACING_MM <= 0:
        raise ValueError("P.CL_SPIRAL_LINE_SPACING_MM must be > 0")
    if POINT_SPACING_MM <= 0:
        raise ValueError("P.CL_POINT_SPACING_MM must be > 0")
    if SURFACE_OFFSET_MM < 0:
        raise ValueError("P.CL_SURFACE_OFFSET_MM should be >= 0")
    if NUMBER_OF_SPIRAL_LAYERS < 1:
        raise ValueError("P.CL_NUMBER_OF_SPIRAL_LAYERS must be >= 1")
    if FIRST_DOME_CLEARANCE_MM < 0:
        raise ValueError("P.CL_FIRST_DOME_CLEARANCE_MM should be >= 0")
    if LAYER_HEIGHT_MM <= 0:
        raise ValueError("P.CL_LAYER_HEIGHT_MM must be > 0")

    theta_start = np.deg2rad(THETA_START_DEG)
    theta_end = np.deg2rad(THETA_END_DEG)

    if theta_start < 0:
        raise ValueError("P.CL_THETA_START_DEG must be >= 0")
    if theta_end <= theta_start:
        raise ValueError("P.CL_THETA_END_DEG must be larger than P.CL_THETA_START_DEG")
    if theta_end > 0.5 * np.pi + 1e-9:
        raise ValueError("P.CL_THETA_END_DEG should not exceed 90 degrees")

    meridian_length = radius * (theta_end - theta_start)
    turns = meridian_length / SPIRAL_LINE_SPACING_MM

    n_raw = max(1000, int(turns * 360))

    if PRINT_TOP_TO_BOTTOM:
        u = np.linspace(0.0, 1.0, n_raw)
    else:
        u = np.linspace(1.0, 0.0, n_raw)

    theta = theta_start + u * (theta_end - theta_start)

    phase_offset = layer_index * np.deg2rad(LAYER_PHASE_SHIFT_DEG)
    phi = phase_offset + 2.0 * np.pi * turns * u

    p_surface, n_out = _hemisphere_point_and_normal(
        theta=theta,
        phi=phi,
        sphere_radius_mm=radius,
    )

    p_tool_raw = p_surface + SURFACE_OFFSET_MM * n_out

    p_tool, _ = _resample_polyline_by_arclength(
        p_tool_raw,
        ds=POINT_SPACING_MM,
    )

    center = _sphere_center()

    n_out_rs = p_tool - center[None, :]
    n_out_rs = _normalize_rows(n_out_rs)

    ijk = NORMAL_SIGN * n_out_rs

    return p_tool, ijk, turns


def _make_safe_reposition_rows(
    current_point: np.ndarray,
    next_start_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p0 = np.asarray(current_point, dtype=float).reshape(3)
    p1 = np.asarray(next_start_point, dtype=float).reshape(3)

    lifted_z = max(p0[2], p1[2]) + INTER_LAYER_Z_LIFT_MM

    p_up = np.array([p0[0], p0[1], lifted_z], dtype=float)
    p_xy = np.array([p1[0], p1[1], lifted_z], dtype=float)
    p_down = np.array([p1[0], p1[1], p1[2]], dtype=float)

    points = np.vstack((p_up, p_xy, p_down))
    ijk = np.tile(TRAVEL_IJK.reshape(1, 3), (3, 1))

    return points, ijk


def generate_spiral_cldata_rows() -> tuple[np.ndarray, list[int], list[tuple[int, float, int, float]]]:
    all_points = []
    all_ijk = []

    no_extrude_row_indices = []
    layer_info = []

    for layer_idx in range(NUMBER_OF_SPIRAL_LAYERS):
        layer_radius = _layer_radius(layer_idx)

        p_layer, ijk_layer, turns = _generate_one_spiral_layer(
            sphere_radius_mm=layer_radius,
            layer_index=layer_idx,
        )

        layer_info.append((layer_idx + 1, layer_radius, len(p_layer), turns))

        if layer_idx > 0 and ADD_SAFE_REPOSITION_BETWEEN_LAYERS:
            current_last = all_points[-1][-1]
            next_start = p_layer[0]

            p_travel, ijk_travel = _make_safe_reposition_rows(
                current_point=current_last,
                next_start_point=next_start,
            )

            start_idx = sum(len(block) for block in all_points)

            no_extrude_row_indices.extend([
                start_idx,
                start_idx + 1,
                start_idx + 2,
            ])

            all_points.append(p_travel)
            all_ijk.append(ijk_travel)

        all_points.append(p_layer)
        all_ijk.append(ijk_layer)

    p_out = np.vstack(all_points)
    tool_dir = np.vstack(all_ijk)

    bed_t_mm = BED_SIZE_MM[2]
    bed_top_z_mm = bed_t_mm / 2.0 + Z_LIFT_MM

    if MAP_LOWEST_POINT_TO_BED_TOP:
        z_min = p_out[:, 2].min()
        p_out[:, 2] += bed_top_z_mm - z_min

    rows = np.column_stack((
        p_out[:, 0],
        p_out[:, 1],
        p_out[:, 2],
        tool_dir[:, 0],
        tool_dir[:, 1],
        tool_dir[:, 2],
    ))

    return rows, no_extrude_row_indices, layer_info


# ============================================================
# Visualization
# ============================================================

def _make_base_geometry_for_plot():
    radius = PREVIEW_CYLINDER_RADIUS_MM
    height = PREVIEW_CYLINDER_HEIGHT_MM

    ph = np.linspace(0.0, 2.0 * np.pi, 80)
    z_cyl = np.linspace(0.0, height, 8)

    cyl_points = []

    for z in z_cyl:
        x = radius * np.cos(ph)
        y = radius * np.sin(ph)
        zz = np.full_like(x, z + CL_Z_OFFSET_MM)
        cyl_points.append(np.column_stack((x, y, zz)))

    theta_vals = np.linspace(0.0, 0.5 * np.pi, 10)
    hemi_points = []

    for theta_value in theta_vals:
        theta = np.full_like(ph, theta_value)

        p_hemi, _ = _hemisphere_point_and_normal(
            theta=theta,
            phi=ph,
            sphere_radius_mm=SPIRAL_MANDREL_RADIUS_MM,
        )

        p_hemi[:, 2] += CL_Z_OFFSET_MM
        hemi_points.append(p_hemi)

    return cyl_points, hemi_points


def plot_all(
    cura_rows: np.ndarray,
    spiral_rows_shifted: np.ndarray,
    combined_rows: np.ndarray,
    combined_no_extrude_indices: list[int],
):
    if not SHOW_PLOT:
        return

    cura_xyz = cura_rows[:, :3]
    spiral_xyz = spiral_rows_shifted[:, :3]
    combined_xyz = combined_rows[:, :3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if SHOW_BASE_GEOMETRY:
        cyl_points, hemi_points = _make_base_geometry_for_plot()

        for ring in cyl_points:
            ax.plot(
                ring[:, 0],
                ring[:, 1],
                ring[:, 2],
                linestyle=":",
                linewidth=0.8,
            )

        for idx_ring, ring in enumerate(hemi_points):
            ax.plot(
                ring[:, 0],
                ring[:, 1],
                ring[:, 2],
                linestyle=":",
                linewidth=0.8,
                label="nominal mandrel preview" if idx_ring == 0 else None,
            )

    ax.plot(
        cura_xyz[:, 0],
        cura_xyz[:, 1],
        cura_xyz[:, 2],
        linestyle="-",
        marker=".",
        markersize=0.6,
        linewidth=0.4,
        label="Cura base CLdata",
    )

    ax.plot(
        spiral_xyz[:, 0],
        spiral_xyz[:, 1],
        spiral_xyz[:, 2],
        linestyle="-",
        marker=".",
        markersize=1.5,
        linewidth=0.8,
        label="two-layer hemisphere spiral CLdata",
    )

    if len(combined_no_extrude_indices) > 0:
        idx = np.array(combined_no_extrude_indices, dtype=int)
        ax.scatter(
            combined_xyz[idx, 0],
            combined_xyz[idx, 1],
            combined_xyz[idx, 2],
            s=45,
            marker="x",
            label="no-extrude travel rows",
        )

    total_span = max(
        combined_xyz[:, 0].max() - combined_xyz[:, 0].min(),
        combined_xyz[:, 1].max() - combined_xyz[:, 1].min(),
        combined_xyz[:, 2].max() - combined_xyz[:, 2].min(),
        1.0,
    )

    arrow_length = max(total_span * 0.04, 1.0)

    if SHOW_TOOL_VECTORS_SPIRAL:
        spiral_start = len(cura_rows)

        for local_idx in range(0, len(spiral_rows_shifted), max(1, VIZ_STEP_SPIRAL)):
            global_idx = spiral_start + local_idx

            ax.quiver(
                combined_rows[global_idx, 0],
                combined_rows[global_idx, 1],
                combined_rows[global_idx, 2],
                combined_rows[global_idx, 3],
                combined_rows[global_idx, 4],
                combined_rows[global_idx, 5],
                length=arrow_length,
                normalize=True,
            )

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title("Combined Cura base + two-layer hemisphere spiral CLdata")
    ax.legend()

    _set_equal_aspect_3d(
        ax,
        combined_xyz[:, 0],
        combined_xyz[:, 1],
        combined_xyz[:, 2],
    )

    ax.set_proj_type("ortho")

    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

def generate_full_combined_cldata():
    print(f"Input: {INPUT_GCODE.name}")

    # --------------------------------------------------------
    # 1. Convert Cura base to CLdata
    # --------------------------------------------------------
    cura_rows = convert_cura_gcode_to_cldata_rows(INPUT_GCODE)

    write_cldata(
        OUTPUT_CURA_CLDATA,
        cura_rows,
        header_comment=(
            "Cura base CLdata generated from G-code\n"
            f"Source G-code: {INPUT_GCODE}\n"
            f"Coordinate mapping: X_out = X_gcode - {BED_CENTER_X_MM:.3f}, "
            f"Y_out = Y_gcode - {BED_CENTER_Y_MM:.3f}, "
            f"Z_out = Z_gcode + {CL_Z_OFFSET_MM:.3f}\n"
            f"Tool orientation: IJK = {tuple(TOOL_IJK_BASE)}"
        ),
    )

    # --------------------------------------------------------
    # 2. Generate spiral CLdata
    # --------------------------------------------------------
    spiral_rows_unshifted, spiral_no_extrude_local, layer_info = generate_spiral_cldata_rows()

    write_cldata(
        OUTPUT_SPIRAL_CLDATA,
        spiral_rows_unshifted,
        header_comment=(
            "Hemisphere spiral CLdata generated directly\n"
            f"Spiral sphere center: "
            f"[{SPIRAL_SPHERE_CENTER_X_MM:.6f}, "
            f"{SPIRAL_SPHERE_CENTER_Y_MM:.6f}, "
            f"{SPIRAL_SPHERE_CENTER_Z_MM:.6f}]\n"
            f"Spiral mandrel radius: {SPIRAL_MANDREL_RADIUS_MM:.6f}\n"
            f"First dome clearance: {FIRST_DOME_CLEARANCE_MM:.6f}\n"
            f"Layer height: {LAYER_HEIGHT_MM:.6f}\n"
            f"Number of spiral layers: {NUMBER_OF_SPIRAL_LAYERS}\n"
            f"Spiral line spacing: {SPIRAL_LINE_SPACING_MM:.6f}\n"
            f"Point spacing: {POINT_SPACING_MM:.6f}\n"
            f"Surface offset: {SURFACE_OFFSET_MM:.6f}\n"
            "Note: this standalone spiral file is unshifted in Z. "
            "The combined file applies CL_Z_OFFSET_MM."
        ),
    )

    write_index_file(
        OUTPUT_SPIRAL_NO_EXTRUDE_INDEX_FILE,
        spiral_no_extrude_local,
        header_comment=f"Local no-extrude row indices for {OUTPUT_SPIRAL_CLDATA.name}",
    )

    # --------------------------------------------------------
    # 3. Shift spiral into same CL Z convention as Cura base
    # --------------------------------------------------------
    spiral_rows_shifted = spiral_rows_unshifted.copy()
    spiral_rows_shifted[:, 2] += CL_Z_OFFSET_MM

    # --------------------------------------------------------
    # 4. Combine Cura base + shifted spiral
    # --------------------------------------------------------
    combined_rows = np.vstack((cura_rows, spiral_rows_shifted))

    spiral_start_index = len(cura_rows)

    combined_no_extrude_indices = [
        spiral_start_index + idx
        for idx in spiral_no_extrude_local
    ]

    write_cldata(
        OUTPUT_COMBINED_CLDATA,
        combined_rows,
        header_comment=(
            "Combined CLdata: Cura mandrel + hemisphere spiral layers\n"
            f"Cura CLdata source: {OUTPUT_CURA_CLDATA}\n"
            f"Spiral CLdata source: {OUTPUT_SPIRAL_CLDATA}\n"
            f"Spiral physical sphere center: "
            f"[{SPIRAL_SPHERE_CENTER_X_MM:.6f}, "
            f"{SPIRAL_SPHERE_CENTER_Y_MM:.6f}, "
            f"{SPIRAL_SPHERE_CENTER_Z_MM:.6f}]\n"
            f"Spiral physical mandrel radius: {SPIRAL_MANDREL_RADIUS_MM:.6f}\n"
            f"CL_Z_OFFSET_MM applied to Cura and combined spiral: {CL_Z_OFFSET_MM:.6f}\n"
            f"Spiral starts at combined row index: {spiral_start_index}\n"
            "Columns: x y z i j k"
        ),
    )

    write_index_file(
        OUTPUT_COMBINED_NO_EXTRUDE_INDEX_FILE,
        combined_no_extrude_indices,
        header_comment=(
            "Global no-extrude row indices for combined CLdata\n"
            f"Spiral starts at row index: {spiral_start_index}\n"
            "These are local spiral no-extrude indices offset by the Cura base row count."
        ),
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------
    print(f"Wrote {OUTPUT_CURA_CLDATA.name}")
    print(f"Wrote {OUTPUT_SPIRAL_CLDATA.name}")
    print(f"Wrote {OUTPUT_COMBINED_CLDATA.name}")
    print_cldata_diagnostics("Mandrel", cura_rows)
    print_cldata_diagnostics("Dome", spiral_rows_shifted)
    print_cldata_diagnostics("Combined", combined_rows)

    for layer_num, radius, n_points, turns in layer_info:
        print(
            f"Layer {layer_num}: radius {radius:.3f} mm, "
            f"{n_points} rows, {turns:.2f} turns"
        )

    if combined_no_extrude_indices:
        print(f"Travel rows: {combined_no_extrude_indices}")

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    plot_all(
        cura_rows=cura_rows,
        spiral_rows_shifted=spiral_rows_shifted,
        combined_rows=combined_rows,
        combined_no_extrude_indices=combined_no_extrude_indices,
    )

    return cura_rows, spiral_rows_shifted, combined_rows, combined_no_extrude_indices


if __name__ == "__main__":
    generate_full_combined_cldata()