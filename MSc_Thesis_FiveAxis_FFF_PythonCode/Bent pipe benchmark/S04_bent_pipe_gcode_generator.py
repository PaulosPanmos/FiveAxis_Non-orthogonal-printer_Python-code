# -*- coding: utf-8 -*-
"""
S04_bent_pipe_gcode_generator.py

Converts the solved bent-pipe path to Klipper-compatible five-axis G-code.

The solved IK file provides X, Y, z_table, B and C. Extrusion is calculated
from the corresponding CL-path segment lengths. Machine-coordinate offsets and
axis limits are read from S00_bent_pipe_parameters.py.

A new numbered G-code file is created on each run.
"""

from __future__ import annotations
import math
import re
from pathlib import Path
import numpy as np
import S00_bent_pipe_parameters as P


# ============================================================
# Files
# ============================================================

BASE_DIR = Path(P.BASE_DIR)

IK_PATH = Path(P.SOLVED_IK_PATH)
CL_PATH = Path(P.CLDATA_PATH)

GCODE_BASENAME = P.GCODE_BASENAME
GCODE_EXT = P.GCODE_EXT


# ============================================================
# Machine coordinate mapping
# ============================================================

X_CENTER_OFFSET_MM = float(P.GCODE_X_CENTER_OFFSET_MM)
Y_CENTER_OFFSET_MM = float(P.GCODE_Y_CENTER_OFFSET_MM)

ZTABLE_TO_MACHINE_Z_OFFSET = float(P.GCODE_ZTABLE_TO_MACHINE_Z_OFFSET)
ZTABLE_SIGN = float(P.GCODE_ZTABLE_SIGN)

ANGLES_IN_RADIANS = bool(P.GCODE_ANGLES_IN_RADIANS)
OUTPUT_BC_IN_DEGREES = bool(P.GCODE_OUTPUT_BC_IN_DEGREES)

B_OFFSET = float(P.GCODE_B_OFFSET)
C_OFFSET = float(P.GCODE_C_OFFSET)


# ============================================================
# Machine limits
# ============================================================

X_MIN_MM = float(P.GCODE_X_MIN_MM)
X_MAX_MM = float(P.GCODE_X_MAX_MM)

Y_MIN_MM = float(P.GCODE_Y_MIN_MM)
Y_MAX_MM = float(P.GCODE_Y_MAX_MM)

Z_MIN_MM = float(P.GCODE_Z_MIN_MM)
Z_MAX_MM = float(P.GCODE_Z_MAX_MM)

C_MIN_DEG = float(P.GCODE_C_MIN_DEG)
C_MAX_DEG = float(P.GCODE_C_MAX_DEG)

CHECK_B_RANGE = bool(P.GCODE_CHECK_B_RANGE)
B_MIN_DEG = float(P.GCODE_B_MIN_DEG)
B_MAX_DEG = float(P.GCODE_B_MAX_DEG)

# ============================================================
# Print settings
# ============================================================

DRY_RUN = bool(P.GCODE_DRY_RUN)

BED_TEMP_C = float(P.GCODE_BED_TEMP_C)
NOZZLE_TEMP_C = float(P.GCODE_NOZZLE_TEMP_C)

PRINT_SPEED_MM_S = float(P.GCODE_PRINT_SPEED_MM_S)
TRAVEL_SPEED_MM_S = float(P.GCODE_TRAVEL_SPEED_MM_S)

LINE_WIDTH_MM = float(P.GCODE_LINE_WIDTH_MM)
LAYER_HEIGHT_MM = float(P.GCODE_LAYER_HEIGHT_MM)
FILAMENT_DIAMETER_MM = float(P.GCODE_FILAMENT_DIAMETER_MM)
FLOW_MULTIPLIER = float(P.GCODE_FLOW_MULTIPLIER)

PRIME_EXTRUSION_MM = float(P.GCODE_PRIME_EXTRUSION_MM)
EXTRUSION_RELATIVE = bool(P.GCODE_EXTRUSION_RELATIVE)

DECIMALS_POS = int(P.GCODE_DECIMALS_POS)
DECIMALS_E = int(P.GCODE_DECIMALS_E)
DECIMALS_ANGLE = int(P.GCODE_DECIMALS_ANGLE)


# ============================================================
# Helpers
# ============================================================

def load_numeric_file(path: Path, min_cols: int) -> np.ndarray:
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < min_cols:
        raise ValueError(f"{path.name} must have at least {min_cols} columns")
    return data


def bead_area_mm2(line_width: float, layer_height: float) -> float:
    return line_width * layer_height


def filament_area_mm2(filament_diameter: float) -> float:
    r = 0.5 * filament_diameter
    return math.pi * r * r


def extrusion_per_mm(
    line_width: float,
    layer_height: float,
    filament_diameter: float,
    flow_multiplier: float = 1.0,
) -> float:
    return (
        flow_multiplier
        * bead_area_mm2(line_width, layer_height)
        / filament_area_mm2(filament_diameter)
    )


def fmt(v: float, n: int) -> str:
    return f"{v:.{n}f}"


def transform_x(x_centered: float) -> float:
    return X_CENTER_OFFSET_MM + x_centered


def transform_y(y_centered: float) -> float:
    return Y_CENTER_OFFSET_MM + y_centered


def transform_z(z_table: float) -> float:
    return ZTABLE_TO_MACHINE_Z_OFFSET + ZTABLE_SIGN * z_table


def transform_angle(a: float, offset: float = 0.0) -> float:
    if ANGLES_IN_RADIANS and OUTPUT_BC_IN_DEGREES:
        return math.degrees(a) + offset
    return a + offset


def segment_lengths_from_cl_xyz(cl_xyz: np.ndarray) -> np.ndarray:
    d = np.diff(cl_xyz, axis=0)
    return np.linalg.norm(d, axis=1)


def validate_move(x: float, y: float, z: float, b: float, c: float) -> None:
    if not (X_MIN_MM <= x <= X_MAX_MM):
        raise ValueError(f"X out of range: {x:.4f} mm")
    if not (Y_MIN_MM <= y <= Y_MAX_MM):
        raise ValueError(f"Y out of range: {y:.4f} mm")
    if not (Z_MIN_MM <= z <= Z_MAX_MM):
        raise ValueError(f"Z out of range: {z:.4f} mm")
    if not (C_MIN_DEG <= c <= C_MAX_DEG):
        raise ValueError(f"C out of range: {c:.4f} deg")
    if CHECK_B_RANGE and not (B_MIN_DEG <= b <= B_MAX_DEG):
        raise ValueError(f"B out of range: {b:.4f} deg")


def next_indexed_output_path(base_dir: Path, basename: str, ext: str) -> Path:
    """Return the next free numbered output path."""
    pattern = re.compile(rf"^{re.escape(basename)}_(\d+){re.escape(ext)}$")
    max_index = 0

    for p in base_dir.iterdir():
        if not p.is_file():
            continue
        match = pattern.match(p.name)
        if match:
            max_index = max(max_index, int(match.group(1)))

    return base_dir / f"{basename}_{max_index + 1}{ext}"


# ============================================================
# Main
# ============================================================

def main() -> None:
    ik = load_numeric_file(IK_PATH, min_cols=5)   # x y z_table B C
    cl = load_numeric_file(CL_PATH, min_cols=6)   # x y z i j k

    if len(ik) != len(cl):
        raise ValueError(
            f"Row count mismatch: IK has {len(ik)} rows, CL has {len(cl)} rows. "
            "Use matching files from the same solve."
        )

    e_per_mm = extrusion_per_mm(
        line_width=LINE_WIDTH_MM,
        layer_height=LAYER_HEIGHT_MM,
        filament_diameter=FILAMENT_DIAMETER_MM,
        flow_multiplier=FLOW_MULTIPLIER,
    )

    cl_xyz = cl[:, :3]
    seg_len = segment_lengths_from_cl_xyz(cl_xyz)

    feed_print = PRINT_SPEED_MM_S * 60.0
    feed_travel = TRAVEL_SPEED_MM_S * 60.0

    lines: list[str] = []

    # --------------------------------------------------------
    # Header / startup
    # --------------------------------------------------------
    lines.append("; 5-axis generated G-code")
    lines.append(f"; Source IK: {IK_PATH.name}")
    lines.append(f"; Source CL: {CL_PATH.name}")
    lines.append(f"; DRY_RUN = {DRY_RUN}")
    if not DRY_RUN:
        lines.append(f"; e_per_mm = {e_per_mm:.6f}")
    lines.append("")

    lines.append("G21                     ; mm units")
    lines.append("G90                     ; absolute XYZBC")
    if not DRY_RUN and EXTRUSION_RELATIVE:
        lines.append("M83                     ; relative extrusion")

    if not DRY_RUN:
        lines.append(f"M140 S{fmt(BED_TEMP_C, 1)}")
        lines.append(f"M104 S{fmt(NOZZLE_TEMP_C, 1)}")

    lines.append("HOME_ALL")

    if not DRY_RUN:
        lines.append(f"M190 S{fmt(BED_TEMP_C, 1)}")
        lines.append(f"M109 S{fmt(NOZZLE_TEMP_C, 1)}")
        lines.append("G92 E0")

    lines.append("")

    # --------------------------------------------------------
    # First move: go to start point without extrusion
    # --------------------------------------------------------
    x0 = transform_x(float(ik[0, 0]))
    y0 = transform_y(float(ik[0, 1]))
    z0 = transform_z(float(ik[0, 2]))
    b0 = transform_angle(float(ik[0, 3]), B_OFFSET)
    c0 = transform_angle(float(ik[0, 4]), C_OFFSET)

    validate_move(x0, y0, z0, b0, c0)

    lines.append("; Move to first point")
    lines.append(
        "G1 "
        f"X{fmt(x0, DECIMALS_POS)} "
        f"Y{fmt(y0, DECIMALS_POS)} "
        f"Z{fmt(z0, DECIMALS_POS)} "
        f"B{fmt(b0, DECIMALS_ANGLE)} "
        f"C{fmt(c0, DECIMALS_ANGLE)} "
        f"F{fmt(feed_travel, 0)}"
    )

    if not DRY_RUN:
        lines.append(f"G1 E{fmt(PRIME_EXTRUSION_MM, DECIMALS_E)} F300")

    lines.append("")

    # --------------------------------------------------------
    # Print moves
    # --------------------------------------------------------
    for i in range(1, len(ik)):
        x = transform_x(float(ik[i, 0]))
        y = transform_y(float(ik[i, 1]))
        z = transform_z(float(ik[i, 2]))
        b = transform_angle(float(ik[i, 3]), B_OFFSET)
        c = transform_angle(float(ik[i, 4]), C_OFFSET)

        validate_move(x, y, z, b, c)

        if DRY_RUN:
            lines.append(
                "G1 "
                f"X{fmt(x, DECIMALS_POS)} "
                f"Y{fmt(y, DECIMALS_POS)} "
                f"Z{fmt(z, DECIMALS_POS)} "
                f"B{fmt(b, DECIMALS_ANGLE)} "
                f"C{fmt(c, DECIMALS_ANGLE)} "
                f"F{fmt(feed_print, 0)}"
            )
        else:
            e = seg_len[i - 1] * e_per_mm
            lines.append(
                "G1 "
                f"X{fmt(x, DECIMALS_POS)} "
                f"Y{fmt(y, DECIMALS_POS)} "
                f"Z{fmt(z, DECIMALS_POS)} "
                f"B{fmt(b, DECIMALS_ANGLE)} "
                f"C{fmt(c, DECIMALS_ANGLE)} "
                f"E{fmt(e, DECIMALS_E)} "
                f"F{fmt(feed_print, 0)}"
            )

    # --------------------------------------------------------
    # End
    # --------------------------------------------------------
    lines.append("")
    lines.append("; End")

    if not DRY_RUN:
        lines.append("G92 E0")
        lines.append("M104 S0")
        lines.append("M140 S0")
        lines.append("G1 E-1.0000 F300")

    z_end_current = transform_z(float(ik[-1, 2]))
    z_safe = min(max(z_end_current + 10.0, Z_MIN_MM), Z_MAX_MM)
    lines.append(f"G1 Z{fmt(z_safe, DECIMALS_POS)} F{fmt(feed_travel, 0)}")
    lines.append("M84")

    gcode_path = next_indexed_output_path(BASE_DIR, GCODE_BASENAME, GCODE_EXT)
    gcode_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {gcode_path}")
    if not DRY_RUN:
        print(f"Extrusion: {e_per_mm:.6f} mm filament per mm path")


if __name__ == "__main__":
    main()