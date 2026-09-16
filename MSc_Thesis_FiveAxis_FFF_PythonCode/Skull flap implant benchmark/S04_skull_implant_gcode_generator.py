# -*- coding: utf-8 -*-
"""
S04_skull_implant_gcode_generator.py

Converts the solved skull-implant path to Klipper-compatible five-axis G-code.

The original AtomSlicer extrusion values are mapped back onto the matching
CL/IK rows. The solved XYZ path can then be placed on the machine build area
without changing the B/C orientations.

All print settings, coordinate transforms and machine limits are read from
S00_skull_implant_parameters.py.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

import S00_skull_implant_parameters as P


# ============================================================
# Benchmark files
# ============================================================

BASE_DIR = Path(P.BASE_DIR)

IK_PATH = Path(P.SOLVED_IK_PATH)
CL_PATH = Path(P.CLDATA_PATH)
ATOMSLICER_GCODE_PATH = Path(P.ATOMSLICER_GCODE_PATH)

GCODE_BASENAME = str(P.GCODE_BASENAME)
GCODE_EXT = str(P.GCODE_EXT)


# ============================================================
# Print settings
# ============================================================

DRY_RUN = bool(P.GCODE_DRY_RUN)

BED_TEMP_C = float(P.GCODE_BED_TEMP_C)
NOZZLE_TEMP_C = float(P.GCODE_NOZZLE_TEMP_C)

PRINT_SPEED_MM_S = float(P.GCODE_PRINT_SPEED_MM_S)
TRAVEL_SPEED_MM_S = float(P.GCODE_TRAVEL_SPEED_MM_S)

PRIME_EXTRUSION_MM = float(P.GCODE_PRIME_EXTRUSION_MM)
EXTRUSION_RELATIVE = bool(P.GCODE_EXTRUSION_RELATIVE)

DECIMALS_POS = int(P.GCODE_DECIMALS_POS)
DECIMALS_E = int(P.GCODE_DECIMALS_E)
DECIMALS_ANGLE = int(P.GCODE_DECIMALS_ANGLE)


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
# Machine limits and automatic placement
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

AUTO_PLACE_XY = bool(P.GCODE_AUTO_PLACE_XY)
AUTO_PLACE_Z = bool(P.GCODE_AUTO_PLACE_Z)

TARGET_X_CENTER_MM = 0.5 * (X_MIN_MM + X_MAX_MM)
TARGET_Y_CENTER_MM = 0.5 * (Y_MIN_MM + Y_MAX_MM)

Z_PLACEMENT_REFERENCE = str(P.GCODE_Z_PLACEMENT_REFERENCE).strip().lower()
TARGET_Z_MIN_MM = float(P.GCODE_TARGET_Z_MIN_MM)


# ============================================================
# AtomSlicer parsing
# ============================================================

FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
NORMAL_COMMENT_RE = re.compile(
    rf"\(\s*({FLOAT_PATTERN})\s*,\s*({FLOAT_PATTERN})\s*,\s*({FLOAT_PATTERN})\s*\)"
)


def gcode_word_value(code: str, letter: str) -> float | None:
    match = re.search(
        rf"(?:^|\s){re.escape(letter)}({FLOAT_PATTERN})(?=\s|$)",
        code,
    )
    if match is None:
        return None
    return float(match.group(1))


def load_atomslicer_relative_extrusion(path: Path, expected_rows: int) -> np.ndarray:
    """Map AtomSlicer relative extrusion values to the CL/IK rows."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"AtomSlicer G-code not found:\n{path}\n\n"
            "The same AtomSlicer source must be used by S01 and S04."
        )

    e_by_row: list[float] = []

    extrusion_relative = False
    last_abs_e = 0.0

    current_x = None
    current_y = None
    current_z = None
    previous_xyz = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            stripped = raw.strip()

            if not stripped:
                continue

            code, _, comment = stripped.partition(";")
            upper_code = code.strip().upper()

            if upper_code.startswith("M83"):
                extrusion_relative = True
                continue

            if upper_code.startswith("M82"):
                extrusion_relative = False
                continue

            if upper_code.startswith("G92"):
                e_reset = gcode_word_value(upper_code, "E")
                if e_reset is not None:
                    last_abs_e = e_reset
                continue

            if not (upper_code.startswith("G0") or upper_code.startswith("G1")):
                continue

            e_raw = gcode_word_value(upper_code, "E")

            if e_raw is not None and not extrusion_relative:
                e_delta = e_raw - last_abs_e
                last_abs_e = e_raw
            else:
                e_delta = 0.0 if e_raw is None else e_raw

            x_word = gcode_word_value(upper_code, "X")
            y_word = gcode_word_value(upper_code, "Y")
            z_word = gcode_word_value(upper_code, "Z")

            if x_word is not None:
                current_x = x_word
            if y_word is not None:
                current_y = y_word
            if z_word is not None:
                current_z = z_word

            if current_x is None or current_y is None or current_z is None:
                continue

            if NORMAL_COMMENT_RE.search(comment) is None:
                continue

            # Match the extrusion-only filter used by S01.
            if P.CL_ONLY_EXTRUSION_MOVES:
                if e_raw is None or e_raw <= 0.0:
                    continue

            current_xyz = (current_x, current_y, current_z)

            if P.CL_REMOVE_DUPLICATE_POINTS and previous_xyz is not None:
                same_point = (
                    abs(current_xyz[0] - previous_xyz[0]) < 1e-9
                    and abs(current_xyz[1] - previous_xyz[1]) < 1e-9
                    and abs(current_xyz[2] - previous_xyz[2]) < 1e-9
                )
                if same_point:
                    continue

            e_by_row.append(float(e_delta))
            previous_xyz = current_xyz

    e = np.asarray(e_by_row, dtype=float)

    if len(e) != expected_rows:
        raise ValueError(
            "AtomSlicer extrusion row count does not match the CL/IK path.\n"
            f"Mapped AtomSlicer rows: {len(e)}\n"
            f"Expected CL/IK rows:    {expected_rows}\n"
            "Make sure S01 and S04 use the same AtomSlicer source and conversion "
            "settings."
        )

    return e


# ============================================================
# File and formatting helpers
# ============================================================

def load_numeric_file(path: Path, min_cols: int) -> np.ndarray:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found:\n{path}")

    data = np.loadtxt(path, comments="#")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.ndim != 2 or data.shape[1] < min_cols:
        raise ValueError(f"{path.name} must have at least {min_cols} columns")

    return data


def fmt(v: float, n: int) -> str:
    return f"{v:.{n}f}"


def next_indexed_output_path(base_dir: Path, basename: str, ext: str) -> Path:
    """Return the next free numbered output path."""
    base_dir = Path(base_dir)

    pattern = re.compile(rf"^{re.escape(basename)}_(\d+){re.escape(ext)}$")
    max_index = 0

    for path in base_dir.iterdir():
        if not path.is_file():
            continue

        match = pattern.match(path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))

    return base_dir / f"{basename}_{max_index + 1}{ext}"


# ============================================================
# Machine transforms
# ============================================================

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


def transformed_row(ik_row: np.ndarray) -> tuple[float, float, float, float, float]:
    """Map one solved IK row x y z_table B C to machine X Y Z B C."""
    return (
        transform_x(float(ik_row[0])),
        transform_y(float(ik_row[1])),
        transform_z(float(ik_row[2])),
        transform_angle(float(ik_row[3]), B_OFFSET),
        transform_angle(float(ik_row[4]), C_OFFSET),
    )


def transformed_rows(ik: np.ndarray) -> np.ndarray:
    return np.array([transformed_row(row) for row in ik], dtype=float)


def xyz_bounds(machine_rows: np.ndarray) -> dict[str, float]:
    x = machine_rows[:, 0]
    y = machine_rows[:, 1]
    z = machine_rows[:, 2]

    return {
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "x_center": float(0.5 * (np.min(x) + np.max(x))),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "y_center": float(0.5 * (np.min(y) + np.max(y))),
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
    }


def first_print_row_index(extrusion_by_row: np.ndarray) -> int:
    """Return the first positive-extrusion row after the initial path point."""
    for row_idx in range(1, len(extrusion_by_row)):
        if extrusion_by_row[row_idx] > 0.0:
            return row_idx

    raise ValueError(
        "No positive-extrusion move found after row 0 in the AtomSlicer path."
    )


def z_placement_reference_value(
    machine_rows: np.ndarray,
    bounds: dict[str, float],
    first_print_idx: int,
) -> tuple[float, str]:
    if Z_PLACEMENT_REFERENCE in {"min", "lowest", "lowest_z"}:
        return bounds["z_min"], "minimum path Z"

    if Z_PLACEMENT_REFERENCE in {"first_path", "first", "row0"}:
        return float(machine_rows[0, 2]), "first path row Z"

    if Z_PLACEMENT_REFERENCE in {"first_print", "first_extrusion", "print"}:
        return (
            float(machine_rows[first_print_idx, 2]),
            f"first positive-extrusion row Z, row {first_print_idx}",
        )

    raise ValueError(
        "Invalid P.GCODE_Z_PLACEMENT_REFERENCE. "
        "Use 'min', 'first_path', or 'first_print'. "
        f"Got: {Z_PLACEMENT_REFERENCE!r}"
    )


def compute_auto_placement_offset(
    machine_rows: np.ndarray,
    first_print_idx: int,
) -> tuple[np.ndarray, float, str]:
    """Compute the constant XYZ placement offset for the solved path."""
    bounds = xyz_bounds(machine_rows)

    z_ref, z_ref_label = z_placement_reference_value(
        machine_rows=machine_rows,
        bounds=bounds,
        first_print_idx=first_print_idx,
    )

    dx = TARGET_X_CENTER_MM - bounds["x_center"] if AUTO_PLACE_XY else 0.0
    dy = TARGET_Y_CENTER_MM - bounds["y_center"] if AUTO_PLACE_XY else 0.0
    dz = TARGET_Z_MIN_MM - z_ref if AUTO_PLACE_Z else 0.0

    return np.array([dx, dy, dz], dtype=float), z_ref, z_ref_label


def apply_xyz_offset(machine_rows: np.ndarray, xyz_offset: np.ndarray) -> np.ndarray:
    shifted = np.array(machine_rows, dtype=float, copy=True)
    shifted[:, 0:3] += xyz_offset.reshape(1, 3)
    return shifted


# ============================================================
# Validation
# ============================================================

def validate_move(
    x: float,
    y: float,
    z: float,
    b: float,
    c: float,
    row_idx: int | None = None,
) -> None:
    prefix = "" if row_idx is None else f"row {row_idx}: "

    if not (X_MIN_MM <= x <= X_MAX_MM):
        raise ValueError(f"{prefix}X out of range: {x:.4f} mm")

    if not (Y_MIN_MM <= y <= Y_MAX_MM):
        raise ValueError(f"{prefix}Y out of range: {y:.4f} mm")

    if not (Z_MIN_MM <= z <= Z_MAX_MM):
        raise ValueError(f"{prefix}Z out of range: {z:.4f} mm")

    if not (C_MIN_DEG <= c <= C_MAX_DEG):
        raise ValueError(f"{prefix}C out of range: {c:.4f} deg")

    if CHECK_B_RANGE and not (B_MIN_DEG <= b <= B_MAX_DEG):
        raise ValueError(f"{prefix}B out of range: {b:.4f} deg")


def validate_all_moves(machine_rows: np.ndarray) -> None:
    for row_idx, row in enumerate(machine_rows):
        validate_move(
            x=float(row[0]),
            y=float(row[1]),
            z=float(row[2]),
            b=float(row[3]),
            c=float(row[4]),
            row_idx=row_idx,
        )


# ============================================================
# G-code formatting
# ============================================================

def g1_xyzbc(
    x: float,
    y: float,
    z: float,
    b: float,
    c: float,
    feed: float,
    e: float | None = None,
) -> str:
    line = (
        "G1 "
        f"X{fmt(x, DECIMALS_POS)} "
        f"Y{fmt(y, DECIMALS_POS)} "
        f"Z{fmt(z, DECIMALS_POS)} "
        f"B{fmt(b, DECIMALS_ANGLE)} "
        f"C{fmt(c, DECIMALS_ANGLE)}"
    )

    if e is not None:
        line += f" E{fmt(e, DECIMALS_E)}"

    line += f" F{fmt(feed, 0)}"
    return line


def extrusion_line(e_value: float, feed: float = 300.0) -> str:
    return f"G1 E{fmt(e_value, DECIMALS_E)} F{fmt(feed, 0)}"


# ============================================================
# Main
# ============================================================

def main() -> None:
    if not EXTRUSION_RELATIVE:
        raise ValueError(
            "The skull-implant generator preserves AtomSlicer relative E values, "
            "so P.GCODE_EXTRUSION_RELATIVE must be True."
        )

    ik = load_numeric_file(IK_PATH, min_cols=5)
    cl = load_numeric_file(CL_PATH, min_cols=6)

    if len(ik) != len(cl):
        raise ValueError(
            f"Row count mismatch: IK has {len(ik)} rows and CL has {len(cl)} rows. "
            "Use matching files from the same solve."
        )

    extrusion_by_row = load_atomslicer_relative_extrusion(
        ATOMSLICER_GCODE_PATH,
        expected_rows=len(cl),
    )

    first_print_idx = first_print_row_index(extrusion_by_row)

    machine_rows_unplaced = transformed_rows(ik)
    xyz_offset, _, _ = compute_auto_placement_offset(
        machine_rows=machine_rows_unplaced,
        first_print_idx=first_print_idx,
    )

    machine_rows = apply_xyz_offset(machine_rows_unplaced, xyz_offset)
    bounds_after = xyz_bounds(machine_rows)

    validate_all_moves(machine_rows)


    feed_print = PRINT_SPEED_MM_S * 60.0
    feed_travel = TRAVEL_SPEED_MM_S * 60.0

    lines: list[str] = []

    # Header
    lines.append("; Skull implant five-axis G-code")
    lines.append("; Parameters source: S00_skull_implant_parameters.py")
    lines.append(f"; Source IK: {IK_PATH.name}")
    lines.append(f"; Source CL: {CL_PATH.name}")
    lines.append(f"; AtomSlicer source: {ATOMSLICER_GCODE_PATH.name}")
    lines.append("; Extrusion source: original AtomSlicer E values")
    lines.append(f"; DRY_RUN = {DRY_RUN}")
    lines.append(
        f"; Auto-place XYZ offset: "
        f"X{xyz_offset[0]:.4f} Y{xyz_offset[1]:.4f} Z{xyz_offset[2]:.4f}"
    )
    lines.append(
        f"; Bounds after placement: "
        f"X[{bounds_after['x_min']:.4f}, {bounds_after['x_max']:.4f}] "
        f"Y[{bounds_after['y_min']:.4f}, {bounds_after['y_max']:.4f}] "
        f"Z[{bounds_after['z_min']:.4f}, {bounds_after['z_max']:.4f}]"
    )
    lines.append(
        f"; First positive-extrusion row: {first_print_idx}, "
        f"Z{float(machine_rows[first_print_idx, 2]):.4f}"
    )
    lines.append("")

    # Startup
    lines.append("G21                     ; mm units")
    lines.append("G90                     ; absolute XYZBC")

    if not DRY_RUN:
        lines.append("M83                     ; relative extrusion")
        lines.append(f"M140 S{fmt(BED_TEMP_C, 1)}        ; set bed temperature")
        lines.append(f"M104 S{fmt(NOZZLE_TEMP_C, 1)}     ; set nozzle temperature")

    lines.append("HOME_ALL")

    if not DRY_RUN:
        lines.append(f"M190 S{fmt(BED_TEMP_C, 1)}        ; wait for bed temperature")
        lines.append(f"M109 S{fmt(NOZZLE_TEMP_C, 1)}     ; wait for nozzle temperature")
        lines.append("G92 E0")

    lines.append("")

    # Move to row 0 without extrusion.
    x0, y0, z0, b0, c0 = machine_rows[0]
    lines.append("; Move to first path point")
    lines.append(g1_xyzbc(x0, y0, z0, b0, c0, feed_travel))

    if not DRY_RUN:
        lines.append("; Prime filament")
        lines.append(extrusion_line(PRIME_EXTRUSION_MM, feed=300.0))
        lines.append("G92 E0")

    lines.append("")

    # Main solved path.
    for row_idx in range(1, len(machine_rows)):
        x, y, z, b, c = machine_rows[row_idx]

        if DRY_RUN:
            lines.append(g1_xyzbc(x, y, z, b, c, feed_print))
        else:
            e = float(extrusion_by_row[row_idx])
            lines.append(g1_xyzbc(x, y, z, b, c, feed_print, e=e))

    # End
    lines.append("")
    lines.append("; End")

    if not DRY_RUN:
        lines.append("G92 E0")
        lines.append("G1 E-1.00000 F300")
        lines.append("M104 S0")
        lines.append("M140 S0")

    z_end_current = float(machine_rows[-1, 2])
    z_safe = min(max(z_end_current + 10.0, Z_MIN_MM), Z_MAX_MM)

    lines.append(f"G1 Z{fmt(z_safe, DECIMALS_POS)} F{fmt(feed_travel, 0)}")
    lines.append("M84")

    output_path = next_indexed_output_path(BASE_DIR, GCODE_BASENAME, GCODE_EXT)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {output_path}")
    print(f"Rows: {len(cl)}, first print row: {first_print_idx}")
    print(
        f"Placement offset: X {xyz_offset[0]:+.3f}, "
        f"Y {xyz_offset[1]:+.3f}, Z {xyz_offset[2]:+.3f} mm"
    )
    print(
        f"Placed bounds: X[{bounds_after['x_min']:.3f}, {bounds_after['x_max']:.3f}] "
        f"Y[{bounds_after['y_min']:.3f}, {bounds_after['y_max']:.3f}] "
        f"Z[{bounds_after['z_min']:.3f}, {bounds_after['z_max']:.3f}] mm"
    )
    print(
        f"Extrusion range/sum: {float(np.min(extrusion_by_row)):.6f} to "
        f"{float(np.max(extrusion_by_row)):.6f} / "
        f"{float(np.sum(extrusion_by_row)):.3f} mm"
    )



if __name__ == "__main__":
    main()
