# -*- coding: utf-8 -*-
"""
S01_skull_implant_cldata_generator.py

Converts the AtomSlicer skull-implant path to x y z i j k CL data.

The parser tracks modal XYZ coordinates and reads the surface-normal comment
attached to each accepted G0/G1 row. Optional filtering, centering and normal
handling are controlled by S00_skull_implant_parameters.py.
"""
from __future__ import annotations

import re
import math
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import S00_skull_implant_parameters as P


# ============================================================
# Files and conversion settings
# ============================================================

INPUT_GCODE = Path(P.ATOMSLICER_GCODE_PATH)
OUTPUT_CLDATA = Path(P.CLDATA_PATH)

XY_CENTER_MODE = str(P.CL_XY_CENTER_MODE)
Z_OFFSET_MM = float(P.CL_Z_OFFSET_MM)

FLIP_NORMALS = bool(P.CL_FLIP_NORMALS)
ONLY_EXTRUSION_MOVES = bool(P.CL_ONLY_EXTRUSION_MOVES)
REMOVE_DUPLICATE_POINTS = bool(P.CL_REMOVE_DUPLICATE_POINTS)

XYZ_DECIMALS = int(P.CL_XYZ_DECIMALS)
IJK_DECIMALS = int(P.CL_IJK_DECIMALS)


# ============================================================
# Parsing expressions
# ============================================================

FLOAT_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

# Normal-vector comment, for example:
# ; (0.000000, 0.000000, 1.000000)
NORMAL_RE = re.compile(
    rf";.*?\(\s*({FLOAT_RE})\s*,\s*({FLOAT_RE})\s*,\s*({FLOAT_RE})\s*\)"
)

# G-code words such as X152.7, Y140.9, Z0.2, E0.01 and F600.
WORD_RE = re.compile(rf"([A-Za-z])\s*({FLOAT_RE})")


def parse_gcode_words(line: str) -> Dict[str, float]:
    """Extract numeric G-code words from one line."""
    words: Dict[str, float] = {}
    for letter, value in WORD_RE.findall(line):
        words[letter.upper()] = float(value)
    return words


def parse_normal(line: str) -> Optional[Tuple[float, float, float]]:
    """Extract an (i, j, k) normal vector from the line comment."""
    match = NORMAL_RE.search(line)
    if not match:
        return None

    i = float(match.group(1))
    j = float(match.group(2))
    k = float(match.group(3))
    return i, j, k


def normalize_vector(i: float, j: float, k: float) -> Tuple[float, float, float]:
    """Normalize a surface-normal vector."""
    length = math.sqrt(i * i + j * j + k * k)

    if length < 1e-12:
        raise ValueError("Found zero-length normal vector.")

    return i / length, j / length, k / length


def almost_same_point(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    tol: float = 1e-9,
) -> bool:
    return (
        abs(a[0] - b[0]) < tol
        and abs(a[1] - b[1]) < tol
        and abs(a[2] - b[2]) < tol
    )


def read_gcode_as_cl_rows(input_file: Path) -> List[Tuple[float, float, float, float, float, float]]:
    """Read AtomSlicer motion rows as x, y, z, i, j, k tuples."""
    rows: List[Tuple[float, float, float, float, float, float]] = []

    current_x: Optional[float] = None
    current_y: Optional[float] = None
    current_z: Optional[float] = None

    with input_file.open("r", encoding="utf-8", errors="replace") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line:
                continue

            # Only parse G0/G1 motion lines.
            if not (line.startswith("G0") or line.startswith("G1")):
                continue

            normal = parse_normal(line)
            if normal is None:
                continue

            words = parse_gcode_words(line)

            # Ignore malformed or unsupported motion commands.
            g_value = words.get("G")
            if g_value not in (0.0, 1.0):
                continue

            # Update modal XYZ coordinates; omitted axes retain their previous value.
            if "X" in words:
                current_x = words["X"]
            if "Y" in words:
                current_y = words["Y"]
            if "Z" in words:
                current_z = words["Z"]

            # CL data require a complete XYZ position.
            if current_x is None or current_y is None or current_z is None:
                continue

            # Optional extrusion-only filtering.
            if ONLY_EXTRUSION_MOVES:
                e_value = words.get("E", 0.0)
                if e_value <= 0.0:
                    continue

            i, j, k = normalize_vector(*normal)

            if FLIP_NORMALS:
                i, j, k = -i, -j, -k

            x = current_x
            y = current_y
            z = current_z + Z_OFFSET_MM

            if REMOVE_DUPLICATE_POINTS and rows:
                previous_xyz = rows[-1][0:3]
                current_xyz = (x, y, z)
                if almost_same_point(previous_xyz, current_xyz):
                    continue

            rows.append((x, y, z, i, j, k))

    return rows


def apply_xy_centering(
    rows: List[Tuple[float, float, float, float, float, float]],
    mode: str,
) -> Tuple[List[Tuple[float, float, float, float, float, float]], Tuple[float, float]]:
    """Apply the selected XY centering mode and return the removed offset."""
    if not rows:
        return rows, (0.0, 0.0)

    mode = mode.lower().strip()

    if mode == "none":
        return rows, (0.0, 0.0)

    if mode == "first_point_xy":
        x_offset = rows[0][0]
        y_offset = rows[0][1]

    elif mode == "bbox_center_xy":
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]

        x_offset = 0.5 * (min(xs) + max(xs))
        y_offset = 0.5 * (min(ys) + max(ys))

    else:
        raise ValueError(
            f"Unknown XY_CENTER_MODE: {mode!r}. "
            "Use 'none', 'bbox_center_xy', or 'first_point_xy'."
        )

    centered_rows = [
        (x - x_offset, y - y_offset, z, i, j, k)
        for x, y, z, i, j, k in rows
    ]

    return centered_rows, (x_offset, y_offset)


def write_cldata(
    output_file: Path,
    rows: List[Tuple[float, float, float, float, float, float]],
    input_file: Path,
    xy_offset: Tuple[float, float],
) -> None:
    """Write the converted CL data file."""
    xyz_fmt = f"{{:.{XYZ_DECIMALS}f}}"
    ijk_fmt = f"{{:.{IJK_DECIMALS}f}}"

    with output_file.open("w", encoding="utf-8") as f:
        f.write("# CLdata converted from AtomSlicer 3-axis G-code\n")
        f.write(f"# Source G-code: {input_file.name}\n")
        f.write("# Columns: x y z i j k\n")
        f.write(f"# XY_CENTER_MODE: {XY_CENTER_MODE}\n")
        f.write(f"# X offset subtracted: {xy_offset[0]:.6f}\n")
        f.write(f"# Y offset subtracted: {xy_offset[1]:.6f}\n")
        f.write(f"# Z_OFFSET_MM added: {Z_OFFSET_MM:.6f}\n")
        f.write(f"# FLIP_NORMALS: {FLIP_NORMALS}\n")
        f.write(f"# ONLY_EXTRUSION_MOVES: {ONLY_EXTRUSION_MOVES}\n")
        f.write(f"# Number of rows: {len(rows)}\n")
        f.write("# x y z i j k\n")

        for x, y, z, i, j, k in rows:
            f.write(
                f"{xyz_fmt.format(x)} "
                f"{xyz_fmt.format(y)} "
                f"{xyz_fmt.format(z)} "
                f"{ijk_fmt.format(i)} "
                f"{ijk_fmt.format(j)} "
                f"{ijk_fmt.format(k)}\n"
            )


def main() -> None:
    if not INPUT_GCODE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_GCODE.resolve()}\n\n"
            "Check ATOMSLICER_GCODE_PATH in S00_skull_implant_parameters.py."
        )

    rows = read_gcode_as_cl_rows(INPUT_GCODE)

    if not rows:
        raise RuntimeError(
            "No CLdata rows were parsed.\n"
            "This usually means the script did not find G0/G1 lines with comments like:\n"
            "G1 X... Y... Z... E... ; (i, j, k)"
        )

    rows, xy_offset = apply_xy_centering(rows, XY_CENTER_MODE)

    write_cldata(
        output_file=OUTPUT_CLDATA,
        rows=rows,
        input_file=INPUT_GCODE,
        xy_offset=xy_offset,
    )

    print(f"Wrote {len(rows)} CL rows to {OUTPUT_CLDATA.name}")
    print(
        f"XY offset: X {xy_offset[0]:.3f}, Y {xy_offset[1]:.3f} mm; "
        f"Z offset: {Z_OFFSET_MM:.3f} mm"
    )


if __name__ == "__main__":
    main()