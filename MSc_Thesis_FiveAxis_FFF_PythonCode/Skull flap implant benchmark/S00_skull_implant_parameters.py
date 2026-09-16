# -*- coding: utf-8 -*-
"""
S00_skull_implant_parameters.py

Shared configuration for the skull-implant benchmark.

The AtomSlicer converter, IK solvers, visualizer and G-code generator read
their shared paths and machine settings from this file. Paths are relative to
this file so the benchmark folder can be moved without changing absolute paths.

Units:
- lengths: mm
- angles: radians unless the name ends in _DEG
- CL data: x y z i j k
- solved IK: x y z_table B C
"""

from __future__ import annotations

import os
import numpy as np


# ============================================================
# Benchmark files
# ============================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = THIS_DIR

# Original AtomSlicer source. The same file is used by S01 to create CL data
# and by S04 to preserve AtomSlicer's relative extrusion values.
INPUT_GCODE_FILENAME = "skull_implant_atomslicer.gcode"
ATOMSLICER_GCODE_PATH = os.path.join(BASE_DIR, INPUT_GCODE_FILENAME)

CLDATA_FILENAME = "01_skull_implant_cldata.txt"
CLDATA_PATH = os.path.join(BASE_DIR, CLDATA_FILENAME)

SOLVED_IK_FILENAME = "02_skull_implant_solved_ik.txt"
SOLVED_IK_PATH = os.path.join(BASE_DIR, SOLVED_IK_FILENAME)


# ============================================================
# AtomSlicer -> CL-data conversion
# ============================================================

# Coordinate handling:
# "none"           : keep AtomSlicer X/Y unchanged
# "bbox_center_xy" : subtract the path bounding-box center
# "first_point_xy" : subtract the first parsed X/Y point
CL_XY_CENTER_MODE = "none"

CL_Z_OFFSET_MM = 0.0

# Surface-normal handling.
CL_FLIP_NORMALS = False

# False keeps every G0/G1 path row carrying a normal comment.
# True keeps only rows with positive extrusion.
CL_ONLY_EXTRUSION_MOVES = False

# Remove consecutive duplicate XYZ rows. S04 mirrors this setting when mapping
# AtomSlicer extrusion values back onto the CL/IK rows.
CL_REMOVE_DUPLICATE_POINTS = True

CL_XYZ_DECIMALS = 6
CL_IJK_DECIMALS = 6


# ============================================================
# Coordinate conventions
# ============================================================

# CL X/Y coordinates are measured from the bed center.
XY_ORIGIN = "center"

# Positive z_table moves the bed downward in the world frame.
ZTABLE_POSITIVE_DOWN = True

# CL Z=0 refers to the bed top surface rather than the bed center.
CL_Z_ZERO_AT_BED_TOP = True


# ============================================================
# Machine envelope and frame geometry
# ============================================================

BASE_SIZE = np.array([500.0, 500.0, 30.0], dtype=float)
BED_SIZE = np.array([150.0, 150.0, 10.0], dtype=float)

Z_BASE_TOP = float(BASE_SIZE[2])
SPINDLE_HEIGHT = 500.0
GANTRY_Z = Z_BASE_TOP + SPINDLE_HEIGHT + 150.0

FRAME_X = 460.0
FRAME_Y = 460.0
POST_RADIUS = 12.0
BEAM_RADIUS = 10.0
SPINDLE_RADIUS = 20.0

CARRIAGE_SIZE = np.array([109.0, 100.0, 100.0], dtype=float)


# ============================================================
# Five-axis head geometry
# Must match both IK solvers and the visualizer.
# ============================================================

# B-axis pivot relative to the XY-carriage center.
B_PIVOT_FROM_CARRIAGE_CENTER = np.array([
    -26.0,
      0.0,
    -66.0,
], dtype=float)

# B bracket: B pivot -> C pivot.
B_BRACKET_ZERO_TILT_DEG = +45.0
B_BRACKET_LEN = 77.7

B_BRACKET_SIZE = np.array([18.0, 18.0, B_BRACKET_LEN], dtype=float)

B_BRACKET_CENTER_FROM_B_LOCAL = np.array([
    0.0,
    0.0,
    -0.5 * B_BRACKET_LEN,
], dtype=float)

C_FROM_BRACKET_END_LOCAL = np.array([
    0.0,
    0.0,
    0.0,
], dtype=float)

# C-axis arm: C pivot -> tool frame.
C_ARM_LEN = 43.5

C_ARM_SIZE = np.array([C_ARM_LEN, 18.0, 18.0], dtype=float)

C_ARM_CENTER_FROM_C_LOCAL = np.array([
    +0.5 * C_ARM_LEN,
    0.0,
    0.0,
], dtype=float)

TOOL_FROM_C_LOCAL = np.array([
    +C_ARM_LEN,
    0.0,
    0.0,
], dtype=float)

# Tool holder and nozzle tip.
# The -45 deg tool rotation cancels the B bracket's +45 deg zero tilt at B=C=0.
TOOL_ZERO_ROT_Y_DEG = -45.0

TOOL_HOLDER_SIZE = np.array([18.0, 18.0, 42.72], dtype=float)

TOOL_TIP_X_CAL_MM = +0.0

TOOL_HOLDER_CENTER_FROM_TOOL_LOCAL = np.array([
    TOOL_TIP_X_CAL_MM,
    0.0,
    0.0,
], dtype=float)

TIP_FROM_TOOL_HOLDER_CENTER_LOCAL = np.array([
    26.8,
    0.59,
    -47.8,
], dtype=float)


# ============================================================
# IK solver
# ============================================================

# Convert commanded B/C angles to physical/model rotation directions.
B_AXIS_SIGN = -1.0
C_AXIS_SIGN = -1.0

IK_TOL_POS_MM = 0.02
IK_TOL_DIR = 0.01

# C-axis range used by the analytical solver.
IK_C_MIN_RAD = np.deg2rad(-190.0)
IK_C_MAX_RAD = np.deg2rad(+90.0)

# Narrower C-axis range used by the numerical comparison solver.
IK_NUMERICAL_C_MIN_RAD = np.deg2rad(-90.0)
IK_NUMERICAL_C_MAX_RAD = np.deg2rad(+90.0)

# Numerical solver settings.
IK_MAX_ITERS = 1000
IK_MU = 0.1
IK_FD_EPS_C_RAD = 1e-4
IK_FD_EPS_B_RAD = 1e-4

IK_PREFERRED_C_SIGN = -1.0
IK_C_POSITIVE_PENALTY_LAMBDA = 1e-3
IK_C_POSITIVE_PENALTY_THRESHOLD_RAD = 0.0
IK_C_PREFERRED_BRANCH_SCORE = 20.0

IK_BC_BRANCH_NUDGE_C_RAD = 1.0
IK_BC_BRANCH_NUDGE_B_RAD = 1.0

IK_BC_BRANCH_SCORE_DIR = 1.0
IK_BC_BRANCH_SCORE_CONTINUITY = 0.1
IK_BC_BRANCH_SCORE_POSITIVE_C = 5.0


# ============================================================
# Visualizer geometry
# ============================================================

VIZ_AXIS_RADIUS = 2.5
VIZ_AXIS_B_LEN = 150.0
VIZ_AXIS_C_LEN = 120.0
VIZ_TIP_SPHERE_RADIUS = 2.5

CAMERA_POSITION = [
    (-700.0, -1300.0, 900.0),
    (0.0, 0.0, 250.0),
    (0.0, 0.0, 1.0),
]


# ============================================================
# IK-to-G-code output and machine mapping
# ============================================================

GCODE_BASENAME = "04_skull_implant_5axis_print"
GCODE_EXT = ".gcode"

# Solved IK X/Y are centered around (0, 0); machine X/Y use 0..150 mm.
GCODE_X_CENTER_OFFSET_MM = 75.0
GCODE_Y_CENTER_OFFSET_MM = 75.0

# Map solved z_table to machine Z.
GCODE_ZTABLE_TO_MACHINE_Z_OFFSET = 5.0
GCODE_ZTABLE_SIGN = 1.0

# IK stores B/C in radians; Klipper commands use degrees.
GCODE_ANGLES_IN_RADIANS = True
GCODE_OUTPUT_BC_IN_DEGREES = True

GCODE_B_OFFSET = 0.0
GCODE_C_OFFSET = 0.0


# ============================================================
# G-code machine limits and automatic placement
# ============================================================

GCODE_X_MIN_MM = 0.0
GCODE_X_MAX_MM = 150.0

GCODE_Y_MIN_MM = 0.0
GCODE_Y_MAX_MM = 150.0

# Tilted five-axis poses can require negative Z-table coordinates while the
# nozzle tip remains above the buildplate/part.
GCODE_Z_MIN_MM = -50.0
GCODE_Z_MAX_MM = 332.8

GCODE_C_MIN_DEG = -190.0
GCODE_C_MAX_DEG = 90.0

# B is intentionally not range-checked in the current machine configuration.
GCODE_CHECK_B_RANGE = False
GCODE_B_MIN_DEG = -1e9
GCODE_B_MAX_DEG = +1e9

GCODE_AUTO_PLACE_XY = True
GCODE_AUTO_PLACE_Z = True

# Z placement is referenced to the first positive-extrusion path row.
GCODE_Z_PLACEMENT_REFERENCE = "first_print"
GCODE_TARGET_Z_MIN_MM = 0.20


# ============================================================
# G-code print settings
# ============================================================

GCODE_DRY_RUN = False

GCODE_BED_TEMP_C = 50.0
GCODE_NOZZLE_TEMP_C = 200.0

GCODE_PRINT_SPEED_MM_S = 8.0
GCODE_TRAVEL_SPEED_MM_S = 40.0

GCODE_PRIME_EXTRUSION_MM = 15.0
GCODE_EXTRUSION_RELATIVE = True

GCODE_DECIMALS_POS = 4
GCODE_DECIMALS_E = 5
GCODE_DECIMALS_ANGLE = 4


# ============================================================
# Derived home geometry
# ============================================================

def compute_home_bed_center_z0() -> float:
    """
    Compute the bed-center world Z coordinate for z_table=0.

    At the zero pose (X=Y=z_table=B=C=0), the nozzle tip is assumed to touch
    the bed top. The calculation mirrors the explicit head-frame geometry used
    by the IK solvers and visualizer.
    """
    carriage_center_z0 = GANTRY_Z - 0.5 * CARRIAGE_SIZE[2]
    p_Baxis_z0 = carriage_center_z0 + B_PIVOT_FROM_CARRIAGE_CENTER[2]

    def Ry_deg(a_deg):
        a = np.deg2rad(a_deg)
        ca, sa = np.cos(a), np.sin(a)
        return np.array([
            [ ca, 0.0, sa],
            [0.0, 1.0, 0.0],
            [-sa, 0.0, ca],
        ], dtype=float)

    R_B = Ry_deg(B_BRACKET_ZERO_TILT_DEG)

    p_b_bracket = (
        np.array([0.0, 0.0, p_Baxis_z0], dtype=float)
        + R_B @ B_BRACKET_CENTER_FROM_B_LOCAL
    )

    p_bracket_end = (
        p_b_bracket
        + R_B @ np.array([0.0, 0.0, -0.5 * B_BRACKET_LEN], dtype=float)
    )

    p_Caxis = p_bracket_end + R_B @ C_FROM_BRACKET_END_LOCAL
    p_tool = p_Caxis + R_B @ TOOL_FROM_C_LOCAL

    R_tool = R_B @ Ry_deg(TOOL_ZERO_ROT_Y_DEG)

    p_tool_holder = p_tool + R_tool @ TOOL_HOLDER_CENTER_FROM_TOOL_LOCAL
    p_tip = p_tool_holder + R_tool @ TIP_FROM_TOOL_HOLDER_CENTER_LOCAL

    bed_top_z0 = p_tip[2]
    bed_center_z0 = bed_top_z0 - 0.5 * BED_SIZE[2]

    return float(bed_center_z0)


BED_CENTER_Z0 = compute_home_bed_center_z0()
