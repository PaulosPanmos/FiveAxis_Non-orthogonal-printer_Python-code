# -*- coding: utf-8 -*-
"""
S02b_skull_implant_ik_solver_analytical.py

Analytical inverse kinematics for the skull-implant benchmark.

B and C are recovered analytically from the requested tool direction. X, Y
and z_table are then calculated so the nozzle tip reaches the requested point.
Candidate B/C branches are scored for continuity and the configured C range.

Input:  P.CLDATA_PATH       (x y z i j k)
Output: P.SOLVED_IK_PATH    (x y z_table B C)

This is the preferred solver for the dense skull-implant path.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

import S00_skull_implant_parameters as P


# ============================================================
# Analytical IK settings
# ============================================================

B_AXIS_SIGN = float(P.B_AXIS_SIGN)
C_AXIS_SIGN = float(P.C_AXIS_SIGN)

IK_C_MIN_RAD = float(P.IK_C_MIN_RAD)
IK_C_MAX_RAD = float(P.IK_C_MAX_RAD)

IK_TOL_DIR = float(P.IK_TOL_DIR)
IK_TOL_POS_MM = float(P.IK_TOL_POS_MM)

IK_DIR_HOLD_EPS = 1e-5
IK_ANALYTIC_CONTINUITY_WEIGHT = 0.25
IK_ANALYTIC_SMALL_C_WEIGHT = 0.01
IK_ANALYTIC_VERTICAL_EPS = 1e-8


# ============================================================
# Basic math
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


def _wrap_angle_near(angle: float, reference: float) -> float:
    """Wrap angle by multiples of 2*pi so it stays close to reference."""
    return reference + ((angle - reference + np.pi) % (2.0 * np.pi) - np.pi)


B_BRACKET_ZERO_ROT = Ry(np.deg2rad(P.B_BRACKET_ZERO_TILT_DEG))

TOOL_ZERO_ROT_FROM_HOLDER = Ry(
    np.deg2rad(float(P.TOOL_ZERO_ROT_Y_DEG))
)


# ============================================================
# Bed transforms
# ============================================================

def bed_center_world(z_table: float) -> np.ndarray:
    """Return the world position of the bed center."""
    z = float(z_table)

    if not P.ZTABLE_POSITIVE_DOWN:
        z = -z

    return np.array([0.0, 0.0, P.BED_CENTER_Z0 - z], dtype=float)


def world_to_bed_center(p_world: np.ndarray, z_table: float) -> np.ndarray:
    return np.asarray(p_world, dtype=float).reshape(3) - bed_center_world(z_table)


def cl_to_bed_center_position(px, py, pz) -> np.ndarray:
    """Convert a CL position to bed-center coordinates."""
    x = float(px)
    y = float(py)
    z = float(pz)

    if P.XY_ORIGIN == "corner":
        x -= 0.5 * float(P.BED_SIZE[0])
        y -= 0.5 * float(P.BED_SIZE[1])

    if P.CL_Z_ZERO_AT_BED_TOP:
        z -= 0.5 * float(P.BED_SIZE[2])

    return np.array([x, y, z], dtype=float)


def bed_center_to_cl_xy(x_center: float, y_center: float) -> tuple[float, float]:
    """Convert bed-center XY back to the CL convention."""
    x = float(x_center)
    y = float(y_center)

    if P.XY_ORIGIN == "corner":
        x += 0.5 * float(P.BED_SIZE[0])
        y += 0.5 * float(P.BED_SIZE[1])

    return x, y


# ============================================================
# Forward kinematics
# q = [x_center, y_center, z_table, C_cmd, B_cmd]
# ============================================================

def fk_tip_world(q):
    """Return nozzle-tip position and tool direction in world coordinates."""
    x, y, zt, c_cmd, b_cmd = map(float, q)

    carriage_center = np.array(
        [
            x,
            y,
            float(P.GANTRY_Z) - 0.5 * float(P.CARRIAGE_SIZE[2]),
        ],
        dtype=float,
    )

    p_Baxis = carriage_center + P.B_PIVOT_FROM_CARRIAGE_CENTER

    # Convert machine command angles to physical/model rotation angles.
    B_rot = B_AXIS_SIGN * b_cmd
    C_rot = C_AXIS_SIGN * c_cmd

    # B rotation carries the 45-degree B bracket.
    R_B = Rz(B_rot) @ B_BRACKET_ZERO_ROT

    p_b_bracket = p_Baxis + R_B @ P.B_BRACKET_CENTER_FROM_B_LOCAL

    p_b_bracket_end = (
        p_b_bracket
        + R_B @ np.array(
            [0.0, 0.0, -0.5 * P.B_BRACKET_LEN],
            dtype=float,
        )
    )

    p_Caxis = p_b_bracket_end + R_B @ P.C_FROM_BRACKET_END_LOCAL

    # C rotation carries the C-axis arm and tool holder.
    R_C = R_B @ Rx(C_rot)

    p_tool = p_Caxis + R_C @ P.TOOL_FROM_C_LOCAL
    R_tool = R_C @ TOOL_ZERO_ROT_FROM_HOLDER

    p_tool_holder = p_tool + R_tool @ P.TOOL_HOLDER_CENTER_FROM_TOOL_LOCAL

    p_tip = p_tool_holder + R_tool @ P.TIP_FROM_TOOL_HOLDER_CENTER_LOCAL

    # Tool/nozzle direction.
    d_tool_world = _unit(R_tool @ np.array([0.0, 0.0, -1.0], dtype=float))

    return p_tip, d_tool_world


def fk_tip_bed_center(q):
    p_world, d_world = fk_tip_world(q)
    return world_to_bed_center(p_world, q[2]), _unit(d_world)


def fk_dir_from_bc(c_cmd: float, b_cmd: float) -> np.ndarray:
    """Return the tool direction for a B/C command pair."""
    B_rot = B_AXIS_SIGN * float(b_cmd)
    C_rot = C_AXIS_SIGN * float(c_cmd)

    R_B = Rz(B_rot) @ B_BRACKET_ZERO_ROT
    R_C = R_B @ Rx(C_rot)
    R_tool = R_C @ TOOL_ZERO_ROT_FROM_HOLDER

    d_tool_world = _unit(R_tool @ np.array([0.0, 0.0, -1.0], dtype=float))
    return d_tool_world


def dir_error_2(d_tool, d_des):
    """
    2D direction error projected onto plane perpendicular to desired direction.
    The error is evaluated in the plane normal to the desired direction.
    """
    d_des = _unit(d_des)

    helper = np.array([1.0, 0.0, 0.0])

    if abs(np.dot(helper, d_des)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])

    u = _unit(np.cross(d_des, helper))
    v = _unit(np.cross(d_des, u))

    return np.array([np.dot(d_tool, u), np.dot(d_tool, v)], dtype=float)


def bc_dir_error(cb: np.ndarray, d_des_world: np.ndarray) -> np.ndarray:
    """
    cb = [C_cmd, B_cmd]
    """
    c_cmd, b_cmd = map(float, cb)
    d_tool_world = fk_dir_from_bc(c_cmd, b_cmd)
    return dir_error_2(d_tool_world, d_des_world)


# ============================================================
# Analytical BC inverse
# ============================================================

def solve_bc_analytic_for_direction(
    d_des_world: np.ndarray,
    cb_ref: np.ndarray | None = None,
) -> tuple[np.ndarray, bool, int, float]:
    """
    Closed-form BC solve for the current explicit-frame head geometry.

    Solves:
        d_tool_world =
            Rz(B_rot) @ Ry(45) @ Rx(C_rot) @ Ry(-45) @ [0,0,-1]

    For the specific +45/-45 geometry:
        d_z = -0.5 * (1 + cos(C_rot))

    Then B_rot is found from the horizontal direction angle.

    There are two C branches:
        C_rot = +acos(...)
        C_rot = -acos(...)

    We evaluate both and choose the one with:
        - low direction error
        - good continuity to previous [C, B]
        - small C magnitude

    Returns:
        cb_sol, ok, iters, dir_err

    cb order:
        cb = [C_cmd, B_cmd]
    """
    d = _unit(d_des_world)

    # The mechanism's nominal tool direction lives in lower hemisphere.
    # Numerical clipping avoids acos domain errors.
    dz = float(np.clip(d[2], -1.0, 0.0))

    # From:
    #   d_z = -0.5 * (1 + cos(C_rot))
    # Therefore:
    #   cos(C_rot) = -2*d_z - 1
    cos_c = float(np.clip(-2.0 * dz - 1.0, -1.0, 1.0))
    c_abs = float(np.arccos(cos_c))

    h_norm = float(np.linalg.norm(d[:2]))

    # Near vertical:
    # B is ambiguous and C is near zero.
    # Best practical behavior: keep previous B/C, or use zero.
    if h_norm < IK_ANALYTIC_VERTICAL_EPS:
        if cb_ref is not None:
            cb = np.asarray(cb_ref, dtype=float).copy()
        else:
            cb = np.array([0.0, 0.0], dtype=float)

        cb[0] = np.clip(cb[0], IK_C_MIN_RAD, IK_C_MAX_RAD)

        # If C is near zero, this should align with vertical direction.
        err = float(np.linalg.norm(bc_dir_error(cb, d)))
        ok = err < IK_TOL_DIR
        return cb, ok, 0, err

    target_angle = float(np.arctan2(d[1], d[0]))

    candidates = []

    # Two C_rot branches.
    for c_rot in (+c_abs, -c_abs):
        sin_c = float(np.sin(c_rot))

        # Direction before Rz(B_rot):
        #   d_pre_xy = [A, Y]
        #
        # With +45/-45 geometry:
        #   A = 0.5 * (1 - cos(C_rot))
        #   Y = sqrt(0.5) * sin(C_rot)
        A = 0.5 * (1.0 - cos_c)
        Y = np.sqrt(0.5) * sin_c

        base_angle = float(np.arctan2(Y, A))

        # Rz(B_rot) rotates base horizontal direction into target horizontal direction.
        b_rot = target_angle - base_angle

        # Convert physical/model rotations back to machine command angles.
        c_cmd = c_rot / C_AXIS_SIGN
        b_cmd = b_rot / B_AXIS_SIGN

        if cb_ref is not None:
            b_cmd = _wrap_angle_near(b_cmd, cb_ref[1])

        cb = np.array([c_cmd, b_cmd], dtype=float)

        # Respect C command limits.
        if cb[0] < IK_C_MIN_RAD or cb[0] > IK_C_MAX_RAD:
            continue

        dir_err = float(np.linalg.norm(bc_dir_error(cb, d)))

        if cb_ref is None:
            continuity = 0.0
        else:
            cb_tmp = cb.copy()
            cb_tmp[1] = _wrap_angle_near(cb_tmp[1], cb_ref[1])
            continuity = float(np.linalg.norm(cb_tmp - cb_ref))

        score = (
            100.0 * dir_err
            + IK_ANALYTIC_CONTINUITY_WEIGHT * continuity
            + IK_ANALYTIC_SMALL_C_WEIGHT * abs(cb[0])
        )

        candidates.append((score, cb, dir_err))

    # If both C branches were outside the allowed C range:
    # keep previous solution if possible and mark as failed.
    if not candidates:
        if cb_ref is not None:
            cb = np.asarray(cb_ref, dtype=float).copy()
        else:
            cb = np.array([0.0, 0.0], dtype=float)

        cb[0] = np.clip(cb[0], IK_C_MIN_RAD, IK_C_MAX_RAD)
        err = float(np.linalg.norm(bc_dir_error(cb, d)))
        return cb, False, 0, err

    candidates.sort(key=lambda item: item[0])
    _, cb_best, dir_err_best = candidates[0]

    ok = dir_err_best < IK_TOL_DIR
    return cb_best, ok, 0, dir_err_best


# ============================================================
# XYZ solve from position only
# ============================================================

def solve_xyz_from_position(
    p_des_bed_center: np.ndarray,
    c_cmd: float,
    b_cmd: float,
) -> np.ndarray:
    """
    Once BC command angles are fixed, solve XYZ directly from desired nozzle tip position.

    Since XYZ do not affect orientation, they can be chosen to place the tip exactly.
    """
    q_zero_xyz = np.array([0.0, 0.0, 0.0, c_cmd, b_cmd], dtype=float)
    p0_bed, _ = fk_tip_bed_center(q_zero_xyz)

    x_center = float(p_des_bed_center[0] - p0_bed[0])
    y_center = float(p_des_bed_center[1] - p0_bed[1])

    # In bed coordinates, z_table contributes directly with +1.
    z_table = float(p_des_bed_center[2] - p0_bed[2])

    return np.array([x_center, y_center, z_table], dtype=float)


# ============================================================
# Optional diagnostics
# ============================================================

def print_input_diagnostics(cl: np.ndarray) -> None:
    xyz = cl[:, 0:3]
    norms = np.linalg.norm(cl[:, 3:6], axis=1)

    print(
        f"CL bounds: X[{xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}] "
        f"Y[{xyz[:, 1].min():.2f}, {xyz[:, 1].max():.2f}] "
        f"Z[{xyz[:, 2].min():.2f}, {xyz[:, 2].max():.2f}] mm"
    )
    print(f"Normal length: {norms.min():.6f} to {norms.max():.6f}")


# ============================================================
# Batch solve
# ============================================================

def solve_cl_to_q5(
    cl_path: str | Path = P.CLDATA_PATH,
    out_path: str | Path = P.SOLVED_IK_PATH,
):
    cl_path = Path(cl_path)
    out_path = Path(out_path)

    if not cl_path.exists():
        raise FileNotFoundError(f"CLdata file not found:\n{cl_path}")

    cl = np.loadtxt(cl_path, comments="#")

    if cl.ndim == 1:
        cl = cl.reshape(1, -1)

    if cl.ndim != 2 or cl.shape[1] < 6:
        raise ValueError("CL file must have at least 6 columns: x y z i j k")

    n_rows = cl.shape[0]

    # Output file order:
    #   x y z_table B C
    Q_out = np.zeros((n_rows, 5), dtype=float)

    # Internal BC order:
    #   cb = [C_cmd, B_cmd]
    cb_prev = np.array([0.0, 0.0], dtype=float)
    d_prev = None

    fails = 0
    fail_indices = []
    max_pos_err = 0.0
    max_dir_err = 0.0

    print(f"Analytical IK: {n_rows} rows")
    print(
        f"C range: {np.rad2deg(IK_C_MIN_RAD):.1f} to "
        f"{np.rad2deg(IK_C_MAX_RAD):.1f} deg"
    )
    print_input_diagnostics(cl)

    for row_idx in range(n_rows):
        px, py, pz = cl[row_idx, 0:3]
        dx, dy, dz = cl[row_idx, 3:6]

        p_des_bed_center = cl_to_bed_center_position(px, py, pz)

        # CL IJK is the outward surface normal; the nozzle points opposite to it.
        d_des_world = -_unit([dx, dy, dz])

        # ----------------------------------------------------
        # Stage 1: analytical BC solve from direction only
        # ----------------------------------------------------
        if row_idx == 0:
            cb_ref = None
        else:
            cb_ref = cb_prev.copy()

        if (
            d_prev is not None
            and np.linalg.norm(d_des_world - d_prev) < IK_DIR_HOLD_EPS
        ):
            cb_sol = cb_prev.copy()
            ok_bc = True
            it_bc = 0
            dir_err_bc = float(np.linalg.norm(bc_dir_error(cb_sol, d_des_world)))
        else:
            cb_sol, ok_bc, it_bc, dir_err_bc = solve_bc_analytic_for_direction(
                d_des_world=d_des_world,
                cb_ref=cb_ref,
            )

        cb_sol = np.asarray(cb_sol, dtype=float).copy()

        # Final C safety clamp.
        cb_sol[0] = np.clip(cb_sol[0], IK_C_MIN_RAD, IK_C_MAX_RAD)

        c_cmd = float(cb_sol[0])
        b_cmd = float(cb_sol[1])

        # ----------------------------------------------------
        # Stage 2: direct XYZ solve from position only
        # ----------------------------------------------------
        xyz = solve_xyz_from_position(p_des_bed_center, c_cmd, b_cmd)
        x_center, y_center, z_table = map(float, xyz)

        q = np.array([x_center, y_center, z_table, c_cmd, b_cmd], dtype=float)

        # Evaluate final achieved pose.
        p_ach_bed, d_ach_world = fk_tip_bed_center(q)

        pos_err = float(np.linalg.norm(p_ach_bed - p_des_bed_center))
        dir_err_final = float(np.linalg.norm(dir_error_2(d_ach_world, d_des_world)))

        max_pos_err = max(max_pos_err, pos_err)
        max_dir_err = max(max_dir_err, dir_err_final)

        ok = (
            ok_bc
            and (dir_err_final < IK_TOL_DIR)
            and (pos_err < IK_TOL_POS_MM)
        )

        cb_prev = cb_sol.copy()
        d_prev = d_des_world.copy()

        x_out, y_out = bed_center_to_cl_xy(x_center, y_center)

        # Output machine command angles.
        # Internal q order is [x, y, z_table, C, B].
        # Output file order is [x, y, z_table, B, C].
        Q_out[row_idx] = np.array(
            [x_out, y_out, z_table, b_cmd, c_cmd],
            dtype=float,
        )

        if not ok:
            fails += 1
            fail_indices.append(row_idx)

        if row_idx % 1000 == 0 or row_idx == n_rows - 1:
            print(
                f"[{row_idx + 1:6d}/{n_rows}] "
                f"pos={pos_err:.6f} mm "
                f"dir={dir_err_final:.6f} "
                f"B={np.rad2deg(b_cmd): .3f} deg "
                f"C={np.rad2deg(c_cmd): .3f} deg "
                f"{'OK' if ok else 'FAIL'} "
                f"fails={fails}"
            )

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, Q_out, fmt="%.8f", header="x y z_table B C")

    print()
    print(f"IK complete: {n_rows} rows, {fails} failed")
    print(f"Max errors: {max_pos_err:.6f} mm position, {max_dir_err:.6f} direction")
    print(f"Wrote {out_path}")

    if fails > 0:
        fail_path = out_path.with_name(out_path.stem + "_fail_indices.txt")
        np.savetxt(fail_path, np.asarray(fail_indices, dtype=int), fmt="%d")
        print(f"Wrote failed row indices to {fail_path}")

    return Q_out


if __name__ == "__main__":
    solve_cl_to_q5()
    