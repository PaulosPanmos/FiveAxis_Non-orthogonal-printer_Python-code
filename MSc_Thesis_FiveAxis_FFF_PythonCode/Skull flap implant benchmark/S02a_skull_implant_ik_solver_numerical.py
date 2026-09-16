# -*- coding: utf-8 -*-
"""
S02a_skull_implant_ik_solver_numerical.py

Numerical inverse kinematics for the skull-implant benchmark.

For each CL row, B and C are solved from the requested tool direction. X, Y
and z_table are then calculated so the nozzle tip reaches the requested point.
Multiple B/C seeds are checked to reduce branch switching.

Input:  P.CLDATA_PATH       (x y z i j k)
Output: P.SOLVED_IK_PATH    (x y z_table B C)

This is the numerical alternative to S02b_skull_implant_ik_solver_analytical.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import S00_skull_implant_parameters as P


# ============================================================
# Numerical IK settings
# ============================================================

IK_BC_MAX_ITERS = int(P.IK_MAX_ITERS)
IK_BC_TOL_DIR = float(P.IK_TOL_DIR)
IK_BC_MU = float(P.IK_MU)

B_AXIS_SIGN = float(P.B_AXIS_SIGN)
C_AXIS_SIGN = float(P.C_AXIS_SIGN)

# Continuity settings used by the numerical solver.
IK_BC_CONTINUITY_LAMBDA = 1e-5
IK_BC_CONTINUITY_WEIGHTS = np.array([10.0, 10.0], dtype=float)
IK_BC_PREDICT_ALPHA = 1.0
IK_DIR_HOLD_EPS = 1e-8

# C-branch preference and range.
IK_PREFERRED_C_SIGN = float(P.IK_PREFERRED_C_SIGN)
IK_C_MIN_RAD = float(P.IK_NUMERICAL_C_MIN_RAD)
IK_C_MAX_RAD = float(P.IK_NUMERICAL_C_MAX_RAD)

IK_C_POSITIVE_PENALTY_LAMBDA = float(P.IK_C_POSITIVE_PENALTY_LAMBDA)
IK_C_POSITIVE_PENALTY_THRESHOLD_RAD = float(P.IK_C_POSITIVE_PENALTY_THRESHOLD_RAD)
IK_C_PREFERRED_BRANCH_SCORE = float(P.IK_C_PREFERRED_BRANCH_SCORE)

IK_BC_BRANCH_NUDGE_C_RAD = float(P.IK_BC_BRANCH_NUDGE_C_RAD)
IK_BC_BRANCH_NUDGE_B_RAD = float(P.IK_BC_BRANCH_NUDGE_B_RAD)
IK_BC_BRANCH_SCORE_DIR = float(P.IK_BC_BRANCH_SCORE_DIR)
IK_BC_BRANCH_SCORE_CONTINUITY = float(P.IK_BC_BRANCH_SCORE_CONTINUITY)
IK_BC_BRANCH_SCORE_POSITIVE_C = float(P.IK_BC_BRANCH_SCORE_POSITIVE_C)

IK_FD_EPS_C_RAD = float(P.IK_FD_EPS_C_RAD)
IK_FD_EPS_B_RAD = float(P.IK_FD_EPS_B_RAD)


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


def _wrap_angle_near(angle: float, reference: float) -> float:
    """Wrap angle by multiples of 2*pi so it stays close to reference."""
    return reference + ((angle - reference + np.pi) % (2.0 * np.pi) - np.pi)


B_BRACKET_ZERO_ROT = Ry(np.deg2rad(P.B_BRACKET_ZERO_TILT_DEG))
TOOL_ZERO_ROT_FROM_HOLDER = Ry(np.deg2rad(P.TOOL_ZERO_ROT_Y_DEG))


# ============================================================
# Bed transforms
# ============================================================

def bed_center_world(z_table: float) -> np.ndarray:
    z = float(z_table)

    if not P.ZTABLE_POSITIVE_DOWN:
        z = -z

    return np.array([0.0, 0.0, P.BED_CENTER_Z0 - z], dtype=float)


def world_to_bed_center(p_world: np.ndarray, z_table: float) -> np.ndarray:
    return np.asarray(p_world, float).reshape(3) - bed_center_world(z_table)


def cl_to_bed_center_position(px, py, pz) -> np.ndarray:
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

    d_tool_world = _unit(R_tool @ np.array([0.0, 0.0, -1.0], dtype=float))

    return p_tip, d_tool_world


def fk_tip_bed_center(q):
    p_world, d_world = fk_tip_world(q)
    return world_to_bed_center(p_world, q[2]), _unit(d_world)


# ============================================================
# BC-only orientation model
# ============================================================

def fk_dir_from_bc(c_cmd: float, b_cmd: float) -> np.ndarray:
    """Tool direction depends only on B and C command angles."""
    B_rot = B_AXIS_SIGN * float(b_cmd)
    C_rot = C_AXIS_SIGN * float(c_cmd)

    R_B = Rz(B_rot) @ B_BRACKET_ZERO_ROT
    R_C = R_B @ Rx(C_rot)
    R_tool = R_C @ TOOL_ZERO_ROT_FROM_HOLDER

    d_tool_world = _unit(R_tool @ np.array([0.0, 0.0, -1.0], dtype=float))
    return d_tool_world


def dir_error_2(d_tool, d_des):
    d_des = _unit(d_des)

    helper = np.array([1.0, 0.0, 0.0])

    if abs(np.dot(helper, d_des)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])

    u = _unit(np.cross(d_des, helper))
    v = _unit(np.cross(d_des, u))

    return np.array([np.dot(d_tool, u), np.dot(d_tool, v)], dtype=float)


def bc_dir_error(cb: np.ndarray, d_des_world: np.ndarray) -> np.ndarray:
    """cb = [C_cmd, B_cmd], returns 2D direction error."""
    c_cmd, b_cmd = map(float, cb)
    d_tool_world = fk_dir_from_bc(c_cmd, b_cmd)
    return dir_error_2(d_tool_world, d_des_world)


def bc_jacobian_fd(cb: np.ndarray, d_des_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference Jacobian of 2D direction error wrt [C_cmd, B_cmd]."""
    cb = np.asarray(cb, dtype=float).copy()
    e0 = bc_dir_error(cb, d_des_world)

    J = np.zeros((2, 2), dtype=float)
    eps = np.array([IK_FD_EPS_C_RAD, IK_FD_EPS_B_RAD], dtype=float)

    for i in range(2):
        dcb = np.zeros(2, dtype=float)
        dcb[i] = eps[i]
        e1 = bc_dir_error(cb + dcb, d_des_world)
        J[:, i] = (e1 - e0) / eps[i]

    return J, e0


# ============================================================
# C side penalties / scores
# ============================================================

def c_positive_penalty_terms(c_cmd: float) -> tuple[float, float]:
    """Return gradient and Hessian terms for the positive-C penalty."""
    if c_cmd > IK_C_POSITIVE_PENALTY_THRESHOLD_RAD:
        dc = c_cmd - IK_C_POSITIVE_PENALTY_THRESHOLD_RAD
        grad = 2.0 * IK_C_POSITIVE_PENALTY_LAMBDA * dc
        hess = 2.0 * IK_C_POSITIVE_PENALTY_LAMBDA
        return grad, hess

    return 0.0, 0.0


def c_positive_branch_score(c_cmd: float) -> float:
    """Scalar score used when comparing BC branch candidates."""
    if c_cmd > IK_C_POSITIVE_PENALTY_THRESHOLD_RAD:
        dc = c_cmd - IK_C_POSITIVE_PENALTY_THRESHOLD_RAD
        return float(dc * dc)

    return 0.0


def c_preferred_side_score(c_cmd: float) -> float:
    """Return the penalty for the non-preferred C branch."""
    if IK_PREFERRED_C_SIGN < 0.0:
        return float(max(0.0, c_cmd) ** 2)

    return float(max(0.0, -c_cmd) ** 2)


# ============================================================
# BC solve
# ============================================================

def solve_bc_for_direction(
    d_des_world: np.ndarray,
    cb0: np.ndarray,
    cb_ref: np.ndarray | None = None,
    max_iters: int = IK_BC_MAX_ITERS,
    tol_dir: float = IK_BC_TOL_DIR,
    mu: float = IK_BC_MU,
) -> tuple[np.ndarray, bool, int, float]:
    """
    Solve BC command angles from desired tool direction.

    cb order:
        cb = [C_cmd, B_cmd]
    """
    cb = np.asarray(cb0, dtype=float).copy()
    d_des_world = _unit(d_des_world)

    use_cont = cb_ref is not None

    if use_cont:
        cb_ref = np.asarray(cb_ref, dtype=float).copy()
        cb[1] = _wrap_angle_near(cb[1], cb_ref[1])
        W_cont = np.diag(IK_BC_CONTINUITY_WEIGHTS)
        lam_cont = IK_BC_CONTINUITY_LAMBDA
    else:
        W_cont = None
        lam_cont = 0.0

    dir_err = np.inf

    for it in range(max_iters):
        J, e = bc_jacobian_fd(cb, d_des_world)
        dir_err = float(np.linalg.norm(e))

        if dir_err < tol_dir:
            return cb, True, it, dir_err

        JTJ = J.T @ J
        A = JTJ + mu * np.eye(2)
        rhs = J.T @ e

        if use_cont and lam_cont > 0.0:
            A = A + lam_cont * W_cont
            rhs = rhs + lam_cont * (W_cont @ (cb - cb_ref))

        grad_positive_c, hess_positive_c = c_positive_penalty_terms(cb[0])

        if hess_positive_c > 0.0:
            A[0, 0] += hess_positive_c
            rhs[0] += grad_positive_c

        dcb = -np.linalg.solve(A, rhs)
        cb += dcb

        # Enforce hard C command limits used by the IK solve.
        # Clamp C to the numerical solver range.
        cb[0] = np.clip(cb[0], IK_C_MIN_RAD, IK_C_MAX_RAD)

        # Keep B on the nearest periodic branch.
        if use_cont:
            cb[1] = _wrap_angle_near(cb[1], cb_ref[1])

    return cb, False, max_iters, dir_err


def score_bc_candidate(
    cb_sol: np.ndarray,
    cb_ref: np.ndarray | None,
    d_des_world: np.ndarray,
) -> float:
    """
    Score a BC command candidate for branch selection.

    Lower is better.

    Includes:
    - direction error
    - continuity distance to previous BC command
    - positive-C branch penalty
    - preferred-C branch score
    """
    cb_sol = np.asarray(cb_sol, dtype=float)

    dir_norm = float(np.linalg.norm(bc_dir_error(cb_sol, d_des_world)))

    if cb_ref is None:
        cont_norm = 0.0
        b_ref = cb_sol[1]
    else:
        cb_ref = np.asarray(cb_ref, dtype=float)
        cb_tmp = cb_sol.copy()
        cb_tmp[1] = _wrap_angle_near(cb_tmp[1], cb_ref[1])
        cont_norm = float(np.linalg.norm(cb_tmp - cb_ref))
        b_ref = cb_ref[1]

    b_wrapped = _wrap_angle_near(cb_sol[1], b_ref)
    cb_wrapped = np.array([cb_sol[0], b_wrapped], dtype=float)

    positive_c_score = c_positive_branch_score(cb_wrapped[0])
    preferred_score = c_preferred_side_score(cb_wrapped[0])

    score = (
        IK_BC_BRANCH_SCORE_DIR * dir_norm
        + IK_BC_BRANCH_SCORE_CONTINUITY * cont_norm
        + IK_BC_BRANCH_SCORE_POSITIVE_C * positive_c_score
        + IK_C_PREFERRED_BRANCH_SCORE * preferred_score
    )

    return float(score)


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

    if cl.ndim != 2 or cl.shape[1] < 6:
        raise ValueError("CL file must have at least 6 columns: px py pz dx dy dz")

    n_rows = cl.shape[0]

    # Output file order:
    #   x y z_table B C
    Q_out = np.zeros((n_rows, 5), dtype=float)

    # Internal BC order:
    #   cb = [C_cmd, B_cmd]
    cb_prev = np.array([0.0, 0.0], dtype=float)
    cb_prev2 = None
    d_prev = None

    fails = 0

    print(f"Numerical IK: {n_rows} rows")
    print(
        f"C range: {np.rad2deg(IK_C_MIN_RAD):.1f} to "
        f"{np.rad2deg(IK_C_MAX_RAD):.1f} deg"
    )

    for k in range(n_rows):
        px, py, pz = cl[k, 0:3]
        dx, dy, dz = cl[k, 3:6]

        p_des_bed_center = cl_to_bed_center_position(px, py, pz)

        # Current convention:
        # CL IJK describes surface normal / outward direction.
        # Nozzle direction is the opposite.
        d_des_world = -_unit([dx, dy, dz])

        # ----------------------------------------------------
        # Stage 1: solve BC command angles from direction only
        # ----------------------------------------------------
        if k == 0:
            cb_seed = cb_prev.copy()
            cb_ref = None

            # Compare zero and preferred-C seeds on the first row.
            seed_zero = cb_seed.copy()
            seed_zero[0] = np.clip(seed_zero[0], IK_C_MIN_RAD, IK_C_MAX_RAD)

            seed_pref = np.array(
                [
                    IK_PREFERRED_C_SIGN * abs(IK_BC_BRANCH_NUDGE_C_RAD),
                    0.0,
                ],
                dtype=float,
            )
            seed_pref[0] = np.clip(seed_pref[0], IK_C_MIN_RAD, IK_C_MAX_RAD)

            cand_zero = solve_bc_for_direction(
                d_des_world=d_des_world,
                cb0=seed_zero,
                cb_ref=cb_ref,
            )

            cand_pref = solve_bc_for_direction(
                d_des_world=d_des_world,
                cb0=seed_pref,
                cb_ref=cb_ref,
            )

            cb_zero, ok_zero, it_zero, dir_zero = cand_zero
            cb_pref, ok_pref, it_pref, dir_pref = cand_pref

            score_zero = score_bc_candidate(cb_zero, cb_ref, d_des_world)
            score_pref = score_bc_candidate(cb_pref, cb_ref, d_des_world)

            if score_pref < score_zero:
                cb_sol, ok_bc, it_bc, dir_err = cb_pref, ok_pref, it_pref, dir_pref
            else:
                cb_sol, ok_bc, it_bc, dir_err = cb_zero, ok_zero, it_zero, dir_zero

        else:
            # Reuse B/C when the requested direction is effectively unchanged.
            if d_prev is not None and np.linalg.norm(d_des_world - d_prev) < IK_DIR_HOLD_EPS:
                cb_sol = cb_prev.copy()
                ok_bc = True
                it_bc = 0
                dir_err = float(np.linalg.norm(bc_dir_error(cb_sol, d_des_world)))

            else:
                if cb_prev2 is None:
                    cb_base = cb_prev.copy()
                else:
                    cb_base = cb_prev + IK_BC_PREDICT_ALPHA * (cb_prev - cb_prev2)

                cb_base[0] = np.clip(cb_base[0], IK_C_MIN_RAD, IK_C_MAX_RAD)
                cb_base[1] = _wrap_angle_near(cb_base[1], cb_prev[1])

                cb_ref = cb_prev.copy()
                cb_ref[0] = np.clip(cb_ref[0], IK_C_MIN_RAD, IK_C_MAX_RAD)

                # Use several seeds to reduce branch switching.
                #
                # The C part of these seeds is clipped to IK_C_MIN_RAD..IK_C_MAX_RAD.
                # When IK_C_MAX_RAD = 0, any positive-C seed becomes C = 0,
                # so the solver cannot intentionally start on the bad positive branch.
                cb_seed_pos = cb_base + np.array(
                    [+IK_BC_BRANCH_NUDGE_C_RAD, +IK_BC_BRANCH_NUDGE_B_RAD],
                    dtype=float,
                )

                cb_seed_neg = cb_base + np.array(
                    [-IK_BC_BRANCH_NUDGE_C_RAD, -IK_BC_BRANCH_NUDGE_B_RAD],
                    dtype=float,
                )

                cb_seed_pref = cb_base + np.array(
                    [
                        IK_PREFERRED_C_SIGN * abs(IK_BC_BRANCH_NUDGE_C_RAD),
                        0.0,
                    ],
                    dtype=float,
                )

                cb_seed_opp = cb_base + np.array(
                    [
                        -IK_PREFERRED_C_SIGN * abs(IK_BC_BRANCH_NUDGE_C_RAD),
                        0.0,
                    ],
                    dtype=float,
                )

                seeds = [
                    cb_seed_pos,
                    cb_seed_neg,
                    cb_seed_pref,
                    cb_seed_opp,
                ]

                candidates = []

                for seed in seeds:
                    seed = np.asarray(seed, dtype=float).copy()

                    # Apply the configured C range to each seed.
                    # This prevents positive-C branch seeds when IK_C_MAX_RAD = 0.
                    seed[0] = np.clip(seed[0], IK_C_MIN_RAD, IK_C_MAX_RAD)

                    # Keep B seed on same periodic branch as previous.
                    seed[1] = _wrap_angle_near(seed[1], cb_prev[1])

                    cand = solve_bc_for_direction(
                        d_des_world=d_des_world,
                        cb0=seed,
                        cb_ref=cb_ref,
                    )

                    cb_cand, ok_cand, it_cand, dir_cand = cand

                    # Clamp the selected candidate to the configured C range.
                    cb_cand = np.asarray(cb_cand, dtype=float).copy()
                    cb_cand[0] = np.clip(cb_cand[0], IK_C_MIN_RAD, IK_C_MAX_RAD)

                    score_cand = score_bc_candidate(cb_cand, cb_ref, d_des_world)

                    candidates.append(
                        (score_cand, cb_cand, ok_cand, it_cand, dir_cand)
                    )

                candidates.sort(key=lambda item: item[0])

                best_score, cb_sol, ok_bc, it_bc, dir_err = candidates[0]

        cb_sol = np.asarray(cb_sol, dtype=float).copy()

        # Clamp C before evaluating the final pose.
        cb_sol[0] = np.clip(cb_sol[0], IK_C_MIN_RAD, IK_C_MAX_RAD)

        c_cmd = float(cb_sol[0])
        b_cmd = float(cb_sol[1])

        # ----------------------------------------------------
        # Stage 2: solve XYZ from position only
        # ----------------------------------------------------
        xyz = solve_xyz_from_position(p_des_bed_center, c_cmd, b_cmd)
        x_center, y_center, z_table = map(float, xyz)

        q = np.array([x_center, y_center, z_table, c_cmd, b_cmd], dtype=float)

        # Evaluate final achieved pose.
        p_ach_bed, d_ach_world = fk_tip_bed_center(q)

        pos_err = float(np.linalg.norm(p_ach_bed - p_des_bed_center))
        dir_err_final = float(np.linalg.norm(dir_error_2(d_ach_world, d_des_world)))

        ok = (
            ok_bc
            and (dir_err_final < P.IK_TOL_DIR)
            and (pos_err < P.IK_TOL_POS_MM)
        )

        cb_prev2 = cb_prev.copy()
        cb_prev = cb_sol.copy()
        d_prev = d_des_world.copy()

        x_out, y_out = bed_center_to_cl_xy(x_center, y_center)

        # Output machine command angles.
        # Internal q order is [x, y, z_table, C, B].
        # Output file order is [x, y, z_table, B, C].
        Q_out[k] = np.array([x_out, y_out, z_table, b_cmd, c_cmd], dtype=float)

        if not ok:
            fails += 1

        if k % 50 == 0:
            print(
                f"[{k+1:5d}/{n_rows}] "
                f"pos={pos_err:.6f} mm "
                f"dir={dir_err_final:.6f} "
                f"C={np.rad2deg(c_cmd): .2f} deg "
                f"B={np.rad2deg(b_cmd): .2f} deg "
                f"{'OK' if ok else 'FAIL'} "
                f"fails={fails}"
            )

    np.savetxt(out_path, Q_out, fmt="%.8f", header="x y z_table B C")

    print()
    print(f"IK complete: {n_rows} rows, {fails} failed")
    print(f"Wrote {out_path}")

    return Q_out


if __name__ == "__main__":
    solve_cl_to_q5()