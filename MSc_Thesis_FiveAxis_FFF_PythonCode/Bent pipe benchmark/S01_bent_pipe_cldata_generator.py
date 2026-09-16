# -*- coding: utf-8 -*-
"""
S01_bent_pipe_cldata_generator.py

Generates the CL toolpath for the bent-pipe benchmark.

The path follows a vertical section, a circular bend and a final straight
section. A helix is built around this centerline and written as x y z i j k
CL data. A Matplotlib preview is shown after generation.

Settings are read from S00_bent_pipe_parameters.py.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

import S00_bent_pipe_parameters as P


# ============================================================
# Utilities
# ============================================================

def _normalize_rows(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.where(n < eps, 1.0, n)
    return v / n


def _arc_lengths(poly: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.zeros(poly.shape[0], dtype=float)
    s[1:] = np.cumsum(d)
    return s


def _resample_polyline_by_arclength(poly: np.ndarray, ds: float) -> Tuple[np.ndarray, np.ndarray]:
    if ds <= 0:
        raise ValueError("ds must be > 0")

    s = _arc_lengths(poly)
    L = s[-1]

    if L <= 0:
        return poly.copy(), s.copy()

    n = int(np.floor(L / ds)) + 1
    n = max(n, 2)

    s_rs = np.linspace(0.0, L, n, endpoint=True)

    x = np.interp(s_rs, s, poly[:, 0])
    y = np.interp(s_rs, s, poly[:, 1])
    z = np.interp(s_rs, s, poly[:, 2])

    poly_rs = np.column_stack((x, y, z))
    return poly_rs, s_rs


def _parallel_transport_frames(axis: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build smooth normal/binormal frames along the centerline."""
    N = axis.shape[0]
    n = np.zeros_like(axis)
    b = np.zeros_like(axis)

    ref = np.array([0.0, 1.0, 0.0], dtype=float)
    if np.linalg.norm(np.cross(ref, t[0])) < 1e-6:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)

    n0 = ref - np.dot(ref, t[0]) * t[0]
    n0_norm = np.linalg.norm(n0)

    if n0_norm < 1e-12:
        n0 = np.array([0.0, 0.0, 1.0], dtype=float)
        n0_norm = 1.0

    n[0] = n0 / n0_norm
    b[0] = np.cross(t[0], n[0])

    for idx in range(1, N):
        v = np.cross(t[idx - 1], t[idx])
        v_norm = np.linalg.norm(v)

        if v_norm < 1e-10:
            n[idx] = n[idx - 1]
            b[idx] = b[idx - 1]
            continue

        v_unit = v / v_norm
        cos_ang = np.clip(np.dot(t[idx - 1], t[idx]), -1.0, 1.0)
        ang = np.arccos(cos_ang)
        sin_ang = np.sin(ang)

        n_prev = n[idx - 1]
        n_rot = (
            n_prev * cos_ang
            + np.cross(v_unit, n_prev) * sin_ang
            + v_unit * np.dot(v_unit, n_prev) * (1.0 - cos_ang)
        )

        n_rot_norm = np.linalg.norm(n_rot)
        if n_rot_norm < 1e-12:
            n_rot = n_prev
            n_rot_norm = np.linalg.norm(n_rot) or 1.0

        n[idx] = n_rot / n_rot_norm
        b[idx] = np.cross(t[idx], n[idx])

    return n, b


def _set_equal_aspect_3d(ax, xs, ys, zs):
    """Use equal axis scaling for the 3D preview."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    zs = np.asarray(zs, dtype=float)

    x_mid = 0.5 * (xs.max() + xs.min())
    y_mid = 0.5 * (ys.max() + ys.min())
    z_mid = 0.5 * (zs.max() + zs.min())

    x_range = xs.max() - xs.min()
    y_range = ys.max() - ys.min()
    z_range = zs.max() - zs.min()

    max_range = max(x_range, y_range, z_range)
    half = 0.5 * max_range

    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)

    ax.set_box_aspect((1, 1, 1))


def _smoothstep01(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


# ============================================================
# Geometry
# ============================================================

def _build_bent_axis_raw(
    axis_z_length: float,
    bend_radius: float,
    bend_angle_deg: float,
    axis_x_length: float,
    raw_step: float,
) -> np.ndarray:
    """Build the vertical, bend and final straight centerline sections."""
    bend_angle_rad = np.deg2rad(bend_angle_deg)

    n_z_raw = max(2, int(np.ceil(axis_z_length / raw_step)) + 1)
    n_bend_raw = max(10, int(np.ceil((bend_radius * bend_angle_rad) / raw_step)) + 1)
    n_x_raw = max(2, int(np.ceil(axis_x_length / raw_step)) + 1)

    # Straight vertical section
    z1 = np.linspace(0.0, axis_z_length, n_z_raw)
    x1 = np.zeros_like(z1)
    y1 = np.zeros_like(z1)

    # Circular bend
    phi = np.linspace(0.0, bend_angle_rad, n_bend_raw)
    x2 = x1[-1] + bend_radius * (1.0 - np.cos(phi))
    y2 = np.full_like(x2, y1[-1])
    z2 = z1[-1] + bend_radius * np.sin(phi)

    # Straight section after bend
    s3 = np.linspace(0.0, axis_x_length, n_x_raw)
    x3 = x2[-1] + s3 * np.sin(bend_angle_rad)
    y3 = np.full_like(x3, y2[-1])
    z3 = z2[-1] + s3 * np.cos(bend_angle_rad)

    return np.column_stack((
        np.concatenate([x1, x2[1:], x3[1:]]),
        np.concatenate([y1, y2[1:], y3[1:]]),
        np.concatenate([z1, z2[1:], z3[1:]]),
    ))


# ============================================================
# Main
# ============================================================

def generate_bent_helix_toolpath(
    axis_z_length=P.CL_AXIS_Z_LENGTH,
    bend_radius=P.CL_BEND_RADIUS,
    bend_angle_deg=P.CL_BEND_ANGLE_DEG,
    axis_x_length=P.CL_AXIS_X_LENGTH,
    helix_radius=P.CL_HELIX_RADIUS,
    final_pitch_mm=P.CL_FINAL_PITCH_MM,
    points_per_turn=P.CL_POINTS_PER_TURN,
    outfile=P.CLDATA_PATH,
    viz_step=P.CL_VIZ_STEP,
    bed_size_mm=P.BED_SIZE,
    bed_margin_mm=P.CL_BED_MARGIN_MM,
    z_lift_mm=P.CL_Z_LIFT_MM,
    auto_fit_to_bed=P.CL_AUTO_FIT_TO_BED,
    flat_first_turns: float = P.CL_FLAT_FIRST_TURNS,
    blend_turns: float = P.CL_BLEND_TURNS,
):
    if final_pitch_mm <= 0:
        raise ValueError("final_pitch_mm must be > 0")
    if points_per_turn < 6:
        raise ValueError("points_per_turn should be at least 6")
    if helix_radius <= 0:
        raise ValueError("helix_radius must be > 0")
    if flat_first_turns < 0:
        raise ValueError("flat_first_turns must be >= 0")
    if blend_turns < 0:
        raise ValueError("blend_turns must be >= 0")

    # Centerline sampling follows the requested pitch resolution.
    sample_ds_mm = final_pitch_mm / points_per_turn

    # --------------------------------------------------------
    # Build curved centerline
    # --------------------------------------------------------
    raw_step = min(0.5, max(sample_ds_mm * 2.0, 0.05))

    axis_raw = _build_bent_axis_raw(
        axis_z_length=axis_z_length,
        bend_radius=bend_radius,
        bend_angle_deg=bend_angle_deg,
        axis_x_length=axis_x_length,
        raw_step=raw_step,
    )

    # --------------------------------------------------------
    # Optional auto-fit to bed
    # --------------------------------------------------------
    fit_scale = 1.0

    if auto_fit_to_bed:
        bed_w_mm, bed_d_mm, _ = bed_size_mm
        usable_x_mm = bed_w_mm - 2.0 * bed_margin_mm
        usable_y_mm = bed_d_mm - 2.0 * bed_margin_mm

        x_min = axis_raw[:, 0].min() - helix_radius
        x_max = axis_raw[:, 0].max() + helix_radius
        y_min = axis_raw[:, 1].min() - helix_radius
        y_max = axis_raw[:, 1].max() + helix_radius

        span_x = max(1e-12, x_max - x_min)
        span_y = max(1e-12, y_max - y_min)

        sx = usable_x_mm / span_x
        sy = usable_y_mm / span_y

        fit_scale = min(sx, sy, 1.0)

    axis_scaled_raw = axis_raw * fit_scale
    helix_radius_scaled = helix_radius * fit_scale

    # --------------------------------------------------------
    # Resample centerline by arclength
    # --------------------------------------------------------
    axis, s = _resample_polyline_by_arclength(axis_scaled_raw, ds=sample_ds_mm)
    N = axis.shape[0]

    # --------------------------------------------------------
    # Compute local tangent frames
    # --------------------------------------------------------
    tang = np.zeros_like(axis)
    tang[1:-1] = axis[2:] - axis[:-2]
    tang[0] = axis[1] - axis[0]
    tang[-1] = axis[-1] - axis[-2]

    t = _normalize_rows(tang)

    # Smooth local frames around the centerline.
    n_vec, b_vec = _parallel_transport_frames(axis, t)

    # --------------------------------------------------------
    # Helical phase and startup logic
    # --------------------------------------------------------
    omega = 2.0 * np.pi / final_pitch_mm
    phase = omega * s
    turns_progress = s / final_pitch_mm

    flat_end_turn = flat_first_turns
    blend_end_turn = flat_first_turns + blend_turns

    alpha = np.zeros_like(turns_progress)

    if blend_turns > 0.0:
        in_blend = (turns_progress > flat_end_turn) & (turns_progress < blend_end_turn)
        alpha[in_blend] = _smoothstep01(
            (turns_progress[in_blend] - flat_end_turn) / blend_turns
        )

    alpha[turns_progress >= blend_end_turn] = 1.0

    # Hold the first loop at the start, then blend onto the centerline.
    s_eff = alpha * s

    # --------------------------------------------------------
    # Interpolate effective centerline and frames
    # --------------------------------------------------------
    axis_x = np.interp(s_eff, s, axis[:, 0])
    axis_y = np.interp(s_eff, s, axis[:, 1])
    axis_z = np.interp(s_eff, s, axis[:, 2])
    axis_eff = np.column_stack((axis_x, axis_y, axis_z))

    n_x = np.interp(s_eff, s, n_vec[:, 0])
    n_y = np.interp(s_eff, s, n_vec[:, 1])
    n_z = np.interp(s_eff, s, n_vec[:, 2])
    n_eff = _normalize_rows(np.column_stack((n_x, n_y, n_z)))

    b_x = np.interp(s_eff, s, b_vec[:, 0])
    b_y = np.interp(s_eff, s, b_vec[:, 1])
    b_z = np.interp(s_eff, s, b_vec[:, 2])
    b_eff = _normalize_rows(np.column_stack((b_x, b_y, b_z)))

    t_x = np.interp(s_eff, s, t[:, 0])
    t_y = np.interp(s_eff, s, t[:, 1])
    t_z = np.interp(s_eff, s, t[:, 2])
    tool_dir = _normalize_rows(np.column_stack((t_x, t_y, t_z)))

    # --------------------------------------------------------
    # Generate helix around curved centerline
    # --------------------------------------------------------
    radial = np.cos(phase)[:, None] * n_eff + np.sin(phase)[:, None] * b_eff
    helix_mm = axis_eff + helix_radius_scaled * radial

    # --------------------------------------------------------
    # Map helix to bed frame
    # --------------------------------------------------------
    P_out = helix_mm.copy()

    _, _, bed_t_mm = bed_size_mm
    bed_top_z_mm = bed_t_mm / 2.0 + z_lift_mm

    # Center XY on the bed
    x_mid = 0.5 * (P_out[:, 0].min() + P_out[:, 0].max())
    y_mid = 0.5 * (P_out[:, 1].min() + P_out[:, 1].max())

    P_out[:, 0] -= x_mid
    P_out[:, 1] -= y_mid

    # Put lowest point on bed top
    z_min = P_out[:, 2].min()
    P_out[:, 2] += (bed_top_z_mm - z_min)

    x, y, z = P_out[:, 0], P_out[:, 1], P_out[:, 2]
    i, j, k = tool_dir[:, 0], tool_dir[:, 1], tool_dir[:, 2]

    # Centerline in the same frame as the plotted toolpath.
    axis_out = axis_eff.copy()
    axis_out[:, 0] -= x_mid
    axis_out[:, 1] -= y_mid
    axis_out[:, 2] += (bed_top_z_mm - z_min)

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------
    seg_mm = np.linalg.norm(np.diff(helix_mm, axis=0), axis=1)

    turns = s[-1] / final_pitch_mm
    pts_per_turn_int = max(1, int(round(final_pitch_mm / sample_ds_mm)))

    if pts_per_turn_int < len(helix_mm):
        d_turn_mm = np.linalg.norm(
            helix_mm[pts_per_turn_int:] - helix_mm[:-pts_per_turn_int],
            axis=1,
        )
        d_turn_mean = d_turn_mm.mean()
        d_turn_min = d_turn_mm.min()
        d_turn_max = d_turn_mm.max()
    else:
        d_turn_mean = d_turn_min = d_turn_max = np.nan

    # --------------------------------------------------------
    # Write CL-data
    # --------------------------------------------------------
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("# x y z i j k\n")
        for xi, yi, zi, ii, ji, ki in zip(x, y, z, i, j, k):
            f.write(f"{xi:.6f} {yi:.6f} {zi:.6f} {ii:.6f} {ji:.6f} {ki:.6f}\n")

    print(f"Wrote {N} CL points to {outfile}")
    print(f"Helix radius / pitch: {helix_radius_scaled:.3f} / {final_pitch_mm:.3f} mm")
    print(f"Centerline length: {s[-1]:.3f} mm, turns: {turns:.2f}")
    print(
        f"Segment length mean/min/max: {seg_mm.mean():.4f} / "
        f"{seg_mm.min():.4f} / {seg_mm.max():.4f} mm"
    )
    print(
        f"Turn spacing mean/min/max: {d_turn_mean:.4f} / "
        f"{d_turn_min:.4f} / {d_turn_max:.4f} mm"
    )
    print(
        f"XYZ range: X[{x.min():.3f}, {x.max():.3f}] "
        f"Y[{y.min():.3f}, {y.max():.3f}] "
        f"Z[{z.min():.3f}, {z.max():.3f}] mm"
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Use the same coordinate frame for both centerline and toolpath.
    ax.plot(
        axis_out[:, 0],
        axis_out[:, 1],
        axis_out[:, 2],
        linestyle="--",
        label="Centerline",
    )

    ax.plot(
        x,
        y,
        z,
        linestyle="-",
        marker=".",
        markersize=2,
        label="Toolpath",
    )

    total_span = max(
        (x.max() - x.min()),
        (y.max() - y.min()),
        (z.max() - z.min()),
    )
    arrow_length = max(total_span * 0.05, 1.0)

    for idx in range(0, N, max(1, viz_step)):
        ax.quiver(
            x[idx],
            y[idx],
            z[idx],
            i[idx],
            j[idx],
            k[idx],
            length=arrow_length,
            normalize=True,
        )

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title("Bent-pipe benchmark CL toolpath")
    ax.legend()

    _set_equal_aspect_3d(
        ax,
        np.concatenate([axis_out[:, 0], x]),
        np.concatenate([axis_out[:, 1], y]),
        np.concatenate([axis_out[:, 2], z]),
    )

    # Orthographic projection makes geometry inspection less visually distorted.
    ax.set_proj_type("ortho")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    generate_bent_helix_toolpath()