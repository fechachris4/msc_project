"""Base motion vs. end-effector error: the thesis success criterion made
visible. Read-only, same pattern as analysis/diagnose.py and
analysis/validate_velocity.py: the controller runs unmodified, every
logged quantity comes from the public functions apply_ctrl itself calls
(servo.pose_error), nothing re-derived.

    python -m analysis.base_vs_error

Outputs: analysis/output/base_vs_error.png and a per-side RMS/peak/
rejection-% table on stdout. Units are SI internally; mm on the figure
and in the printed table.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from controller import desired_pos, frames, servo
from sim import motion, world

# Base-motion scenario for this run — same values main.py runs today.
SCENARIO = dict(
    linear_amplitude=np.array([0.1, 0.3, 0.0]),   # m, world xyz
    linear_frequency=0.1,                          # Hz
    rotational_amplitude=np.zeros(3),              # rad, rpy
    rotational_frequency=0.2,                      # Hz
)
T = 40.0  # sim seconds, 4 periods at 0.1 Hz (diagnose.py convention)
# Static settle before the scenario starts (validate_velocity.py's
# pattern): the arms boot from a default configuration far from
# desired_pos.apply()'s targets, so the first ~1 s is a large one-off
# convergence transient, not disturbance response. Without settling
# first, that transient — not the base sway — would set the peak error
# and swamp the rejection-% this figure exists to show.
SETTLE_SECONDS = 2.0

OUT = Path("analysis/output")

# Okabe-Ito per axis
C_XYZ = ("#D55E00", "#009E73", "#0072B2")
C_BASE = "0.6"
AXES = ("x", "y", "z")


def run():
    """Closed-loop rollout under SCENARIO; return {t, base_disp, right_e,
    left_e} arrays (base_disp and *_e in meters, world frame)."""
    mujoco.mj_resetData(world.model, world.data)
    mujoco.mj_forward(world.model, world.data)
    desired_pos.apply()
    servo.init_ctrl()

    dt = world.model.opt.timestep
    zero_twist = (np.zeros(3), np.zeros(3))
    for _ in range(int(SETTLE_SECONDS / dt)):
        servo.apply_ctrl(dt, zero_twist)
        mujoco.mj_step(world.model, world.data)

    n = int(T / dt)
    home_pos = motion.HOME_POS
    t_start = world.data.time  # phase 0 at motion start: no teleport

    log = {
        "t": np.empty(n),
        "base_disp": np.empty((n, 3)),
        "right_e": np.empty((n, 3)),
        "left_e": np.empty((n, 3)),
    }

    for k in range(n):
        t = world.data.time - t_start
        motion.set_torso_pose(t, **SCENARIO)
        # refresh xpos/xmat so the logged state sees the torso pose at
        # t, not the previous step's (main.py / diagnose.py pattern)
        mujoco.mj_kinematics(world.model, world.data)
        # set_torso_pose (mocap write) and torso_twist_at (feedforward)
        # must stay a matched pair — same scenario, same instant t.
        base_twist = motion.torso_twist_at(t, **SCENARIO)

        base_pos, _ = frames.torso_pose()
        log["t"][k] = t
        log["base_disp"][k] = base_pos - home_pos
        for side, key in (("right", "right_e"), ("left", "left_e")):
            e_pos, _ = servo.pose_error(side)
            log[key][k] = e_pos

        servo.apply_ctrl(dt, base_twist)
        mujoco.mj_step(world.model, world.data)

    return log


def stats(log):
    """{side: {rms (3,) mm, peak (3,) mm, peak_norm mm, rejection %}} plus
    "peak_base" (mm, norm) — rejection = 1 - peak|e|/peak|base disp|,
    the norm-based thesis success metric."""
    base_mm = log["base_disp"] * 1000.0
    peak_base = float(np.linalg.norm(base_mm, axis=1).max())

    out = {"peak_base": peak_base}
    for side, key in (("right", "right_e"), ("left", "left_e")):
        e_mm = log[key] * 1000.0
        e_norm = np.linalg.norm(e_mm, axis=1)
        rms = np.sqrt(np.mean(e_mm**2, axis=0))
        peak = np.max(np.abs(e_mm), axis=0)
        peak_norm = float(e_norm.max())
        rejection = (1.0 - peak_norm / peak_base) * 100.0
        out[side] = dict(rms=rms, peak=peak, peak_norm=peak_norm,
                         rejection=rejection)
    return out


def print_stats(st):
    print(f"{'':8s}{'RMS x':>10s}{'y':>10s}{'z':>10s}"
          f"{'peak x':>10s}{'y':>10s}{'z':>10s}"
          f"{'|e| peak':>12s}{'rejection':>12s}")
    for side in world.SIDES:
        d = st[side]
        cells = "".join(f"{v:10.2f}" for v in (*d["rms"], *d["peak"]))
        print(f"{side:8s}{cells}{d['peak_norm']:12.2f}"
              f"{d['rejection']:11.1f}%  [mm]")
    print(f"peak |base disp| = {st['peak_base']:.2f} mm")


def make_figures(log, st):
    OUT.mkdir(parents=True, exist_ok=True)
    t = log["t"]
    base_mm = log["base_disp"] * 1000.0
    right_mm = log["right_e"] * 1000.0
    left_mm = log["left_e"] * 1000.0

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 8),
                             layout="constrained")
    for i, ax in enumerate(axes):
        ax.plot(t, base_mm[:, i], color=C_BASE, linewidth=1.2,
                label="base disp" if i == 0 else None)
        ax.plot(t, right_mm[:, i], color=C_XYZ[i], linewidth=1.2,
                label="right e_pos" if i == 0 else None)
        ax.plot(t, left_mm[:, i], color=C_XYZ[i], linestyle="--",
                linewidth=1.2, label="left e_pos" if i == 0 else None)
        ax.axhline(0, color="0.85", linewidth=0.5, zorder=0)
        ax.set_ylabel(f"{AXES[i]} [mm]")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")

    fig.suptitle(
        f"Base motion vs. EE error, world frame — rejection: "
        f"right {st['right']['rejection']:.0f}%, "
        f"left {st['left']['rejection']:.0f}% "
        f"(peak base disp {st['peak_base']:.0f} mm)"
    )
    fig.savefig(OUT / "base_vs_error.png", dpi=200)
    plt.close(fig)


def main():
    log = run()
    st = stats(log)
    print_stats(st)
    make_figures(log, st)
    print(f"\nSaved {OUT}/base_vs_error.png")


if __name__ == "__main__":
    main()
