"""Validate frames.ee_velocity against measured ground truth. Read-only:
nothing in the controller changes — the closed loop runs as-is under a
pinned base sway while three velocity signals are logged per arm:

  composed   frames.ee_velocity(side, motion.torso_twist_at(t))
  measured   central finite difference of frames.measured_ee_pose —
             MuJoCo ground truth, includes the mocap teleports
  direct     mj_objectVelocity on the pinch site — what MuJoCo believes;
             the mocap base has no velocity state, so this misses the
             base contribution entirely (the reason ee_velocity exists)

    python -m analysis.validate_velocity

Outputs: three figures (PNG) in analysis/output/ and an error-statistics
table on stdout. Units are SI internally; mm/s and deg/s on the figures.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pinocchio as pin

from controller import frames, servo
from controller.state import Twist
from plotting.style import C_XYZ
from sim import motion, targets, world

# Pinned scenario (independent of the motion module's research levers):
# rotation nonzero to exercise the w x r transport term, but modest —
# the right arm hits the torso near +15.6 deg roll
# (docs/diagnosis.md) and this comparison wants no contact.
SCENARIO = dict(
    linear_amplitude=np.array([0.05, 0.02, 0.01]),   # m
    linear_frequency=0.5,                            # Hz
    rotational_amplitude=np.radians([5.0, 3.0, 8.0]),
    rotational_frequency=0.3,                        # Hz
)
HOME = [0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109,
        1.57079633]
SETTLE_SECONDS = 2.0
MOTION_SECONDS = 5.0

OUT = Path("analysis/output")


def run():
    """Closed-loop rollout under the pinned sway; return per-side logs."""
    mujoco.mj_resetData(world.model, world.data)
    for side in world.SIDES:
        world.data.qpos[world.qpos_adrs[side]] = HOME
    mujoco.mj_forward(world.model, world.data)

    for side in world.SIDES:
        state = frames.arm_controller_state(
            world.read_state(Twist.zero()),
            side,
            world.MOUNT_CALIBRATION,
        )
        pos = state.ee_pose_world.position_m
        rot = state.ee_pose_world.rotation
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot.flatten())
        targets.set_target(side, pos)
        targets.set_target_quat(side, quat)
    pipeline = servo.ReactivePositionPipeline(
        world.read_state(Twist.zero()), world.PIPELINE_SETUP)
    world.apply_command(pipeline.command())

    dt = world.model.opt.timestep
    zero_twist = (np.zeros(3), np.zeros(3))
    for _ in range(int(SETTLE_SECONDS / dt)):
        plant = world.read_state(Twist(*zero_twist))
        command, _ = pipeline.step(
            frames.controller_states(plant, world.MOUNT_CALIBRATION),
            targets.world_targets(),
            dt,
        )
        world.apply_command(command)
        mujoco.mj_step(world.model, world.data)

    n = int(MOTION_SECONDS / dt)
    t_start = world.data.time  # phase 0 at motion start: no teleport
    log = {s: {
        "t": np.empty(n),
        "p": np.empty((n, 3)),        # measured EE position
        "R": np.empty((n, 3, 3)),     # measured EE rotation
        "v": np.empty((n, 3)),        # composed linear
        "w": np.empty((n, 3)),        # composed angular
        "v_direct": np.empty((n, 3)),  # mj_objectVelocity linear
        "w_direct": np.empty((n, 3)),  # mj_objectVelocity angular
    } for s in world.SIDES}

    vel6 = np.zeros(6)  # mj_objectVelocity order: [angular; linear]
    for k in range(n):
        tau = world.data.time - t_start
        motion.set_torso_pose(tau, **SCENARIO)
        # Re-run the kinematic/velocity pipeline so every sampled signal
        # refers to the same instant: fresh mocap, current qpos/qvel.
        mujoco.mj_kinematics(world.model, world.data)
        mujoco.mj_comPos(world.model, world.data)
        mujoco.mj_comVel(world.model, world.data)

        base_twist = motion.torso_twist_at(tau, **SCENARIO)
        plant = world.read_state(Twist(*base_twist))
        for s in world.SIDES:
            L = log[s]
            L["t"][k] = tau
            measured = world.measured_ee_pose(s)
            L["p"][k] = measured.position_m
            L["R"][k] = measured.rotation
            state = frames.arm_controller_state(
                plant, s, world.MOUNT_CALIBRATION)
            L["v"][k] = state.ee_twist_world.linear_m_s
            L["w"][k] = state.ee_twist_world.angular_rad_s
            mujoco.mj_objectVelocity(world.model, world.data,
                                     mujoco.mjtObj.mjOBJ_SITE,
                                     world.ee_site_id[s], vel6, 0)
            L["w_direct"][k] = vel6[:3]
            L["v_direct"][k] = vel6[3:]

        command, _ = pipeline.step(
            frames.controller_states(plant, world.MOUNT_CALIBRATION),
            targets.world_targets(),
            dt,
        )
        world.apply_command(command)
        mujoco.mj_step(world.model, world.data)

    # Ground truth by central differences of the measured pose; trim the
    # logs to the interior samples so every signal shares one time grid.
    for s in world.SIDES:
        L = log[s]
        L["v_fd"] = (L["p"][2:] - L["p"][:-2]) / (2.0 * dt)
        w_fd = np.empty_like(L["v_fd"])
        for k in range(len(w_fd)):
            w_fd[k] = pin.log3(L["R"][k + 2] @ L["R"][k].T) / (2.0 * dt)
        L["w_fd"] = w_fd
        for key in ("t", "v", "w", "v_direct", "w_direct"):
            L[key] = L[key][1:-1]
    return log


def stats(log):
    """{side: {signal: (rmse (3,), max (3,))}} vs the FD ground truth."""
    out = {}
    for s in world.SIDES:
        L = log[s]
        out[s] = {}
        for name, sig, ref in (("v", L["v"], L["v_fd"]),
                               ("w", L["w"], L["w_fd"]),
                               ("v_direct", L["v_direct"], L["v_fd"]),
                               ("w_direct", L["w_direct"], L["w_fd"])):
            err = sig - ref
            out[s][name] = (np.sqrt((err**2).mean(axis=0)),
                            np.abs(err).max(axis=0))
    return out


def print_stats(st):
    print(f"{'':14s}{'RMSE x':>10s}{'y':>10s}{'z':>10s}"
          f"{'max x':>10s}{'y':>10s}{'z':>10s}")
    for s in world.SIDES:
        for name, unit, scale in (("v", "mm/s", 1e3),
                                  ("w", "deg/s", np.degrees(1.0)),
                                  ("v_direct", "mm/s", 1e3),
                                  ("w_direct", "deg/s", np.degrees(1.0))):
            rmse, mx = st[s][name]
            cells = "".join(f"{x * scale:10.2f}" for x in (*rmse, *mx))
            print(f"{s:6s}{name:8s}{cells}  [{unit}]")


def _overlay_figure(L, comp_key, fd_key, unit, scale, labels, title, path):
    """3 axis panels (composed vs FD ground truth) + residual-norm panel."""
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 10),
                             layout="constrained")
    for i, ax in enumerate(axes[:3]):
        ax.plot(L["t"], L[fd_key][:, i] * scale, color="black",
                linestyle="--", linewidth=1.0, label="measured (FD)")
        ax.plot(L["t"], L[comp_key][:, i] * scale, color=C_XYZ[i],
                linewidth=1.2, alpha=0.9, label="composed")
        ax.set_ylabel(f"{labels[i]} [{unit}]")
        ax.axhline(0, color="0.7", linewidth=0.5, zorder=0)
    axes[0].legend(loc="upper right")
    resid = np.linalg.norm(L[comp_key] - L[fd_key], axis=1) * scale
    axes[3].plot(L["t"], resid, color="black", linewidth=1.0)
    axes[3].set_ylabel(f"|composed − FD| [{unit}]")
    axes[3].set_xlabel("time since motion start [s]")
    fig.suptitle(title)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_figures(log, st):
    OUT.mkdir(parents=True, exist_ok=True)
    L = log["right"]

    # 1/2 — does the composed velocity match ground truth?
    peak_v = np.linalg.norm(L["v_fd"], axis=1).max() * 1e3
    max_v = st["right"]["v"][1].max() * 1e3
    _overlay_figure(
        L, "v", "v_fd", "mm/s", 1e3, ("vx", "vy", "vz"),
        f"Right EE linear velocity: composed matches measured ground "
        f"truth\n(max residual {max_v:.1f} mm/s of {peak_v:.0f} mm/s peak)",
        OUT / "velocity_1_linear_composed_vs_fd.png",
    )
    peak_w = np.degrees(np.linalg.norm(L["w_fd"], axis=1).max())
    max_w = np.degrees(st["right"]["w"][1].max())
    _overlay_figure(
        L, "w", "w_fd", "deg/s", np.degrees(1.0), ("wx", "wy", "wz"),
        f"Right EE angular velocity: composed matches measured ground "
        f"truth\n(max residual {max_w:.2f} deg/s of {peak_w:.0f} deg/s peak)",
        OUT / "velocity_2_angular_composed_vs_fd.png",
    )

    # 3 — what does MuJoCo's own readout miss? (the base contribution)
    gap_v = st["right"]["v_direct"][1].max() * 1e3
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 5),
                             layout="constrained")
    for ax, key, direct, i, unit, scale in (
        (axes[0], "v", "v_direct", 0, "mm/s", 1e3),
        (axes[1], "w", "w_direct", 0, "deg/s", np.degrees(1.0)),
    ):
        ax.plot(L["t"], L[key + "_fd"][:, i] * scale,
                color="black", linestyle="--", linewidth=1.0,
                label="measured (FD)")
        ax.plot(L["t"], L[key][:, i] * scale, color=C_XYZ[0],
                linewidth=1.2, alpha=0.9, label="composed")
        ax.plot(L["t"], L[direct][:, i] * scale, color="0.5",
                linewidth=1.2, label="MuJoCo mj_objectVelocity")
        ax.axhline(0, color="0.7", linewidth=0.5, zorder=0)
    axes[0].set_ylabel("vx [mm/s]")
    axes[1].set_ylabel("wx [deg/s]")
    axes[1].set_xlabel("time since motion start [s]")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("MuJoCo's direct readout misses the base contribution "
                 f"(up to {gap_v:.0f} mm/s on vx)\n"
                 "— the mocap torso has no velocity state")
    fig.savefig(OUT / "velocity_3_mujoco_gap.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    log = run()
    st = stats(log)
    print_stats(st)
    make_figures(log, st)
    print(f"\nfigures -> {OUT}/velocity_[123]_*.png")
