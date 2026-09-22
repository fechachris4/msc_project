"""Headless scripted-mount-disturbance run: report figure, GIF and CSV.

Engineering question: do both arms hold world-fixed end-effector targets
while the mount is disturbed, and how much residual error remains compared with
doing nothing (arms rigid on the torso)?

Protocol: settle on the static torso until both arms are within tolerance,
then apply the scripted disturbance from tools/mount_disturbance.py for EVALUATION_S seconds.
Outputs go to analysis/output/disturbance/.

usage: python tools/disturbance_run.py [right|left|both] [--no-gif]
         [--speed=V | --speeds=V1,V2,...] [--scale=S]
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import report_style  # noqa: E402
import mount_disturbance  # noqa: E402
from controller import desired_pos, frames, reactive_controller  # noqa: E402
from controller.runner import ReactivePositionRunner  # noqa: E402
from controller.transforms import rotation_from_rpy  # noqa: E402
from sim import motion, world  # noqa: E402

OUT = Path("analysis/output/disturbance")
EVALUATION_S = 8.0
SETTLE_TIMEOUT_S = 15.0
SETTLE_POS_TOL_M = 0.002
SETTLE_ROT_TOL_RAD = np.deg2rad(0.5)
SETTLE_DWELL_S = 0.5
GIF_LEAD_S = 1.0          # static torso shown before the disturbance starts
GIF_FRAME_S = 0.04        # 25 fps
GIF_SIZE = (640, 480)

OI = {"x": "#D55E00", "y": "#009E73", "z": "#0072B2"}
ARM_STYLE = {"right": dict(color="black", linestyle="-"),
             "left": dict(color="0.45", linestyle="-")}


class WalkingTorsoDriver:
    """Static until start_walking(t0); then the disturbance, phase-aligned to t0."""

    def __init__(self, scale, speed):
        self.scale = scale
        self.speed = speed
        self.t0 = None

    def start_walking(self, t0):
        self.t0 = float(t0)

    def pose_at(self, t):
        if self.t0 is None or t < self.t0:
            return motion.HOME_POS.copy(), motion.HOME_RPY.copy()
        return mount_disturbance.torso_pose_at(t - self.t0, scale=self.scale, speed=self.speed)

    def twist_at(self, t):
        if self.t0 is None or t < self.t0:
            return np.zeros(3), np.zeros(3)
        return mount_disturbance.torso_twist_at(t - self.t0, scale=self.scale, speed=self.speed)


def _within_tolerance(runner, targets, arms):
    plant = runner.current_state
    resolved = frames.resolve_targets_world(
        plant, world.MOUNT_CALIBRATION, targets)
    states = frames.controller_states(plant, world.MOUNT_CALIBRATION)
    for side in arms:
        e_pos, e_rot = reactive_controller.pose_error(
            states.for_arm(side), resolved.for_arm(side))
        if (np.linalg.norm(e_pos) > SETTLE_POS_TOL_M
                or np.linalg.norm(e_rot) > SETTLE_ROT_TOL_RAD):
            return False
    return True


def _camera():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.3, 0.0, 1.3]
    cam.distance = 2.0
    cam.azimuth = -135.0
    cam.elevation = -15.0
    return cam


def run(arms, scale, speed, record_gif=True, controller_config=None):
    world.backend.release()
    world.backend.configure_torso_driver(None, None)
    world.backend.reset()
    targets = desired_pos.apply()
    driver = WalkingTorsoDriver(scale, speed)
    world.backend.configure_torso_driver(driver.pose_at, driver.twist_at)
    runner_kwargs = {}
    if controller_config is not None:
        runner_kwargs["controller_config"] = controller_config
    runner = ReactivePositionRunner(
        world.backend, world.MOUNT_CALIBRATION, world.PIPELINE_SETUP,
        targets, arms, **runner_kwargs)
    runner.start()
    mount_disturbance.hide_look_at_object()
    renderer = None
    if record_gif:
        renderer = mujoco.Renderer(world.model, GIF_SIZE[1], GIF_SIZE[0])
    cam = _camera()
    frames_out = []
    rows = []
    rigid_offset = {}
    dt = runner.current_state.nominal_dt_s
    try:
        # --- settle on the static torso ------------------------------------
        t_start = runner.current_state.sample_time_s
        dwell_start = None
        settled = False
        while runner.current_state.sample_time_s - t_start < SETTLE_TIMEOUT_S:
            t = runner.current_state.sample_time_s
            if _within_tolerance(runner, targets, arms):
                dwell_start = t if dwell_start is None else dwell_start
                if t - dwell_start >= SETTLE_DWELL_S:
                    settled = True
                    break
            else:
                dwell_start = None
            runner.cycle()
        settle_s = runner.current_state.sample_time_s - t_start
        # Rigid-arm counterfactual: an EE welded to the mount exactly at
        # the target, so the settle residual (<= SETTLE_POS_TOL_M) does not
        # bias the uncompensated disturbance.
        plant0 = runner.current_state
        resolved0 = frames.resolve_targets_world(
            plant0, world.MOUNT_CALIBRATION, targets)
        torso0 = plant0.torso_pose_world
        for side in arms:
            rigid_offset[side] = torso0.rotation.T @ (
                resolved0.for_arm(side).pose_world.position_m
                - torso0.position_m)

        # --- lead-in frames, then walk -------------------------------------
        t_walk = runner.current_state.sample_time_s + GIF_LEAD_S
        driver.start_walking(t_walk)
        next_frame = runner.current_state.sample_time_s
        n_steps = int(np.ceil((GIF_LEAD_S + EVALUATION_S) / dt))
        for _ in range(n_steps):
            cycle = runner.cycle()
            plant = cycle.input_state
            t_rel = plant.sample_time_s - t_walk
            torso = plant.torso_pose_world
            rpy = _rpy_from_rotation(torso.rotation)
            row = {
                "t": t_rel,
                "torso_dx": torso.position_m - motion.HOME_POS,
                "torso_rpy": rpy,
            }
            R_T = torso.rotation
            rigid_rot = float(np.arccos(np.clip(
                (np.trace(R_T) - 1.0) / 2.0, -1.0, 1.0)))
            for side in arms:
                trace = cycle.traces[side]
                ee = cycle.controller_states.for_arm(side).ee_pose_world
                rigid = torso.position_m + torso.rotation @ rigid_offset[side]
                target = cycle.resolved_targets.for_arm(side).pose_world
                row[side] = {
                    "e_pos": trace.e_pos,
                    "e_rot": trace.e_rot,
                    "ee": ee.position_m,
                    "rigid_err": rigid - target.position_m,
                    "rigid_rot_err": rigid_rot,
                    "safety_active": bool(np.any(
                        cycle.human_safety_states.for_arm(side)
                        .constraints.active)),
                    "qdot_max": float(np.abs(trace.qdot_effective).max()),
                    "qdot_raw_max": float(np.abs(trace.qdot_raw).max()),
                    "qdot_rms": float(np.sqrt(np.mean(
                        trace.qdot_effective ** 2))),
                    "speed_saturated": bool(np.any(
                        trace.qdot_speed_clipped != trace.qdot_raw)),
                }
            rows.append(row)
            if renderer is not None and plant.sample_time_s >= next_frame:
                desired_pos.show_targets(cycle.resolved_targets)
                mujoco.mj_forward(world.model, world.data)
                renderer.update_scene(world.data, camera=cam)
                frames_out.append(Image.fromarray(renderer.render().copy()))
                next_frame += GIF_FRAME_S
    finally:
        runner.close()
        world.backend.configure_torso_driver(None, None)
        if renderer is not None:
            renderer.close()
    return settled, settle_s, rows, frames_out


def _rpy_from_rotation(R):
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def _stack(rows, key, side=None):
    if side is None:
        return np.array([r[key] for r in rows])
    return np.array([r[side][key] for r in rows])


def figure(rows, arms, path, speed):
    t = _stack(rows, "t")
    walking = t >= 0.0
    torso_mm = 1e3 * _stack(rows, "torso_dx")
    torso_deg = np.degrees(_stack(rows, "torso_rpy"))
    report_style.apply()
    fig, axes = plt.subplots(4, 1, sharex=True,
                             figsize=(report_style.FULL_WIDTH_IN, 7.0),
                             layout="constrained")
    ax = axes[0]
    for i, (name, col) in enumerate(OI.items()):
        ax.plot(t, torso_mm[:, i], color=col, linewidth=1.5, label=name)
    ax.set_ylabel("mount displacement [mm]")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    ax = axes[1]
    for i, (name, col) in enumerate(zip(("roll", "pitch", "yaw"), OI.values())):
        ax.plot(t, torso_deg[:, i], color=col, linewidth=1.5, label=name)
    ax.set_ylabel("mount rotation [deg]")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)

    summary = {}
    ax = axes[2]
    for side in arms:
        e = 1e3 * np.linalg.norm(_stack(rows, "e_pos", side), axis=1)
        rigid = 1e3 * np.linalg.norm(_stack(rows, "rigid_err", side), axis=1)
        ax.plot(t, e, label=f"{side} arm, controlled", **ARM_STYLE[side])
        ax.plot(t, rigid, color=ARM_STYLE[side]["color"], linestyle="--",
                linewidth=1.0, label=f"{side} arm, rigid (no control)")
        summary[side] = dict(
            rms=np.sqrt(np.mean(e[walking] ** 2)),
            peak=e[walking].max(),
            rigid_rms=np.sqrt(np.mean(rigid[walking] ** 2)),
            rigid_peak=rigid[walking].max(),
        )
    ax.set_ylabel("EE position error [mm]")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax = axes[3]
    for side in arms:
        e_rot = np.degrees(np.linalg.norm(_stack(rows, "e_rot", side), axis=1))
        ax.plot(t, e_rot, label=f"{side} arm", **ARM_STYLE[side])
        summary[side]["rot_rms"] = np.sqrt(np.mean(e_rot[walking] ** 2))
        summary[side]["rot_peak"] = e_rot[walking].max()
    ax.set_ylabel("EE orientation error [deg]")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("time since disturbance onset [s]")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    for ax in axes:
        ax.axvline(0.0, color="0.6", linewidth=0.8, linestyle=":")
        ax.axhline(0.0, color="0.7", linewidth=0.5, zorder=0)
    report_style.panel_letters(axes, x=-0.07)
    worst = max(summary.values(), key=lambda s: s["rms"])
    reduction = 100.0 * (1.0 - worst["rms"] / worst["rigid_rms"])
    fig.savefig(path, dpi=200)
    fig.savefig(Path(path).with_suffix(".pdf"))
    plt.close(fig)
    return summary


def save_csv(rows, arms, path):
    header = ["t_s", "torso_dx_m", "torso_dy_m", "torso_dz_m",
              "torso_roll_rad", "torso_pitch_rad", "torso_yaw_rad"]
    for side in arms:
        header += [f"{side}_e{a}_m" for a in "xyz"]
        header += [f"{side}_erot_rad", f"{side}_rigid_err_m",
                   f"{side}_safety_active"]
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            vals = [r["t"], *r["torso_dx"], *r["torso_rpy"]]
            for side in arms:
                a = r[side]
                vals += [*a["e_pos"], np.linalg.norm(a["e_rot"]),
                         np.linalg.norm(a["rigid_err"]),
                         int(a["safety_active"])]
            f.write(",".join(f"{v:.6g}" for v in vals) + "\n")


SPEED_COLORS = {0.5: "#0072B2", 1.0: "#009E73", 1.5: "#D55E00"}


def _per_stride_rms(t, series, stride_s):
    """RMS of `series` over each complete pattern period in t >= 0."""
    out = []
    k = 0
    while (k + 1) * stride_s <= t.max() + 1e-9:
        m = (t >= k * stride_s) & (t < (k + 1) * stride_s)
        out.append(np.sqrt(np.mean(series[m] ** 2)))
        k += 1
    return np.array(out)


def comparison_figure(results, arms, path):
    """One figure across disturbance levels: error traces and RMS per level."""
    side = arms[0]
    fig = plt.figure(figsize=(9, 8.5), layout="constrained")
    grid = fig.add_gridspec(3, 2)
    ax_pos = fig.add_subplot(grid[0, :])
    ax_rot = fig.add_subplot(grid[1, :], sharex=ax_pos)
    ax_rms = fig.add_subplot(grid[2, 0])
    ax_rot_rms = fig.add_subplot(grid[2, 1])
    speeds = sorted(results)
    for speed in speeds:
        rows = results[speed]["rows"]
        t = _stack(rows, "t")
        col = SPEED_COLORS.get(speed, "black")
        e = 1e3 * np.linalg.norm(_stack(rows, "e_pos", side), axis=1)
        rigid = 1e3 * np.linalg.norm(_stack(rows, "rigid_err", side), axis=1)
        e_rot = np.degrees(np.linalg.norm(_stack(rows, "e_rot", side), axis=1))
        rigid_rot = np.degrees(_stack(rows, "rigid_rot_err", side))
        ax_pos.plot(t, e, color=col, linewidth=1.5,
                    label=mount_disturbance.level_label(speed))
        ax_pos.plot(t, rigid, color=col, linewidth=0.8, linestyle="--")
        ax_rot.plot(t, e_rot, color=col, linewidth=1.5,
                    label=mount_disturbance.level_label(speed))
        ax_rot.plot(t, rigid_rot, color=col, linewidth=0.8, linestyle="--")
    ax_pos.plot([], [], color="0.3", linewidth=0.8, linestyle="--",
                label="rigid arm (EE fixed to mount, no control)")
    rigid_handle = ax_rot.plot(
        [], [], color="0.3", linewidth=0.8, linestyle="--",
        label="rigid arm (= mount rotation angle)")[0]
    ax_rot.legend(handles=[rigid_handle], loc="upper right", frameon=False,
                  fontsize=8)
    other = [a for a in arms if a != side]
    if other:
        gap = max(abs(results[v]["summary"][side]["rms"]
                      - results[v]["summary"][other[0]]["rms"])
                  for v in speeds)
        print(f"comparison figure: {side} arm traces shown; {other[0]} arm "
              f"RMS within {max(gap, 0.05):.1f} mm at every level "
              "(state this in the caption)")
    ax_pos.set_ylabel(f"{side}-arm EE position error\n"
                      "$|p_{ref}-p|$ [mm]")
    ax_pos.set_ylim(bottom=0)
    ax_pos.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
                  frameon=False)
    ax_rot.set_ylabel(f"{side}-arm EE orientation error\n"
                      "$|\\log(R_{ref}R^{\\top})|$ [deg]")
    ax_rot.set_ylim(bottom=0)
    ax_rot.set_xlabel("time since disturbance onset [s] (t < 0: static mount)")
    for ax in (ax_pos, ax_rot):
        ax.axvline(0.0, color="0.6", linewidth=0.8, linestyle=":")
    # Summary panels: worst arm at each level, per-period RMS mean +- SD
    # over the complete pattern periods (2/f) in the disturbance window
    # (deterministic sim, so the spread is period-to-period variation incl.
    # the onset period).
    stats = {k: [] for k in ("ctrl", "rigid", "rot", "rigid_rot")}
    n_strides = []
    for speed in speeds:
        rows = results[speed]["rows"]
        t = _stack(rows, "t")
        stride_s = 1.0 / mount_disturbance.walk_params(speed=speed)[
            "linear_frequency"][1]
        worst = max(arms, key=lambda a: results[speed]["summary"][a]["rms"])
        series = {
            "ctrl": 1e3 * np.linalg.norm(
                _stack(rows, "e_pos", worst), axis=1),
            "rigid": 1e3 * np.linalg.norm(
                _stack(rows, "rigid_err", worst), axis=1),
            "rot": np.degrees(np.linalg.norm(
                _stack(rows, "e_rot", worst), axis=1)),
            "rigid_rot": np.degrees(_stack(rows, "rigid_rot_err", worst)),
        }
        w = t >= 0.0
        for k, v in series.items():
            per = _per_stride_rms(t[w], v[w], stride_s)
            stats[k].append((per.mean(), per.std(ddof=1)))
            n_strides.append(len(per))
    cols = [SPEED_COLORS.get(v, "black") for v in speeds]

    def band(ax, key, marker, style, label):
        m = np.array([x[0] for x in stats[key]])
        sd = np.array([x[1] for x in stats[key]])
        ax.errorbar(speeds, m, yerr=sd, color="black", linestyle=style,
                    capsize=3, linewidth=1.0, zorder=1, label=label)
        ax.scatter(speeds, m, c=cols, marker=marker, s=45, zorder=2)

    band(ax_rms, "rigid", "s", "--", "rigid arm (EE fixed to mount)")
    band(ax_rms, "ctrl", "o", "-", "reactive control")
    ax_rms.set_ylabel("position error RMS [mm]\n"
                      f"per-period mean $\\pm$ SD, "
                      f"{min(n_strides)}\u2013{max(n_strides)} periods")
    ax_rms.legend(loc="upper left", frameon=False)
    band(ax_rot_rms, "rigid_rot", "s", "--", "rigid arm (mount rotation)")
    band(ax_rot_rms, "rot", "o", "-", "reactive control")
    ax_rot_rms.set_ylabel("orientation error RMS [deg]\n"
                          "per-period mean $\\pm$ SD")
    ax_rot_rms.legend(loc="upper left", frameon=False)
    for ax in (ax_rms, ax_rot_rms):
        ax.set_xlabel("disturbance condition")
        ax.set_ylim(bottom=0)
        ax.set_xticks(speeds)
        ax.set_xticklabels([mount_disturbance.level_label(v) for v in speeds])
    fig.savefig(path, dpi=200)
    fig.savefig(Path(path).with_suffix(".pdf"))
    plt.close(fig)


def _report_one(arms, scale, speed, record_gif):
    print(mount_disturbance.describe(scale, speed))
    tag = f"v{speed:g}" + (f"_scale{scale:g}" if scale != 1.0 else "")
    settled, settle_s, rows, frames_out = run(arms, scale, speed, record_gif)
    print(f"settled: {settled} after {settle_s:.2f} s; "
          f"disturbed {EVALUATION_S:.0f} s, {len(rows)} samples")
    summary = figure(rows, arms, OUT / f"disturbance_rejection_{tag}.png", speed)
    save_csv(rows, arms, OUT / f"disturbance_log_{tag}.csv")
    if frames_out:
        frames_out[0].save(
            OUT / f"disturbance_{tag}.gif", save_all=True,
            append_images=frames_out[1:],
            duration=int(1000 * GIF_FRAME_S), loop=0, optimize=True)
        print(f"gif: {len(frames_out)} frames -> {OUT / f'disturbance_{tag}.gif'}")
    for side, s in summary.items():
        print(f"{side:5s} controlled |e| RMS {s['rms']:.1f} mm, peak "
              f"{s['peak']:.1f} mm | rigid RMS {s['rigid_rms']:.1f} mm, "
              f"peak {s['rigid_peak']:.1f} mm | orient RMS "
              f"{s['rot_rms']:.2f} deg, peak {s['rot_peak']:.2f} deg | "
              f"safety active "
              f"{100*np.mean(_stack(rows,'safety_active',side)):.0f}% "
              "of samples")
    print(f"figure: {OUT / f'disturbance_rejection_{tag}.png'}")
    return dict(rows=rows, summary=summary)


def main(argv):
    record_gif = "--no-gif" not in argv
    argv = [a for a in argv if a != "--no-gif"]
    scale = mount_disturbance.pop_float_option(argv, "scale", mount_disturbance.GAIT_SCALE)
    speeds = [mount_disturbance.DEFAULT_SPEED_M_S]
    for a in list(argv):
        if a.startswith("--speeds="):
            speeds = [float(v) for v in a.split("=", 1)[1].split(",")]
            argv.remove(a)
    speed = mount_disturbance.pop_float_option(argv, "speed", None)
    if speed is not None:
        speeds = [speed]
    choice = argv[0] if argv else "both"
    arms = world.SIDES if choice == "both" else (choice,)
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for v in speeds:
        results[v] = _report_one(arms, scale, v, record_gif)
    if len(speeds) > 1:
        path = OUT / "disturbance_level_comparison.png"
        comparison_figure(results, arms, path)
        print(f"comparison figure: {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
