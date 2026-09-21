"""Mount-disturbance feedforward vs baseline: same scripted-disturbance scenario as
walk_report.py, run twice with controller.reactive_pose.velocity_feedforward_
enabled off (baseline PD) and on (torso-twist cancellation added to the task
twist), so the effect can be read directly off matching axes.

The target is world-fixed (see [targets.<arm>] reference_frame = "world" in
config/control.toml) and the torso is disturbed, so target.twist_world is always
zero: the entire feedforward contribution here comes from the mount-
disturbance-cancellation term in ReactiveController.compute (subtracting the
currently-known torso-induced EE velocity from the commanded task twist).
This isolates that source; see tools/velocity_ff_demo.py for the other
source (a target that itself carries nonzero path velocity).

usage: python tools/walk_ff_compare.py [right|left|both]
         [--speed=V | --speeds=V1,V2,...] [--scale=S] [--no-gif]
"""

import dataclasses
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import walk_report  # noqa: E402
import walk_sim  # noqa: E402
from controller import servo  # noqa: E402
from sim import world  # noqa: E402

OUT = Path("analysis/output/disturbance")
VARIANTS = (
    ("baseline", False, "0.35", "-"),
    ("velocity_feedforward", True, "black", "-"),
)


def _stack(rows, key, side):
    return np.array([r[side][key] for r in rows])


def _run_variant(arms, scale, speed, enabled, record_gif):
    config = dataclasses.replace(
        servo.CONTROL, velocity_feedforward_enabled=enabled)
    settled, settle_s, rows, frames_out = walk_report.run(
        arms, scale, speed, record_gif=record_gif, controller_config=config)
    return settled, settle_s, rows, frames_out


def _save_gif(frames_out, path):
    frames_out[0].save(
        path, save_all=True, append_images=frames_out[1:],
        duration=int(1000 * walk_report.GIF_FRAME_S), loop=0, optimize=True)
    print(f"  gif: {len(frames_out)} frames -> {path}")


def compare_one_speed(arms, scale, speed, record_gif):
    print(walk_sim.describe(scale, speed))
    tag = f"v{speed:g}" + (f"_scale{scale:g}" if scale != 1.0 else "")
    results = {}
    for name, enabled, color, style in VARIANTS:
        settled, settle_s, rows, frames_out = _run_variant(
            arms, scale, speed, enabled, record_gif)
        print(f"  [{name}] settled: {settled} after {settle_s:.2f} s")
        results[name] = rows
        if frames_out:
            _save_gif(frames_out, OUT / f"disturbance_ff_{name}_{tag}.gif")

    side = arms[0]
    t = np.array([r["t"] for r in results["baseline"]])
    walking = t >= 0.0

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
                             layout="constrained")
    summary = {}
    for name, enabled, color, style in VARIANTS:
        rows = results[name]
        e = 1e3 * np.linalg.norm(_stack(rows, "e_pos", side), axis=1)
        e_rot = np.degrees(np.linalg.norm(_stack(rows, "e_rot", side), axis=1))
        axes[0].plot(t, e, color=color, linestyle=style, linewidth=1.5,
                     label=name)
        axes[1].plot(t, e_rot, color=color, linestyle=style, linewidth=1.5,
                     label=name)
        summary[name] = dict(
            rms=np.sqrt(np.mean(e[walking] ** 2)),
            peak=e[walking].max(),
            rot_rms=np.sqrt(np.mean(e_rot[walking] ** 2)),
        )
    axes[0].set_ylabel(f"{side} EE position error [mm]")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                   frameon=False)
    axes[1].set_ylabel(f"{side} EE orientation error [deg]")
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlabel("time since disturbance onset [s]")
    for ax in axes:
        ax.axvline(0.0, color="0.6", linewidth=0.8, linestyle=":")

    base_rms = summary["baseline"]["rms"]
    ff_rms = summary["velocity_feedforward"]["rms"]
    change = 100.0 * (1.0 - ff_rms / base_rms) if base_rms else 0.0
    fig.suptitle(
        f"Level {walk_sim.level_label(speed)}: mount-disturbance feedforward changes "
        f"{side} RMS position error {base_rms:.1f} -> {ff_rms:.1f} mm "
        f"({change:+.0f}%)"
    )
    path = OUT / f"disturbance_ff_compare_{tag}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  baseline  |e| RMS {summary['baseline']['rms']:.1f} mm, peak "
          f"{summary['baseline']['peak']:.1f} mm, rot RMS "
          f"{summary['baseline']['rot_rms']:.2f} deg")
    print(f"  ff        |e| RMS {summary['velocity_feedforward']['rms']:.1f} mm, "
          f"peak {summary['velocity_feedforward']['peak']:.1f} mm, rot RMS "
          f"{summary['velocity_feedforward']['rot_rms']:.2f} deg")
    print(f"  figure: {path}")
    return summary


def main(argv):
    record_gif = "--no-gif" not in argv
    argv = [a for a in argv if a != "--no-gif"]
    scale = walk_sim.pop_float_option(argv, "scale", walk_sim.GAIT_SCALE)
    speeds = [walk_sim.DEFAULT_SPEED_M_S]
    for a in list(argv):
        if a.startswith("--speeds="):
            speeds = [float(v) for v in a.split("=", 1)[1].split(",")]
            argv.remove(a)
    speed = walk_sim.pop_float_option(argv, "speed", None)
    if speed is not None:
        speeds = [speed]
    choice = argv[0] if argv else "both"
    arms = world.SIDES if choice == "both" else (choice,)
    OUT.mkdir(parents=True, exist_ok=True)
    for v in speeds:
        compare_one_speed(arms, scale, v, record_gif)


if __name__ == "__main__":
    main(sys.argv[1:])
