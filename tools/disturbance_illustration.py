"""Report illustration of the task: the mount moves, the end-effectors hold.

One walk_report.run with the fixed-amplitude disturbance of
disturbance_freq_sweep.py at F_HZ, amplitudes multiplied by SCALE so the
motion is visible in print (stated on the figure). Frames are taken at the
four extreme mount poses of one pattern period (2/f), where z and y are both
at +-peak, plus a time-average of the whole period: moving parts smear,
parts held still in the world stay sharp.

Outputs (analysis/output/disturbance/):
  task_illustration.png / .pdf
  task_illustration.gif

usage: python tools/disturbance_illustration.py [--scale=3] [--f=1.0]
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import disturbance_freq_sweep as sweep  # noqa: E402
import walk_report  # noqa: E402
import walk_sim  # noqa: E402

OUT = Path("analysis/output/disturbance")
START_PERIODS = 2          # skip the onset transient
CORNERS = (("up + left", 1 / 8), ("down + left", 3 / 8),
           ("up + right", 5 / 8), ("down + right", 7 / 8))


def close_camera():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.2, 0.0, 1.22]
    cam.distance = 1.6
    cam.azimuth = -135.0
    cam.elevation = -12.0
    return cam


if __name__ == "__main__":
    argv = sys.argv[1:]
    scale = walk_sim.pop_float_option(argv, "scale", 3.0)
    f_hz = walk_sim.pop_float_option(argv, "f", 1.0)
    OUT.mkdir(parents=True, exist_ok=True)

    walk_report._camera = close_camera
    walk_report.GIF_FRAME_S = 0.02
    sweep._fundamental_hz[0] = f_hz
    walk_sim.walk_params = sweep.fixed_amplitude_params
    settled, settle_s, rows, frames = walk_report.run(
        walk_report.world.SIDES, scale, sweep.AMPLITUDE_KEY, record_gif=True)
    p = walk_sim.walk_params(scale=scale)
    walk_sim.walk_params = sweep._table_params

    t = walk_report._stack(rows, "t")
    on = t >= 0.0
    rms = max(np.sqrt(np.mean((1e3 * np.linalg.norm(
        walk_report._stack(rows, "e_pos", s), axis=1))[on] ** 2))
        for s in walk_report.world.SIDES)
    free = max(np.sqrt(np.mean((1e3 * np.linalg.norm(
        walk_report._stack(rows, "rigid_err", s), axis=1))[on] ** 2))
        for s in walk_report.world.SIDES)
    safety = max(np.mean(walk_report._stack(rows, "safety_active", s))
                 for s in walk_report.world.SIDES)
    print(f"settled {settled}; scale {scale:g}, f = {f_hz:g} Hz: EE error RMS "
          f"{rms:.1f} mm vs {free:.1f} mm uncontrolled; safety active "
          f"{100 * safety:.0f}%")

    period_s = 2.0 / f_hz

    def frame_at(t_s):
        k = int(round((walk_report.GIF_LEAD_S + t_s) / walk_report.GIF_FRAME_S))
        return np.asarray(frames[min(k, len(frames) - 1)], dtype=float)

    t0 = START_PERIODS * period_s
    n = int(round(period_s / walk_report.GIF_FRAME_S))
    blur = np.mean([frame_at(t0 + i * walk_report.GIF_FRAME_S)
                    for i in range(n)], axis=0)

    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.7), layout="constrained")
    for ax, (name, frac) in zip(axes, CORNERS):
        ax.imshow(frame_at(t0 + frac * period_s).astype(np.uint8))
        ax.set_title(f"mount {name}", fontsize=9.5)
    axes[4].imshow(blur.astype(np.uint8))
    axes[4].set_title("one full cycle averaged:\nmount blurs, hands stay sharp",
                      fontsize=9.5)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "bottom", "left", "right"):
        axes[4].spines[side].set_color("#0072B2")
        axes[4].spines[side].set_linewidth(2)
    lin = 1e3 * p["linear_amplitude"].max()
    rot = np.degrees(p["rotational_amplitude"]).max()
    fig.suptitle(
        "The task: the mount (brown block) is moved, the arms keep both "
        "end-effectors on fixed world targets (red, green)\n"
        f"mount motion shown {scale:g}x larger than in the experiments so it is "
        f"visible (here up to {lin:.0f} mm, {rot:.0f}°; f = {f_hz:g} Hz); "
        f"end-effector error {rms:.0f} mm RMS", fontsize=9.5)
    fig.savefig(OUT / "task_illustration.png", dpi=300)
    fig.savefig(OUT / "task_illustration.pdf")
    plt.close(fig)

    k0 = int(round(walk_report.GIF_LEAD_S / walk_report.GIF_FRAME_S))
    clip = frames[k0:k0 + 2 * n:2]
    clip[0].save(OUT / "task_illustration.gif", save_all=True,
                 append_images=clip[1:], duration=40, loop=0, optimize=True)
    print(f"figure: {OUT / 'task_illustration.png'}")
