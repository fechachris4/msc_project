"""Report illustration of the task: the mount moves, the end-effectors hold.

One disturbance_run.run with the fixed-amplitude disturbance of
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
import report_style  # noqa: E402
import disturbance_run  # noqa: E402
import mount_disturbance  # noqa: E402

OUT = Path("analysis/output/disturbance")
RENDER_SIZE = (1280, 960)
CROP = (slice(60, 700), slice(300, 1160))   # rows, cols around the arms
START_PERIODS = 2          # skip the onset transient
CORNERS = (("up + left", 1 / 8), ("down + left", 3 / 8),
           ("up + right", 5 / 8), ("down + right", 7 / 8))


def close_camera():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.2, 0.0, 1.15]
    cam.distance = 1.75
    cam.azimuth = -135.0
    cam.elevation = -32.0   # looks down: floor fills the frame, no horizon
    return cam


if __name__ == "__main__":
    argv = sys.argv[1:]
    scale = mount_disturbance.pop_float_option(argv, "scale", 3.0)
    f_hz = mount_disturbance.pop_float_option(argv, "f", 1.0)
    OUT.mkdir(parents=True, exist_ok=True)
    disturbance_run.world.model.site_rgba[:, 3] = 0.0   # hide frame-origin marker sites

    disturbance_run._camera = close_camera
    disturbance_run.GIF_SIZE = RENDER_SIZE
    disturbance_run.world.model.vis.global_.offwidth = RENDER_SIZE[0]
    disturbance_run.world.model.vis.global_.offheight = RENDER_SIZE[1]
    disturbance_run.GIF_FRAME_S = 0.02
    settled, settle_s, rows, frames = disturbance_run.run(
        disturbance_run.world.SIDES, scale, sweep.AMPLITUDE_KEY, record_gif=True,
        f_hz=f_hz)
    p = mount_disturbance.disturbance_params(scale=scale, speed=sweep.AMPLITUDE_KEY,
                                             f_hz=f_hz)

    t = disturbance_run._stack(rows, "t")
    on = t >= 0.0
    rms = max(np.sqrt(np.mean((1e3 * np.linalg.norm(
        disturbance_run._stack(rows, "e_pos", s), axis=1))[on] ** 2))
        for s in disturbance_run.world.SIDES)
    free = max(np.sqrt(np.mean((1e3 * np.linalg.norm(
        disturbance_run._stack(rows, "rigid_err", s), axis=1))[on] ** 2))
        for s in disturbance_run.world.SIDES)
    safety = max(np.mean(disturbance_run._stack(rows, "safety_active", s))
                 for s in disturbance_run.world.SIDES)
    print(f"settled {settled}; scale {scale:g}, f = {f_hz:g} Hz: EE error RMS "
          f"{rms:.1f} mm vs {free:.1f} mm uncontrolled; safety active "
          f"{100 * safety:.0f}%")

    period_s = 2.0 / f_hz

    def frame_at(t_s):
        k = int(round((disturbance_run.GIF_LEAD_S + t_s) / disturbance_run.GIF_FRAME_S))
        return np.asarray(frames[min(k, len(frames) - 1)], dtype=float)

    t0 = START_PERIODS * period_s
    n = int(round(period_s / disturbance_run.GIF_FRAME_S))
    blur = np.mean([frame_at(t0 + i * disturbance_run.GIF_FRAME_S)
                    for i in range(n)], axis=0)

    report_style.apply()
    fig, axes = plt.subplots(1, 5, figsize=(report_style.FULL_WIDTH_IN, 1.35),
                             layout="constrained")
    for ax, (name, frac) in zip(axes, CORNERS):
        ax.imshow(frame_at(t0 + frac * period_s).astype(np.uint8)[CROP])
        ax.set_xlabel(f"mount {name}", fontsize=7.5)
    axes[4].imshow(blur.astype(np.uint8)[CROP])
    axes[4].set_xlabel("one cycle averaged", fontsize=7.5)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
    report_style.panel_letters(axes, x=0.16, y=1.02)
    lin = 1e3 * p["linear_amplitude"].max()
    rot = np.degrees(p["rotational_amplitude"]).max()
    print(f"caption facts: mount motion {scale:g}x the experiments (up to "
          f"{lin:.0f} mm, {rot:.0f} deg), f = {f_hz:g} Hz, EE error "
          f"{rms:.0f} mm RMS vs {free:.0f} mm with no control")
    report_style.save(fig, OUT / "task_illustration.png")

    k0 = int(round(disturbance_run.GIF_LEAD_S / disturbance_run.GIF_FRAME_S))
    clip = [f.resize((640, 480)) for f in frames[k0:k0 + 2 * n:2]]
    clip[0].save(OUT / "task_illustration.gif", save_all=True,
                 append_images=clip[1:], duration=40, loop=0, optimize=True)
    print(f"figure: {OUT / 'task_illustration.png'}")
