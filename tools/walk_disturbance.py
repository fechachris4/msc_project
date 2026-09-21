"""Document the scripted mount disturbance: parameter table and profiles.

Outputs (analysis/output/disturbance/):
  base_disturbance.png       mount displacement, linear and angular speed over
                             two pattern periods at each disturbance level
  base_disturbance_table.md  amplitudes, frequencies, peak velocities per level

usage: python tools/walk_disturbance.py [--speeds=0.5,1.0,1.5]
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import walk_sim  # noqa: E402
from sim import motion  # noqa: E402

OUT = Path("analysis/output/disturbance")
OI = ("#D55E00", "#009E73", "#0072B2")
SPEED_COLORS = {0.5: "#0072B2", 1.0: "#009E73", 1.5: "#D55E00"}
# Axis styles are monochrome so the speed colours above mean only speed.
AXIS_STYLE = (("black", "-"), ("0.55", "-"), ("black", "--"))


def profile(speed, strides=2, n=800):
    p = walk_sim.walk_params(speed=speed)
    stride_s = 1.0 / p["linear_frequency"][1]
    t = np.linspace(0.0, strides * stride_s, n)
    pos = np.array([walk_sim.torso_pose_at(s, speed=speed)[0] for s in t])
    rpy = np.array([walk_sim.torso_pose_at(s, speed=speed)[1] for s in t])
    v = np.array([walk_sim.torso_twist_at(s, speed=speed)[0] for s in t])
    w = np.array([walk_sim.torso_twist_at(s, speed=speed)[1] for s in t])
    return p, t, pos - motion.HOME_POS, rpy, v, w


def table_rows(speeds):
    rows = []
    for speed in speeds:
        p, t, d, rpy, v, w = profile(speed)
        a = 1e3 * p["linear_amplitude"]
        f = p["linear_frequency"]
        ra = np.degrees(p["rotational_amplitude"])
        rf = p["rotational_frequency"]
        rows.append(dict(
            speed=speed, step_hz=f[0], stride_hz=f[1],
            amp_mm=a, amp_deg=ra, lin_hz=f, rot_hz=rf,
            peak_v=np.linalg.norm(v, axis=1).max(),
            peak_w=np.degrees(np.linalg.norm(w, axis=1).max()),
            peak_disp=np.linalg.norm(d, axis=1).max() * 1e3,
            peak_acc=max(np.linalg.norm(
                (2 * np.pi * f) ** 2 * p["linear_amplitude"])
                for _ in [0]),
        ))
    return rows


def write_table(rows, path):
    lines = [
        "| disturbance level | f / f/2 | x, y, z amplitude "
        "(freq) | roll, pitch, yaw amplitude (freq) | peak |Δp| | "
        "peak |v| | peak |ω| |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lin = ", ".join(f"{a:.0f} mm ({f:.2g} Hz)"
                        for a, f in zip(r["amp_mm"], r["lin_hz"]))
        rot = ", ".join(f"{a:.1f}° ({f:.2g} Hz)"
                        for a, f in zip(r["amp_deg"], r["rot_hz"]))
        lines.append(
            f"| {walk_sim.LEVEL_NAMES.get(r['speed'], r['speed'])} | {r['step_hz']:.2g} / {r['stride_hz']:.2g} Hz "
            f"| {lin} | {rot} | {r['peak_disp']:.0f} mm | "
            f"{r['peak_v']:.2f} m/s | {r['peak_w']:.0f} °/s |")
    lines += [
        "",
        "Mount frame at rest is the world frame: x forward, y left, z up. "
        "Each axis is A·sin(2πft); no net translation. x, z and pitch "
        "oscillate at the fundamental f; y, roll and yaw at the sub-harmonic "
        "f/2. Scripted, hand-chosen values (tools/walk_sim.py); not a gait "
        "model.",
    ]
    path.write_text("\n".join(lines) + "\n")


def figure(speeds, path):
    """Three rows (mount displacement, linear speed, angular speed), one
    column per disturbance level. Captions live in the report."""
    fig, axes = plt.subplots(3, len(speeds), figsize=(9, 7.2), sharey="row",
                             sharex=True, layout="constrained")
    for j, speed in enumerate(speeds):
        p, t, d, rpy, v, w = profile(speed)
        stride = t / (1.0 / p["linear_frequency"][1])
        ax = axes[0, j]
        for i, (lab, (col, ls)) in enumerate(zip(
                ("x (forward)", "y (left)", "z (up)"), AXIS_STYLE)):
            ax.plot(stride, 1e3 * d[:, i], color=col, linestyle=ls,
                    linewidth=1.5, label=lab)
        ax.set_title(f"level {walk_sim.LEVEL_NAMES.get(speed, speed)}\n"
                     f"f = {p['linear_frequency'][0]:.2g} Hz, "
                     f"f/2 = {p['linear_frequency'][1]:.2g} Hz", fontsize=9.5)
        axes[1, j].plot(stride, np.linalg.norm(v, axis=1), color="black",
                        linewidth=1.5)
        axes[2, j].plot(stride, np.degrees(np.linalg.norm(w, axis=1)),
                        color="black", linewidth=1.5)
        axes[2, j].set_xlabel("time [pattern periods, 2/f]")
    axes[0, 0].set_ylabel("mount displacement\nfrom rest pose [mm]")
    axes[1, 0].set_ylabel("mount linear speed\n|v| [m/s]")
    axes[2, 0].set_ylabel("mount angular speed\n|\u03c9| [deg/s]")
    axes[1, 0].set_ylim(0, 0.5)
    axes[2, 0].set_ylim(0, 50)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3,
               frameon=False, fontsize=8.5,
               title="mount displacement along world axes "
                     "(mount frame = world frame at rest)",
               title_fontsize=8.5)
    for ax in axes.flat:
        ax.axhline(0, color="0.7", linewidth=0.5, zorder=0)
    fig.savefig(path, dpi=200)
    fig.savefig(Path(path).with_suffix(".pdf"))
    plt.close(fig)


def main(argv):
    speeds = [0.5, 1.0, 1.5]
    for a in argv:
        if a.startswith("--speeds="):
            speeds = [float(v) for v in a.split("=", 1)[1].split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = table_rows(speeds)
    write_table(rows, OUT / "base_disturbance_table.md")
    figure(speeds, OUT / "base_disturbance.png")
    print((OUT / "base_disturbance_table.md").read_text())
    print(f"figure: {OUT / 'base_disturbance.png'}")


if __name__ == "__main__":
    main(sys.argv[1:])
