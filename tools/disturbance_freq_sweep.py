"""Reactive disturbance rejection vs disturbance frequency (one thing varies).

Every run uses the SAME mount-motion amplitudes (AMPLITUDE_KEY row of
walk_sim._SPEED_TABLE); only the fundamental frequency f changes. x, z and
pitch oscillate at f; y, roll and yaw at f/2. Each run is one
walk_report.run: settle on the static mount, then 8 s of disturbance.

Check: at f = 1.8 Hz this is identical to walk_report.py --speed=1.0.

Outputs (analysis/output/disturbance/):
  freq_sweep.png / .pdf   position and orientation error RMS vs frequency
  freq_sweep.csv

usage: python tools/disturbance_freq_sweep.py [--freqs=0.5,1,1.5,1.8,2,2.5,3] [--replot]
"""

import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import walk_report  # noqa: E402
import walk_sim  # noqa: E402

OUT = Path("analysis/output/disturbance")
AMPLITUDE_KEY = 1.0        # table row whose amplitudes are held fixed
FREQS_HZ = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]   # add 1.8 to re-check vs --speed=1.0
NO_CONTROL = "#D55E00"
REACTIVE = "#0072B2"

_table_params = walk_sim.walk_params
_fundamental_hz = [None]


def fixed_amplitude_params(scale=walk_sim.GAIT_SCALE, speed=None):
    p = _table_params(scale=scale, speed=AMPLITUDE_KEY)
    f = _fundamental_hz[0]
    p["linear_frequency"] = np.array([f, 0.5 * f, f])
    p["rotational_frequency"] = np.array([0.5 * f, f, 0.5 * f])
    return p


def run_one(f_hz, arms):
    _fundamental_hz[0] = f_hz
    walk_sim.walk_params = fixed_amplitude_params
    try:
        settled, settle_s, rows, _ = walk_report.run(
            arms, 1.0, AMPLITUDE_KEY, record_gif=False)
    finally:
        walk_sim.walk_params = _table_params
    t = walk_report._stack(rows, "t")
    on = t >= 0.0
    out = dict(f_hz=f_hz, settled=int(settled), pos=0.0, pos_free=0.0,
               rot=0.0, rot_free=0.0)
    for side in arms:     # report the worst arm
        rms = lambda v: float(np.sqrt(np.mean(v[on] ** 2)))  # noqa: E731
        out["pos"] = max(out["pos"], rms(1e3 * np.linalg.norm(
            walk_report._stack(rows, "e_pos", side), axis=1)))
        out["pos_free"] = max(out["pos_free"], rms(1e3 * np.linalg.norm(
            walk_report._stack(rows, "rigid_err", side), axis=1)))
        out["rot"] = max(out["rot"], rms(np.degrees(np.linalg.norm(
            walk_report._stack(rows, "e_rot", side), axis=1))))
        out["rot_free"] = max(out["rot_free"], rms(np.degrees(
            walk_report._stack(rows, "rigid_rot_err", side))))
    return out


def figure(rows, path):
    f = np.array([r["f_hz"] for r in rows])
    p = _table_params(speed=AMPLITUDE_KEY)
    amp = (f"same motion size in every run: "
           f"{1e3 * p['linear_amplitude'].max():.0f} mm, "
           f"{np.degrees(p['rotational_amplitude']).max():.0f}° peak per axis")
    fig = plt.figure(figsize=(9, 6.2), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.9])
    ax_how = fig.add_subplot(grid[0, :])
    axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    # How the runs differ: same mount motion, played slower or faster.
    ts = np.linspace(0.0, 2.0, 600)
    for f_hz, col, ls in ((f.min(), "0.55", "-"), (f.max(), "black", "-")):
        _fundamental_hz[0] = f_hz
        walk_sim.walk_params = fixed_amplitude_params
        z = [walk_sim.torso_pose_at(s, speed=AMPLITUDE_KEY)[0][2] for s in ts]
        walk_sim.walk_params = _table_params
        ax_how.plot(ts, 1e3 * (np.array(z) - z[0]), color=col, linestyle=ls,
                    linewidth=1.5, label=f"f = {f_hz:g} Hz")
    ax_how.set_xlabel("time [s]   (vertical axis shown as the example; "
                      "the other five axes are sped up the same way)")
    ax_how.set_ylabel("mount vertical\ndisplacement [mm]")
    ax_how.set_title("What changes between runs: the mount moves the same "
                     "distance, only faster", fontsize=9.5, loc="left")
    ax_how.legend(loc="lower right", bbox_to_anchor=(1.0, 1.08), ncol=2,
                  frameon=False, fontsize=9, title="slowest and fastest run",
                  title_fontsize=8.5)
    ax_how.spines[["top", "right"]].set_visible(False)
    for ax, key, unit, name in ((axes[0], "pos", "mm", "position"),
                                (axes[1], "rot", "deg", "orientation")):
        free = np.array([r[key + "_free"] for r in rows])
        ctrl = np.array([r[key] for r in rows])
        ax.plot(f, free, color=NO_CONTROL, linestyle="--", marker="s",
                linewidth=1.5, label="no control (arm rigid on mount)")
        ax.plot(f, ctrl, color=REACTIVE, marker="o", linewidth=1.8,
                label="reactive control")
        ax.fill_between(f, ctrl, free, color=REACTIVE, alpha=0.10,
                        linewidth=0)
        for x, c, u in zip(f, ctrl, free):
            ax.annotate(f"{100 * (1 - c / u):.0f}%", (x, c),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8, color=REACTIVE)
        ax.set_xlabel("mount disturbance frequency f [Hz]")
        ax.set_ylabel(f"end-effector {name} error, RMS [{unit}]")
        ax.set_ylim(bottom=0)
        ax.set_xticks(f)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="center left", frameon=False, fontsize=9,
                   title="% = share of the error removed by control",
                   title_fontsize=8.5, alignment="left")
    axes[0].set_title("Result: the faster the mount moves, the more error "
                      "is left over", fontsize=9.5, loc="left")
    fig.suptitle(f"Every run uses the same mount motion (up to "
                 f"{1e3 * p['linear_amplitude'].max():.0f} mm and "
                 f"{np.degrees(p['rotational_amplitude']).max():.0f}\u00b0 "
                 "per axis); only its frequency f changes",
                 fontsize=9, color="0.3", x=0.01, ha="left")
    fig.savefig(path, dpi=200)
    fig.savefig(Path(path).with_suffix(".pdf"))
    plt.close(fig)



def write_report_text(rows, path):
    """Markdown table, LaTeX tabular and caption for the report."""
    p = _table_params(speed=AMPLITUDE_KEY)
    lin = 1e3 * p["linear_amplitude"]
    rot = np.degrees(p["rotational_amplitude"])
    md = ["| f [Hz] | position RMS [mm] | removed | orientation RMS [deg] "
          "| removed |", "|---|---|---|---|---|"]
    tex = ["\\begin{tabular}{rrrrr}", "\\toprule",
           "$f$ [Hz] & pos.\\ RMS [mm] & removed & ori.\\ RMS [deg] & "
           "removed \\\\", "\\midrule"]
    for r in rows:
        rp = 100 * (1 - r["pos"] / r["pos_free"])
        rr = 100 * (1 - r["rot"] / r["rot_free"])
        md.append(f"| {r['f_hz']:g} | {r['pos']:.1f} | {rp:.0f}% | "
                  f"{r['rot']:.2f} | {rr:.0f}% |")
        tex.append(f"{r['f_hz']:g} & {r['pos']:.1f} & {rp:.0f}\\% & "
                   f"{r['rot']:.2f} & {rr:.0f}\\% \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    free = rows[0]
    text = [
        "# Disturbance frequency sweep", "",
        "## Disturbance definition (identical in every run)", "",
        "Each mount axis follows A sin(2 pi f_axis t) about the rest pose; "
        "no net translation. Only f changes between runs.", "",
        "| axis | amplitude | frequency |", "|---|---|---|",
        f"| x (forward) | {lin[0]:g} mm | f |",
        f"| y (left) | {lin[1]:g} mm | f/2 |",
        f"| z (up) | {lin[2]:g} mm | f |",
        f"| roll | {rot[0]:g} deg | f/2 |",
        f"| pitch | {rot[1]:g} deg | f |",
        f"| yaw | {rot[2]:g} deg | f/2 |", "",
        f"With no control (arm rigid on the mount) this gives "
        f"{free['pos_free']:.1f} mm position and {free['rot_free']:.2f} deg "
        "orientation error RMS at the end-effector, at every frequency.", "",
        "## Results (worst arm, RMS over 8 s after onset)", "", *md, "",
        "## LaTeX", "", "```latex", *tex, "```", "",
        "## Caption", "",
        "Reactive disturbance rejection against mount disturbance frequency. "
        "The same scripted six-axis mount motion is applied in every run and "
        "only its frequency f is varied (top: slowest and fastest run, "
        "vertical axis). Bottom: end-effector position and orientation error "
        "RMS over 8 s with the arm rigid on the mount (no control, dashed) and "
        "with the reactive controller (solid); labels give the share of the "
        "uncontrolled error removed. Worst of the two arms is shown; the "
        "world-frame target is fixed. Residual error grows with frequency "
        "because the controller acts only after error appears.", ""]
    Path(path).write_text("\n".join(text))

if __name__ == "__main__":
    freqs = FREQS_HZ
    for a in sys.argv[1:]:
        if a.startswith("--freqs="):
            freqs = [float(v) for v in a.split("=", 1)[1].split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    if "--replot" in sys.argv:
        with open(OUT / "freq_sweep.csv") as fh:
            rows = [{k: float(v) for k, v in r.items()}
                    for r in csv.DictReader(fh)]
        rows = [r for r in rows if r["f_hz"] in FREQS_HZ]
        freqs = []
    for f_hz in freqs:
        rows.append(run_one(f_hz, walk_report.world.SIDES))
        r = rows[-1]
        print(f"f = {f_hz:g} Hz: pos {r['pos']:.1f} / {r['pos_free']:.1f} mm, "
              f"rot {r['rot']:.2f} / {r['rot_free']:.2f} deg "
              f"(control / none), settled {r['settled']}")
    if freqs:
        with open(OUT / "freq_sweep.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    figure(rows, OUT / "freq_sweep.png")
    write_report_text(rows, OUT / "freq_sweep_report.md")
    print(f"figure: {OUT / 'freq_sweep.png'}")
