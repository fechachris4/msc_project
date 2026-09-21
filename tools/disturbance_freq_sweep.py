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

import report_style  # noqa: E402
import walk_report  # noqa: E402
import walk_sim  # noqa: E402

OUT = Path("analysis/output/disturbance")
AMPLITUDE_KEY = 1.0        # table row whose amplitudes are held fixed
FREQS_HZ = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]   # add 1.8 to re-check vs --speed=1.0

_table_params = walk_sim.walk_params
_fundamental_hz = [None]


def fixed_amplitude_params(scale=walk_sim.GAIT_SCALE, speed=None):
    p = _table_params(scale=scale, speed=AMPLITUDE_KEY)
    f = _fundamental_hz[0]
    p["linear_frequency"] = np.array([f, 0.5 * f, f])
    p["rotational_frequency"] = np.array([0.5 * f, f, 0.5 * f])
    return p


def _per_period_rms(t, series, period_s, dt=0.002):
    """RMS over each complete pattern period; the last sample sits one step
    before the end of the window, hence the one-step tolerance."""
    out = []
    k = 0
    while (k + 1) * period_s <= t.max() + dt + 1e-9:
        m = (t >= k * period_s) & (t < (k + 1) * period_s)
        out.append(np.sqrt(np.mean(series[m] ** 2)))
        k += 1
    return np.array(out)


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
    # Worst arm. Value = RMS over the whole 8 s; *_sd = SD of the RMS of each
    # complete pattern period (2/f), n_periods of them.
    period_s = 2.0 / f_hz
    out = dict(f_hz=f_hz, settled=int(settled), n_periods=0,
               saturated_pct=0.0, qdot_peak_deg_s=0.0)
    for key in ("pos", "pos_free", "rot", "rot_free"):
        out[key] = 0.0
        out[key + "_sd"] = 0.0
    for side in arms:
        out["saturated_pct"] = max(out["saturated_pct"], 100.0 * float(np.mean(
            walk_report._stack(rows, "speed_saturated", side)[on])))
        out["qdot_peak_deg_s"] = max(out["qdot_peak_deg_s"], float(np.degrees(
            walk_report._stack(rows, "qdot_max", side)[on].max())))
        series = dict(
            pos=1e3 * np.linalg.norm(
                walk_report._stack(rows, "e_pos", side), axis=1),
            pos_free=1e3 * np.linalg.norm(
                walk_report._stack(rows, "rigid_err", side), axis=1),
            rot=np.degrees(np.linalg.norm(
                walk_report._stack(rows, "e_rot", side), axis=1)),
            rot_free=np.degrees(walk_report._stack(rows, "rigid_rot_err", side)))
        for key, v in series.items():
            rms = float(np.sqrt(np.mean(v[on] ** 2)))
            if rms > out[key]:
                per = _per_period_rms(t[on], v[on], period_s)
                out[key] = rms
                out[key + "_sd"] = float(per.std(ddof=1))
                out["n_periods"] = len(per)
    return out


def figure(rows, path):
    """(a) what varies between runs; (b, c) error RMS against frequency.
    No title: the caption is in freq_sweep_report.md."""
    report_style.apply()
    f = np.array([r["f_hz"] for r in rows])
    fig = plt.figure(figsize=(report_style.FULL_WIDTH_IN, 4.6),
                     layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.7])
    ax_how = fig.add_subplot(grid[0, :])
    axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    ts = np.linspace(0.0, 2.0, 600)
    for f_hz, col in ((f.min(), "0.6"), (f.max(), "black")):
        _fundamental_hz[0] = f_hz
        walk_sim.walk_params = fixed_amplitude_params
        z = [walk_sim.torso_pose_at(s, speed=AMPLITUDE_KEY)[0][2] for s in ts]
        walk_sim.walk_params = _table_params
        ax_how.plot(ts, 1e3 * (np.array(z) - z[0]), color=col, linewidth=1.4,
                    label=f"f = {f_hz:g} Hz")
    ax_how.set_xlabel("time [s]")
    ax_how.set_ylabel("mount vertical\ndisplacement [mm]")
    ax_how.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    for ax, key, unit, name in ((axes[0], "pos", "mm", "position"),
                                (axes[1], "rot", "deg", "orientation")):
        free = np.array([r[key + "_free"] for r in rows])
        ctrl = np.array([r[key] for r in rows])
        ax.errorbar(f, free, yerr=[r[key + "_free_sd"] for r in rows],
                    marker="s", capsize=2, label=report_style.NO_CONTROL_LABEL,
                    **report_style.NO_CONTROL)
        ax.errorbar(f, ctrl, yerr=[r[key + "_sd"] for r in rows], marker="o",
                    capsize=2, label=report_style.REACTIVE_LABEL,
                    **report_style.REACTIVE)
        sat = np.array([r["saturated_pct"] > 0.0 for r in rows])
        ax.plot(f[sat], ctrl[sat], linestyle="none", marker="o",
                markerfacecolor="white", zorder=3,
                markeredgecolor=report_style.REACTIVE["color"],
                label="joint-speed limit reached" if sat.any() else None)
        ax.set_xlabel("mount disturbance frequency f [Hz]")
        ax.set_ylabel(f"end-effector {name}\nerror RMS [{unit}]")
        ax.set_ylim(bottom=0)
        ax.set_xticks(f)
    axes[0].legend(loc="center left")
    report_style.panel_letters([ax_how] + axes, x=-0.09)
    report_style.save(fig, path)


def write_report_text(rows, path):
    """Markdown table, LaTeX tabular and caption for the report."""
    p = _table_params(speed=AMPLITUDE_KEY)
    lin = 1e3 * p["linear_amplitude"]
    rot = np.degrees(p["rotational_amplitude"])
    md = ["| f [Hz] | position RMS [mm] | removed | orientation RMS [deg] "
          "| removed | peak joint speed [deg/s] | samples at speed limit |",
          "|---|---|---|---|---|---|---|"]
    tex = ["\\begin{tabular}{rrrrr}", "\\toprule",
           "$f$ [Hz] & pos.\\ RMS [mm] & removed & ori.\\ RMS [deg] & "
           "removed \\\\", "\\midrule"]
    for r in rows:
        rp = 100 * (1 - r["pos"] / r["pos_free"])
        rr = 100 * (1 - r["rot"] / r["rot_free"])
        md.append(f"| {r['f_hz']:g} | {r['pos']:.1f} | {rp:.0f}% | "
                  f"{r['rot']:.2f} | {rr:.0f}% | {r['qdot_peak_deg_s']:.0f} | "
                  f"{r['saturated_pct']:.0f}% |")
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
        "only its frequency f is varied. (a) Vertical mount displacement in "
        "the slowest and fastest run; the other five axes are sped up in the "
        "same way. (b) End-effector position and (c) orientation error, RMS "
        "over 8 s after disturbance onset, with the arm rigid on the mount "
        "(no control) and with the reactive controller; worst of the two "
        "arms, world-frame target fixed. Error bars: SD of the per-period "
        f"RMS over the n = {int(min(r['n_periods'] for r in rows))} to "
        f"{int(max(r['n_periods'] for r in rows))} complete pattern periods "
        "(2/f) in each run; the simulation is deterministic, so they show "
        "period-to-period variation including the onset period. The "
        "controller removes "
        f"{100 * (1 - rows[0]['pos'] / rows[0]['pos_free']):.0f}% of the "
        f"position error at {rows[0]['f_hz']:g} Hz but only "
        f"{100 * (1 - rows[-1]['pos'] / rows[-1]['pos_free']):.0f}% at "
        f"{rows[-1]['f_hz']:g} Hz.", ""]
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
        print(f"f = {f_hz:g} Hz: sat {r['saturated_pct']:.1f}%, qdot peak "
              f"{r['qdot_peak_deg_s']:.1f} deg/s, "
              f"pos {r['pos']:.1f} / {r['pos_free']:.1f} mm, "
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
