"""Kp x Kd sweep of the reactive controller under the scripted mount disturbance.

Each episode is one tools/walk_report.run: settle on the static mount,
then disturb at level key SPEED_M_S for 8 s with the given position gains (rotation,
null-space and damping gains stay at the config values). Metrics are the
worst-arm position RMS/peak, orientation RMS, joint-speed saturation, and
whether the arm settled at all. Results append to a CSV after every
episode, so an interrupted sweep resumes.

Outputs (analysis/output/disturbance/gain_sweep/):
  sweep.csv                    one row per episode
  gain_sweep_heatmaps.png      position RMS, orientation RMS, saturation
  gain_sweep_kp_curves.png     position RMS vs Kp per Kd, with the lag
                               model prediction
  gain_sweep_qdot_curves.png   peak / RMS commanded joint speed vs Kp,
                               against the robot and sim velocity caps

usage: python tools/walk_gain_sweep.py [--speed=1.0] [--workers=4]
                                       [--fresh] [--plots-only]
"""

import csv
import multiprocessing as mp
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = Path("analysis/output/disturbance/gain_sweep")
SPEED_M_S = 1.0
KP_POS_GRID = [4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0]
KD_POS_GRID = [0.0, 0.3, 0.6, 0.9, 1.5]
FIELDS = ["kp_pos", "kd_pos", "speed", "settled", "settle_s",
          "pos_rms_mm", "pos_peak_mm", "rot_rms_deg", "rot_peak_deg",
          "saturated_pct", "qdot_max_deg_s", "qdot_onset_max_deg_s",
          "qdot_raw_max_deg_s", "qdot_rms_deg_s", "over_hw_cap_pct",
          "rigid_rms_mm"]
ONSET_S = 0.5   # joint-speed peaks are taken after this, excluding the
                # step transient of an abruptly started disturbance
HW_QDOT_CAP_DEG_S = 45.0     # Christian_control kModelVelocityLimitsDegS
SIM_QDOT_CAP_DEG_S = (69.9, 79.6)  # config joint_velocity_rad_s, small/large


def episode(job):
    kp, kd, speed = job
    import walk_report  # noqa: E402  (fresh MuJoCo world per process)
    from runtime_config import CONFIG, control_with_legacy_overrides
    control = control_with_legacy_overrides(
        CONFIG.reactive_pose, {"KP_POS": kp, "KD_POS": kd})
    arms = ("right", "left")
    settled, settle_s, rows, _ = walk_report.run(
        arms, 1.0, speed, record_gif=False, controller_config=control)
    t = walk_report._stack(rows, "t")
    walking = t >= 0.0
    steady = t >= ONSET_S
    worst = dict(pos_rms=0.0, pos_peak=0.0, rot_rms=0.0, rot_peak=0.0,
                 sat=0.0, qdot=0.0, qdot_onset=0.0, qdot_raw=0.0,
                 qdot_rms=0.0, over=0.0, rigid=0.0)
    for side in arms:
        e = 1e3 * np.linalg.norm(
            walk_report._stack(rows, "e_pos", side), axis=1)[walking]
        r = np.degrees(np.linalg.norm(
            walk_report._stack(rows, "e_rot", side), axis=1))[walking]
        rigid = 1e3 * np.linalg.norm(
            walk_report._stack(rows, "rigid_err", side), axis=1)[walking]
        sat = walk_report._stack(rows, "speed_saturated", side)[walking]
        qd_all = walk_report._stack(rows, "qdot_max", side)
        qd = qd_all[steady]
        worst["qdot_onset"] = max(
            worst["qdot_onset"], np.degrees(qd_all[walking].max()))
        worst["pos_rms"] = max(worst["pos_rms"], np.sqrt(np.mean(e**2)))
        worst["pos_peak"] = max(worst["pos_peak"], e.max())
        worst["rot_rms"] = max(worst["rot_rms"], np.sqrt(np.mean(r**2)))
        worst["rot_peak"] = max(worst["rot_peak"], r.max())
        worst["sat"] = max(worst["sat"], 100.0 * np.mean(sat))
        worst["qdot"] = max(worst["qdot"], np.degrees(qd.max()))
        qraw = walk_report._stack(rows, "qdot_raw_max", side)[steady]
        qrms = walk_report._stack(rows, "qdot_rms", side)[steady]
        worst["qdot_raw"] = max(worst["qdot_raw"], np.degrees(qraw.max()))
        worst["qdot_rms"] = max(
            worst["qdot_rms"], np.degrees(np.sqrt(np.mean(qrms**2))))
        worst["over"] = max(worst["over"], 100.0 * np.mean(
            np.degrees(qd) > HW_QDOT_CAP_DEG_S))
        worst["rigid"] = max(worst["rigid"], np.sqrt(np.mean(rigid**2)))
    return dict(kp_pos=kp, kd_pos=kd, speed=speed, settled=int(settled),
                settle_s=settle_s, pos_rms_mm=worst["pos_rms"],
                pos_peak_mm=worst["pos_peak"], rot_rms_deg=worst["rot_rms"],
                rot_peak_deg=worst["rot_peak"], saturated_pct=worst["sat"],
                qdot_max_deg_s=worst["qdot"],
                qdot_onset_max_deg_s=worst["qdot_onset"],
                qdot_raw_max_deg_s=worst["qdot_raw"],
                qdot_rms_deg_s=worst["qdot_rms"],
                over_hw_cap_pct=worst["over"], rigid_rms_mm=worst["rigid"])


def load_rows(path):
    if not path.exists():
        return []
    with open(path) as f:
        return [{k: float(v) for k, v in r.items()}
                for r in csv.DictReader(f)]


def append_row(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row[k] for k in FIELDS})


def grid(rows, key):
    table = np.full((len(KD_POS_GRID), len(KP_POS_GRID)), np.nan)
    for r in rows:
        if r["kp_pos"] in KP_POS_GRID and r["kd_pos"] in KD_POS_GRID:
            i = KD_POS_GRID.index(r["kd_pos"])
            j = KP_POS_GRID.index(r["kp_pos"])
            table[i, j] = r[key]
    return table


def heatmaps(rows, speed, path):
    panels = [("pos_rms_mm", "position error RMS [mm]", "viridis_r"),
              ("rot_rms_deg", "orientation error RMS [deg]", "viridis_r"),
              ("qdot_max_deg_s", "peak commanded joint speed [deg/s]",
               "magma_r")]
    import report_style
    report_style.apply()
    fig, axes = plt.subplots(3, 1, figsize=(report_style.FULL_WIDTH_IN, 6.6),
                             layout="constrained")
    settled = grid(rows, "settled")
    for ax, (key, label, cmap) in zip(axes, panels):
        table = grid(rows, key)
        im = ax.imshow(table, origin="lower", cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(KP_POS_GRID)),
                      [f"{v:g}" for v in KP_POS_GRID])
        ax.set_yticks(range(len(KD_POS_GRID)),
                      [f"{v:g}" for v in KD_POS_GRID])
        ax.set_xlabel("Kp position [1/s]")
        ax.set_ylabel("Kd position [-]")
        fig.colorbar(im, ax=ax, shrink=0.9, label=label)
        for i in range(len(KD_POS_GRID)):
            for j in range(len(KP_POS_GRID)):
                v = table[i, j]
                if np.isnan(v):
                    continue
                txt = f"{v:.1f}" if key != "qdot_max_deg_s" else f"{v:.0f}"
                if settled[i, j] == 0:
                    txt += "*"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="white" if v > np.nanmax(table) * 0.6
                        else "black")
    best = min((r for r in rows if r["settled"]),
               key=lambda r: r["pos_rms_mm"])
    report_style.panel_letters(axes, x=-0.06)
    print(f"caption facts: disturbance {_level(speed)}; best position RMS "
          f"{best['pos_rms_mm']:.1f} mm at Kp={best['kp_pos']:g}, "
          f"Kd={best['kd_pos']:g}; * = did not settle before disturbance onset")
    report_style.save(fig, path)
    return best


def _level(speed):
    import walk_sim
    return walk_sim.level_label(speed)


def _conditions(speed):
    from runtime_config import CONFIG
    c = CONFIG.reactive_pose
    return (f"disturbance {_level(speed)}, both arms, world-fixed targets\n"
            f"held fixed: $K_R$ = {c.kp_rotation_s_inv:g} 1/s, "
            f"$K_d$ (angular rows) = {c.kd_rotation:g}, "
            f"centring $k_n$ = {c.null_gain_s_inv:g} 1/s, "
            f"$\\lambda$ = {c.dls_damping:g}")


def kp_curves(rows, speed, path):
    import walk_sim
    p = walk_sim.walk_params(speed=speed)
    import report_style
    report_style.apply()
    fig, ax = plt.subplots(figsize=(report_style.FULL_WIDTH_IN, 3.9),
                           layout="constrained")
    cmap = plt.get_cmap("viridis")
    for i, kd in enumerate(KD_POS_GRID):
        pts = sorted((r["kp_pos"], r["pos_rms_mm"], r["settled"])
                     for r in rows if r["kd_pos"] == kd)
        if not pts:
            continue
        kp = np.array([x[0] for x in pts])
        rms = np.array([x[1] for x in pts])
        ok = np.array([x[2] for x in pts], dtype=bool)
        col = cmap(i / max(1, len(KD_POS_GRID) - 1))
        ax.plot(kp, rms, marker="o", color=col, label=f"$K_d$ = {kd:g}")
        if (~ok).any():
            ax.plot(kp[~ok], rms[~ok], "x", color="red", markersize=9)
    # Model prediction. From the implemented law with an ideal inner servo,
    # (1+Kd) e_dot + Kp e = -v_rigid(t), where v_rigid is the velocity the
    # EE would have if welded to the mount at the target. Integrate that
    # filter on the analytic disturbance for each Kd and compare RMS to the sweep.
    from controller import transforms
    from sim import motion
    from runtime_config import CONFIG
    target = np.asarray(CONFIG.right_target.position_m, dtype=float)
    r0 = target - motion.HOME_POS
    dt = 0.002
    t = np.arange(0.0, 8.0, dt)
    rigid = np.array([
        walk_sim.torso_pose_at(s, speed=speed)[0]
        + transforms.rotation_from_rpy(
            walk_sim.torso_pose_at(s, speed=speed)[1]) @ r0
        for s in t]) - target
    v_rigid = np.gradient(rigid, dt, axis=0)
    kp_line = np.geomspace(min(KP_POS_GRID), max(KP_POS_GRID), 30)
    for i, kd in enumerate(KD_POS_GRID):
        pred = []
        for kp in kp_line:
            e = np.zeros(3)
            acc = 0.0
            for k in range(len(t)):
                e = e + dt * (-v_rigid[k] - kp * e) / (1.0 + kd)
                acc += float(e @ e)
            pred.append(1e3 * np.sqrt(acc / len(t)))
        col = cmap(i / max(1, len(KD_POS_GRID) - 1))
        ax.plot(kp_line, pred, color=col, linestyle="--", linewidth=1.0,
                label=("lag model $(1+K_d)\\,\\dot e + K_p e = -v_{mount}(EE)$"
                       ", dashed per $K_d$") if i == 0 else None)
    rigid_rms = max(r["rigid_rms_mm"] for r in rows)
    ax.axhline(rigid_rms, label=report_style.NO_CONTROL_LABEL,
               **report_style.NO_CONTROL)
    ax.set_xscale("log")
    ax.set_xticks(KP_POS_GRID, [f"{v:g}" for v in KP_POS_GRID])
    ax.minorticks_off()
    ax.set_xlabel("position gain $K_p$ [1/s]  (log scale)")
    ax.set_ylabel("end-effector position\nerror RMS [mm]")
    ax.set_ylim(0, 1.5 * rigid_rms)
    ax.legend(loc="upper right", fontsize=7.5, ncol=2)
    print("caption facts: " + _conditions(speed).replace("\n", "; "))
    report_style.save(fig, path)
    plt.close(fig)


def qdot_curves(rows, speed, path):
    """Steady-state peak commanded joint speed vs Kp, per Kd, against the
    robot and simulation joint-speed caps."""
    import report_style
    report_style.apply()
    fig, ax = plt.subplots(figsize=(report_style.FULL_WIDTH_IN, 4.4),
                           layout="constrained")
    cmap = plt.get_cmap("viridis")
    for i, kd in enumerate(KD_POS_GRID):
        pts = sorted((r["kp_pos"], r["qdot_max_deg_s"])
                     for r in rows if r["kd_pos"] == kd)
        if not pts:
            continue
        ax.plot([x[0] for x in pts], [x[1] for x in pts], marker="o",
                color=cmap(i / max(1, len(KD_POS_GRID) - 1)),
                label=f"$K_d$ = {kd:g}")
    ax.axhline(HW_QDOT_CAP_DEG_S, color="red", linewidth=1.2,
               label=f"robot clip as configured now, {HW_QDOT_CAP_DEG_S:g} deg/s")
    ax.axhline(SIM_QDOT_CAP_DEG_S[0], color="red", linewidth=0.8,
               linestyle="--",
               label=f"Gen3 limits {SIM_QDOT_CAP_DEG_S[1]:g} / "
                     f"{SIM_QDOT_CAP_DEG_S[0]:g} deg/s (sim clip; rig runs 79.2)")
    ax.axhline(SIM_QDOT_CAP_DEG_S[1], color="red", linewidth=0.8,
               linestyle="--")
    ax.set_xscale("log")
    ax.set_xticks(KP_POS_GRID, [f"{v:g}" for v in KP_POS_GRID])
    ax.minorticks_off()
    ax.set_xlabel("position gain $K_p$ [1/s]  (log scale)")
    ax.set_ylabel("peak commanded joint speed [deg/s]\n"
                  f"(max over joints and arms, t > {ONSET_S:g} s)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7.5, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.17))
    report_style.save(fig, path)
    plt.close(fig)


def main(argv):
    speed = SPEED_M_S
    workers = 4
    fresh = "--fresh" in argv
    plots_only = "--plots-only" in argv
    for a in argv:
        if a.startswith("--speed="):
            speed = float(a.split("=", 1)[1])
        if a.startswith("--workers="):
            workers = int(a.split("=", 1)[1])
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "sweep.csv"
    if fresh and csv_path.exists():
        csv_path.unlink()
    rows = load_rows(csv_path)
    done = {(r["kp_pos"], r["kd_pos"]) for r in rows}
    jobs = [(kp, kd, speed) for kp in KP_POS_GRID for kd in KD_POS_GRID
            if (kp, kd) not in done]
    if not plots_only and jobs:
        print(f"{len(jobs)} episodes to run, {len(done)} already done, "
              f"{workers} workers", flush=True)
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for row in pool.imap_unordered(episode, jobs):
                append_row(csv_path, row)
                rows.append(row)
                print(f"Kp={row['kp_pos']:5.1f} Kd={row['kd_pos']:.1f} "
                      f"settled={row['settled']} pos RMS "
                      f"{row['pos_rms_mm']:5.1f} mm  rot RMS "
                      f"{row['rot_rms_deg']:4.2f} deg  sat "
                      f"{row['saturated_pct']:4.0f}%", flush=True)
    best = heatmaps(rows, speed, OUT / "gain_sweep_heatmaps.png")
    kp_curves(rows, speed, OUT / "gain_sweep_kp_curves.png")
    qdot_curves(rows, speed, OUT / "gain_sweep_qdot_curves.png")
    print(f"best settled: Kp={best['kp_pos']:g} Kd={best['kd_pos']:g} -> "
          f"{best['pos_rms_mm']:.1f} mm RMS, {best['rot_rms_deg']:.2f} deg")
    print(f"outputs: {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
