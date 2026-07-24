"""Canonical MuJoCo validation for the current reactive baseline.

Produces one provenance-stamped run plus a thesis-ready figure containing the
world-frame tracking traces and mean/RMSE/peak position-error summary.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import cartesian_path, live, metrics
from plotting.style import C_BASE, C_LEFT, C_RIGHT
from runtime_config import CONFIG, print_effective_config
from sim import motion


DEFAULT_OUTPUT = Path("analysis/output/reactive_baseline")

# Two complete periods of the current default translation/rotation scenario.
# The 10 mm settling gate only selects the evaluation start; raw error remains
# fully reported. It clears the known ~6 mm static residual at the configured
# left task point without hiding it from the metrics.
SCENARIO = live.ExperimentConfig(
    arms=("right", "left"),
    linear_amplitude=np.array(
        motion.LINEAR_AMPLITUDE, dtype=float, copy=True),
    linear_frequency=float(motion.LINEAR_FREQUENCY),
    rotational_amplitude=np.array(
        motion.ROTATIONAL_AMPLITUDE, dtype=float, copy=True),
    rotational_frequency=float(motion.ROTATIONAL_FREQUENCY),
    evaluation_seconds=4.0,
    settle_pos_tol=0.010,
    settle_rot_tol=np.deg2rad(0.5),
    settle_dwell=0.5,
    settle_timeout=20.0,
)


def make_figure(log, run_metrics, path):
    """Write world-frame error traces and mean/RMSE/peak summary."""
    mask = log.evaluation_mask
    time_s = log.eval_time[mask]
    base_norm_mm = np.linalg.norm(
        log.base_displacement[mask], axis=1) * 1000.0

    fig, (trace_ax, metric_ax) = plt.subplots(
        2, 1, figsize=(9, 8), layout="constrained")
    trace_ax.plot(
        time_s,
        base_norm_mm,
        color=C_BASE,
        linewidth=1.0,
        label="|base displacement|",
    )
    for side, color, style in (
        ("right", C_RIGHT, "-"),
        ("left", C_LEFT, "--"),
    ):
        error_norm_mm = np.linalg.norm(
            log.arm_data[side]["e_pos"][mask], axis=1) * 1000.0
        trace_ax.plot(
            time_s,
            error_norm_mm,
            color=color,
            linestyle=style,
            linewidth=1.2,
            label=f"{side} |position error|",
        )
    trace_ax.set_xlabel("evaluation time [s]")
    trace_ax.set_ylabel("world-frame magnitude [mm]")
    trace_ax.set_title("Reactive world-frame pose hold under base motion")
    trace_ax.legend()
    trace_ax.grid(alpha=0.2)

    labels = ("mean", "RMSE", "peak")
    x = np.arange(len(labels))
    width = 0.34
    for offset, (side, color) in (
        (-0.5 * width, ("right", C_RIGHT)),
        (0.5 * width, ("left", C_LEFT)),
    ):
        arm = run_metrics["arms"][side]
        values_mm = 1000.0 * np.array([
            arm["position_error_norm_mean_m"],
            arm["position_error_norm_rmse_m"],
            arm["position_error_norm_peak_m"],
        ])
        bars = metric_ax.bar(
            x + offset, values_mm, width, color=color, label=side)
        metric_ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    metric_ax.set_xticks(x, labels)
    metric_ax.set_ylabel("position error norm [mm]")
    metric_ax.set_title("Evaluation summary")
    metric_ax.legend()
    metric_ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"Reactive baseline · config {CONFIG.source_sha256[:12]}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def run_validation(output_root=DEFAULT_OUTPUT, canonical=False):
    print_effective_config(CONFIG)
    log = live.run_experiment(SCENARIO)
    run_metrics = metrics.experiment_metrics(log)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tracking_figure = make_figure(
        log, run_metrics, output_root / "reactive_baseline_tracking.png")
    path_figure = cartesian_path.make_cartesian_path_figure(
        log,
        output_root / "reactive_baseline_cartesian_path.png",
    )
    run_dir = live.save_run(
        log,
        SCENARIO,
        output_root,
        canonical=canonical,
        final_outputs=(tracking_figure, path_figure),
    )

    print(
        f"settled={log.settled} accepted={log.accepted} "
        f"warnings={list(log.warning_reasons)}"
    )
    for side in log.arms:
        arm = run_metrics["arms"][side]
        print(
            f"{side}: mean={arm['position_error_norm_mean_m'] * 1000:.1f} "
            f"mm  RMSE={arm['position_error_norm_rmse_m'] * 1000:.1f} "
            f"mm  peak={arm['position_error_norm_peak_m'] * 1000:.1f} mm"
        )
    print(f"run: {run_dir}")
    print(f"tracking figure: {tracking_figure}")
    print(f"Cartesian path figure: {path_figure}")
    return log, run_metrics, run_dir, tracking_figure


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--canonical", action="store_true")
    args = parser.parse_args(argv)
    run_validation(args.output, args.canonical)


if __name__ == "__main__":
    main()
