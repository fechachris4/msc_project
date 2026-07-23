"""Base motion vs. end-effector error, live: the thesis success criterion
made visible while the sim runs. Read-only, same pattern as
analysis/diagnose.py and analysis/validate_velocity.py: the controller
runs unmodified, and logged errors come from the same explicit state and
pure world-frame error function used by the controller.

    python -m analysis.base_vs_error               # live, both arms
    python -m analysis.base_vs_error --save 30      # headless, 30 sim-seconds

Plain python (no MuJoCo viewer, no mjpython). Live mode opens a window
with a ~30 s rolling view of base displacement vs. per-axis EE error,
world frame, error = ref - actual. The sim runs until the
window is closed (or Ctrl-C); then a full-run RMS/peak/rejection-%
table prints on stdout and the final window is snapshot to
analysis/output/base_vs_error.png. --save mode settles, runs a fixed
duration headlessly, then does the same
table + save. Units are SI internally; mm on the figure and in the
printed table.
"""

import sys
from pathlib import Path

import matplotlib

if "--save" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from analysis import metrics
from controller import desired_pos
from controller.runner import ReactivePositionRunner
from plotting.live_plot import LivePlot
from plotting.style import C_BASE, C_RIGHT, C_LEFT
from sim import motion, world

# Static settle before the scenario starts (validate_velocity.py's
# pattern): the arms boot from a default configuration far from
# desired_pos.apply()'s targets, so the first ~1 s is a large one-off
# convergence transient, not disturbance response. Without settling
# first, that transient — not the base sway — would set the peak error
# and swamp the rejection-% this figure exists to show.
SETTLE_SECONDS = 2.0
WINDOW_S = 30.0  # rolling on-screen window: 3 periods at 0.1 Hz

OUT = Path("analysis/output")


class _DelayedDefaultMotion:
    def __init__(self):
        self._start_time_s = None

    def start_at(self, sample_time_s):
        self._start_time_s = float(sample_time_s)

    def pose_at(self, sample_time_s):
        if self._start_time_s is None:
            return motion.HOME_POS.copy(), motion.HOME_RPY.copy()
        return motion.torso_pose_at(sample_time_s - self._start_time_s)

    def twist_at(self, sample_time_s):
        if self._start_time_s is None:
            return np.zeros(3), np.zeros(3)
        return motion.torso_twist_at(sample_time_s - self._start_time_s)


def run(save_seconds=None):
    """Closed-loop rollout under sim.motion defaults with a live rolling plot.
    save_seconds: if given, run headless for that many sim-seconds; otherwise
    run live until the window is closed (or Ctrl-C). Returns (log, plot)
    where log = {t, base_disp, right_e, left_e} full-run arrays
    (base_disp and *_e in meters, world frame)."""
    world.backend.release()
    world.backend.configure_torso_driver(None, None)
    world.backend.reset()
    source_targets = desired_pos.apply()
    driver = _DelayedDefaultMotion()
    world.backend.configure_torso_driver(driver.pose_at, driver.twist_at)
    runner = ReactivePositionRunner(
        world.backend,
        world.MOUNT_CALIBRATION,
        world.PIPELINE_SETUP,
        source_targets,
    )
    runner.start()

    dt = runner.current_state.nominal_dt_s
    settle_steps = int(SETTLE_SECONDS / dt)
    for step in range(settle_steps):
        if step == settle_steps - 1:
            driver.start_at(
                runner.current_state.sample_time_s + dt)
        runner.cycle()

    home_pos = motion.HOME_POS
    t_start = runner.current_state.sample_time_s
    n_steps = int(save_seconds / dt) if save_seconds is not None else None

    amp_mm = motion.LINEAR_AMPLITUDE * 1000.0
    plot = LivePlot(
        rows=["world x [mm]", "world y [mm]", "world z [mm]"],
        signals={
            "base disp": {"color": C_BASE},
            "right EE error": {"color": C_RIGHT},
            "left EE error": {"style": "--", "color": C_LEFT},
        },
        window=int(WINDOW_S / dt),
        title=(f"Base motion vs. EE error, world frame "
               f"(error = ref − actual)\n"
               f"{motion.LINEAR_FREQUENCY:g} Hz, "
               f"±[{amp_mm[0]:.0f}, {amp_mm[1]:.0f}, "
               f"{amp_mm[2]:.0f}] mm"),
    )

    # Full-run history for the stats table — the LivePlot ring buffers
    # only keep the last WINDOW_S seconds.
    log = {"t": [], "base_disp": [], "right_e": [], "left_e": []}

    step = 0
    try:
        while True:
            if n_steps is not None:
                if step >= n_steps:
                    break
            elif not plot.is_open():
                break

            cycle = runner.cycle()
            plant = cycle.input_state
            t = plant.sample_time_s - t_start
            base_pos = plant.torso_pose_world.position_m
            base_disp = base_pos - home_pos
            right_e = cycle.traces.right.e_pos
            left_e = cycle.traces.left.e_pos

            log["t"].append(t)
            log["base_disp"].append(base_disp)
            log["right_e"].append(right_e)
            log["left_e"].append(left_e)
            plot.add(t, {
                "base disp": base_disp * 1000.0,
                "right EE error": right_e * 1000.0,
                "left EE error": left_e * 1000.0,
            })

            step += 1
    except KeyboardInterrupt:
        pass  # fall through to the stats table
    finally:
        runner.close()
        world.backend.configure_torso_driver(None, None)

    return {k: np.asarray(v) for k, v in log.items()}, plot


def main(save_seconds=None):
    log, plot = run(save_seconds)
    if len(log["t"]) == 0:
        print("Window closed before any samples were logged.")
        return
    st = metrics.stats(log)
    print(f"\nFull run: {log['t'][-1]:.1f} s "
          f"({len(log['t'])} samples)")
    metrics.print_stats(st)
    OUT.mkdir(parents=True, exist_ok=True)
    plot.save(OUT / "base_vs_error.png")
    print(f"\nSaved {OUT}/base_vs_error.png (last {WINDOW_S:.0f} s window)")


if __name__ == "__main__":
    _args = sys.argv[1:]
    _save_seconds = None
    if "--save" in _args:
        _i = _args.index("--save")
        _save_seconds = float(_args[_i + 1])
    main(_save_seconds)
