"""Base motion vs. end-effector error, live: the thesis success criterion
made visible while the sim runs. Read-only, same pattern as
analysis/diagnose.py and analysis/validate_velocity.py: the controller
runs unmodified, every logged quantity comes from the public functions
apply_ctrl itself calls (servo.pose_error), nothing re-derived.

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
import mujoco
import numpy as np

from analysis import metrics
from controller import desired_pos, frames, servo
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


def run(save_seconds=None):
    """Closed-loop rollout under sim.motion defaults with a live rolling plot.
    save_seconds: if given, run headless for that many sim-seconds; otherwise
    run live until the window is closed (or Ctrl-C). Returns (log, plot)
    where log = {t, base_disp, right_e, left_e} full-run arrays
    (base_disp and *_e in meters, world frame)."""
    mujoco.mj_resetData(world.model, world.data)
    mujoco.mj_forward(world.model, world.data)
    desired_pos.apply()
    servo.init_ctrl()

    dt = world.model.opt.timestep
    zero_twist = (np.zeros(3), np.zeros(3))
    for _ in range(int(SETTLE_SECONDS / dt)):
        servo.apply_ctrl(dt, zero_twist)
        mujoco.mj_step(world.model, world.data)

    home_pos = motion.HOME_POS
    t_start = world.data.time  # phase 0 at motion start: no teleport
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

            t = world.data.time - t_start
            motion.set_torso_pose(t)
            # refresh xpos/xmat so the logged state sees the torso pose
            # at t, not the previous step's (main.py/diagnose.py pattern)
            mujoco.mj_kinematics(world.model, world.data)
            # set_torso_pose (mocap write) and torso_twist_at
            # (feedforward) must stay a matched pair — same scenario,
            # same instant t.
            base_twist = motion.torso_twist_at(t)

            base_pos, _ = frames.torso_pose()
            base_disp = base_pos - home_pos
            right_e, _ = servo.pose_error("right")
            left_e, _ = servo.pose_error("left")

            log["t"].append(t)
            log["base_disp"].append(base_disp)
            log["right_e"].append(right_e)
            log["left_e"].append(left_e)
            plot.add(t, {
                "base disp": base_disp * 1000.0,
                "right EE error": right_e * 1000.0,
                "left EE error": left_e * 1000.0,
            })

            servo.apply_ctrl(dt, base_twist)
            mujoco.mj_step(world.model, world.data)
            step += 1
    except KeyboardInterrupt:
        pass  # fall through to the stats table

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
