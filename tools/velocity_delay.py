"""What a late mount-velocity signal costs: the README disturbance at f = 1.8 Hz,
right arm, with the mount's contribution to the measured end-effector twist
delayed by n control cycles before the controller sees it.

This mirrors the hardware structure: the controller's velocity terms (the Kd
term and, when enabled, the feedforward) use the world-frame tool twist, whose
mount part comes from an estimator and can arrive late, while the arm's own
part (J q_dot) is current. In simulation the mount part is exact unless
delayed here. Only this tool changes; the controller is untouched.

usage: python tools/velocity_delay.py [--delays=0,10,20,40,60,80]
writes analysis/output/disturbance/velocity_delay.png and .csv
"""

import collections
import csv
import dataclasses
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import report_style  # noqa: E402
import disturbance_run  # noqa: E402
import mount_disturbance  # noqa: E402
from controller import reactive_controller, servo  # noqa: E402
from controller.state import Twist  # noqa: E402

OUT = Path("analysis/output/disturbance")
F_HZ = 1.8
SIDE = "right"
DT_S = 0.002
_compute = reactive_controller.ReactiveController.compute
_delay_cycles = 0


def _delayed_compute(self, state, target):
    """ReactiveController.compute with the mount part of the twist delayed."""
    arm = state.jacobian_world @ state.joints.velocity_rad_s
    total = np.concatenate([state.ee_twist_world.linear_m_s,
                            state.ee_twist_world.angular_rad_s])
    buf = self.__dict__.setdefault(
        "_mount_twist_history", collections.deque(maxlen=_delay_cycles + 1))
    buf.append(total - arm)
    seen = arm + buf[0]          # buf[0] is _delay_cycles old once full
    state = dataclasses.replace(
        state, ee_twist_world=Twist(seen[:3], seen[3:]))
    return _compute(self, state, target)


def run_one(delay_ms, feedforward):
    global _delay_cycles
    _delay_cycles = int(round(delay_ms * 1e-3 / DT_S))
    reactive_controller.ReactiveController.compute = _delayed_compute
    try:
        config = dataclasses.replace(
            servo.CONTROL, velocity_feedforward_enabled=feedforward)
        _, _, rows, _ = disturbance_run.run(
            (SIDE,), mount_disturbance.AMPLITUDE_SCALE,
            mount_disturbance.DEFAULT_LEVEL, record_gif=False,
            controller_config=config, f_hz=F_HZ)
    finally:
        reactive_controller.ReactiveController.compute = _compute
    t = np.array([r["t"] for r in rows])
    e = 1e3 * np.linalg.norm(np.array([r[SIDE]["e_pos"] for r in rows]), axis=1)
    return float(np.sqrt(np.mean(e[t >= 0.0] ** 2)))


def main(argv):
    delays = [0, 10, 20, 40, 60, 80]
    for a in argv:
        if a.startswith("--delays="):
            delays = [float(v) for v in a.split("=", 1)[1].split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for ff in (False, True):
        for d in delays:
            results[ff, d] = run_one(d, ff)
            print(f"  feedforward={ff!s:5}  delay {d:4.0f} ms  "
                  f"position error RMS {results[ff, d]:.2f} mm")
    with open(OUT / "velocity_delay.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["delay_ms", "rms_mm_reactive", "rms_mm_feedforward"])
        for d in delays:
            w.writerow([d, f"{results[False, d]:.3f}", f"{results[True, d]:.3f}"])

    report_style.apply()
    fig, ax = plt.subplots(figsize=(report_style.FULL_WIDTH_IN * 0.62, 2.8),
                           layout="constrained")
    ax.plot(delays, [results[False, d] for d in delays], marker="o",
            label=report_style.REACTIVE_LABEL, **report_style.REACTIVE)
    ax.plot(delays, [results[True, d] for d in delays], marker="o",
            label=report_style.FEEDFORWARD_LABEL, **report_style.FEEDFORWARD)
    # Delay measured on the hardware's mount-velocity path (twist-estimate
    # lag + sensing age + joint response), per participant.
    ax.axvspan(61, 71, color="0.88", linewidth=0, zorder=0)
    ax.text(66, 0.4, "measured\non hardware", ha="center", va="bottom",
            fontsize=8, color="0.35")
    ax.set_xticks(delays)
    ax.set_xlabel("delay on the mount-velocity signal [ms]")
    ax.set_ylabel("position error RMS [mm]")
    top = max(results.values())
    ax.set_ylim(0, 1.3 * top)          # headroom for the legend
    ax.legend(loc="upper left")
    path = OUT / "velocity_delay.png"
    report_style.save(fig, path)
    print(f"  figure: {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
