"""What a late mount-velocity signal costs: the README disturbance at f = 1.8 Hz,
right arm, with the mount's contribution to the measured end-effector twist
delayed by n control cycles before the controller sees it.

This mirrors the hardware: the controller's velocity terms (the K_d term and,
when enabled, the feedforward) use the world-frame tool twist, whose mount part
comes from an estimator and can arrive late, while the arm's own part (J q_dot)
is current. In simulation the mount part is exact unless delayed here.

Three controllers: K_d term only (as on hardware), K_d term + feedforward, and,
as a reference, no velocity term at all (K_d = 0, no feedforward).

usage: python tools/velocity_delay.py [--delays=0,10,20,40,60,80] [--plot-only]
writes analysis/output/disturbance/velocity_delay.png, .pdf and .csv
"""

import collections
import contextlib
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
HARDWARE_DELAY_MS = (61, 71)   # twist-estimate lag + sensing age + joint response


@contextlib.contextmanager
def delayed_mount_twist(delay_ms):
    """Inside the block, every ReactiveController sees the mount part of the
    measured EE twist delay_ms late; the arm's own part stays current."""
    cycles = int(round(delay_ms * 1e-3 / DT_S))
    original = reactive_controller.ReactiveController.compute
    history = {}

    def compute(self, state, target):
        arm = state.jacobian_world @ state.joints.velocity_rad_s
        total = np.concatenate([state.ee_twist_world.linear_m_s,
                                state.ee_twist_world.angular_rad_s])
        buf = history.setdefault(id(self), collections.deque(maxlen=cycles + 1))
        buf.append(total - arm)
        seen = arm + buf[0]      # buf[0] is `cycles` old once the buffer is full
        state = dataclasses.replace(state, ee_twist_world=Twist(seen[:3], seen[3:]))
        return original(self, state, target)

    reactive_controller.ReactiveController.compute = compute
    try:
        yield
    finally:
        reactive_controller.ReactiveController.compute = original


def position_rms_mm(config, delay_ms=0.0):
    with delayed_mount_twist(delay_ms):
        _, _, rows, _ = disturbance_run.run(
            (SIDE,), mount_disturbance.AMPLITUDE_SCALE,
            mount_disturbance.DEFAULT_LEVEL, record_gif=False,
            controller_config=config, f_hz=F_HZ)
    t = np.array([r["t"] for r in rows])
    e = 1e3 * np.linalg.norm(np.array([r[SIDE]["e_pos"] for r in rows]), axis=1)
    return float(np.sqrt(np.mean(e[t >= 0.0] ** 2)))


def measure(delays):
    no_velocity = position_rms_mm(dataclasses.replace(
        servo.CONTROL, velocity_enabled=False, velocity_feedforward_enabled=False))
    print(f"  no velocity term (K_d = 0)        position error RMS {no_velocity:.2f} mm")
    rows = []
    for d in delays:
        kd, ff = (position_rms_mm(dataclasses.replace(
            servo.CONTROL, velocity_feedforward_enabled=on), d) for on in (False, True))
        print(f"  delay {d:4.0f} ms  position error RMS: K_d term {kd:.2f} mm, "
              f"+ feedforward {ff:.2f} mm")
        rows.append((d, kd, ff, no_velocity))
    with open(OUT / "velocity_delay.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["delay_ms", "rms_mm_kd", "rms_mm_kd_feedforward",
                    "rms_mm_no_velocity_term"])
        w.writerows([[d, f"{a:.3f}", f"{b:.3f}", f"{c:.3f}"] for d, a, b, c in rows])


def plot():
    with open(OUT / "velocity_delay.csv") as fh:
        rows = [[float(v) for v in r] for r in list(csv.reader(fh))[1:]]
    d, kd, ff, none = (np.array(c) for c in zip(*rows))
    report_style.apply()
    fig, ax = plt.subplots(figsize=(report_style.FULL_WIDTH_IN, 2.6),
                           layout="constrained")
    ax.axvspan(*HARDWARE_DELAY_MS, color="0.9", linewidth=0, zorder=0)
    ax.text(sum(HARDWARE_DELAY_MS) / 2, 0.3, "measured\non hardware",
            ha="center", va="bottom", fontsize=7.5, color="0.35")
    ax.axhline(none[0], color="0.35", linestyle=":", linewidth=1.2,
               label="no velocity term ($K_d$ = 0)")
    ax.plot(d, kd, marker="o", label="$K_d$ term only, as on hardware",
            **report_style.REACTIVE)
    ax.plot(d, ff, marker="o", label="$K_d$ term + mount-velocity feedforward",
            **report_style.FEEDFORWARD)
    ax.set_xticks(d)
    ax.set_xlim(-3, d.max() + 3)
    ax.set_xlabel("delay on the mount-velocity signal [ms]")
    ax.set_ylabel("position error RMS [mm]")
    ax.set_ylim(0, 1.2 * max(kd.max(), ff.max(), none[0]))   # room for the legend
    ax.legend(loc="upper left")
    path = OUT / "velocity_delay.png"
    report_style.save(fig, path)
    print(f"  figure: {path}")


def main(argv):
    """--plot-only redraws the figure from the saved CSV."""
    delays = [0, 10, 20, 40, 60, 80]
    for a in argv:
        if a.startswith("--delays="):
            delays = [float(v) for v in a.split("=", 1)[1].split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    if "--plot-only" not in argv:
        measure(delays)
    plot()


if __name__ == "__main__":
    main(sys.argv[1:])
