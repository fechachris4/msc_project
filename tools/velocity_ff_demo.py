"""Trajectory-velocity feedforward vs baseline: one arm tracks a constant
Cartesian-velocity WORLD target with a static torso, run twice with
controller.reactive_pose.velocity_feedforward_enabled off (baseline PD-on-
error) and on (target.twist_world added to the commanded task twist).

Static torso means the mount-disturbance-cancellation source of feedforward
(see tools/feedforward_compare.py) is exactly zero here — this isolates the
other source: a target that itself carries nonzero path velocity.

Classic PD-only steady-state tracking lag for a constant-velocity target is
e_ss = v / Kp along the moving axis (Kd/DLS/null-space terms don't remove
it, since d(e)/dt = 0 at steady state only if e itself is nonzero and
constant). Feedforward removes that term algebraically: with the desired
velocity added straight into the commanded task twist, Kp only has to
correct residual/model error, not the whole reference velocity.

usage: python tools/velocity_ff_demo.py [right|left] [--speed=V] [--duration=S]
"""

import dataclasses
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller import desired_pos, frames, reactive_controller, servo  # noqa: E402
from controller.runner import ReactivePositionRunner  # noqa: E402
from controller.state import FramedTarget, Pose, TargetFrame, Twist  # noqa: E402
from controller.trajectory import (  # noqa: E402
    IndependentArmTargetSource,
    StaticTargetSource,
)
from sim import world  # noqa: E402

OUT = Path("analysis/output/velocity_ff")
SETTLE_TIMEOUT_S = 15.0
SETTLE_POS_TOL_M = 0.002
SETTLE_ROT_TOL_RAD = np.deg2rad(0.5)
SETTLE_DWELL_S = 0.5
DEFAULT_SPEED_M_S = 0.10
DEFAULT_DURATION_S = 3.0
STEADY_STATE_FRACTION = 0.5  # ignore the initial transient when averaging


@dataclasses.dataclass(frozen=True, slots=True)
class ConstantVelocityTargetSource:
    """A WORLD-frame target moving at a fixed Cartesian velocity from t0.

    ``t0`` is the runner's elapsed time (seconds since ``runner.start()``)
    at which the ramp should begin; ``sample`` is still called with the
    runner's absolute elapsed time, exactly like every other TargetSource.
    """

    start_pose: Pose
    linear_velocity_m_s: np.ndarray
    t0: float

    @property
    def reference_frame(self):
        return TargetFrame.WORLD

    def sample(self, elapsed_time_s):
        ramp_time = max(0.0, elapsed_time_s - self.t0)
        position = (
            self.start_pose.position_m
            + self.linear_velocity_m_s * ramp_time
        )
        return FramedTarget(
            TargetFrame.WORLD,
            Pose(position, self.start_pose.rotation),
            Twist(self.linear_velocity_m_s, np.zeros(3)),
        )


def _within_tolerance(runner, targets, side):
    plant = runner.current_state
    resolved = frames.resolve_targets_world(
        plant, world.MOUNT_CALIBRATION, targets)
    states = frames.controller_states(plant, world.MOUNT_CALIBRATION)
    e_pos, e_rot = reactive_controller.pose_error(
        states.for_arm(side), resolved.for_arm(side))
    return (
        np.linalg.norm(e_pos) <= SETTLE_POS_TOL_M
        and np.linalg.norm(e_rot) <= SETTLE_ROT_TOL_RAD
    )


def _settle(runner, targets, side):
    t_start = runner.current_state.sample_time_s
    dwell_start = None
    while runner.current_state.sample_time_s - t_start < SETTLE_TIMEOUT_S:
        t = runner.current_state.sample_time_s
        if _within_tolerance(runner, targets, side):
            dwell_start = t if dwell_start is None else dwell_start
            if t - dwell_start >= SETTLE_DWELL_S:
                return True
        else:
            dwell_start = None
        runner.cycle()
    return False


def run(side, speed, duration_s, enabled):
    world.backend.release()
    world.backend.configure_torso_driver(None, None)
    world.backend.reset()
    static_targets = desired_pos.apply()

    config = dataclasses.replace(
        servo.CONTROL, velocity_feedforward_enabled=enabled)
    runner = ReactivePositionRunner(
        world.backend, world.MOUNT_CALIBRATION, world.PIPELINE_SETUP,
        static_targets, (side,), controller_config=config)
    runner.start()
    try:
        settled = _settle(runner, static_targets, side)

        plant = runner.current_state
        states = frames.controller_states(plant, world.MOUNT_CALIBRATION)
        start_pose = states.for_arm(side).ee_pose_world
        t0 = runner.target_elapsed_time_s
        moving = ConstantVelocityTargetSource(
            start_pose, np.array([speed, 0.0, 0.0]), t0)
        other_side = "left" if side == "right" else "right"
        source = IndependentArmTargetSource(**{
            side: moving,
            other_side: StaticTargetSource(static_targets.for_arm(other_side)),
        })

        n_steps = int(np.ceil(
            duration_s / runner.current_state.nominal_dt_s))
        rows = []
        for _ in range(n_steps):
            cycle = runner.cycle(source_targets=source)
            trace = cycle.traces[side]
            rows.append(dict(
                t=cycle.target_elapsed_time_s - t0,
                e_pos=trace.e_pos,
            ))
    finally:
        runner.close()
    return settled, rows


def compare(side, speed, duration_s):
    print(f"velocity feedforward demo: {side} arm, "
          f"target speed {speed*1e3:.0f} mm/s, static torso")
    results = {}
    for name, enabled in (("baseline", False), ("velocity_feedforward", True)):
        settled, rows = run(side, speed, duration_s, enabled)
        results[name] = rows
        print(f"  [{name}] settled before ramp: {settled}")

    kp = servo.CONTROL.kp_position_s_inv
    predicted_lag_mm = 1e3 * speed / kp
    print(f"  analytic PD-only steady-state lag e_ss = v/Kp = "
          f"{predicted_lag_mm:.1f} mm (Kp={kp:g} s^-1)")

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    summary = {}
    for name, color in (("baseline", "0.35"), ("velocity_feedforward", "black")):
        rows = results[name]
        t = np.array([r["t"] for r in rows])
        e = 1e3 * np.linalg.norm(np.array([r["e_pos"] for r in rows]), axis=1)
        ax.plot(t, e, color=color, linewidth=1.5, label=name)
        steady = t >= STEADY_STATE_FRACTION * duration_s
        summary[name] = dict(rms=np.sqrt(np.mean(e[steady] ** 2)),
                              final=float(e[-1]))
    ax.axhline(predicted_lag_mm, color="0.7", linestyle=":", linewidth=1.0,
               label="predicted PD-only lag (v/Kp)")
    ax.set_xlabel("time since ramp onset [s]")
    ax.set_ylabel(f"{side} EE position error [mm]")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=False)
    change = 100.0 * (
        1.0 - summary["velocity_feedforward"]["rms"] / summary["baseline"]["rms"]
    ) if summary["baseline"]["rms"] else 0.0
    ax.set_title(
        f"{speed*1e3:.0f} mm/s constant-velocity target: steady-state RMS "
        f"{summary['baseline']['rms']:.1f} -> "
        f"{summary['velocity_feedforward']['rms']:.1f} mm ({change:+.0f}%)"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"velocity_ff_compare_{side}_{speed*1e3:.0f}mms.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  baseline  steady-state RMS {summary['baseline']['rms']:.1f} mm "
          f"(final {summary['baseline']['final']:.1f} mm)")
    print(f"  ff        steady-state RMS "
          f"{summary['velocity_feedforward']['rms']:.1f} mm "
          f"(final {summary['velocity_feedforward']['final']:.1f} mm)")
    print(f"  figure: {path}")


def main(argv):
    side = argv[0] if argv and not argv[0].startswith("--") else "right"
    speed = DEFAULT_SPEED_M_S
    duration = DEFAULT_DURATION_S
    for a in list(argv):
        if a.startswith("--speed="):
            speed = float(a.split("=", 1)[1])
        elif a.startswith("--duration="):
            duration = float(a.split("=", 1)[1])
    compare(side, speed, duration)


if __name__ == "__main__":
    main(sys.argv[1:])
