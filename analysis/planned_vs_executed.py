"""Closed-loop verification of the planning layer against the real controller.

The planner is only interesting if the arm actually goes where the plan
says.  Everything in ``planning/`` can be unit-tested in the abstract --
splines, obstacle distances, least-squares convergence -- without ever
answering the question this file exists to answer: when the optimised
path is handed to the REAL ``ReactivePositionRunner`` driving the real
MuJoCo plant, how far does the measured end effector end up from the
plan, and does the whole-arm safety filter have to intervene?

Three world-frame position signals are recorded every control cycle and
they are deliberately kept apart, because the gap between each
neighbouring pair measures a different thing:

    PLANNED    ``source.planned_targets(elapsed)`` -- the optimiser's
               path, before any prefilter.  This is the thesis claim.
    DELIVERED  what the Runner actually sampled and resolved to world.
               It differs from PLANNED by exactly the lead compensation
               (``planning/plan_source.py``), which is an inverse model
               of the controller's known steady-state lag.
    MEASURED   ``cycle.controller_states.for_arm(side).ee_pose_world``,
               the controller's own forward kinematics.

MEASURED - PLANNED is therefore the number that matters, and
MEASURED - DELIVERED is the residual the lead compensation could not
remove.  ``--no-lead`` reruns the identical plan with the prefilter off
so the two are directly comparable rather than argued about.

The cylinder keep-out router is explicitly DISABLED for every run here.
That router replaces the reference position while passing the reference
twist through unchanged, so with it enabled the executed path would be
the router's and not the planner's -- ``planning/planner.py`` refuses to
plan alongside it, and this harness must make the same refusal true of
the execution side.

Usage (headless, from the repository root)::

    .venv/bin/python -m analysis.planned_vs_executed
    .venv/bin/python -m analysis.planned_vs_executed --no-lead
    .venv/bin/python -m analysis.planned_vs_executed --base-motion --arm both

SI everywhere; millimetres and degrees appear only in printed and
plotted output.
"""

import argparse
import dataclasses
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from controller.cylinder_router import CylinderKeepout
from controller.runner import ReactivePositionRunner
from controller.state import TargetFrame, Twist
from planning.planner import (
    plan_from_state,
    planned_sides,
    remaining_clearance,
)
from plotting.style import C_BASE, SIDE_COLOR
from runtime_config import CONFIG, PlanningConfig, print_effective_config
from sim import motion, world


DEFAULT_OUTPUT = Path("analysis/output/planning")

# The scripted torso disturbance for ``--base-motion``.  ``sim/motion.py``
# ships all-zero amplitudes (static base), so a non-zero scenario has to
# be stated somewhere; these are the values the existing gain sweeps and
# the golden trace already use (analysis/position_gain_sweep.py,
# tests/golden_trace.py), so the planning results stay comparable with
# the reactive baseline results rather than inventing a new scenario.
BASE_LINEAR_AMPLITUDE_M = np.array([0.18, 0.04, 0.05])
BASE_ROTATIONAL_AMPLITUDE_RAD = np.array([0.0, 0.0, -0.2])
BASE_FREQUENCY_HZ = 0.5

# The plan is deterministic, so repeating it changes nothing about the
# executed path.  The repeats exist only to turn the planning wall clock
# into a sample large enough for a mean and a p99 to mean anything --
# that is the number which decides whether this optimiser could ever run
# inside a replan loop.
DEFAULT_PLAN_SAMPLES = 5

_PLANNED_STYLE = {"linestyle": "-", "linewidth": 1.8, "alpha": 0.95}
_DELIVERED_STYLE = {"linestyle": ":", "linewidth": 1.2, "alpha": 0.9}
_MEASURED_STYLE = {"linestyle": "--", "linewidth": 1.4, "alpha": 0.95}


@dataclasses.dataclass(frozen=True, slots=True)
class ArmTrace:
    """One planned arm's per-cycle record, world frame, SI units."""

    side: str
    plan: object
    planned_position_m: np.ndarray
    delivered_position_m: np.ndarray
    measured_position_m: np.ndarray
    rotation_deviation_rad: np.ndarray
    min_clearance_constrained_m: np.ndarray
    min_clearance_all_points_m: np.ndarray
    safety_intervened: np.ndarray
    safety_stopped: np.ndarray

    def __post_init__(self):
        count = len(self.planned_position_m)
        for name in (
            "delivered_position_m",
            "measured_position_m",
            "rotation_deviation_rad",
            "min_clearance_constrained_m",
            "min_clearance_all_points_m",
            "safety_intervened",
            "safety_stopped",
        ):
            if len(getattr(self, name)) != count:
                raise ValueError(f"{name} must hold {count} samples")

    @property
    def planned_deviation_m(self):
        """Per-cycle ``|measured - planned|`` -- the headline signal."""
        return np.linalg.norm(
            self.measured_position_m - self.planned_position_m, axis=1
        )

    @property
    def delivered_deviation_m(self):
        return np.linalg.norm(
            self.measured_position_m - self.delivered_position_m, axis=1
        )


@dataclasses.dataclass(frozen=True, slots=True)
class RunLog:
    """Everything one closed-loop execution recorded."""

    sample_time_s: np.ndarray
    elapsed_time_s: np.ndarray
    arms: tuple
    lead_compensation: bool
    base_motion: bool
    plan_torso_pose_world: object

    def for_side(self, side):
        for trace in self.arms:
            if trace.side == side:
                return trace
        raise ValueError(f"{side!r} was not a planned arm")


@dataclasses.dataclass(frozen=True, slots=True)
class ArmMetrics:
    """Per-arm verdict; every distance is metres."""

    side: str
    planned_mean_m: float
    planned_rmse_m: float
    planned_peak_m: float
    delivered_mean_m: float
    delivered_rmse_m: float
    delivered_peak_m: float
    rotation_mean_rad: float
    rotation_peak_rad: float
    achieved_min_clearance_m: float
    achieved_min_clearance_all_points_m: float
    planned_min_clearance_m: float
    safety_intervention_cycles: int
    safety_stop_cycles: int
    final_goal_error_m: float
    plan_duration_s: float
    plan_iterations: int
    plan_success: bool


@dataclasses.dataclass(frozen=True, slots=True)
class RunMetrics:
    """One run's metrics, the value ``run()`` returns."""

    arms: tuple
    lead_compensation: bool
    base_motion: bool
    cycle_count: int
    simulated_seconds: float
    plan_wall_mean_s: float
    plan_wall_p99_s: float
    plan_wall_samples: int
    plan_clearance_final_torso_m: float
    config_sha256: str
    figure_path: object = None

    def for_side(self, side):
        for arm in self.arms:
            if arm.side == side:
                return arm
        raise ValueError(f"{side!r} was not a planned arm")

    @property
    def sides(self):
        return tuple(arm.side for arm in self.arms)


def disabled_cylinder_keepout(config=None):
    """Copy the configured keep-out geometry but force it off.

    Field-for-field rather than ``dataclasses.replace`` so this stays
    readable next to ``controller/runner.py:keepout_from_config``, which
    is the only other place the TOML record is translated.  Nothing here
    imports the planning layer: the planner refuses an ENABLED keep-out,
    so the disabling has to happen before the planner is ever called.
    """
    config = CONFIG.cylinder_keepout if config is None else config
    return CylinderKeepout(
        enabled=False,
        center_xy_m=(
            config.cylinder_keepout_center_x_m,
            config.cylinder_keepout_center_y_m,
        ),
        radius_m=config.cylinder_keepout_radius_m,
        z_min_m=config.cylinder_keepout_z_min_m,
        z_max_m=config.cylinder_keepout_z_max_m,
        clearance_m=config.cylinder_keepout_clearance_m,
        waypoint_tolerance_m=config.cylinder_waypoint_tolerance_m,
    )


def planning_config_for(arm=None, lead_compensation=True):
    """The configured planning setup with this run's two overrides."""
    if arm is not None and arm not in ("right", "left", "both"):
        raise ValueError("arm must be right, left, both, or None")
    config = CONFIG.planning
    if arm is not None:
        config = dataclasses.replace(config, arm=arm)
    return dataclasses.replace(
        config, lead_compensation_enabled=bool(lead_compensation)
    )


def torso_driver(base_motion):
    """Return ``(pose_at, twist_at)`` for ``configure_torso_driver``.

    ``sim/motion.py`` defaults every amplitude to zero, so the scenario
    amplitudes must be passed explicitly or the "base motion" run would
    silently be a static-base run.
    """
    if not base_motion:
        return None, None

    def pose_at(sample_time_s):
        return motion.torso_pose_at(
            sample_time_s,
            BASE_LINEAR_AMPLITUDE_M,
            BASE_FREQUENCY_HZ,
            BASE_ROTATIONAL_AMPLITUDE_RAD,
            BASE_FREQUENCY_HZ,
        )

    def twist_at(sample_time_s):
        return motion.torso_twist_at(
            sample_time_s,
            BASE_LINEAR_AMPLITUDE_M,
            BASE_FREQUENCY_HZ,
            BASE_ROTATIONAL_AMPLITUDE_RAD,
            BASE_FREQUENCY_HZ,
        )

    return pose_at, twist_at


def prepare_simulation(pose_at=None, twist_at=None):
    """Reset the shared backend and seed the configured start posture.

    ``release()`` first is what makes a rerun in the same process legal:
    the backend refuses a second ``takeover()`` while it still believes a
    Runner owns it, and ``reset()`` refuses to touch an active backend.
    """
    world.backend.release()
    world.backend.configure_torso_driver(None, None)
    world.backend.reset()
    for side in world.SIDES:
        initial = CONFIG.simulation.initial_joint_position(side)
        if initial is not None:
            world.backend.data.qpos[world.backend.qpos_adrs[side]] = (
                np.asarray(initial, dtype=float)
            )
    world.backend.data.qvel[:] = 0.0
    mujoco.mj_forward(world.backend.model, world.backend.data)
    world.backend.configure_torso_driver(pose_at, twist_at)


def read_plant_for_planning(twist_at=None):
    """Plant sample used to seed the plan, before any Runner exists.

    The torso twist is not stored in MuJoCo (the torso is a mocap body),
    so it comes from the scripted driver.  At ``t = 0`` a sine's rate is
    at its maximum, so defaulting it to zero here would quietly misreport
    the plant state even though the plan itself only reads the pose.
    """
    sample_time_s = float(world.backend.data.time)
    if twist_at is None:
        torso_twist = Twist.zero()
    else:
        torso_twist = Twist(*twist_at(sample_time_s))
    return world.backend.read_state(torso_twist)


def build_plan(plant, planning_config, keepout, plan_samples):
    """Plan once, then time repeats of the same call.

    Returns ``(outcome, wall_times_s)``.  Only the first outcome is ever
    executed; the repeats are a cost measurement, not a search.
    """
    if not isinstance(planning_config, PlanningConfig):
        raise TypeError("planning_config must be a PlanningConfig")
    samples = int(plan_samples)
    if samples < 1:
        raise ValueError("plan_samples must be at least 1")

    outcome = None
    wall_times_s = []
    for _ in range(samples):
        started_s = time.perf_counter()
        result = plan_from_state(
            plant,
            world.MOUNT_CALIBRATION,
            planning_config,
            cylinder_keepout=keepout,
        )
        wall_times_s.append(time.perf_counter() - started_s)
        if outcome is None:
            outcome = result
    return outcome, np.asarray(wall_times_s, dtype=float)


def _rotation_deviation_rad(measured_rotation, planned_rotation):
    """Angle of the rotation taking planned onto measured, radians."""
    relative = measured_rotation.T @ planned_rotation
    cosine = 0.5 * (np.trace(relative) - 1.0)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _check_world_frame(sampled_target):
    """Guard the assumption that lets PLANNED and MEASURED be compared.

    The planner emits a world-frame path, so the delivered reference can
    be read straight out of ``cycle.resolved_targets``.  If a future
    source ever delivered torso-frame targets this comparison would be
    mixing frames, so it fails loudly instead.
    """
    if sampled_target.reference_frame is not TargetFrame.WORLD:
        raise ValueError(
            "the planned source must deliver WORLD-frame targets; got "
            f"{sampled_target.reference_frame}"
        )


def execute(
    outcome, planning_config, keepout, max_seconds=None, base_motion=False
):
    """Run the real Runner over the plan and record every cycle."""
    sides = planned_sides(planning_config)
    duration_s = max(
        outcome.source.for_arm(side).duration_s for side in sides
    )
    if max_seconds is not None:
        duration_s = min(duration_s, float(max_seconds))

    runner = ReactivePositionRunner(
        world.backend,
        world.MOUNT_CALIBRATION,
        world.PIPELINE_SETUP,
        outcome.source,
        sides,
        cylinder_keepout=keepout,
    )
    start_state = runner.start()
    try:
        dt_s = start_state.nominal_dt_s
        steps = max(1, int(math.ceil(duration_s / dt_s)) + 1)
        sample_time_s = []
        elapsed_time_s = []
        records = {side: _blank_record() for side in sides}

        for _ in range(steps):
            cycle = runner.cycle()
            elapsed = cycle.target_elapsed_time_s
            planned = outcome.source.planned_targets(elapsed)
            sample_time_s.append(cycle.input_state.sample_time_s)
            elapsed_time_s.append(elapsed)
            for side in sides:
                _append_record(records[side], cycle, planned, side)

        # The plan is world-frame, so it does not follow the wearer; this
        # is what decays under base motion and what a replan trigger
        # would watch (planning/planner.py:remaining_clearance).
        final_clearance_m = remaining_clearance(
            outcome, runner.current_state, planning_config
        )
    finally:
        runner.close()

    traces = tuple(
        _build_trace(side, outcome.for_side(side), records[side])
        for side in sides
    )
    log = RunLog(
        sample_time_s=np.asarray(sample_time_s, dtype=float),
        elapsed_time_s=np.asarray(elapsed_time_s, dtype=float),
        arms=traces,
        lead_compensation=planning_config.lead_compensation_enabled,
        base_motion=bool(base_motion),
        plan_torso_pose_world=outcome.torso_pose_world,
    )
    return log, final_clearance_m


def _blank_record():
    return {
        "planned": [],
        "delivered": [],
        "measured": [],
        "rotation": [],
        "clearance": [],
        "clearance_all": [],
        "intervened": [],
        "stopped": [],
    }


def _append_record(record, cycle, planned_targets, side):
    """Store one cycle for one arm; all three signals share the tick."""
    _check_world_frame(cycle.sampled_targets.for_arm(side))
    planned_target = planned_targets.for_arm(side)
    planned_position = np.asarray(
        planned_target.pose.position_m, dtype=float
    )
    delivered_position = np.asarray(
        cycle.resolved_targets.for_arm(side).pose_world.position_m,
        dtype=float,
    )
    measured_pose = cycle.controller_states.for_arm(side).ee_pose_world
    constraints = cycle.human_safety_states.for_arm(side).constraints
    # The mount spheres sit inside the wearer envelope by construction
    # (controller/link_spheres.py), so the unfiltered minimum is a large
    # constant negative number.  Both are kept: the constrained minimum is
    # the safety filter's own quantity, the raw one is the literal record.
    free = ~constraints.mount_exempt
    status = cycle.human_safety_statuses.get(side)

    record["planned"].append(planned_position)
    record["delivered"].append(delivered_position)
    record["measured"].append(
        np.asarray(measured_pose.position_m, dtype=float)
    )
    record["rotation"].append(
        _rotation_deviation_rad(
            np.asarray(measured_pose.rotation, dtype=float),
            np.asarray(planned_target.pose.rotation, dtype=float),
        )
    )
    record["clearance"].append(
        float(np.min(constraints.signed_clearance_m[free]))
    )
    record["clearance_all"].append(
        float(np.min(constraints.signed_clearance_m))
    )
    record["intervened"].append(
        False if status is None else bool(status.human_adjusted)
    )
    record["stopped"].append(
        False if status is None else bool(status.stopped)
    )


def _build_trace(side, plan, record):
    return ArmTrace(
        side=side,
        plan=plan,
        planned_position_m=np.asarray(record["planned"], dtype=float),
        delivered_position_m=np.asarray(record["delivered"], dtype=float),
        measured_position_m=np.asarray(record["measured"], dtype=float),
        rotation_deviation_rad=np.asarray(
            record["rotation"], dtype=float
        ),
        min_clearance_constrained_m=np.asarray(
            record["clearance"], dtype=float
        ),
        min_clearance_all_points_m=np.asarray(
            record["clearance_all"], dtype=float
        ),
        safety_intervened=np.asarray(record["intervened"], dtype=bool),
        safety_stopped=np.asarray(record["stopped"], dtype=bool),
    )


def _summary(values):
    return (
        float(np.mean(values)),
        float(np.sqrt(np.mean(np.square(values)))),
        float(np.max(values)),
    )


def arm_metrics(trace):
    """Reduce one arm's trace to the numbers the thesis quotes."""
    planned_mean, planned_rmse, planned_peak = _summary(
        trace.planned_deviation_m
    )
    delivered_mean, delivered_rmse, delivered_peak = _summary(
        trace.delivered_deviation_m
    )
    goal_position = np.asarray(
        trace.plan.goal_pose_world.position_m, dtype=float
    )
    return ArmMetrics(
        side=trace.side,
        planned_mean_m=planned_mean,
        planned_rmse_m=planned_rmse,
        planned_peak_m=planned_peak,
        delivered_mean_m=delivered_mean,
        delivered_rmse_m=delivered_rmse,
        delivered_peak_m=delivered_peak,
        rotation_mean_rad=float(np.mean(trace.rotation_deviation_rad)),
        rotation_peak_rad=float(np.max(trace.rotation_deviation_rad)),
        achieved_min_clearance_m=float(
            np.min(trace.min_clearance_constrained_m)
        ),
        achieved_min_clearance_all_points_m=float(
            np.min(trace.min_clearance_all_points_m)
        ),
        planned_min_clearance_m=float(
            trace.plan.result.final_min_clearance_m
        ),
        safety_intervention_cycles=int(np.count_nonzero(
            trace.safety_intervened
        )),
        safety_stop_cycles=int(np.count_nonzero(trace.safety_stopped)),
        final_goal_error_m=float(np.linalg.norm(
            trace.measured_position_m[-1] - goal_position
        )),
        plan_duration_s=float(trace.plan.result.duration_s),
        plan_iterations=int(trace.plan.result.iterations),
        plan_success=bool(trace.plan.result.success),
    )


def run_metrics(log, plan_wall_times_s, plan_clearance_final_torso_m):
    return RunMetrics(
        arms=tuple(arm_metrics(trace) for trace in log.arms),
        lead_compensation=log.lead_compensation,
        base_motion=log.base_motion,
        cycle_count=len(log.sample_time_s),
        simulated_seconds=float(
            log.sample_time_s[-1] - log.sample_time_s[0]
        ),
        plan_wall_mean_s=float(np.mean(plan_wall_times_s)),
        plan_wall_p99_s=float(np.percentile(plan_wall_times_s, 99.0)),
        plan_wall_samples=int(len(plan_wall_times_s)),
        plan_clearance_final_torso_m=float(plan_clearance_final_torso_m),
        config_sha256=CONFIG.source_sha256,
    )


def scenario_name(lead_compensation, base_motion):
    return "{}_{}".format(
        "lead" if lead_compensation else "nolead",
        "basemotion" if base_motion else "static",
    )


def _draw_paths(axis, log, first_index, second_index, labels):
    """One 2-D projection of the three paths for every planned arm."""
    for trace in log.arms:
        color = SIDE_COLOR[trace.side]
        for positions, style, role in (
            (trace.planned_position_m, _PLANNED_STYLE, "planned"),
            (trace.delivered_position_m, _DELIVERED_STYLE, "delivered"),
            (trace.measured_position_m, _MEASURED_STYLE, "measured"),
        ):
            axis.plot(
                positions[:, first_index],
                positions[:, second_index],
                color=color,
                label=f"{trace.side} {role}",
                **style,
            )
        goal = np.asarray(
            trace.plan.goal_pose_world.position_m, dtype=float
        )
        axis.plot(
            trace.planned_position_m[0, first_index],
            trace.planned_position_m[0, second_index],
            marker="o", color=color, markersize=6, linestyle="none",
            label=f"{trace.side} start",
        )
        axis.plot(
            goal[first_index], goal[second_index],
            marker="*", color=color, markersize=12, linestyle="none",
            label=f"{trace.side} goal",
        )
    axis.set_xlabel(f"world {labels[0]} [m]")
    axis.set_ylabel(f"world {labels[1]} [m]")
    axis.grid(alpha=0.2)
    axis.set_aspect("equal", adjustable="datalim")


def _draw_wearer_footprint(axis, log):
    """Top-down wearer envelope at PLAN TIME, inflated by clearance."""
    human = CONFIG.human_safety
    torso = log.plan_torso_pose_world
    center = np.asarray(torso.position_m, dtype=float) + (
        np.asarray(torso.rotation, dtype=float)
        @ np.array([
            human.center_xy_torso_m[0], human.center_xy_torso_m[1], 0.0
        ])
    )
    radius = human.radius_m + human.clearance_m
    angle = np.linspace(0.0, 2.0 * np.pi, 181)
    axis.plot(
        center[0] + radius * np.cos(angle),
        center[1] + radius * np.sin(angle),
        color=C_BASE, linewidth=1.0, linestyle="-",
        label="wearer envelope (plan time)",
    )


def _draw_deviation(axis, log):
    for trace in log.arms:
        color = SIDE_COLOR[trace.side]
        axis.plot(
            log.elapsed_time_s, trace.planned_deviation_m * 1000.0,
            color=color, label=f"{trace.side} vs planned", **_PLANNED_STYLE,
        )
        axis.plot(
            log.elapsed_time_s, trace.delivered_deviation_m * 1000.0,
            color=color, label=f"{trace.side} vs delivered",
            **_DELIVERED_STYLE,
        )
    axis.set_xlabel("plan time [s]")
    axis.set_ylabel("position deviation [mm]")
    axis.set_title("Executed minus reference")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)


def _draw_clearance(axis, log):
    human = CONFIG.human_safety
    for trace in log.arms:
        color = SIDE_COLOR[trace.side]
        clearance_mm = trace.min_clearance_constrained_m * 1000.0
        axis.plot(
            log.elapsed_time_s, clearance_mm, color=color,
            label=f"{trace.side} achieved (whole arm)", **_MEASURED_STYLE,
        )
        axis.axhline(
            trace.plan.result.final_min_clearance_m * 1000.0,
            color=color, linewidth=1.0, alpha=0.5,
            label=f"{trace.side} planned (end effector)",
        )
        axis.fill_between(
            log.elapsed_time_s, 0.0, 1.0,
            where=trace.safety_intervened,
            transform=axis.get_xaxis_transform(),
            color=color, alpha=0.07, linewidth=0.0,
            label=f"{trace.side} safety filter active",
        )
    axis.axhline(
        human.control_margin_m * 1000.0, color=C_BASE, linewidth=1.0,
        linestyle="--", label="control margin",
    )
    axis.axhline(
        human.activation_distance_m * 1000.0, color=C_BASE, linewidth=1.0,
        linestyle=":", label="activation distance",
    )
    axis.set_xlabel("plan time [s]")
    axis.set_ylabel("signed clearance [mm]")
    axis.set_title("Human clearance")
    axis.legend(fontsize=7)
    axis.grid(alpha=0.2)


def make_figure(log, metrics, path):
    """Write the planned/delivered/measured paths plus clearance."""
    figure, axes = plt.subplots(
        2, 2, figsize=(13, 9.5), layout="constrained"
    )
    top_view, side_view = axes[0]
    deviation_axis, clearance_axis = axes[1]

    _draw_wearer_footprint(top_view, log)
    _draw_paths(top_view, log, 0, 1, ("x", "y"))
    top_view.set_title("Cartesian path, top view")
    top_view.legend(fontsize=7, loc="best")

    _draw_paths(side_view, log, 0, 2, ("x", "z"))
    side_view.set_title("Cartesian path, side view")

    _draw_deviation(deviation_axis, log)
    _draw_clearance(clearance_axis, log)

    figure.suptitle(
        "Planned vs executed  ·  lead={}  base motion={}  ·  config {}"
        .format(
            "on" if metrics.lead_compensation else "off",
            "on" if metrics.base_motion else "off",
            metrics.config_sha256[:12],
        )
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def print_metrics(metrics):
    """Human-facing report; millimetres and degrees only in here."""
    print(
        f"lead compensation={'on' if metrics.lead_compensation else 'off'}  "
        f"base motion={'on' if metrics.base_motion else 'off'}  "
        f"cycles={metrics.cycle_count}  "
        f"simulated={metrics.simulated_seconds:.2f} s"
    )
    print(
        f"planning wall clock: mean={metrics.plan_wall_mean_s * 1000:.1f} ms "
        f"p99={metrics.plan_wall_p99_s * 1000:.1f} ms "
        f"over {metrics.plan_wall_samples} plan(s)"
    )
    print(
        "plan clearance under the FINAL torso pose: "
        f"{metrics.plan_clearance_final_torso_m * 1000:.1f} mm"
    )
    for arm in metrics.arms:
        print(f"  [{arm.side}]")
        print(
            f"    measured vs planned    "
            f"mean={arm.planned_mean_m * 1000:8.2f} mm  "
            f"RMSE={arm.planned_rmse_m * 1000:8.2f} mm  "
            f"peak={arm.planned_peak_m * 1000:8.2f} mm"
        )
        print(
            f"    measured vs delivered  "
            f"mean={arm.delivered_mean_m * 1000:8.2f} mm  "
            f"RMSE={arm.delivered_rmse_m * 1000:8.2f} mm  "
            f"peak={arm.delivered_peak_m * 1000:8.2f} mm"
        )
        print(
            f"    orientation deviation  "
            f"mean={np.degrees(arm.rotation_mean_rad):8.2f} deg  "
            f"peak={np.degrees(arm.rotation_peak_rad):8.2f} deg"
        )
        print(
            f"    clearance  planned (end effector)="
            f"{arm.planned_min_clearance_m * 1000:.1f} mm  "
            f"achieved (whole arm)="
            f"{arm.achieved_min_clearance_m * 1000:.1f} mm  "
            f"achieved (incl. mount points)="
            f"{arm.achieved_min_clearance_all_points_m * 1000:.1f} mm"
        )
        print(
            f"    safety filter  interventions="
            f"{arm.safety_intervention_cycles}  "
            f"stops={arm.safety_stop_cycles}  "
            f"final error to goal="
            f"{arm.final_goal_error_m * 1000:.1f} mm"
        )
        print(
            f"    plan  duration={arm.plan_duration_s:.2f} s  "
            f"iterations={arm.plan_iterations}  "
            f"success={arm.plan_success}"
        )


def print_comparison(with_lead, without_lead):
    """The mm table that makes the prefilter's effect arguable-free."""
    if with_lead.sides != without_lead.sides:
        raise ValueError("the two runs must plan the same arms")
    print("")
    print("lead compensation comparison, measured vs planned [mm]")
    header = (
        f"{'arm':6s} {'mean on':>9s} {'mean off':>9s} "
        f"{'RMSE on':>9s} {'RMSE off':>9s} "
        f"{'peak on':>9s} {'peak off':>9s}"
    )
    print(header)
    print("-" * len(header))
    for side in with_lead.sides:
        on = with_lead.for_side(side)
        off = without_lead.for_side(side)
        print(
            f"{side:6s} "
            f"{on.planned_mean_m * 1000:9.2f} "
            f"{off.planned_mean_m * 1000:9.2f} "
            f"{on.planned_rmse_m * 1000:9.2f} "
            f"{off.planned_rmse_m * 1000:9.2f} "
            f"{on.planned_peak_m * 1000:9.2f} "
            f"{off.planned_peak_m * 1000:9.2f}"
        )


def run(
    arm=None,
    lead_compensation=True,
    base_motion=False,
    max_seconds=None,
    plan_samples=DEFAULT_PLAN_SAMPLES,
    output_dir=DEFAULT_OUTPUT,
    make_figures=True,
):
    """Plan, execute against the real Runner, and reduce to metrics.

    Returns a ``RunMetrics``.  The shared MuJoCo backend is reset on the
    way in and left driver-free on the way out, so calling this twice in
    one process (which ``--no-lead`` does) is legal.
    """
    if max_seconds is not None:
        max_seconds = float(max_seconds)
        if not np.isfinite(max_seconds) or max_seconds <= 0.0:
            raise ValueError("max_seconds must be finite and positive")

    planning_config = planning_config_for(arm, lead_compensation)
    keepout = disabled_cylinder_keepout()
    pose_at, twist_at = torso_driver(bool(base_motion))

    prepare_simulation(pose_at, twist_at)
    try:
        plant = read_plant_for_planning(twist_at)
        outcome, plan_wall_times_s = build_plan(
            plant, planning_config, keepout, plan_samples
        )
        log, plan_clearance_m = execute(
            outcome,
            planning_config,
            keepout,
            max_seconds,
            base_motion=bool(base_motion),
        )
    finally:
        world.backend.configure_torso_driver(None, None)

    metrics = run_metrics(log, plan_wall_times_s, plan_clearance_m)
    if not make_figures:
        return metrics

    figure_path = make_figure(
        log,
        metrics,
        Path(output_dir)
        / "planned_vs_executed_{}.png".format(
            scenario_name(metrics.lead_compensation, metrics.base_motion)
        ),
    )
    return dataclasses.replace(metrics, figure_path=figure_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=("right", "left", "both"), default=None,
        help="override [planning].arm for this run",
    )
    parser.add_argument(
        "--no-lead", action="store_true",
        help="rerun with lead compensation off and print the comparison",
    )
    parser.add_argument(
        "--base-motion", action="store_true",
        help="enable the scripted torso disturbance (the thesis scenario)",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=None,
        help="cap the simulated duration below the plan duration",
    )
    parser.add_argument(
        "--plan-samples", type=int, default=DEFAULT_PLAN_SAMPLES,
        help="repeats used to sample the planning wall clock",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    print_effective_config(CONFIG)
    with_lead = run(
        arm=args.arm,
        lead_compensation=True,
        base_motion=args.base_motion,
        max_seconds=args.max_seconds,
        plan_samples=args.plan_samples,
        output_dir=args.output,
    )
    print_metrics(with_lead)
    print(f"figure: {with_lead.figure_path}")

    if not args.no_lead:
        return with_lead, None

    print("")
    without_lead = run(
        arm=args.arm,
        lead_compensation=False,
        base_motion=args.base_motion,
        max_seconds=args.max_seconds,
        plan_samples=args.plan_samples,
        output_dir=args.output,
    )
    print_metrics(without_lead)
    print(f"figure: {without_lead.figure_path}")
    print_comparison(with_lead, without_lead)
    return with_lead, without_lead


if __name__ == "__main__":
    main()
