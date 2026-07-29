"""Collision-aware placement of minimum-jerk spline knots.

The adapted idea, taken from HumanSL's GPMP2 planner and not its code:

- a smoothness prior over the trajectory states,
- a hinged obstacle cost that is zero beyond a margin,
- and the cost evaluated at DENSE points between the knots, not only at
  the knots themselves (``TrajectoryOptimization.cpp:83-96``), because a
  sparse plan is executed densely.

What is deliberately different.  The trajectory representation is this
project's existing globally-solved minimum-jerk C2 spline, so the
optimiser never has to model smoothness, timing, or Cartesian rate
limits.  Because that spline's sampled position is exactly LINEAR in its
knot positions (its coefficients solve a linear system whose right-hand
side is linear in the knots), the whole problem is a small nonlinear
least-squares whose only nonlinearity is the obstacle hinge, with an
exact analytic Jacobian.  ``scipy.optimize.least_squares`` solves it with
the same Gauss-Newton/Levenberg-Marquardt family GTSAM uses.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from controller.state import Pose, TargetFrame
from controller.trajectory import (
    CartesianWaypointTrajectory,
    CartesianWaypoint,
    TrajectoryLimits,
    _rotation_from_vector,
    _rotation_vector,
    timed_waypoint_trajectory,
)
from planning.obstacles import ObstacleSet

# Slack allowed between the requested margin and the margin measured on
# the delivered trajectory before a re-solve on the real timing.
_RETIME_TOLERANCE_M = 1e-4

# Bounded passes over the coupled knot-geometry / delivered-timing fixed
# point. Each pass is a full solve, so this is a cost ceiling as well as a
# termination guarantee.
_MAX_RETIME_PASSES = 3


@dataclass(frozen=True, slots=True)
class OptimizerParams:
    """Weights and sampling for the knot-placement problem."""

    waypoint_count: int = 3
    dense_samples: int = 60
    clearance_margin_m: float = 0.05
    smoothness_weight: float = 1.0
    obstacle_weight: float = 40.0
    max_iterations: int = 200
    tool_radius_m: float = 0.0
    # Without a deviation prior the cost has no finite basin: the obstacle
    # hinge can drive knots arbitrarily far (a 0.40 m reach was observed
    # becoming a 5.27 m path with a knot 1.44 m behind the wearer). The
    # prior gives the problem a minimum; the reach allowance bounds it
    # outright.
    deviation_weight: float = 0.5
    reach_allowance_m: float = 0.45

    def __post_init__(self):
        if int(self.waypoint_count) < 1:
            raise ValueError("waypoint_count must be at least 1")
        if int(self.dense_samples) < 2:
            raise ValueError("dense_samples must be at least 2")
        for name in (
            "clearance_margin_m",
            "smoothness_weight",
            "obstacle_weight",
            "tool_radius_m",
            "deviation_weight",
            "reach_allowance_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "waypoint_count", int(self.waypoint_count))
        object.__setattr__(self, "dense_samples", int(self.dense_samples))
        object.__setattr__(
            self, "max_iterations", int(self.max_iterations)
        )


@dataclass(frozen=True, slots=True)
class PlanResult:
    """An optimised world-frame path plus the evidence it was checked."""

    trajectory: CartesianWaypointTrajectory
    knots_world_m: np.ndarray
    knots_torso_m: np.ndarray
    initial_min_clearance_m: float
    final_min_clearance_m: float
    iterations: int
    success: bool
    collision_free: bool
    margin_met: bool
    message: str

    @property
    def duration_s(self):
        return self.trajectory.duration_s


def _pose_to_torso(pose_world, torso_pose_world):
    rotation = torso_pose_world.rotation.T
    position = rotation @ (
        pose_world.position_m - torso_pose_world.position_m
    )
    return Pose(position, rotation @ pose_world.rotation)


def _position_to_world(position_torso, torso_pose_world):
    return (
        torso_pose_world.rotation @ np.asarray(position_torso, dtype=float)
        + torso_pose_world.position_m
    )


def _uniform_times(count, duration_s):
    return np.linspace(0.0, float(duration_s), count)


def _spline_basis(times_s, sample_times_s):
    """Matrix ``B`` with ``dense_positions = B @ knot_positions``.

    Built by evaluating the real trajectory class once per knot with unit
    knot positions, so the basis is by construction the same spline the
    controller will later be given.
    """
    knot_count = len(times_s)
    identity_rotation = np.eye(3)
    basis = np.empty((len(sample_times_s), knot_count))
    for index in range(knot_count):
        positions = np.zeros((knot_count, 3))
        positions[index, 0] = 1.0
        trajectory = CartesianWaypointTrajectory(
            TargetFrame.WORLD,
            tuple(
                CartesianWaypoint(
                    float(time_s), Pose(position, identity_rotation)
                )
                for time_s, position in zip(times_s, positions)
            ),
        )
        for row, sample_time in enumerate(sample_times_s):
            basis[row, index] = trajectory.sample(
                float(sample_time)
            ).pose.position_m[0]
    return basis


def _second_difference_matrix(count):
    if count < 3:
        return np.zeros((0, count))
    matrix = np.zeros((count - 2, count))
    for row in range(count - 2):
        matrix[row, row] = 1.0
        matrix[row, row + 1] = -2.0
        matrix[row, row + 2] = 1.0
    return matrix


def _minimum_clearance(points_torso_m, obstacles, tool_radius_m):
    distance, _ = obstacles.distance_and_gradient(points_torso_m)
    return float(np.min(distance) - tool_radius_m)


def trajectory_min_clearance(
    trajectory, torso_pose_world, obstacles, tool_radius_m, samples=120
):
    """Clearance of the trajectory that will ACTUALLY be delivered.

    The optimiser places knots on a uniformly timed spline, but the
    delivered trajectory may be re-timed from the Cartesian limits, and a
    minimum-jerk spline's shape depends on its segment durations.  The
    reported clearance is therefore always measured on the final object,
    never on the optimiser's internal prediction.
    """
    times = np.linspace(0.0, trajectory.duration_s, int(samples))
    points_world = np.stack([
        trajectory.sample(float(time_s)).pose.position_m for time_s in times
    ])
    rotation = torso_pose_world.rotation
    points_torso = (points_world - torso_pose_world.position_m) @ rotation
    return _minimum_clearance(points_torso, obstacles, tool_radius_m)


def optimize_path(
    start_pose_world,
    goal_pose_world,
    torso_pose_world,
    obstacles,
    limits,
    params=OptimizerParams(),
    duration_s=None,
):
    """Place spline knots so the executed path keeps clearance.

    Returns a :class:`PlanResult` whose trajectory is expressed in the
    WORLD frame, because that is the frame the task and the controller
    seam both require.  Optimisation happens in the TORSO frame, where
    the human envelope is static.
    """
    for name, pose in (
        ("start_pose_world", start_pose_world),
        ("goal_pose_world", goal_pose_world),
        ("torso_pose_world", torso_pose_world),
    ):
        if not isinstance(pose, Pose):
            raise TypeError(f"{name} must be a Pose")
    if not isinstance(obstacles, ObstacleSet):
        raise TypeError("obstacles must be an ObstacleSet")
    if not isinstance(limits, TrajectoryLimits):
        raise TypeError("limits must be TrajectoryLimits")
    if not isinstance(params, OptimizerParams):
        raise TypeError("params must be OptimizerParams")

    start_torso = _pose_to_torso(start_pose_world, torso_pose_world)
    goal_torso = _pose_to_torso(goal_pose_world, torso_pose_world)

    knot_count = params.waypoint_count + 2
    straight = np.linspace(
        start_torso.position_m, goal_torso.position_m, knot_count
    )
    free_index = np.arange(1, knot_count - 1)
    fixed_index = np.array([0, knot_count - 1])

    # Geometry is optimised on a nominal parameterisation; real timing is
    # derived afterwards from the Cartesian limits.  A minimum-jerk
    # spline's SHAPE depends on its segment durations, so if the derived
    # timing is not the timing the knots were placed against, the plan is
    # re-solved once on the real times and the reported clearance is
    # always measured on the delivered trajectory.
    nominal_duration_s = 1.0 if duration_s is None else float(duration_s)
    times = _uniform_times(knot_count, nominal_duration_s)

    smoothness = _second_difference_matrix(knot_count)
    smoothness_free = smoothness[:, free_index]
    smoothness_fixed = smoothness[:, fixed_index] @ straight[fixed_index]

    basis_free = None
    fixed_contribution = None

    def set_times(knot_times_s):
        """Rebuild the (linear) spline basis for one knot time vector."""
        nonlocal basis_free, fixed_contribution
        span = float(knot_times_s[-1])
        sample_times = np.linspace(0.0, span, params.dense_samples)
        basis = _spline_basis(knot_times_s, sample_times)
        basis_free = basis[:, free_index]
        fixed_contribution = basis[:, fixed_index] @ straight[fixed_index]

    def dense_points(free_knots):
        return basis_free @ free_knots + fixed_contribution

    def residuals(flat):
        free_knots = flat.reshape(-1, 3)
        points = dense_points(free_knots)
        distance, _ = obstacles.distance_and_gradient(points)
        clearance = distance - params.tool_radius_m
        hinge = np.maximum(0.0, params.clearance_margin_m - clearance)
        smooth = (
            smoothness_free @ free_knots + smoothness_fixed
        ).reshape(-1)
        deviation = (free_knots - straight[free_index]).reshape(-1)
        return np.concatenate([
            params.smoothness_weight * smooth,
            params.deviation_weight * deviation,
            params.obstacle_weight * hinge,
        ])

    def jacobian(flat):
        free_knots = flat.reshape(-1, 3)
        points = dense_points(free_knots)
        distance, gradient = obstacles.distance_and_gradient(points)
        clearance = distance - params.tool_radius_m
        active = clearance < params.clearance_margin_m

        variable_count = free_knots.size
        smooth_rows = smoothness_free.shape[0] * 3
        deviation_rows = variable_count
        obstacle_start = smooth_rows + deviation_rows
        block = np.zeros((obstacle_start + len(points), variable_count))

        # Smoothness block: linear, one identical copy per axis.
        for axis in range(3):
            rows = np.arange(smoothness_free.shape[0]) * 3 + axis
            columns = np.arange(free_knots.shape[0]) * 3 + axis
            block[np.ix_(rows, columns)] = (
                params.smoothness_weight * smoothness_free
            )

        # Deviation block: identity, one row per variable.
        block[
            smooth_rows:obstacle_start, :
        ] = params.deviation_weight * np.eye(variable_count)

        # Obstacle block: d(hinge)/dx = -grad . dp/dx while active.
        for sample in np.flatnonzero(active):
            row = obstacle_start + sample
            for knot in range(free_knots.shape[0]):
                weight = basis_free[sample, knot]
                if weight == 0.0:
                    continue
                for axis in range(3):
                    block[row, knot * 3 + axis] = (
                        -params.obstacle_weight
                        * gradient[sample, axis]
                        * weight
                    )
        return block

    initial = straight[free_index].reshape(-1)
    set_times(times)
    initial_clearance = _minimum_clearance(
        dense_points(straight[free_index]), obstacles, params.tool_radius_m
    )

    # Hard box bound on every free knot: the start/goal bounding box grown
    # by the reach allowance. Without it the hinge can route the path far
    # outside the arm's reachable workspace and still report success.
    corner_low = np.minimum(
        start_torso.position_m, goal_torso.position_m
    ) - params.reach_allowance_m
    corner_high = np.maximum(
        start_torso.position_m, goal_torso.position_m
    ) + params.reach_allowance_m
    bounds = (
        np.tile(corner_low, len(free_index)),
        np.tile(corner_high, len(free_index)),
    )

    def solve():
        solution = least_squares(
            residuals,
            initial,
            jac=jacobian,
            bounds=bounds,
            max_nfev=params.max_iterations,
            method="trf",
        )
        knots = straight.copy()
        knots[free_index] = solution.x.reshape(-1, 3)
        return knots, solution

    def build(knots_torso, explicit_durations):
        knots_world = np.stack([
            _position_to_world(point, torso_pose_world)
            for point in knots_torso
        ])
        relative = _rotation_vector(
            start_pose_world.rotation.T @ goal_pose_world.rotation
        )
        fractions = np.linspace(0.0, 1.0, knot_count)
        poses = tuple(
            Pose(
                knots_world[index],
                start_pose_world.rotation
                @ _rotation_from_vector(fractions[index] * relative),
            )
            for index in range(knot_count)
        )
        return knots_world, timed_waypoint_trajectory(
            TargetFrame.WORLD, poses, explicit_durations, limits
        )

    explicit = (
        None if duration_s is None
        else _segment_durations(duration_s, knot_count)
    )

    # Knot geometry and delivered timing are coupled: the limits derive the
    # timing from the path, and a minimum-jerk spline's shape depends on its
    # segment durations.  Iterate that fixed point a bounded number of
    # times, ALWAYS keeping the best plan measured on the delivered object.
    # (An earlier version adopted the re-solve unconditionally and could
    # replace a collision-free plan with a colliding one.)
    best = None
    total_evaluations = 0
    passes = 0
    for _ in range(_MAX_RETIME_PASSES):
        passes += 1
        set_times(times)
        knots_torso, solution = solve()
        knots_world, trajectory = build(knots_torso, explicit)
        clearance = trajectory_min_clearance(
            trajectory, torso_pose_world, obstacles, params.tool_radius_m
        )
        total_evaluations += int(solution.nfev)
        if best is None or clearance > best[0]:
            best = (
                clearance, knots_torso, knots_world, trajectory, solution
            )
        if clearance >= params.clearance_margin_m - _RETIME_TOLERANCE_M:
            break
        actual_times = np.array(
            [waypoint.time_s for waypoint in trajectory.waypoints]
        )
        if np.allclose(actual_times, times, rtol=0.0, atol=1e-9):
            break
        times = actual_times

    final_clearance, knots_torso, knots_world, trajectory, solution = best
    collision_free = final_clearance >= 0.0
    margin_met = (
        final_clearance >= params.clearance_margin_m - _RETIME_TOLERANCE_M
    )
    message = str(solution.message)
    if passes > 1:
        message = f"{message} ({passes} timing passes, best kept)"

    return PlanResult(
        trajectory=trajectory,
        knots_world_m=knots_world,
        knots_torso_m=knots_torso,
        initial_min_clearance_m=initial_clearance,
        final_min_clearance_m=final_clearance,
        iterations=total_evaluations,
        success=bool(solution.success) and margin_met,
        collision_free=collision_free,
        margin_met=margin_met,
        message=message,
    )


def _segment_durations(duration_s, knot_count):
    total = float(duration_s)
    return [total / (knot_count - 1)] * (knot_count - 1)
