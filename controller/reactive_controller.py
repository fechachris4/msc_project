"""The complete mathematical policy for the reactive controller.

Read the functions below in order:

1. world-frame pose error,
2. world-frame twist error,
3. proportional and derivative task twist,
4. damped least-squares inverse kinematics,
5. null-space joint-limit avoidance (deadband),
6. requested joint velocity.
7. safety-priority projection of that velocity.

This file deliberately contains no frame conversion, command integration,
timing, backend access, or lifecycle handling.  The reusable QP workspace is
the only persistent numerical state and changes no controller semantics.
Changing the controller equations belongs here; changing numerical gains
belongs in ``config/control.toml``.
"""

from dataclasses import dataclass

import numpy as np
import osqp
import pinocchio as pin
from scipy import sparse

from controller.state import (
    ArmControllerState,
    ArmHumanSafetyState,
    WorldTarget,
)
from runtime_config import HumanSafetyConfig, ReactivePoseConfig


class SafetyVelocityProjector:
    """Reusable fixed-structure OSQP workspace for one arm."""

    def __init__(self, max_human_constraints, config):
        count = int(max_human_constraints)
        if count <= 0:
            raise ValueError("max_human_constraints must be positive")
        if not isinstance(config, HumanSafetyConfig):
            raise TypeError("config must be a HumanSafetyConfig")
        self._max_human_constraints = count

        # Dense human rows plus one identity row per joint.  The sparsity
        # pattern never changes, so each cycle updates only numeric values.
        row_indices = []
        column_starts = [0]
        for joint in range(7):
            row_indices.extend(range(count))
            row_indices.append(count + joint)
            column_starts.append(len(row_indices))
        constraint_matrix = sparse.csc_matrix(
            (
                np.ones(len(row_indices)),
                np.asarray(row_indices, dtype=np.int32),
                np.asarray(column_starts, dtype=np.int32),
            ),
            shape=(count + 7, 7),
        )
        self._solver = osqp.OSQP()
        self._solver.setup(
            P=sparse.eye(7, format="csc"),
            q=np.zeros(7),
            A=constraint_matrix,
            l=np.full(count + 7, -np.inf),
            u=np.full(count + 7, np.inf),
            verbose=False,
            eps_abs=config.constraint_tolerance_m_s,
            eps_rel=0.0,
            max_iter=config.projection_iterations,
            polishing=False,
            warm_starting=True,
            adaptive_rho=True,
            rho=1.0,
            check_termination=5,
        )

    def project(self, requested, A, b, lower, upper):
        count = A.shape[0]
        if count > self._max_human_constraints:
            raise ValueError("human constraint count exceeds fixed capacity")
        padded_A = np.zeros((self._max_human_constraints, 7))
        padded_A[:count] = A
        matrix_values = np.concatenate([
            np.concatenate((padded_A[:, joint], [1.0]))
            for joint in range(7)
        ])
        lower_bounds = np.concatenate((
            b,
            np.full(self._max_human_constraints - count, -np.inf),
            lower,
        ))
        upper_bounds = np.concatenate((
            np.full(self._max_human_constraints, np.inf),
            upper,
        ))
        self._solver.update(
            q=-requested,
            l=lower_bounds,
            u=upper_bounds,
            Ax=matrix_values,
        )
        result = self._solver.solve(raise_error=False)
        candidate_available = (
            result.x is not None
            and np.all(np.isfinite(result.x))
        )
        return (
            np.asarray(result.x, dtype=float)
            if candidate_available else np.zeros(7),
            candidate_available,
            int(result.info.iter),
        )


def pose_error(state, target):
    """Equation 1: world-frame pose error, reference minus actual."""
    if not isinstance(state, ArmControllerState):
        raise TypeError("state must be an ArmControllerState")
    if not isinstance(target, WorldTarget):
        raise TypeError("target must be a WorldTarget")
    return (
        target.pose_world.position_m - state.ee_pose_world.position_m,
        pin.log3(target.pose_world.rotation @ state.ee_pose_world.rotation.T),
    )


def twist_error(state, target):
    """Equation 2: world-frame twist error, reference minus actual."""
    if not isinstance(state, ArmControllerState):
        raise TypeError("state must be an ArmControllerState")
    if not isinstance(target, WorldTarget):
        raise TypeError("target must be a WorldTarget")
    return (
        target.twist_world.linear_m_s - state.ee_twist_world.linear_m_s,
        target.twist_world.angular_rad_s - state.ee_twist_world.angular_rad_s,
    )


def task_twist_terms(e_pos, e_rot, e_v, e_w, config):
    """Equation 3: xdot_task = Kp * pose_error + Kd * twist_error."""
    p_twist = np.concatenate([
        config.kp_position_s_inv * e_pos
        if config.position_enabled else np.zeros(3),
        config.kp_rotation_s_inv * e_rot
        if config.orientation_enabled else np.zeros(3),
    ])
    d_twist = (
        np.concatenate([
            config.kd_position * e_v,
            config.kd_rotation * e_w,
        ])
        if config.velocity_enabled else np.zeros(6)
    )
    return p_twist, d_twist


def solve_reactive_velocity(
    jacobian_world,
    e_pos,
    e_rot,
    e_v,
    e_w,
    joint_position_rad,
    joint_limit_rad,
    joint_zone_rad,
    null_gain_s_inv,
    config,
    damping=None,
):
    """Equations 3-6: PD + DLS + null-space requested joint velocity."""
    damping = config.dls_damping if damping is None else damping

    # Equation 3: desired world-frame task twist.
    p_twist, d_twist = task_twist_terms(
        e_pos, e_rot, e_v, e_w, config)
    task_twist = p_twist + d_twist

    # Equation 4: damped least-squares inverse kinematics.
    qdot_task = jacobian_world.T @ np.linalg.solve(
        jacobian_world @ jacobian_world.T
        + damping**2 * np.eye(6),
        task_twist,
    )

    # Equation 5: deadband joint-limit avoidance. Zero across the whole
    # working range; a linear inward push once a bounded joint enters the
    # activation zone [limit - zone, limit]. Unbounded joints have limit 0.
    limit = np.asarray(joint_limit_rad, dtype=float)
    signed = np.remainder(
        np.asarray(joint_position_rad, dtype=float) + np.pi, 2.0 * np.pi
    ) - np.pi
    excess = np.abs(signed) - (limit - float(joint_zone_rad))
    qdot_null_objective = np.where(
        (limit > 0.0) & (excess > 0.0),
        -np.asarray(null_gain_s_inv) * excess * np.sign(signed),
        0.0,
    )

    # Equation 6: project the push into the Jacobian null space.
    qdot_null_projected = (
        np.eye(7) - np.linalg.pinv(jacobian_world) @ jacobian_world
    ) @ qdot_null_objective
    return ReactiveSolve(
        p_twist=p_twist,
        d_twist=d_twist,
        task_twist=task_twist,
        qdot_task=qdot_task,
        qdot_null_objective=qdot_null_objective,
        qdot_null_projected=qdot_null_projected,
        qdot_raw=qdot_task + qdot_null_projected,
    )


def constrain_velocity_for_human(
    requested_velocity_rad_s,
    lower_velocity_rad_s,
    upper_velocity_rad_s,
    safety_state,
    config,
    measured_velocity_rad_s=None,
    projector=None,
):
    """Equation 7: project requested qdot into limits and safe half-spaces.

    For every active arm sphere, the control-barrier inequality is

        distance_jacobian @ qdot >= -recovery_gain * signed_clearance.

    Positive clearance permits bounded approach; penetration requires outward
    recovery.  A fixed-structure OSQP solve finds the minimum change to the
    requested seven-joint velocity.  The same solver has a C/C++ interface for
    the hardware port.  A result is used only after every constraint is
    independently checked.  Otherwise the filter returns a hold/catch-up
    command and marks a genuine safety stop.
    """
    if not isinstance(safety_state, ArmHumanSafetyState):
        raise TypeError("safety_state must be an ArmHumanSafetyState")
    if not isinstance(config, HumanSafetyConfig):
        raise TypeError("config must be a HumanSafetyConfig")

    requested = _finite_vector7(
        requested_velocity_rad_s, "requested_velocity_rad_s"
    )
    lower = _finite_vector7(
        lower_velocity_rad_s, "lower_velocity_rad_s"
    )
    upper = _finite_vector7(
        upper_velocity_rad_s, "upper_velocity_rad_s"
    )
    measured = (
        np.zeros(7)
        if measured_velocity_rad_s is None
        else _finite_vector7(
            measured_velocity_rad_s, "measured_velocity_rad_s"
        )
    )
    if np.any(lower > upper):
        raise ValueError(
            "lower_velocity_rad_s must not exceed upper_velocity_rad_s"
        )

    bounded_request = np.clip(requested, lower, upper)
    if not config.enabled:
        return SafetyVelocitySolve(
            qdot_safe=requested,
            minimum_clearance_m=safety_state.minimum_clearance_m,
            active_constraint_count=0,
            projection_iterations=0,
            max_constraint_violation_m_s=0.0,
            human_adjusted=False,
            limit_adjusted=False,
            stopped=False,
            reason="disabled",
            limiting_points=(),
        )
    constraints = safety_state.constraints
    active_indices = (
        np.empty(0, dtype=int)
        if constraints is None
        else np.flatnonzero(constraints.active)
    )
    minimum_clearance = safety_state.minimum_clearance_m
    if active_indices.size == 0:
        return SafetyVelocitySolve(
            qdot_safe=bounded_request,
            minimum_clearance_m=minimum_clearance,
            active_constraint_count=0,
            projection_iterations=0,
            max_constraint_violation_m_s=0.0,
            human_adjusted=False,
            limit_adjusted=not np.array_equal(
                requested, bounded_request
            ),
            stopped=False,
            reason="clear",
            limiting_points=(),
        )

    A = constraints.distance_jacobian_m_rad[active_indices]
    active_clearance = constraints.signed_clearance_m[active_indices]
    measured_distance_rate = A @ measured
    b = (
        -config.recovery_gain_s_inv
        * (active_clearance - config.control_margin_m)
        - config.approach_velocity_damping
        * np.minimum(measured_distance_rate, 0.0)
    )
    bounded_slack = A @ bounded_request - b
    if np.min(bounded_slack) >= -config.constraint_tolerance_m_s:
        return SafetyVelocitySolve(
            qdot_safe=bounded_request,
            minimum_clearance_m=minimum_clearance,
            active_constraint_count=len(active_indices),
            projection_iterations=0,
            max_constraint_violation_m_s=0.0,
            human_adjusted=False,
            limit_adjusted=not np.array_equal(
                requested, bounded_request
            ),
            stopped=False,
            reason=(
                "joint_limit_filtered"
                if not np.array_equal(requested, bounded_request)
                else "clear"
            ),
            limiting_points=tuple(
                constraints.point_names[index]
                for local_index, index in enumerate(active_indices)
                if bounded_slack[local_index]
                <= 10.0 * config.constraint_tolerance_m_s
            ),
        )
    tolerance = config.constraint_tolerance_m_s
    x, iterations = _repair_constraint_feasibility(
        bounded_request,
        A,
        b,
        lower,
        upper,
        tolerance,
        max_iterations=4,
    )
    violation = _maximum_constraint_violation(x, A, b, lower, upper)
    candidate_available = violation <= tolerance
    if not candidate_available:
        row_scale = np.maximum(np.linalg.norm(A, axis=1), 1e-12)
        solver_A = A / row_scale[:, None]
        solver_b = b / row_scale
        active_projector = (
            SafetyVelocityProjector(len(active_indices), config)
            if projector is None else projector
        )
        x, candidate_available, solver_iterations = (
            active_projector.project(
                requested, solver_A, solver_b, lower, upper
            )
        )
        iterations += solver_iterations
        violation = _maximum_constraint_violation(
            x, A, b, lower, upper
        )
        if candidate_available and violation > tolerance:
            x, repair_iterations = _repair_constraint_feasibility(
                x,
                A,
                b,
                lower,
                upper,
                tolerance,
                max_iterations=16,
            )
            iterations += repair_iterations
            violation = _maximum_constraint_violation(
                x, A, b, lower, upper
            )
    feasible = candidate_available and violation <= tolerance
    stopped = not feasible
    if stopped:
        # Holding joint position is the fail-safe action.  If the persistent
        # command must catch up to measured/position bounds, use only that
        # required velocity and still report the stop.
        x = np.clip(np.zeros(7), lower, upper)
        violation = _maximum_constraint_violation(x, A, b, lower, upper)

    human_adjusted = not np.allclose(
        x, bounded_request, rtol=0.0, atol=tolerance
    )
    limit_adjusted = not np.array_equal(requested, bounded_request)
    safe_slack = A @ x - b
    limiting = tuple(
        constraints.point_names[index]
        for local_index, index in enumerate(active_indices)
        if safe_slack[local_index] <= 10.0 * tolerance
    )
    if stopped:
        reason = (
            "unsafe_initial_state_hold"
            if np.any(active_clearance < 0.0)
            else "constraint_projection_failed_hold"
        )
    elif human_adjusted:
        reason = "filtered"
    elif limit_adjusted:
        reason = "joint_limit_filtered"
    else:
        reason = "clear"
    return SafetyVelocitySolve(
        qdot_safe=x,
        minimum_clearance_m=minimum_clearance,
        active_constraint_count=len(active_indices),
        projection_iterations=iterations,
        max_constraint_violation_m_s=violation,
        human_adjusted=human_adjusted,
        limit_adjusted=limit_adjusted,
        stopped=stopped,
        reason=reason,
        limiting_points=limiting,
    )


def _finite_vector7(value, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (7,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite shape-(7,) array")
    return vector


def _maximum_constraint_violation(value, A, b, lower, upper):
    violations = [
        float(np.max(lower - value)),
        float(np.max(value - upper)),
    ]
    if A.size:
        violations.append(float(np.max(b - A @ value)))
    return max(0.0, *violations)


def _repair_constraint_feasibility(
    candidate,
    A,
    b,
    lower,
    upper,
    tolerance,
    max_iterations,
):
    """Repair a near-feasible QP result; never bypass final verification."""
    value = np.clip(np.asarray(candidate, dtype=float), lower, upper)
    row_norm_squared = np.einsum("ij,ij->i", A, A)
    completed = 0
    for completed in range(1, max_iterations + 1):
        for index in range(len(A)):
            deficit = b[index] - A[index] @ value
            if (
                deficit > tolerance
                and row_norm_squared[index] > np.finfo(float).eps
            ):
                value += (
                    deficit / row_norm_squared[index]
                ) * A[index]
        value = np.clip(value, lower, upper)
        if _maximum_constraint_violation(
            value, A, b, lower, upper
        ) <= tolerance:
            break
    return value, completed


class ReactiveController:
    """Pure controller policy; it has no state that persists between cycles."""

    def __init__(self, config, avoidance):
        if not isinstance(config, ReactivePoseConfig):
            raise TypeError("config must be a ReactivePoseConfig")
        if not isinstance(avoidance, JointLimitAvoidance):
            raise TypeError("avoidance must be JointLimitAvoidance")
        self._config = config
        self._avoidance = avoidance

    def compute(self, state, target):
        e_pos, e_rot = pose_error(state, target)
        e_v, e_w = twist_error(state, target)
        solve = solve_reactive_velocity(
            state.jacobian_world,
            e_pos,
            e_rot,
            e_v,
            e_w,
            state.joints.position_rad,
            self._avoidance.limit_rad,
            self._avoidance.zone_rad,
            self._config.null_gain_s_inv,
            self._config,
        )
        return ReactiveOutput(e_pos, e_rot, e_v, e_w, solve)


# Controller-specific inputs and outputs live below the equations so opening
# this file presents the mathematical policy first.


def _read_only(value):
    result = np.asarray(value).copy()
    return np.frombuffer(result.tobytes(), dtype=result.dtype).reshape(
        result.shape
    )


@dataclass(frozen=True, slots=True)
class JointLimitAvoidance:
    limit_rad: np.ndarray  # shape (7,), 0 = unbounded joint
    zone_rad: float

    def __post_init__(self):
        limit = np.asarray(self.limit_rad, dtype=float)
        if limit.shape != (7,) or not np.all(np.isfinite(limit)) or np.any(limit < 0.0):
            raise ValueError("limit_rad must be a finite non-negative shape-(7,) array")
        zone = float(self.zone_rad)
        if not np.isfinite(zone) or zone <= 0.0:
            raise ValueError("zone_rad must be a finite positive float")
        object.__setattr__(self, "limit_rad", _read_only(limit))
        object.__setattr__(self, "zone_rad", zone)


@dataclass(frozen=True, slots=True)
class ReactiveSolve:
    p_twist: np.ndarray
    d_twist: np.ndarray
    task_twist: np.ndarray
    qdot_task: np.ndarray
    qdot_null_objective: np.ndarray
    qdot_null_projected: np.ndarray
    qdot_raw: np.ndarray

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _read_only(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class ReactiveOutput:
    e_pos: np.ndarray
    e_rot: np.ndarray
    e_v: np.ndarray
    e_w: np.ndarray
    solve: ReactiveSolve

    def __post_init__(self):
        for name in ("e_pos", "e_rot", "e_v", "e_w"):
            object.__setattr__(self, name, _read_only(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class SafetyVelocitySolve:
    qdot_safe: np.ndarray
    minimum_clearance_m: float
    active_constraint_count: int
    projection_iterations: int
    max_constraint_violation_m_s: float
    human_adjusted: bool
    limit_adjusted: bool
    stopped: bool
    reason: str
    limiting_points: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(
            self, "qdot_safe", _read_only(self.qdot_safe)
        )
        if self.qdot_safe.shape != (7,) or not np.all(
            np.isfinite(self.qdot_safe)
        ):
            raise ValueError("qdot_safe must be a finite shape-(7,) array")
        minimum = float(self.minimum_clearance_m)
        if np.isnan(minimum):
            raise ValueError("minimum_clearance_m must not be NaN")
        object.__setattr__(self, "minimum_clearance_m", minimum)
        for name in (
            "active_constraint_count",
            "projection_iterations",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        violation = float(self.max_constraint_violation_m_s)
        if not np.isfinite(violation) or violation < 0.0:
            raise ValueError(
                "max_constraint_violation_m_s must be finite and non-negative"
            )
        object.__setattr__(
            self, "max_constraint_violation_m_s", violation
        )
        for name in ("human_adjusted", "limit_adjusted", "stopped"):
            object.__setattr__(self, name, bool(getattr(self, name)))
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")
        points = tuple(self.limiting_points)
        if any(not isinstance(name, str) or not name for name in points):
            raise ValueError(
                "limiting_points must contain non-empty strings"
            )
        object.__setattr__(self, "limiting_points", points)
