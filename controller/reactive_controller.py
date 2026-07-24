"""The complete mathematical policy for the reactive controller.

Read the functions below in order:

1. world-frame pose error,
2. world-frame twist error,
3. proportional and derivative task twist,
4. damped least-squares inverse kinematics,
5. null-space joint centering,
6. requested joint velocity.

This file deliberately contains no frame conversion, command integration,
limits, timing, backend access, lifecycle handling, or persistent state.
Changing the controller equations belongs here; changing numerical gains
belongs in ``config/control.toml``.
"""

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from controller.state import ArmControllerState, WorldTarget
from runtime_config import ReactivePoseConfig


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
    joint_midpoint_rad,
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

    # Equation 5: joint-centering objective.
    qdot_null_objective = (
        -np.asarray(null_gain_s_inv)
        * (joint_position_rad - np.asarray(joint_midpoint_rad))
    )

    # Equation 6: project centering into the Jacobian null space.
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


class ReactiveController:
    """Pure controller policy; it has no state that persists between cycles."""

    def __init__(self, config, centering):
        if not isinstance(config, ReactivePoseConfig):
            raise TypeError("config must be a ReactivePoseConfig")
        if not isinstance(centering, JointCentering):
            raise TypeError("centering must be JointCentering")
        self._config = config
        self._centering = centering

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
            self._centering.midpoint_rad,
            self._centering.enabled * self._config.null_gain_s_inv,
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
class JointCentering:
    midpoint_rad: np.ndarray
    enabled: np.ndarray

    def __post_init__(self):
        midpoint = np.asarray(self.midpoint_rad, dtype=float)
        enabled = np.asarray(self.enabled, dtype=bool)
        if midpoint.shape != (7,) or not np.all(np.isfinite(midpoint)):
            raise ValueError("midpoint_rad must be a finite shape-(7,) array")
        if enabled.shape != (7,):
            raise ValueError("enabled must be a shape-(7,) boolean array")
        object.__setattr__(self, "midpoint_rad", _read_only(midpoint))
        object.__setattr__(self, "enabled", _read_only(enabled))


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
