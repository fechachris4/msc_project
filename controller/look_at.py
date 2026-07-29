"""World-frame look-at orientation policies for Cartesian target sources.

Position paths remain in ``controller.trajectory``. This module decorates one
world-frame path with an independently configured orientation policy and
provides the structured point-source seam for future object tracking.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from controller.state import FramedTarget, Pose, TargetFrame, Twist
from controller.trajectory import (
    KinematicTargetSample,
    TargetSource,
    TrajectoryRateBounds,
)


_LOOK_AT_DIRECTION_EPS = 1e-9
_LOOK_AT_COLLINEAR_SIN_EPS = 1e-8


def _non_negative_time(value, name):
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _read_only_vector(value, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite shape-(3,) vector")
    copy = vector.copy()
    return np.frombuffer(copy.tobytes(), dtype=copy.dtype)


def _read_only_matrix(value, name):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite shape-(3, 3) matrix")
    copy = matrix.copy()
    return np.frombuffer(copy.tobytes(), dtype=copy.dtype).reshape(3, 3)


def _clean_small(value, tolerance=1e-14):
    result = np.asarray(value, dtype=float).copy()
    result[np.abs(result) < tolerance] = 0.0
    return result


def _vee(skew):
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])


def _source_kinematics(source, elapsed_time_s):
    method = getattr(source, "sample_kinematics", None)
    if method is None:
        target = source.sample(elapsed_time_s)
        return KinematicTargetSample(target, np.zeros(3), np.zeros(3))
    result = method(elapsed_time_s)
    if not isinstance(result, KinematicTargetSample):
        raise TypeError(
            "sample_kinematics must return KinematicTargetSample"
        )
    return result


def _normalised_vector_kinematics(
    value,
    velocity,
    acceleration,
    *,
    minimum_norm,
    error_message,
):
    """Unit vector and its first two derivatives."""
    vector = np.asarray(value, dtype=float)
    first = np.asarray(velocity, dtype=float)
    second = np.asarray(acceleration, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < minimum_norm:
        raise ValueError(error_message)
    unit = vector / norm
    norm_rate = float(unit @ first)
    unit_rate = (first - unit * norm_rate) / norm
    norm_acceleration = (
        float(first @ first) - norm_rate**2
    ) / norm + float(unit @ second)
    unit_acceleration = (
        second
        - unit * norm_acceleration
        - 2.0 * unit_rate * norm_rate
    ) / norm
    return unit, unit_rate, unit_acceleration


@dataclass(frozen=True, slots=True)
class WorldPointKinematics:
    """A sampled world point and its first two derivatives.

    The fixed TOML policy uses zero derivatives. This record is the explicit
    seam for a future measured moving object; trajectory code never reads a
    sensor or interprets natural language.
    """

    position_world_m: np.ndarray
    velocity_world_m_s: np.ndarray
    acceleration_world_m_s2: np.ndarray

    def __post_init__(self):
        for name in (
            "position_world_m",
            "velocity_world_m_s",
            "acceleration_world_m_s2",
        ):
            object.__setattr__(
                self,
                name,
                _read_only_vector(getattr(self, name), name),
            )


@runtime_checkable
class WorldPointSource(Protocol):
    """Structured world-point source sampled at trajectory elapsed time."""

    def sample_kinematics(
        self, elapsed_time_s: float
    ) -> WorldPointKinematics:
        ...


@dataclass(frozen=True, slots=True)
class FixedWorldPointSource:
    """Stationary world point used by the current TOML look-at policy."""

    position_world_m: np.ndarray

    def __post_init__(self):
        object.__setattr__(
            self,
            "position_world_m",
            _read_only_vector(
                self.position_world_m, "position_world_m"
            ),
        )

    def sample_kinematics(self, elapsed_time_s):
        _non_negative_time(elapsed_time_s, "elapsed_time_s")
        return WorldPointKinematics(
            self.position_world_m, np.zeros(3), np.zeros(3)
        )


@dataclass(frozen=True, slots=True)
class LookAtTargetSource:
    """Aim one declared local tool axis at a sampled world point.

    ``tool_forward_axis`` and ``tool_up_axis`` are expressed in the local
    end-effector (pinch-site) frame. ``world_up_direction`` is world-expressed.
    Roll aligns the tool-up projection about the forward axis with the
    world-up projection about the look direction.
    """

    source: TargetSource
    point_source: WorldPointSource
    tool_forward_axis: np.ndarray
    tool_up_axis: np.ndarray
    world_up_direction: np.ndarray
    _local_basis: np.ndarray = field(init=False, repr=False)
    _world_up_unit: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.source, TargetSource):
            raise TypeError("source must satisfy TargetSource")
        if TargetFrame(self.source.reference_frame) != TargetFrame.WORLD:
            raise ValueError(
                "look-at orientation requires a world-frame target source"
            )
        if not isinstance(self.point_source, WorldPointSource):
            raise TypeError("point_source must satisfy WorldPointSource")

        forward = _read_only_vector(
            self.tool_forward_axis, "tool_forward_axis"
        )
        tool_up = _read_only_vector(
            self.tool_up_axis, "tool_up_axis"
        )
        world_up = _read_only_vector(
            self.world_up_direction, "world_up_direction"
        )
        forward_norm = float(np.linalg.norm(forward))
        world_up_norm = float(np.linalg.norm(world_up))
        if forward_norm < _LOOK_AT_DIRECTION_EPS:
            raise ValueError("tool_forward_axis must be non-zero")
        if world_up_norm < _LOOK_AT_DIRECTION_EPS:
            raise ValueError("world_up_direction must be non-zero")
        forward_unit = forward / forward_norm
        projected_tool_up = (
            tool_up - forward_unit * float(forward_unit @ tool_up)
        )
        projected_norm = float(np.linalg.norm(projected_tool_up))
        if projected_norm < _LOOK_AT_COLLINEAR_SIN_EPS:
            raise ValueError(
                "tool_forward_axis and tool_up_axis must not be collinear"
            )
        tool_up_unit = projected_tool_up / projected_norm
        local_side = np.cross(tool_up_unit, forward_unit)
        local_basis = np.column_stack(
            (local_side, tool_up_unit, forward_unit)
        )
        object.__setattr__(
            self,
            "tool_forward_axis",
            _read_only_vector(forward, "tool_forward_axis"),
        )
        object.__setattr__(
            self,
            "tool_up_axis",
            _read_only_vector(tool_up, "tool_up_axis"),
        )
        object.__setattr__(
            self,
            "world_up_direction",
            _read_only_vector(world_up, "world_up_direction"),
        )
        object.__setattr__(
            self, "_local_basis", _read_only_matrix(
                local_basis, "local_basis"
            )
        )
        object.__setattr__(
            self,
            "_world_up_unit",
            _read_only_vector(
                world_up / world_up_norm, "world_up_direction"
            ),
        )

    @property
    def reference_frame(self):
        return TargetFrame.WORLD

    @property
    def duration_s(self):
        duration = getattr(self.source, "duration_s", None)
        return self.source.period_s if duration is None else duration

    @property
    def boundary_times_s(self):
        return getattr(self.source, "boundary_times_s", ())

    def sample_kinematics(self, elapsed_time_s):
        elapsed = _non_negative_time(elapsed_time_s, "elapsed_time_s")
        trajectory = _source_kinematics(self.source, elapsed)
        point = self.point_source.sample_kinematics(elapsed)
        if not isinstance(point, WorldPointKinematics):
            raise TypeError(
                "point_source.sample_kinematics must return "
                "WorldPointKinematics"
            )

        relative = (
            point.position_world_m
            - trajectory.target.pose.position_m
        )
        relative_velocity = (
            point.velocity_world_m_s
            - trajectory.target.twist.linear_m_s
        )
        relative_acceleration = (
            point.acceleration_world_m_s2
            - trajectory.linear_acceleration_m_s2
        )
        forward, forward_rate, forward_acceleration = (
            _normalised_vector_kinematics(
                relative,
                relative_velocity,
                relative_acceleration,
                minimum_norm=_LOOK_AT_DIRECTION_EPS,
                error_message=(
                    "look-at object point coincides with the "
                    "end-effector position"
                ),
            )
        )

        side_raw = np.cross(self._world_up_unit, forward)
        side_raw_rate = np.cross(self._world_up_unit, forward_rate)
        side_raw_acceleration = np.cross(
            self._world_up_unit, forward_acceleration
        )
        side, side_rate, side_acceleration = (
            _normalised_vector_kinematics(
                side_raw,
                side_raw_rate,
                side_raw_acceleration,
                minimum_norm=_LOOK_AT_COLLINEAR_SIN_EPS,
                error_message=(
                    "look direction and world_up_direction must not "
                    "be collinear"
                ),
            )
        )
        up = np.cross(forward, side)
        up_rate = (
            np.cross(forward_rate, side)
            + np.cross(forward, side_rate)
        )
        up_acceleration = (
            np.cross(forward_acceleration, side)
            + 2.0 * np.cross(forward_rate, side_rate)
            + np.cross(forward, side_acceleration)
        )

        world_basis = np.column_stack((side, up, forward))
        world_basis_rate = np.column_stack(
            (side_rate, up_rate, forward_rate)
        )
        world_basis_acceleration = np.column_stack(
            (side_acceleration, up_acceleration, forward_acceleration)
        )
        rotation = world_basis @ self._local_basis.T
        rotation_rate = world_basis_rate @ self._local_basis.T
        rotation_acceleration = (
            world_basis_acceleration @ self._local_basis.T
        )
        angular_velocity_matrix = rotation_rate @ rotation.T
        angular_acceleration_matrix = (
            rotation_acceleration @ rotation.T
            + rotation_rate @ rotation_rate.T
        )
        angular_velocity = _vee(
            0.5
            * (
                angular_velocity_matrix
                - angular_velocity_matrix.T
            )
        )
        angular_acceleration = _vee(
            0.5
            * (
                angular_acceleration_matrix
                - angular_acceleration_matrix.T
            )
        )
        target = FramedTarget(
            TargetFrame.WORLD,
            Pose(
                trajectory.target.pose.position_m,
                _clean_small(rotation),
            ),
            Twist(
                trajectory.target.twist.linear_m_s,
                _clean_small(angular_velocity),
            ),
        )
        return KinematicTargetSample(
            target,
            trajectory.linear_acceleration_m_s2,
            _clean_small(angular_acceleration),
        )

    def sample(self, elapsed_time_s):
        return self.sample_kinematics(elapsed_time_s).target

    def maximum_rates(self):
        """Return conservative bounds without replaying the trajectory.

        Translation bounds remain exact. Dynamic look-at angular extrema
        depend on minimum object distance and look/up separation over the
        complete path, so ``+inf`` explicitly means no finite safe bound is
        available. A stationary pose aimed at a fixed point is exactly zero.
        """
        bounds = self.source.maximum_rates()
        stationary = (
            isinstance(self.point_source, FixedWorldPointSource)
            and bounds.max_linear_speed_m_s == 0.0
            and bounds.max_linear_acceleration_m_s2 == 0.0
        )
        return TrajectoryRateBounds(
            bounds.max_linear_speed_m_s,
            bounds.max_linear_acceleration_m_s2,
            0.0 if stationary else np.inf,
            0.0 if stationary else np.inf,
        )
