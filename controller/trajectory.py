"""Pure Cartesian target sources and timed waypoint trajectories.

The records in this module have direct C++ equivalents: fixed poses, explicit
times, one declared frame, and a deterministic ``sample(elapsed_time_s)``
boundary.  There are no MuJoCo reads, marker writes, or user callbacks here.
"""

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from controller.state import (
    DualArmFramedTargets,
    FramedTarget,
    Pose,
    TargetFrame,
    Twist,
)


_SMALL_ANGLE_RAD = 1e-8
_NEAR_PI_RAD = 1e-6
_PROGRAM_BOUNDARY_ATOL = 1e-10
_LIMIT_TOLERANCE = 1e-10


def _non_negative_time(value, name):
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _positive_duration(value, name):
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _read_only_vector(value):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("rotation vector must be finite shape-(3,)")
    copy = vector.copy()
    return np.frombuffer(copy.tobytes(), dtype=copy.dtype)


def _vee(skew):
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])


def _canonical_axis(axis):
    """Choose one deterministic sign for the otherwise ambiguous pi axis."""
    result = np.asarray(axis, dtype=float).copy()
    for component in result:
        if abs(component) > 1e-12:
            if component < 0.0:
                result *= -1.0
            break
    return result


def _rotation_vector(rotation):
    """Principal SO(3) logarithm, with norm in [0, pi]."""
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < _SMALL_ANGLE_RAD:
        return 0.5 * _vee(rotation - rotation.T)
    if np.pi - angle < _NEAR_PI_RAD:
        symmetric = 0.5 * (rotation + np.eye(3))
        index = int(np.argmax(np.diag(symmetric)))
        axis = np.zeros(3)
        axis[index] = np.sqrt(max(symmetric[index, index], 0.0))
        if axis[index] < 1e-12:
            raise ValueError("could not determine the axis of a pi rotation")
        for other in range(3):
            if other != index:
                axis[other] = symmetric[index, other] / axis[index]
        axis /= np.linalg.norm(axis)
        skew_axis = _vee(rotation - rotation.T)
        if np.linalg.norm(skew_axis) > 1e-10:
            if np.dot(axis, skew_axis) < 0.0:
                axis *= -1.0
        else:
            axis = _canonical_axis(axis)
        return axis * angle
    return angle / (2.0 * np.sin(angle)) * _vee(rotation - rotation.T)


def _rotation_from_vector(rotation_vector):
    """SO(3) exponential for a rotation vector."""
    vector = np.asarray(rotation_vector, dtype=float)
    angle = float(np.linalg.norm(vector))
    cross = np.array([
        [0.0, -vector[2], vector[1]],
        [vector[2], 0.0, -vector[0]],
        [-vector[1], vector[0], 0.0],
    ])
    if angle < _SMALL_ANGLE_RAD:
        return np.eye(3) + cross + 0.5 * cross @ cross
    return (
        np.eye(3)
        + (np.sin(angle) / angle) * cross
        + ((1.0 - np.cos(angle)) / angle**2) * cross @ cross
    )


def _minimum_jerk(unit_time, duration_s):
    """Quintic progress and its first two time derivatives."""
    u = float(unit_time)
    progress = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    progress_rate = (
        30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
    ) / duration_s
    progress_acceleration = (
        60.0 * u - 180.0 * u**2 + 120.0 * u**3
    ) / duration_s**2
    return progress, progress_rate, progress_acceleration


def _read_only_array(value):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("trajectory coefficients must be finite")
    copy = array.copy()
    return np.frombuffer(copy.tobytes(), dtype=copy.dtype).reshape(copy.shape)


def _clean_small(value, tolerance=1e-14):
    result = np.asarray(value, dtype=float).copy()
    result[np.abs(result) < tolerance] = 0.0
    return result


def _quintic_coefficients(
    start_position,
    end_position,
    start_velocity,
    end_velocity,
    start_acceleration,
    end_acceleration,
    duration_s,
):
    """Ascending normalized-time coefficients for one quintic segment."""
    duration = _positive_duration(duration_s, "duration_s")
    p0 = np.asarray(start_position, dtype=float)
    p1 = np.asarray(end_position, dtype=float)
    v0 = np.asarray(start_velocity, dtype=float)
    v1 = np.asarray(end_velocity, dtype=float)
    a0 = np.asarray(start_acceleration, dtype=float)
    a1 = np.asarray(end_acceleration, dtype=float)
    c0 = p0
    c1 = duration * v0
    c2 = 0.5 * duration**2 * a0
    rhs = np.stack((
        p1 - c0 - c1 - c2,
        duration * v1 - c1 - 2.0 * c2,
        duration**2 * a1 - 2.0 * c2,
    ))
    c3, c4, c5 = np.linalg.solve(
        np.array([
            [1.0, 1.0, 1.0],
            [3.0, 4.0, 5.0],
            [6.0, 12.0, 20.0],
        ]),
        rhs,
    )
    return np.stack((c0, c1, c2, c3, c4, c5), axis=-1)


def _minimum_jerk_translation_coefficients(positions, times_s):
    """Globally minimum-integrated-jerk C2 spline through positions.

    Endpoint velocity and acceleration are fixed to zero. Interior velocity
    and acceleration are solved together, rather than guessed per segment.
    """
    points = np.asarray(positions, dtype=float)
    times = np.asarray(times_s, dtype=float)
    count = len(points)
    if count == 2:
        zeros = np.zeros((2, 3))
        derivatives = zeros
        accelerations = zeros
    else:
        interior = count - 2
        unknown_count = 2 * interior
        hessian = np.zeros((unknown_count, unknown_count))
        gradients = np.zeros((3, unknown_count))
        integral = np.array([
            [1.0, 1.0 / 2.0, 1.0 / 3.0],
            [1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0],
            [1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0],
        ])
        for segment in range(count - 1):
            duration = times[segment + 1] - times[segment]

            def jerk_map(values):
                coefficients = _quintic_coefficients(
                    values[0],
                    values[1],
                    values[2],
                    values[4],
                    values[3],
                    values[5],
                    duration,
                )
                return np.array([
                    6.0 * coefficients[..., 3],
                    24.0 * coefficients[..., 4],
                    60.0 * coefficients[..., 5],
                ])

            basis = np.eye(6)
            mapping = np.stack(
                [jerk_map(item) for item in basis], axis=1
            )
            cost = mapping.T @ integral @ mapping / duration**5
            constant = np.zeros((3, 6))
            constant[:, 0] = points[segment]
            constant[:, 1] = points[segment + 1]
            local_map = np.zeros((6, unknown_count))
            if segment > 0:
                local_map[2, segment - 1] = 1.0
                local_map[3, interior + segment - 1] = 1.0
            if segment + 1 < count - 1:
                local_map[4, segment] = 1.0
                local_map[5, interior + segment] = 1.0
            hessian += local_map.T @ cost @ local_map
            gradients += constant @ cost @ local_map
        solved = np.linalg.solve(hessian, -gradients.T).T
        derivatives = np.zeros((count, 3))
        accelerations = np.zeros((count, 3))
        derivatives[1:-1] = solved[:, :interior].T
        accelerations[1:-1] = solved[:, interior:].T

    coefficients = []
    for index, duration in enumerate(np.diff(times)):
        coefficients.append(
            _quintic_coefficients(
                points[index],
                points[index + 1],
                derivatives[index],
                derivatives[index + 1],
                accelerations[index],
                accelerations[index + 1],
                duration,
            )
        )
    return np.asarray(coefficients)


def _evaluate_polynomial(coefficients, unit_time, derivative_order=0):
    values = np.asarray(coefficients, dtype=float)
    for _ in range(derivative_order):
        powers = np.arange(1, values.shape[-1], dtype=float)
        values = values[..., 1:] * powers
    return np.polynomial.polynomial.polyval(float(unit_time), values.T).T


def _vector_polynomial_maximum(coefficients):
    """Exact maximum norm of a vector polynomial on normalized [0, 1]."""
    values = np.asarray(coefficients, dtype=float)
    derivative = np.stack([
        np.polynomial.polynomial.polyder(component)
        for component in values
    ])
    norm_squared_derivative = np.zeros(
        values.shape[1] + derivative.shape[1] - 1
    )
    for component, component_derivative in zip(values, derivative):
        product = np.polynomial.polynomial.polymul(
            component, component_derivative
        )
        norm_squared_derivative[:len(product)] += 2.0 * product
    roots = np.polynomial.polynomial.polyroots(norm_squared_derivative)
    candidates = [0.0, 1.0]
    candidates.extend(
        float(root.real)
        for root in roots
        if abs(root.imag) < 1e-9
        and _LIMIT_TOLERANCE < root.real < 1.0 - _LIMIT_TOLERANCE
    )
    return max(
        float(np.linalg.norm(_evaluate_polynomial(values, value)))
        for value in candidates
    )


def _zero_twist(target):
    return (
        np.allclose(target.twist.linear_m_s, 0.0, rtol=0.0, atol=0.0)
        and np.allclose(
            target.twist.angular_rad_s, 0.0, rtol=0.0, atol=0.0
        )
    )


@dataclass(frozen=True, slots=True)
class KinematicTargetSample:
    """Target plus acceleration, used to validate C2 program boundaries."""

    target: FramedTarget
    linear_acceleration_m_s2: np.ndarray
    angular_acceleration_rad_s2: np.ndarray

    def __post_init__(self):
        if not isinstance(self.target, FramedTarget):
            raise TypeError("target must be a FramedTarget")
        object.__setattr__(
            self,
            "linear_acceleration_m_s2",
            _read_only_vector(self.linear_acceleration_m_s2),
        )
        object.__setattr__(
            self,
            "angular_acceleration_rad_s2",
            _read_only_vector(self.angular_acceleration_rad_s2),
        )


def _kinematic_sample(source, elapsed_time_s):
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


def _samples_continuous(left, right):
    if left.target.reference_frame != right.target.reference_frame:
        return False
    values = (
        (left.target.pose.position_m, right.target.pose.position_m),
        (left.target.pose.rotation, right.target.pose.rotation),
        (
            left.target.twist.linear_m_s,
            right.target.twist.linear_m_s,
        ),
        (
            left.target.twist.angular_rad_s,
            right.target.twist.angular_rad_s,
        ),
        (
            left.linear_acceleration_m_s2,
            right.linear_acceleration_m_s2,
        ),
        (
            left.angular_acceleration_rad_s2,
            right.angular_acceleration_rad_s2,
        ),
    )
    return all(
        np.allclose(a, b, rtol=0.0, atol=_PROGRAM_BOUNDARY_ATOL)
        for a, b in values
    )


@runtime_checkable
class TargetSource(Protocol):
    """One-arm Cartesian source sampled in trajectory-local elapsed time."""

    @property
    def reference_frame(self) -> TargetFrame:
        ...

    def sample(self, elapsed_time_s: float) -> FramedTarget:
        ...


@runtime_checkable
class DualArmTargetSource(Protocol):
    """Two-arm source sampled exactly once at the Runner boundary."""

    def sample(self, elapsed_time_s: float) -> DualArmFramedTargets:
        ...


@dataclass(frozen=True, slots=True)
class StaticTargetSource:
    """Adapter that presents one retained target through the source boundary."""

    target: FramedTarget

    def __post_init__(self):
        if not isinstance(self.target, FramedTarget):
            raise TypeError("target must be a FramedTarget")
        if not _zero_twist(self.target):
            raise ValueError("a static target must have zero twist")

    @property
    def reference_frame(self):
        return self.target.reference_frame

    def sample(self, elapsed_time_s):
        _non_negative_time(elapsed_time_s, "elapsed_time_s")
        return self.target

    def sample_kinematics(self, elapsed_time_s):
        return KinematicTargetSample(
            self.sample(elapsed_time_s), np.zeros(3), np.zeros(3)
        )


@dataclass(frozen=True, slots=True)
class CartesianWaypoint:
    """A pose at one trajectory-local time; frame belongs to the trajectory."""

    time_s: float
    pose: Pose

    def __post_init__(self):
        object.__setattr__(
            self, "time_s", _non_negative_time(self.time_s, "time_s")
        )
        if not isinstance(self.pose, Pose):
            raise TypeError("pose must be a Pose")


@dataclass(frozen=True, slots=True)
class TrajectoryRateBounds:
    max_linear_speed_m_s: float
    max_linear_acceleration_m_s2: float
    max_angular_speed_rad_s: float
    max_angular_acceleration_rad_s2: float

    def __post_init__(self):
        for name in (
            "max_linear_speed_m_s",
            "max_linear_acceleration_m_s2",
            "max_angular_speed_rad_s",
            "max_angular_acceleration_rad_s2",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class TrajectoryLimits:
    """Optional Cartesian limits used for timing and strict validation."""

    max_linear_speed_m_s: float | None = None
    max_linear_acceleration_m_s2: float | None = None
    max_angular_speed_rad_s: float | None = None
    max_angular_acceleration_rad_s2: float | None = None

    def __post_init__(self):
        for name in (
            "max_linear_speed_m_s",
            "max_linear_acceleration_m_s2",
            "max_angular_speed_rad_s",
            "max_angular_acceleration_rad_s2",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, _positive_duration(value, name)
                )

    def required_time_scale(self, bounds):
        if not isinstance(bounds, TrajectoryRateBounds):
            raise TypeError("bounds must be TrajectoryRateBounds")
        ratios = [1.0]
        pairs = (
            (
                bounds.max_linear_speed_m_s,
                self.max_linear_speed_m_s,
                False,
            ),
            (
                bounds.max_linear_acceleration_m_s2,
                self.max_linear_acceleration_m_s2,
                True,
            ),
            (
                bounds.max_angular_speed_rad_s,
                self.max_angular_speed_rad_s,
                False,
            ),
            (
                bounds.max_angular_acceleration_rad_s2,
                self.max_angular_acceleration_rad_s2,
                True,
            ),
        )
        for actual, limit, square_root in pairs:
            if limit is not None:
                ratio = actual / limit
                ratios.append(np.sqrt(ratio) if square_root else ratio)
        return max(ratios)

    def validate(self, bounds):
        scale = self.required_time_scale(bounds)
        if scale > 1.0 + _LIMIT_TOLERANCE:
            raise ValueError(
                "explicit trajectory timing violates Cartesian limits; "
                f"required time scale={scale:.6f}"
            )


@dataclass(frozen=True, slots=True)
class CartesianWaypointTrajectory:
    """C2 minimum-jerk translation and smooth SO(3) waypoint motion.

    Translation minimizes integrated squared jerk over the complete path.
    Endpoint linear velocity/acceleration are zero; interior values are solved
    globally and shared by adjacent quintics. Orientation reaches each exact
    waypoint using principal-log SO(3) quintics with zero angular velocity and
    acceleration at every orientation knot. Angular quantities are spatial and
    expressed in the declared reference frame.
    """

    reference_frame: TargetFrame
    waypoints: tuple[CartesianWaypoint, ...]
    _times_s: tuple[float, ...] = field(init=False, repr=False)
    _rotation_vectors: tuple[np.ndarray, ...] = field(
        init=False, repr=False
    )
    _translation_coefficients: np.ndarray = field(
        init=False, repr=False
    )

    def __post_init__(self):
        object.__setattr__(
            self, "reference_frame", TargetFrame(self.reference_frame)
        )
        waypoints = tuple(self.waypoints)
        if len(waypoints) < 2:
            raise ValueError("a waypoint trajectory needs at least two poses")
        if any(not isinstance(item, CartesianWaypoint) for item in waypoints):
            raise TypeError("waypoints must contain CartesianWaypoint values")
        if waypoints[0].time_s != 0.0:
            raise ValueError("the first waypoint time must be 0.0 seconds")
        times = tuple(item.time_s for item in waypoints)
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("waypoint times must be strictly increasing")
        rotations = tuple(
            _read_only_vector(
                _rotation_vector(
                    left.pose.rotation.T @ right.pose.rotation
                )
            )
            for left, right in zip(waypoints, waypoints[1:])
        )
        object.__setattr__(self, "waypoints", waypoints)
        object.__setattr__(self, "_times_s", times)
        object.__setattr__(self, "_rotation_vectors", rotations)
        object.__setattr__(
            self,
            "_translation_coefficients",
            _read_only_array(
                _minimum_jerk_translation_coefficients(
                    [item.pose.position_m for item in waypoints],
                    times,
                )
            ),
        )

    @classmethod
    def from_durations(cls, reference_frame, poses, durations_s):
        poses = tuple(poses)
        durations = tuple(
            _positive_duration(value, "duration_s") for value in durations_s
        )
        if len(durations) != len(poses) - 1:
            raise ValueError(
                "durations_s must contain one duration between each pose"
            )
        times = [0.0]
        for duration in durations:
            times.append(times[-1] + duration)
        return cls(
            reference_frame,
            tuple(
                CartesianWaypoint(time_s, pose)
                for time_s, pose in zip(times, poses)
            ),
        )

    @property
    def duration_s(self):
        return self._times_s[-1]

    @property
    def boundary_times_s(self):
        return self._times_s[1:]

    def sample_kinematics(self, elapsed_time_s):
        elapsed = _non_negative_time(elapsed_time_s, "elapsed_time_s")
        if elapsed <= 0.0:
            return KinematicTargetSample(
                FramedTarget(
                    self.reference_frame,
                    self.waypoints[0].pose,
                    Twist.zero(),
                ),
                np.zeros(3),
                np.zeros(3),
            )
        if elapsed >= self.duration_s:
            return KinematicTargetSample(
                FramedTarget(
                    self.reference_frame,
                    self.waypoints[-1].pose,
                    Twist.zero(),
                ),
                np.zeros(3),
                np.zeros(3),
            )

        segment = bisect_right(self._times_s, elapsed) - 1
        left = self.waypoints[segment]
        right = self.waypoints[segment + 1]
        duration = right.time_s - left.time_s
        unit_time = (elapsed - left.time_s) / duration
        progress, progress_rate, progress_acceleration = _minimum_jerk(
            unit_time, duration
        )

        coefficients = self._translation_coefficients[segment]
        position = _evaluate_polynomial(coefficients, unit_time)
        linear_velocity = (
            _evaluate_polynomial(coefficients, unit_time, 1) / duration
        )
        linear_acceleration = (
            _evaluate_polynomial(coefficients, unit_time, 2) / duration**2
        )
        relative_vector = self._rotation_vectors[segment]
        pose = Pose(
            _clean_small(position),
            left.pose.rotation
            @ _rotation_from_vector(progress * relative_vector),
        )
        twist = Twist(
            _clean_small(linear_velocity),
            _clean_small(
                progress_rate * (left.pose.rotation @ relative_vector)
            ),
        )
        return KinematicTargetSample(
            FramedTarget(self.reference_frame, pose, twist),
            _clean_small(linear_acceleration),
            _clean_small(
                progress_acceleration
                * (left.pose.rotation @ relative_vector)
            ),
        )

    def sample(self, elapsed_time_s):
        return self.sample_kinematics(elapsed_time_s).target

    def maximum_rates(self):
        """Return exact segment-polynomial maxima for configured timing."""
        max_linear_speed = 0.0
        max_linear_acceleration = 0.0
        max_angular_speed = 0.0
        max_angular_acceleration = 0.0
        for index, coefficients in enumerate(
            self._translation_coefficients
        ):
            duration = self._times_s[index + 1] - self._times_s[index]
            velocity = np.stack([
                np.polynomial.polynomial.polyder(component)
                for component in coefficients
            ]) / duration
            acceleration = np.stack([
                np.polynomial.polynomial.polyder(component, 2)
                for component in coefficients
            ]) / duration**2
            max_linear_speed = max(
                max_linear_speed,
                _vector_polynomial_maximum(velocity),
            )
            max_linear_acceleration = max(
                max_linear_acceleration,
                _vector_polynomial_maximum(acceleration),
            )
            angle = float(np.linalg.norm(self._rotation_vectors[index]))
            max_angular_speed = max(
                max_angular_speed, 1.875 * angle / duration
            )
            max_angular_acceleration = max(
                max_angular_acceleration,
                10.0 / np.sqrt(3.0) * angle / duration**2,
            )
        return TrajectoryRateBounds(
            max_linear_speed,
            max_linear_acceleration,
            max_angular_speed,
            max_angular_acceleration,
        )


@dataclass(frozen=True, slots=True)
class HoldTrajectory:
    """Finite static segment with zero twist and acceleration."""

    target: FramedTarget
    duration_s: float

    def __post_init__(self):
        if not isinstance(self.target, FramedTarget):
            raise TypeError("target must be a FramedTarget")
        if not _zero_twist(self.target):
            raise ValueError("a hold target must have zero twist")
        object.__setattr__(
            self,
            "duration_s",
            _positive_duration(self.duration_s, "duration_s"),
        )

    @property
    def reference_frame(self):
        return self.target.reference_frame

    @property
    def boundary_times_s(self):
        return (self.duration_s,)

    def sample(self, elapsed_time_s):
        _non_negative_time(elapsed_time_s, "elapsed_time_s")
        return self.target

    def sample_kinematics(self, elapsed_time_s):
        return KinematicTargetSample(
            self.sample(elapsed_time_s), np.zeros(3), np.zeros(3)
        )

    def maximum_rates(self):
        return TrajectoryRateBounds(0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class CircleTrajectory:
    """One or more exact circular revolutions with smooth rest endpoints."""

    reference_frame: TargetFrame
    start_pose: Pose
    radius_m: float
    normal: np.ndarray
    start_direction: np.ndarray
    duration_s: float
    revolutions: int = 1
    clockwise: bool = False
    end_rotation: np.ndarray | None = None
    _normal: np.ndarray = field(init=False, repr=False)
    _radial: np.ndarray = field(init=False, repr=False)
    _tangent: np.ndarray = field(init=False, repr=False)
    _rotation_vector: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        frame = TargetFrame(self.reference_frame)
        if not isinstance(self.start_pose, Pose):
            raise TypeError("start_pose must be a Pose")
        radius = _positive_duration(self.radius_m, "radius_m")
        duration = _positive_duration(self.duration_s, "duration_s")
        if (
            isinstance(self.revolutions, bool)
            or not isinstance(self.revolutions, int)
            or self.revolutions <= 0
        ):
            raise ValueError("revolutions must be a positive integer")
        if not isinstance(self.clockwise, bool):
            raise TypeError("clockwise must be bool")
        normal = _read_only_vector(self.normal)
        radial = _read_only_vector(self.start_direction)
        normal_norm = float(np.linalg.norm(normal))
        radial_norm = float(np.linalg.norm(radial))
        if normal_norm < 1e-12 or radial_norm < 1e-12:
            raise ValueError("circle directions must be non-zero")
        normal = normal / normal_norm
        radial = radial / radial_norm
        if abs(float(np.dot(normal, radial))) > 1e-9:
            raise ValueError(
                "circle normal and start_direction must be orthogonal"
            )
        tangent = np.cross(normal, radial)
        if self.clockwise:
            tangent *= -1.0
        if self.end_rotation is None:
            end_rotation = self.start_pose.rotation
        else:
            end_rotation = np.asarray(self.end_rotation, dtype=float)
            Pose(self.start_pose.position_m, end_rotation)
        rotation_vector = _rotation_vector(
            self.start_pose.rotation.T @ end_rotation
        )
        object.__setattr__(self, "reference_frame", frame)
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "_normal", _read_only_vector(normal))
        object.__setattr__(self, "_radial", _read_only_vector(radial))
        object.__setattr__(self, "_tangent", _read_only_vector(tangent))
        object.__setattr__(
            self, "_rotation_vector", _read_only_vector(rotation_vector)
        )
        object.__setattr__(
            self, "end_rotation", _read_only_array(end_rotation)
        )

    @property
    def boundary_times_s(self):
        return (self.duration_s,)

    @property
    def centre_m(self):
        return self.start_pose.position_m - self.radius_m * self._radial

    def sample_kinematics(self, elapsed_time_s):
        elapsed = _non_negative_time(elapsed_time_s, "elapsed_time_s")
        if elapsed <= 0.0:
            return KinematicTargetSample(
                FramedTarget(
                    self.reference_frame, self.start_pose, Twist.zero()
                ),
                np.zeros(3),
                np.zeros(3),
            )
        clamped = min(elapsed, self.duration_s)
        unit_time = clamped / self.duration_s
        progress, rate, acceleration = _minimum_jerk(
            unit_time, self.duration_s
        )
        total_angle = 2.0 * np.pi * self.revolutions
        angle = total_angle * progress
        angle_rate = total_angle * rate
        angle_acceleration = total_angle * acceleration
        radial = (
            np.cos(angle) * self._radial
            + np.sin(angle) * self._tangent
        )
        tangent = (
            -np.sin(angle) * self._radial
            + np.cos(angle) * self._tangent
        )
        position = self.centre_m + self.radius_m * radial
        linear_velocity = self.radius_m * angle_rate * tangent
        linear_acceleration = self.radius_m * (
            angle_acceleration * tangent - angle_rate**2 * radial
        )
        rotation = self.start_pose.rotation @ _rotation_from_vector(
            progress * self._rotation_vector
        )
        target = FramedTarget(
            self.reference_frame,
            Pose(position, rotation),
            Twist(
                linear_velocity,
                rate
                * (self.start_pose.rotation @ self._rotation_vector),
            ),
        )
        return KinematicTargetSample(
            target,
            linear_acceleration,
            acceleration
            * (self.start_pose.rotation @ self._rotation_vector),
        )

    def sample(self, elapsed_time_s):
        return self.sample_kinematics(elapsed_time_s).target

    def maximum_rates(self):
        total_angle = 2.0 * np.pi * self.revolutions
        speed = (
            self.radius_m * total_angle * 1.875 / self.duration_s
        )
        acceleration = self.radius_m * (
            total_angle * 10.0 / np.sqrt(3.0)
            + (total_angle * 1.875) ** 2
        ) / self.duration_s**2
        orientation_angle = float(np.linalg.norm(self._rotation_vector))
        return TrajectoryRateBounds(
            speed,
            acceleration,
            1.875 * orientation_angle / self.duration_s,
            10.0
            / np.sqrt(3.0)
            * orientation_angle
            / self.duration_s**2,
        )


def _minimum_duration(distance, angle, limits):
    if not isinstance(limits, TrajectoryLimits):
        raise TypeError("limits must be TrajectoryLimits")
    candidates = []
    if distance > _LIMIT_TOLERANCE:
        if (
            limits.max_linear_speed_m_s is None
            and limits.max_linear_acceleration_m_s2 is None
        ):
            raise ValueError(
                "automatic timing for translation needs a linear speed "
                "or acceleration limit"
            )
        if limits.max_linear_speed_m_s is not None:
            candidates.append(
                1.875 * distance / limits.max_linear_speed_m_s
            )
        if limits.max_linear_acceleration_m_s2 is not None:
            candidates.append(
                np.sqrt(
                    10.0
                    / np.sqrt(3.0)
                    * distance
                    / limits.max_linear_acceleration_m_s2
                )
            )
    if angle > _LIMIT_TOLERANCE:
        if (
            limits.max_angular_speed_rad_s is None
            and limits.max_angular_acceleration_rad_s2 is None
        ):
            raise ValueError(
                "automatic timing for rotation needs an angular speed "
                "or acceleration limit"
            )
        if limits.max_angular_speed_rad_s is not None:
            candidates.append(
                1.875 * angle / limits.max_angular_speed_rad_s
            )
        if limits.max_angular_acceleration_rad_s2 is not None:
            candidates.append(
                np.sqrt(
                    10.0
                    / np.sqrt(3.0)
                    * angle
                    / limits.max_angular_acceleration_rad_s2
                )
            )
    if not candidates:
        raise ValueError("automatic timing requires a moving segment")
    return max(candidates)


def timed_waypoint_trajectory(
    reference_frame,
    poses,
    durations_s,
    limits=TrajectoryLimits(),
):
    """Build a waypoint spline, deriving timing when durations are omitted."""
    poses = tuple(poses)
    if len(poses) < 2:
        raise ValueError("a waypoint path needs at least two poses")
    if durations_s is None:
        durations = []
        for left, right in zip(poses, poses[1:]):
            distance = float(
                np.linalg.norm(right.position_m - left.position_m)
            )
            angle = float(
                np.linalg.norm(
                    _rotation_vector(left.rotation.T @ right.rotation)
                )
            )
            durations.append(_minimum_duration(distance, angle, limits))
        explicit = False
    else:
        durations = [
            _positive_duration(item, "duration_s")
            for item in durations_s
        ]
        explicit = True
    trajectory = CartesianWaypointTrajectory.from_durations(
        reference_frame, poses, durations
    )
    scale = limits.required_time_scale(trajectory.maximum_rates())
    if explicit:
        limits.validate(trajectory.maximum_rates())
    elif scale > 1.0:
        trajectory = CartesianWaypointTrajectory.from_durations(
            reference_frame,
            poses,
            [scale * item for item in durations],
        )
        limits.validate(trajectory.maximum_rates())
    return trajectory


def timed_circle_trajectory(
    reference_frame,
    start_pose,
    radius_m,
    normal,
    start_direction,
    duration_s,
    limits=TrajectoryLimits(),
    *,
    revolutions=1,
    clockwise=False,
    end_rotation=None,
):
    """Build an exact circle, deriving duration when it is omitted."""
    if duration_s is None and (
        limits.max_linear_speed_m_s is None
        and limits.max_linear_acceleration_m_s2 is None
    ):
        raise ValueError(
            "automatic circle timing needs a linear speed or "
            "acceleration limit"
        )
    candidate_duration = 1.0 if duration_s is None else duration_s
    trajectory = CircleTrajectory(
        reference_frame,
        start_pose,
        radius_m,
        normal,
        start_direction,
        candidate_duration,
        revolutions,
        clockwise,
        end_rotation,
    )
    scale = limits.required_time_scale(trajectory.maximum_rates())
    if duration_s is None:
        trajectory = CircleTrajectory(
            reference_frame,
            start_pose,
            radius_m,
            normal,
            start_direction,
            scale,
            revolutions,
            clockwise,
            end_rotation,
        )
        limits.validate(trajectory.maximum_rates())
    else:
        limits.validate(trajectory.maximum_rates())
    return trajectory


@dataclass(frozen=True, slots=True)
class TargetProgramSegment:
    """One explicitly timed source within a Cartesian target program."""

    duration_s: float
    source: TargetSource

    def __post_init__(self):
        object.__setattr__(
            self,
            "duration_s",
            _positive_duration(self.duration_s, "duration_s"),
        )
        if not isinstance(self.source, TargetSource):
            raise TypeError("source must satisfy TargetSource")


@dataclass(frozen=True, slots=True)
class TargetProgram:
    """Sequence holds, waypoint trajectories, and future target shapes.

    All segments must declare the same frame and meet continuously in pose and
    twist.  Coordinate-frame changes therefore cannot be mistaken for ordinary
    trajectory continuity; build a separately converted program explicitly.
    """

    reference_frame: TargetFrame
    segments: tuple[TargetProgramSegment, ...]
    _end_times_s: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self):
        frame = TargetFrame(self.reference_frame)
        segments = tuple(self.segments)
        if not segments:
            raise ValueError("a target program needs at least one segment")
        if any(not isinstance(item, TargetProgramSegment) for item in segments):
            raise TypeError("segments must contain TargetProgramSegment values")
        for segment in segments:
            if TargetFrame(segment.source.reference_frame) != frame:
                raise ValueError(
                    "all target program segments must use its declared frame"
                )
        for index, (left, right) in enumerate(
            zip(segments, segments[1:])
        ):
            if not _samples_continuous(
                _kinematic_sample(left.source, left.duration_s),
                _kinematic_sample(right.source, 0.0),
            ):
                raise ValueError(
                    f"target program boundary {index} is not C2 continuous"
                )
        final = _kinematic_sample(
            segments[-1].source, segments[-1].duration_s
        )
        if (
            not _zero_twist(final.target)
            or not np.allclose(
                final.linear_acceleration_m_s2,
                0.0,
                rtol=0.0,
                atol=_PROGRAM_BOUNDARY_ATOL,
            )
            or not np.allclose(
                final.angular_acceleration_rad_s2,
                0.0,
                rtol=0.0,
                atol=_PROGRAM_BOUNDARY_ATOL,
            )
        ):
            raise ValueError(
                "the final program segment must end at rest"
            )

        end_times = []
        elapsed = 0.0
        for segment in segments:
            elapsed += segment.duration_s
            end_times.append(elapsed)
        object.__setattr__(self, "reference_frame", frame)
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "_end_times_s", tuple(end_times))

    @property
    def duration_s(self):
        return self._end_times_s[-1]

    @property
    def boundary_times_s(self):
        return self._end_times_s

    def sample_kinematics(self, elapsed_time_s):
        elapsed = _non_negative_time(elapsed_time_s, "elapsed_time_s")
        if elapsed >= self.duration_s:
            final = self.segments[-1]
            return _kinematic_sample(final.source, final.duration_s)
        index = bisect_right(self._end_times_s, elapsed)
        start = 0.0 if index == 0 else self._end_times_s[index - 1]
        return _kinematic_sample(
            self.segments[index].source, elapsed - start
        )

    def sample(self, elapsed_time_s):
        return self.sample_kinematics(elapsed_time_s).target

    def maximum_rates(self):
        bounds = [
            segment.source.maximum_rates()
            for segment in self.segments
            if hasattr(segment.source, "maximum_rates")
        ]
        if len(bounds) != len(self.segments):
            raise TypeError(
                "every program segment must provide maximum_rates"
            )
        return TrajectoryRateBounds(
            max(item.max_linear_speed_m_s for item in bounds),
            max(item.max_linear_acceleration_m_s2 for item in bounds),
            max(item.max_angular_speed_rad_s for item in bounds),
            max(item.max_angular_acceleration_rad_s2 for item in bounds),
        )


@dataclass(frozen=True, slots=True)
class PeriodicTargetSource:
    """Repeat one closed source with no pose or twist jump at the boundary."""

    source: TargetSource
    period_s: float

    def __post_init__(self):
        if not isinstance(self.source, TargetSource):
            raise TypeError("source must satisfy TargetSource")
        period = _positive_duration(self.period_s, "period_s")
        if not _samples_continuous(
            _kinematic_sample(self.source, 0.0),
            _kinematic_sample(self.source, period),
        ):
            raise ValueError(
                "a periodic target source must close continuously in "
                "frame, pose, twist, and acceleration"
            )
        object.__setattr__(self, "period_s", period)

    @property
    def reference_frame(self):
        return self.source.reference_frame

    def sample(self, elapsed_time_s):
        elapsed = _non_negative_time(elapsed_time_s, "elapsed_time_s")
        return self.source.sample(elapsed % self.period_s)

    def sample_kinematics(self, elapsed_time_s):
        elapsed = _non_negative_time(elapsed_time_s, "elapsed_time_s")
        return _kinematic_sample(
            self.source, elapsed % self.period_s
        )

    @property
    def boundary_times_s(self):
        return getattr(self.source, "boundary_times_s", (self.period_s,))

    def maximum_rates(self):
        return self.source.maximum_rates()


@dataclass(frozen=True, slots=True)
class IndependentArmTargetSource:
    """Compose independently framed right/left Cartesian sources."""

    right: TargetSource
    left: TargetSource

    def __post_init__(self):
        if not isinstance(self.right, TargetSource):
            raise TypeError("right must satisfy TargetSource")
        if not isinstance(self.left, TargetSource):
            raise TypeError("left must satisfy TargetSource")

    def sample(self, elapsed_time_s):
        elapsed = _non_negative_time(elapsed_time_s, "elapsed_time_s")
        return DualArmFramedTargets(
            right=self.right.sample(elapsed),
            left=self.left.sample(elapsed),
        )


@dataclass(frozen=True, slots=True)
class StaticDualArmTargetSource:
    """Compatibility adapter for retained ``DualArmFramedTargets``."""

    targets: DualArmFramedTargets

    def __post_init__(self):
        if not isinstance(self.targets, DualArmFramedTargets):
            raise TypeError("targets must be DualArmFramedTargets")
        for side in ("right", "left"):
            if not _zero_twist(self.targets.for_arm(side)):
                raise ValueError("static dual-arm targets must have zero twist")

    def sample(self, elapsed_time_s):
        _non_negative_time(elapsed_time_s, "elapsed_time_s")
        return self.targets


def as_dual_arm_target_source(value):
    """Adapt retained targets or accept an existing dual-arm source."""
    if isinstance(value, DualArmFramedTargets):
        return StaticDualArmTargetSource(value)
    if isinstance(value, DualArmTargetSource):
        return value
    raise TypeError(
        "target source must be DualArmFramedTargets or satisfy "
        "DualArmTargetSource"
    )


def sample_dual_arm_target_source(source, elapsed_time_s):
    """Sample and validate the source output at the control boundary."""
    elapsed = _non_negative_time(elapsed_time_s, "elapsed_time_s")
    result = source.sample(elapsed)
    if not isinstance(result, DualArmFramedTargets):
        raise TypeError("target source sample must return DualArmFramedTargets")
    return result
