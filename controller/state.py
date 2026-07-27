"""Fixed-shape control data contracts.

These records are deliberately plain: each maps directly to a C++ struct.
All Cartesian quantities are expressed in the world frame after the frame
boundary, and every array is copied and made read-only at construction.
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


ARMS = ("right", "left")


def _array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite shape-{shape} array")
    result = array.copy()
    return np.frombuffer(result.tobytes(), dtype=result.dtype).reshape(
        result.shape
    )


@dataclass(frozen=True, slots=True)
class Pose:
    position_m: np.ndarray
    rotation: np.ndarray

    def __post_init__(self):
        position = _array(self.position_m, (3,), "position_m")
        rotation = _array(self.rotation, (3, 3), "rotation")
        if (
            not np.allclose(
                rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-9
            )
            or not np.isclose(
                np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-9
            )
        ):
            raise ValueError("rotation must be a proper orthonormal matrix")
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "rotation", rotation)


@dataclass(frozen=True, slots=True)
class Twist:
    linear_m_s: np.ndarray
    angular_rad_s: np.ndarray

    def __post_init__(self):
        object.__setattr__(
            self, "linear_m_s", _array(self.linear_m_s, (3,), "linear_m_s")
        )
        object.__setattr__(
            self,
            "angular_rad_s",
            _array(self.angular_rad_s, (3,), "angular_rad_s"),
        )

    @classmethod
    def zero(cls):
        return cls(np.zeros(3), np.zeros(3))


@dataclass(frozen=True, slots=True)
class ArmJointState:
    position_rad: np.ndarray
    velocity_rad_s: np.ndarray

    def __post_init__(self):
        object.__setattr__(
            self,
            "position_rad",
            _array(self.position_rad, (7,), "position_rad"),
        )
        object.__setattr__(
            self,
            "velocity_rad_s",
            _array(self.velocity_rad_s, (7,), "velocity_rad_s"),
        )


@dataclass(frozen=True, slots=True)
class PlantState:
    sample_time_s: float
    nominal_dt_s: float
    torso_pose_world: Pose
    torso_twist_world: Twist
    right: ArmJointState
    left: ArmJointState

    def __post_init__(self):
        sample_time = float(self.sample_time_s)
        nominal_dt = float(self.nominal_dt_s)
        if not np.isfinite(sample_time) or sample_time < 0.0:
            raise ValueError("sample_time_s must be finite and non-negative")
        if not np.isfinite(nominal_dt) or nominal_dt <= 0.0:
            raise ValueError("nominal_dt_s must be finite and positive")
        object.__setattr__(self, "sample_time_s", sample_time)
        object.__setattr__(self, "nominal_dt_s", nominal_dt)
        if not isinstance(self.torso_pose_world, Pose):
            raise TypeError("torso_pose_world must be a Pose")
        if not isinstance(self.torso_twist_world, Twist):
            raise TypeError("torso_twist_world must be a Twist")
        if not isinstance(self.right, ArmJointState):
            raise TypeError("right must be an ArmJointState")
        if not isinstance(self.left, ArmJointState):
            raise TypeError("left must be an ArmJointState")

    def arm(self, side):
        if side == "right":
            return self.right
        if side == "left":
            return self.left
        raise ValueError(f"unknown arm: {side!r}")


class TargetFrame(StrEnum):
    WORLD = "world"
    BASE = "base"
    TORSO = "torso"


@dataclass(frozen=True, slots=True)
class FramedTarget:
    """Target pose/twist relative to and expressed in ``reference_frame``."""

    reference_frame: TargetFrame
    pose: Pose
    twist: Twist

    def __post_init__(self):
        object.__setattr__(
            self, "reference_frame", TargetFrame(self.reference_frame)
        )
        if not isinstance(self.pose, Pose):
            raise TypeError("pose must be a Pose")
        if not isinstance(self.twist, Twist):
            raise TypeError("twist must be a Twist")


@dataclass(frozen=True, slots=True)
class WorldTarget:
    """Controller-facing Cartesian target; pose and twist are world-aligned."""

    pose_world: Pose
    twist_world: Twist

    def __post_init__(self):
        if not isinstance(self.pose_world, Pose):
            raise TypeError("pose_world must be a Pose")
        if not isinstance(self.twist_world, Twist):
            raise TypeError("twist_world must be a Twist")


@dataclass(frozen=True, slots=True)
class DualArmFramedTargets:
    right: FramedTarget
    left: FramedTarget

    def for_arm(self, side):
        if side == "right":
            return self.right
        if side == "left":
            return self.left
        raise ValueError(f"unknown arm: {side!r}")


@dataclass(frozen=True, slots=True)
class DualArmWorldTargets:
    right: WorldTarget
    left: WorldTarget

    def for_arm(self, side):
        if side == "right":
            return self.right
        if side == "left":
            return self.left
        raise ValueError(f"unknown arm: {side!r}")


@dataclass(frozen=True, slots=True)
class MountCalibration:
    """Fixed poses T_T_B for the right and left arm base frames."""

    right_torso_to_base: Pose
    left_torso_to_base: Pose

    def __post_init__(self):
        if not isinstance(self.right_torso_to_base, Pose):
            raise TypeError("right_torso_to_base must be a Pose")
        if not isinstance(self.left_torso_to_base, Pose):
            raise TypeError("left_torso_to_base must be a Pose")

    def for_arm(self, side):
        if side == "right":
            return self.right_torso_to_base
        if side == "left":
            return self.left_torso_to_base
        raise ValueError(f"unknown arm: {side!r}")


def _row_array(value, trailing_shape, name, dtype=float):
    array = np.asarray(value, dtype=dtype)
    if (
        array.ndim != len(trailing_shape) + 1
        or array.shape[1:] != trailing_shape
        or (dtype is float and not np.all(np.isfinite(array)))
    ):
        raise ValueError(
            f"{name} must have shape (N, {', '.join(map(str, trailing_shape))})"
        )
    result = array.copy()
    return np.frombuffer(result.tobytes(), dtype=result.dtype).reshape(
        result.shape
    )


@dataclass(frozen=True, slots=True)
class LinkSafetyPoints:
    """Fixed arrays for all conservative arm spheres in one plant sample."""

    names: tuple[str, ...]
    frame_names: tuple[str, ...]
    position_world_m: np.ndarray
    jacobian_world_m_rad: np.ndarray
    radius_m: np.ndarray
    mount_exempt: np.ndarray

    def __post_init__(self):
        names = tuple(self.names)
        frames = tuple(self.frame_names)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("names must contain non-empty strings")
        if any(not isinstance(name, str) or not name for name in frames):
            raise ValueError("frame_names must contain non-empty strings")
        count = len(names)
        if len(frames) != count:
            raise ValueError("names and frame_names must have equal length")
        position = _row_array(
            self.position_world_m, (3,), "position_world_m"
        )
        jacobian = _row_array(
            self.jacobian_world_m_rad,
            (3, 7),
            "jacobian_world_m_rad",
        )
        radius = np.asarray(self.radius_m, dtype=float)
        exempt = np.asarray(self.mount_exempt, dtype=bool)
        if (
            position.shape[0] != count
            or jacobian.shape[0] != count
            or radius.shape != (count,)
            or exempt.shape != (count,)
        ):
            raise ValueError("all LinkSafetyPoints fields must have N rows")
        if not np.all(np.isfinite(radius)) or np.any(radius <= 0.0):
            raise ValueError("radius_m must be finite and positive")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "frame_names", frames)
        object.__setattr__(self, "position_world_m", position)
        object.__setattr__(self, "jacobian_world_m_rad", jacobian)
        object.__setattr__(
            self,
            "radius_m",
            np.frombuffer(radius.copy().tobytes(), dtype=radius.dtype),
        )
        object.__setattr__(
            self,
            "mount_exempt",
            np.frombuffer(exempt.copy().tobytes(), dtype=exempt.dtype),
        )

    def __len__(self):
        return len(self.names)


@dataclass(frozen=True, slots=True)
class HumanDistanceConstraints:
    """Vectorised distance and joint-rate sensitivity for all arm spheres."""

    point_names: tuple[str, ...]
    signed_clearance_m: np.ndarray
    distance_jacobian_m_rad: np.ndarray
    active: np.ndarray
    mount_exempt: np.ndarray

    def __post_init__(self):
        names = tuple(self.point_names)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("point_names must contain non-empty strings")
        count = len(names)
        clearance = np.asarray(self.signed_clearance_m, dtype=float)
        jacobian = _row_array(
            self.distance_jacobian_m_rad,
            (7,),
            "distance_jacobian_m_rad",
        )
        active = np.asarray(self.active, dtype=bool)
        exempt = np.asarray(self.mount_exempt, dtype=bool)
        if (
            clearance.shape != (count,)
            or jacobian.shape[0] != count
            or active.shape != (count,)
            or exempt.shape != (count,)
        ):
            raise ValueError(
                "all HumanDistanceConstraints fields must have N rows"
            )
        if not np.all(np.isfinite(clearance)):
            raise ValueError("signed_clearance_m must be finite")
        object.__setattr__(self, "point_names", names)
        for name, value in (
            ("signed_clearance_m", clearance),
            ("distance_jacobian_m_rad", jacobian),
            ("active", active),
            ("mount_exempt", exempt),
        ):
            copied = np.asarray(value).copy()
            object.__setattr__(
                self,
                name,
                np.frombuffer(
                    copied.tobytes(), dtype=copied.dtype
                ).reshape(copied.shape),
            )


@dataclass(frozen=True, slots=True)
class ArmHumanSafetyState:
    constraints: HumanDistanceConstraints | None = None

    def __post_init__(self):
        if (
            self.constraints is not None
            and not isinstance(
                self.constraints, HumanDistanceConstraints
            )
        ):
            raise TypeError(
                "constraints must be HumanDistanceConstraints or None"
            )

    @property
    def minimum_clearance_m(self):
        if self.constraints is None:
            return float("inf")
        protected = self.constraints.signed_clearance_m[
            ~self.constraints.mount_exempt
        ]
        return float(np.min(protected)) if protected.size else float("inf")


@dataclass(frozen=True, slots=True)
class DualArmHumanSafetyStates:
    right: ArmHumanSafetyState
    left: ArmHumanSafetyState

    def for_arm(self, side):
        if side == "right":
            return self.right
        if side == "left":
            return self.left
        raise ValueError(f"unknown arm: {side!r}")


@dataclass(frozen=True, slots=True)
class ArmControllerState:
    """World-aligned controller input derived from one explicit plant sample."""

    joints: ArmJointState
    ee_pose_world: Pose
    ee_twist_world: Twist
    jacobian_world: np.ndarray
    link_safety_points: LinkSafetyPoints | None = None

    def __post_init__(self):
        if not isinstance(self.joints, ArmJointState):
            raise TypeError("joints must be an ArmJointState")
        if not isinstance(self.ee_pose_world, Pose):
            raise TypeError("ee_pose_world must be a Pose")
        if not isinstance(self.ee_twist_world, Twist):
            raise TypeError("ee_twist_world must be a Twist")
        object.__setattr__(
            self,
            "jacobian_world",
            _array(self.jacobian_world, (6, 7), "jacobian_world"),
        )
        if (
            self.link_safety_points is not None
            and not isinstance(self.link_safety_points, LinkSafetyPoints)
        ):
            raise TypeError(
                "link_safety_points must be LinkSafetyPoints or None"
            )


@dataclass(frozen=True, slots=True)
class DualArmControllerStates:
    right: ArmControllerState
    left: ArmControllerState

    def for_arm(self, side):
        if side == "right":
            return self.right
        if side == "left":
            return self.left
        raise ValueError(f"unknown arm: {side!r}")


@dataclass(frozen=True, slots=True)
class JointPositionCommand:
    """Backend-facing position command in radians for both seven-DOF arms."""

    right_position_rad: np.ndarray
    left_position_rad: np.ndarray

    def __post_init__(self):
        object.__setattr__(
            self,
            "right_position_rad",
            _array(self.right_position_rad, (7,), "right_position_rad"),
        )
        object.__setattr__(
            self,
            "left_position_rad",
            _array(self.left_position_rad, (7,), "left_position_rad"),
        )

    def for_arm(self, side):
        if side == "right":
            return self.right_position_rad
        if side == "left":
            return self.left_position_rad
        raise ValueError(f"unknown arm: {side!r}")
