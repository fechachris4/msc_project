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


@dataclass(frozen=True, slots=True)
class ArmControllerState:
    """World-aligned controller input derived from one explicit plant sample."""

    joints: ArmJointState
    ee_pose_world: Pose
    ee_twist_world: Twist
    jacobian_world: np.ndarray

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
