"""Target-frame resolution is pure math — check hand-computed poses."""

import unittest

import numpy as np

from controller import frames
from controller.state import (
    ArmJointState,
    FramedTarget,
    MountCalibration,
    PlantState,
    Pose,
    TargetFrame,
    Twist,
)
from controller.transforms import rotation_from_rpy


def _plant(torso_pos, torso_rot):
    arm = ArmJointState(np.zeros(7), np.zeros(7))
    return PlantState(
        0.0,
        0.002,
        Pose(torso_pos, torso_rot),
        Twist.zero(),
        arm,
        arm,
    )


_IDENTITY_MOUNTS = MountCalibration(
    Pose(np.zeros(3), np.eye(3)),
    Pose(np.zeros(3), np.eye(3)),
)


class TestResolveWorld(unittest.TestCase):
    def test_composes_with_torso_pose(self):
        # Torso at (1, 0, 0.5), yawed +90°: torso-x maps to world-y.
        torso_pos = np.array([1.0, 0.0, 0.5])
        torso_rot = rotation_from_rpy([0.0, 0.0, np.pi / 2])
        result = frames.resolve_target_world(
            _plant(torso_pos, torso_rot),
            "right",
            _IDENTITY_MOUNTS,
            FramedTarget(
                TargetFrame.TORSO,
                Pose([0.4, 0.0, 0.1], np.eye(3)),
                Twist.zero(),
            ),
        )
        pos = result.pose_world.position_m
        rot = result.pose_world.rotation
        np.testing.assert_allclose(pos, [1.0, 0.4, 0.6], atol=1e-12)
        np.testing.assert_allclose(rot, torso_rot, atol=1e-12)

    def test_rotations_compose(self):
        # Torso yawed +90°, reference yawed +90° in torso frame: world yaw 180°.
        torso_pos = np.zeros(3)
        torso_rot = rotation_from_rpy([0.0, 0.0, np.pi / 2])
        result = frames.resolve_target_world(
            _plant(torso_pos, torso_rot),
            "right",
            _IDENTITY_MOUNTS,
            FramedTarget(
                TargetFrame.TORSO,
                Pose(
                    np.zeros(3),
                    rotation_from_rpy([0.0, 0.0, np.pi / 2]),
                ),
                Twist.zero(),
            ),
        )
        rot = result.pose_world.rotation
        np.testing.assert_allclose(rot, rotation_from_rpy([0.0, 0.0, np.pi]), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
