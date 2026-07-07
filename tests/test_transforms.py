"""controller.transforms vs Pinocchio — independent cross-validation.

Pinocchio is the library the C++ port will use, so agreement here is the
convention guarantee that matters: quaternion order [w, x, y, z],
Rodrigues rotation, and RPY composed as Rz @ Ry @ Rx.
"""

import unittest

import numpy as np
import pinocchio as pin

from controller.transforms import (
    pose_from_transform,
    rotation_about_axis,
    rotation_from_quat,
    rotation_from_rpy,
    transform_from_pose,
)

N_SAMPLES = 50
ATOL = 1e-12


class TransformsVsPinocchioTest(unittest.TestCase):
    def test_rotation_from_quat_matches_pinocchio(self):
        rng = np.random.default_rng(0)
        for _ in range(N_SAMPLES):
            quat = rng.standard_normal(4)
            quat /= np.linalg.norm(quat)
            w, x, y, z = quat
            np.testing.assert_allclose(
                rotation_from_quat(quat),
                pin.Quaternion(w, x, y, z).matrix(),
                atol=ATOL,
            )

    def test_rotation_about_axis_matches_pinocchio(self):
        rng = np.random.default_rng(1)
        for _ in range(N_SAMPLES):
            axis = rng.standard_normal(3)
            axis /= np.linalg.norm(axis)
            angle = rng.uniform(-np.pi, np.pi)
            np.testing.assert_allclose(
                rotation_about_axis(axis, angle),
                pin.exp3(axis * angle),
                atol=ATOL,
            )

    def test_rotation_from_rpy_matches_pinocchio(self):
        rng = np.random.default_rng(2)
        for _ in range(N_SAMPLES):
            roll, pitch, yaw = rng.uniform(-np.pi, np.pi, 3)
            np.testing.assert_allclose(
                rotation_from_rpy([roll, pitch, yaw]),
                pin.rpy.rpyToMatrix(roll, pitch, yaw),
                atol=ATOL,
            )


class PoseTransformRoundTripTest(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(3)
        for _ in range(N_SAMPLES):
            pos = rng.standard_normal(3)
            rot = rotation_from_rpy(rng.uniform(-np.pi, np.pi, 3))
            out_pos, out_rot = pose_from_transform(transform_from_pose(pos, rot))
            np.testing.assert_allclose(out_pos, pos, atol=ATOL)
            np.testing.assert_allclose(out_rot, rot, atol=ATOL)


if __name__ == "__main__":
    unittest.main()
