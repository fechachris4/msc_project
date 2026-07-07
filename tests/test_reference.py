"""resolve_world is pure math — test it against hand-computed poses."""

import unittest

import numpy as np

from controller.desired_pos import resolve_world
from controller.transforms import rotation_from_rpy


class TestResolveWorld(unittest.TestCase):
    def test_composes_with_torso_pose(self):
        # Torso at (1, 0, 0.5), yawed +90°: torso-x maps to world-y.
        torso_pos = np.array([1.0, 0.0, 0.5])
        torso_rot = rotation_from_rpy([0.0, 0.0, np.pi / 2])
        pos, rot = resolve_world(
            [0.4, 0.0, 0.1], [0.0, 0.0, 0.0], (torso_pos, torso_rot)
        )
        np.testing.assert_allclose(pos, [1.0, 0.4, 0.6], atol=1e-12)
        np.testing.assert_allclose(rot, torso_rot, atol=1e-12)

    def test_rotations_compose(self):
        # Torso yawed +90°, reference yawed +90° in torso frame: world yaw 180°.
        torso_pos = np.zeros(3)
        torso_rot = rotation_from_rpy([0.0, 0.0, np.pi / 2])
        _, rot = resolve_world(
            [0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2], (torso_pos, torso_rot)
        )
        np.testing.assert_allclose(rot, rotation_from_rpy([0.0, 0.0, np.pi]), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
