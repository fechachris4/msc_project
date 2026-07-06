"""resolve_world is pure math — test it against hand-computed poses."""

import unittest

import numpy as np

from controller.reference import resolve_world
from controller.transforms import rotation_from_rpy


class TestResolveWorld(unittest.TestCase):
    def test_world_frame_is_passthrough(self):
        ref = {"frame": "world", "pos": [1.0, 2.0, 3.0], "rpy": [0.1, 0.2, 0.3]}
        pos, rot = resolve_world(ref)
        np.testing.assert_allclose(pos, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(rot, rotation_from_rpy([0.1, 0.2, 0.3]))

    def test_frame_defaults_to_world(self):
        pos, _ = resolve_world({"pos": [0.5, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]})
        np.testing.assert_allclose(pos, [0.5, 0.0, 0.0])

    def test_torso_frame_composes_with_torso_pose(self):
        # Torso at (1, 0, 0.5), yawed +90°: torso-x maps to world-y.
        torso_pos = np.array([1.0, 0.0, 0.5])
        torso_rot = rotation_from_rpy([0.0, 0.0, np.pi / 2])
        ref = {"frame": "torso", "pos": [0.4, 0.0, 0.1], "rpy": [0.0, 0.0, 0.0]}
        pos, rot = resolve_world(ref, (torso_pos, torso_rot))
        np.testing.assert_allclose(pos, [1.0, 0.4, 0.6], atol=1e-12)
        np.testing.assert_allclose(rot, torso_rot, atol=1e-12)

    def test_torso_frame_requires_torso_pose(self):
        ref = {"frame": "torso", "pos": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]}
        with self.assertRaises(ValueError):
            resolve_world(ref)

    def test_unknown_frame_raises(self):
        ref = {"frame": "hand", "pos": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]}
        with self.assertRaises(ValueError):
            resolve_world(ref)


if __name__ == "__main__":
    unittest.main()
