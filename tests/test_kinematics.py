import importlib
import sys
import types
import unittest

import numpy as np


def rotation_z(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


class KinematicsTest(unittest.TestCase):
    def setUp(self):
        self.previous_sim = sys.modules.get("sim")

        world = types.SimpleNamespace()
        world.right_ee_id = 0
        world.left_ee_id = 1
        world.torso_mocap_id = 0
        world.kinova_right_base_id = 1
        world.kinova_left_base_id = 2

        data = types.SimpleNamespace()
        data.site_xpos = np.array([
            [0.8, -0.3, 1.4],
            [0.0, 0.0, 0.0],
        ])
        data.site_xmat = np.array([
            rotation_z(-0.3).reshape(-1),
            np.eye(3).reshape(-1),
        ])
        data.xpos = np.array([
            [0.1, 0.2, 1.0],
            [0.25, -0.15, 1.2],
            [0.0, 0.0, 0.0],
        ])
        data.xmat = np.array([
            rotation_z(0.4).reshape(-1),
            rotation_z(0.9).reshape(-1),
            np.eye(3).reshape(-1),
        ])
        world.data = data

        sim = types.ModuleType("sim")
        sim.world = world
        sys.modules["sim"] = sim
        sys.modules.pop("controller.kinematics", None)

    def tearDown(self):
        sys.modules.pop("controller.kinematics", None)
        if self.previous_sim is None:
            sys.modules.pop("sim", None)
        else:
            sys.modules["sim"] = self.previous_sim

    def test_right_ee_positions_matches_direct_right_ee_pose(self):
        kinematics = importlib.import_module("controller.kinematics")

        pos, rot = kinematics.right_ee_positions()

        expected_pos, expected_rot = kinematics.direct_right_ee_pose()
        np.testing.assert_allclose(pos, expected_pos)
        np.testing.assert_allclose(rot, expected_rot)


class RotationHelpersTest(unittest.TestCase):
    def test_rotation_from_quat_matches_z_rotation(self):
        from controller.kinematics import rotation_from_quat
        # quat [w, x, y, z] for a rotation of 0.6 rad about z
        half = 0.3
        quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
        np.testing.assert_allclose(
            rotation_from_quat(quat), rotation_z(0.6), atol=1e-12
        )

    def test_rotation_about_axis_matches_z_rotation(self):
        from controller.kinematics import rotation_about_axis
        np.testing.assert_allclose(
            rotation_about_axis(np.array([0.0, 0.0, 1.0]), 0.6),
            rotation_z(0.6),
            atol=1e-12,
        )

    def test_rotation_about_arbitrary_axis_is_orthonormal(self):
        from controller.kinematics import rotation_about_axis
        axis = np.array([1.0, 2.0, -0.5])
        axis /= np.linalg.norm(axis)
        rot = rotation_about_axis(axis, 1.234)
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-12)
        np.testing.assert_allclose(rot @ axis, axis, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
