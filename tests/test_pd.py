"""Pose error in controller.pd.

Pure math: zero at coincidence, known injected offsets recovered exactly.
Wrapper integration: drive the real pipeline (target mocap -> pd wrappers
-> frames FK) at randomized arm configurations.
"""

import unittest

import mujoco
import numpy as np

from controller.transforms import rotation_about_axis, rotation_from_rpy

N_SAMPLES = 20
PURE_TOL = 1e-12
WRAP_TOL = 1e-9


def random_axis_angle(rng):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(-0.95 * np.pi, 0.95 * np.pi)
    return axis, angle


class PoseErrorPureTest(unittest.TestCase):
    def test_zero_at_coincidence(self):
        from controller import pd

        rng = np.random.default_rng(7)
        for _ in range(N_SAMPLES):
            pos = rng.uniform(-1.0, 1.0, 3)
            rot = rotation_from_rpy(rng.uniform(-np.pi, np.pi, 3))
            np.testing.assert_allclose(
                pd.position_error(pos, pos), np.zeros(3), atol=PURE_TOL
            )
            np.testing.assert_allclose(
                pd.rotation_error(rot, rot), np.zeros(3), atol=PURE_TOL
            )

    def test_known_offset_recovered(self):
        from controller import pd

        rng = np.random.default_rng(8)
        for _ in range(N_SAMPLES):
            ee_pos = rng.uniform(-1.0, 1.0, 3)
            ee_rot = rotation_from_rpy(rng.uniform(-np.pi, np.pi, 3))
            delta_pos = rng.uniform(-0.5, 0.5, 3)
            axis, angle = random_axis_angle(rng)

            ref_pos = ee_pos + delta_pos
            ref_rot = rotation_about_axis(axis, angle) @ ee_rot

            np.testing.assert_allclose(
                pd.position_error(ref_pos, ee_pos), delta_pos, atol=PURE_TOL
            )
            np.testing.assert_allclose(
                pd.rotation_error(ref_rot, ee_rot), axis * angle, atol=PURE_TOL
            )


class PoseErrorWrapperTest(unittest.TestCase):
    def _randomize_arm(self, rng, prefix):
        from sim import world

        for i in range(1, 8):
            jnt_id = mujoco.mj_name2id(
                world.model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}joint_{i}"
            )
            if world.model.jnt_limited[jnt_id]:
                low, high = world.model.jnt_range[jnt_id]
            else:
                low, high = -np.pi, np.pi
            world.data.qpos[world.model.jnt_qposadr[jnt_id]] = rng.uniform(
                low, high
            )

    def _check_arm(self, rng, ee_pose, set_pos, set_quat, pose_error):
        ee_pos, ee_rot = ee_pose()

        # Target at the FK pose -> zero error.
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, ee_rot.flatten())
        set_pos(ee_pos)
        set_quat(quat)
        e_pos, e_rot = pose_error()
        np.testing.assert_allclose(e_pos, np.zeros(3), atol=WRAP_TOL)
        np.testing.assert_allclose(e_rot, np.zeros(3), atol=WRAP_TOL)

        # Known offset -> recovered.
        delta_pos = rng.uniform(-0.2, 0.2, 3)
        axis, angle = random_axis_angle(rng)
        mujoco.mju_mat2Quat(
            quat, (rotation_about_axis(axis, angle) @ ee_rot).flatten()
        )
        set_pos(ee_pos + delta_pos)
        set_quat(quat)
        e_pos, e_rot = pose_error()
        np.testing.assert_allclose(e_pos, delta_pos, atol=WRAP_TOL)
        np.testing.assert_allclose(e_rot, axis * angle, atol=WRAP_TOL)

    def test_wrappers_recover_offsets(self):
        from controller import frames, pd
        from sim import targets, world

        rng = np.random.default_rng(9)
        for _ in range(5):
            self._randomize_arm(rng, "right_")
            self._randomize_arm(rng, "left_")
            mujoco.mj_kinematics(world.model, world.data)

            self._check_arm(
                rng, frames.right_ee_pose, targets.set_right_target,
                targets.set_right_target_quat, pd.right_pose_error,
            )
            self._check_arm(
                rng, frames.left_ee_pose, targets.set_left_target,
                targets.set_left_target_quat, pd.left_pose_error,
            )


if __name__ == "__main__":
    unittest.main()
