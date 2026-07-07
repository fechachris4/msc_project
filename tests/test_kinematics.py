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


class RotationHelpersTest(unittest.TestCase):
    def test_rotation_from_quat_matches_z_rotation(self):
        from controller.transforms import rotation_from_quat
        # quat [w, x, y, z] for a rotation of 0.6 rad about z
        half = 0.3
        quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
        np.testing.assert_allclose(
            rotation_from_quat(quat), rotation_z(0.6), atol=1e-12
        )

    def test_rotation_about_axis_matches_z_rotation(self):
        from controller.transforms import rotation_about_axis
        np.testing.assert_allclose(
            rotation_about_axis(np.array([0.0, 0.0, 1.0]), 0.6),
            rotation_z(0.6),
            atol=1e-12,
        )

    def test_rotation_about_arbitrary_axis_is_orthonormal(self):
        from controller.transforms import rotation_about_axis
        axis = np.array([1.0, 2.0, -0.5])
        axis /= np.linalg.norm(axis)
        rot = rotation_about_axis(axis, 1.234)
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-12)
        np.testing.assert_allclose(rot @ axis, axis, atol=1e-12)


class AnalyticalFKTest(unittest.TestCase):
    """Validate analytical FK against MuJoCo at random configurations.

    Independent computation: extract_chain reads only MjModel constants
    and qpos; MuJoCo's site_xpos/site_xmat is the reference.
    """

    N_SAMPLES = 50
    ATOL = 1e-9

    @classmethod
    def setUpClass(cls):
        import mujoco
        from sim import world
        cls.mujoco = mujoco
        cls.world = world

    def _random_qpos(self, rng, chain):
        model = self.world.model
        q = {}
        for jnt_id, adr in zip(chain.joint_ids, chain.qpos_adrs):
            if model.jnt_limited[jnt_id]:
                low, high = model.jnt_range[jnt_id]
            else:
                low, high = -np.pi, np.pi
            q[adr] = rng.uniform(low, high)
        return q

    def _check_arm(self, prefix):
        import controller.kinematics as kinematics
        mujoco = self.mujoco
        world = self.world
        chain = kinematics.extract_chain(
            world.model, prefix + "base_link", prefix + "pinch_site"
        )
        base_id = mujoco.mj_name2id(
            world.model, mujoco.mjtObj.mjOBJ_BODY, prefix + "base_link"
        )
        site_id = mujoco.mj_name2id(
            world.model, mujoco.mjtObj.mjOBJ_SITE, prefix + "pinch_site"
        )
        rng = np.random.default_rng(42)
        data = mujoco.MjData(world.model)
        for _ in range(self.N_SAMPLES):
            for adr, value in self._random_qpos(rng, chain).items():
                data.qpos[adr] = value
            mujoco.mj_kinematics(world.model, data)

            T_K_E = kinematics.fk(chain, data.qpos)

            base_pos = data.xpos[base_id]
            base_rot = data.xmat[base_id].reshape(3, 3)
            ee_pos_expected = data.site_xpos[site_id]
            ee_rot_expected = data.site_xmat[site_id].reshape(3, 3)

            np.testing.assert_allclose(
                base_rot @ T_K_E[:3, 3] + base_pos,
                ee_pos_expected,
                atol=self.ATOL,
            )
            np.testing.assert_allclose(
                base_rot @ T_K_E[:3, :3],
                ee_rot_expected,
                atol=self.ATOL,
            )

    def test_right_arm_fk_matches_mujoco(self):
        self._check_arm("right_")

    def test_left_arm_fk_matches_mujoco(self):
        self._check_arm("left_")

    def test_chain_has_seven_joints(self):
        import controller.kinematics as kinematics
        for prefix in ("right_", "left_"):
            chain = kinematics.extract_chain(
                self.world.model, prefix + "base_link", prefix + "pinch_site"
            )
            self.assertEqual(len(chain.joint_ids), 7)


class WorldFrameEETest(unittest.TestCase):
    """frames.right/left_ee_pose must match MuJoCo's world EE pose while
    reading the EE pose only on the comparison side."""

    def _check(self, ee_pose, site_id_name):
        import mujoco
        from sim import world

        site_id = mujoco.mj_name2id(
            world.model, mujoco.mjtObj.mjOBJ_SITE, site_id_name
        )
        rng = np.random.default_rng(7)
        for jnt_id in range(world.model.njnt):
            adr = world.model.jnt_qposadr[jnt_id]
            if world.model.jnt_limited[jnt_id]:
                low, high = world.model.jnt_range[jnt_id]
            else:
                low, high = -np.pi, np.pi
            world.data.qpos[adr] = rng.uniform(low, high)
        mujoco.mj_kinematics(world.model, world.data)

        pos, rot = ee_pose()
        np.testing.assert_allclose(
            pos, world.data.site_xpos[site_id], atol=1e-9
        )
        np.testing.assert_allclose(
            rot, world.data.site_xmat[site_id].reshape(3, 3), atol=1e-9
        )

    def _check_moved_torso(self, ee_pose, site_id_name):
        """Same comparison, but with the torso mocap body at random poses.

        The default torso rotation is identity, at which a transpose or
        missing torso-rotation factor is invisible; this exercises it."""
        import mujoco
        from sim import world

        site_id = mujoco.mj_name2id(
            world.model, mujoco.mjtObj.mjOBJ_SITE, site_id_name
        )
        mocap_idx = world.model.body_mocapid[world.torso_mocap_id]
        init_pos = world.data.mocap_pos[mocap_idx].copy()
        init_quat = world.data.mocap_quat[mocap_idx].copy()

        def restore():
            world.data.mocap_pos[mocap_idx] = init_pos
            world.data.mocap_quat[mocap_idx] = init_quat
            mujoco.mj_kinematics(world.model, world.data)

        self.addCleanup(restore)

        rng = np.random.default_rng(11)
        for _ in range(20):
            world.data.mocap_pos[mocap_idx] = (
                np.array([0.0, 0.0, 1.1]) + rng.uniform(-0.5, 0.5, 3)
            )
            quat = rng.standard_normal(4)
            world.data.mocap_quat[mocap_idx] = quat / np.linalg.norm(quat)
            for jnt_id in range(world.model.njnt):
                adr = world.model.jnt_qposadr[jnt_id]
                if world.model.jnt_limited[jnt_id]:
                    low, high = world.model.jnt_range[jnt_id]
                else:
                    low, high = -np.pi, np.pi
                world.data.qpos[adr] = rng.uniform(low, high)
            mujoco.mj_kinematics(world.model, world.data)

            pos, rot = ee_pose()
            np.testing.assert_allclose(
                pos, world.data.site_xpos[site_id], atol=1e-9
            )
            np.testing.assert_allclose(
                rot, world.data.site_xmat[site_id].reshape(3, 3), atol=1e-9
            )

    def test_right_ee_pose_matches_mujoco(self):
        from controller import frames
        self._check(frames.right_ee_pose, "right_pinch_site")

    def test_left_ee_pose_matches_mujoco(self):
        from controller import frames
        self._check(frames.left_ee_pose, "left_pinch_site")

    def test_right_ee_pose_matches_mujoco_under_moved_torso(self):
        from controller import frames
        self._check_moved_torso(frames.right_ee_pose, "right_pinch_site")

    def test_left_ee_pose_matches_mujoco_under_moved_torso(self):
        from controller import frames
        self._check_moved_torso(frames.left_ee_pose, "left_pinch_site")


if __name__ == "__main__":
    unittest.main()
