import unittest

import mujoco
import numpy as np

from controller import frames
from controller.state import (
    ArmJointState,
    DualArmFramedTargets,
    FramedTarget,
    MountCalibration,
    PlantState,
    Pose,
    TargetFrame,
    Twist,
)
from controller.transforms import rotation_from_rpy
from sim import world
from tests.control_test_support import apply_cycle, reconstruct_pipeline


def _arm():
    return ArmJointState(np.zeros(7), np.zeros(7))


def _plant():
    return PlantState(
        sample_time_s=1.0,
        nominal_dt_s=0.002,
        torso_pose_world=Pose(
            np.array([1.0, 2.0, 3.0]),
            rotation_from_rpy([0.0, 0.0, np.pi / 2.0]),
        ),
        torso_twist_world=Twist(
            np.array([0.1, 0.2, 0.3]),
            np.array([0.0, 0.0, 2.0]),
        ),
        right=_arm(),
        left=_arm(),
    )


class FixedStateContractTest(unittest.TestCase):
    def test_arrays_are_copied_fixed_shape_and_read_only(self):
        position = np.array([1.0, 2.0, 3.0])
        pose = Pose(position, np.eye(3))
        position[:] = 9.0
        np.testing.assert_array_equal(pose.position_m, [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            pose.position_m[0] = 4.0
        with self.assertRaisesRegex(ValueError, "shape"):
            ArmJointState(np.zeros(6), np.zeros(7))

    def test_mujoco_reader_copies_state(self):
        mujoco.mj_forward(world.model, world.data)
        plant = world.read_state(Twist.zero())
        before = plant.right.position_rad.copy()
        address = world.qpos_adrs["right"][0]
        original = float(world.data.qpos[address])
        try:
            world.data.qpos[address] += 0.1
            np.testing.assert_array_equal(plant.right.position_rad, before)
        finally:
            world.data.qpos[address] = original
            mujoco.mj_forward(world.model, world.data)


class TargetFrameBoundaryTest(unittest.TestCase):
    def setUp(self):
        identity = Pose(np.zeros(3), np.eye(3))
        self.identity_calibration = MountCalibration(identity, identity)

    def test_world_target_passes_through(self):
        target = FramedTarget(
            TargetFrame.WORLD,
            Pose(np.array([0.4, -0.2, 0.7]), np.eye(3)),
            Twist(np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])),
        )
        resolved = frames.resolve_target_world(
            _plant(), "right", self.identity_calibration, target)
        np.testing.assert_allclose(
            resolved.pose_world.position_m, target.pose.position_m, atol=1e-12)
        np.testing.assert_allclose(
            resolved.pose_world.rotation, target.pose.rotation, atol=1e-12)
        np.testing.assert_allclose(
            resolved.twist_world.linear_m_s,
            target.twist.linear_m_s,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            resolved.twist_world.angular_rad_s,
            target.twist.angular_rad_s,
            atol=1e-12,
        )

    def test_live_torso_target_includes_frame_transport_twist(self):
        target = FramedTarget(
            TargetFrame.TORSO,
            Pose(np.array([1.0, 0.0, 0.0]), np.eye(3)),
            Twist(np.array([0.5, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        )
        resolved = frames.resolve_target_world(
            _plant(), "right", self.identity_calibration, target)
        np.testing.assert_allclose(
            resolved.pose_world.position_m, [1.0, 3.0, 3.0], atol=1e-12)
        np.testing.assert_allclose(
            resolved.twist_world.linear_m_s, [-1.9, 0.7, 0.3], atol=1e-12)
        np.testing.assert_allclose(
            resolved.twist_world.angular_rad_s, [-1.0, 0.0, 2.0], atol=1e-12)

    def test_base_target_uses_selected_arm_mount(self):
        right_mount = Pose(np.array([0.2, 0.0, 0.0]), np.eye(3))
        calibration = MountCalibration(
            right_mount, Pose(np.array([-0.4, 0.0, 0.0]), np.eye(3)))
        target = FramedTarget(
            TargetFrame.BASE,
            Pose(np.array([0.3, 0.0, 0.0]), np.eye(3)),
            Twist.zero(),
        )
        resolved = frames.resolve_target_world(
            _plant(), "right", calibration, target)
        np.testing.assert_allclose(
            resolved.pose_world.position_m, [1.0, 2.5, 3.0], atol=1e-12)
        np.testing.assert_allclose(
            resolved.twist_world.linear_m_s, [-0.9, 0.2, 0.3], atol=1e-12)
        np.testing.assert_allclose(
            resolved.twist_world.angular_rad_s, [0.0, 0.0, 2.0], atol=1e-12)

    def test_control_path_re_resolves_live_targets_each_cycle(self):
        mujoco.mj_resetData(world.model, world.data)
        mujoco.mj_forward(world.model, world.data)
        mocap_index = world.model.body_mocapid[world.torso_body_id]
        original_position = world.data.mocap_pos[mocap_index].copy()
        original_quaternion = world.data.mocap_quat[mocap_index].copy()

        live_targets = DualArmFramedTargets(
            right=FramedTarget(
                TargetFrame.TORSO,
                Pose(np.array([0.4, -0.2, 0.1]), np.eye(3)),
                Twist.zero(),
            ),
            left=FramedTarget(
                TargetFrame.BASE,
                Pose(np.array([0.3, 0.0, 0.0]), np.eye(3)),
                Twist.zero(),
            ),
        )
        world_sources = DualArmFramedTargets(
            right=FramedTarget(
                TargetFrame.WORLD,
                Pose(np.array([0.4, -0.2, 1.2]), np.eye(3)),
                Twist.zero(),
            ),
            left=FramedTarget(
                TargetFrame.WORLD,
                Pose(np.array([0.4, 0.2, 1.2]), np.eye(3)),
                Twist.zero(),
            ),
        )
        try:
            plant_before = world.read_state(Twist.zero())
            live_before = frames.resolve_targets_world(
                plant_before, world.MOUNT_CALIBRATION, live_targets)
            fixed_before = frames.resolve_targets_world(
                plant_before, world.MOUNT_CALIBRATION, world_sources)

            delta = np.array([0.12, -0.04, 0.03])
            world.data.mocap_pos[mocap_index] += delta
            mujoco.mj_kinematics(world.model, world.data)
            plant_after = world.read_state(Twist.zero())
            live_after = frames.resolve_targets_world(
                plant_after, world.MOUNT_CALIBRATION, live_targets)
            fixed_after = frames.resolve_targets_world(
                plant_after, world.MOUNT_CALIBRATION, world_sources)

            for side in world.SIDES:
                np.testing.assert_allclose(
                    live_after.for_arm(side).pose_world.position_m
                    - live_before.for_arm(side).pose_world.position_m,
                    delta,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    fixed_after.for_arm(side).pose_world.position_m,
                    fixed_before.for_arm(side).pose_world.position_m,
                    atol=1e-12,
                )

            pipeline = reconstruct_pipeline()
            traces = apply_cycle(
                pipeline,
                world.model.opt.timestep,
                (np.zeros(3), np.zeros(3)),
                world_targets=live_after,
            )
            right_state = frames.arm_controller_state(
                plant_after, "right", world.MOUNT_CALIBRATION)
            np.testing.assert_allclose(
                traces["right"].e_pos,
                live_after.right.pose_world.position_m
                - right_state.ee_pose_world.position_m,
                atol=1e-12,
            )
        finally:
            world.data.mocap_pos[mocap_index] = original_position
            world.data.mocap_quat[mocap_index] = original_quaternion
            mujoco.mj_resetData(world.model, world.data)
            mujoco.mj_forward(world.model, world.data)


class ExplicitFrameGroundTruthTest(unittest.TestCase):
    def test_controller_state_pose_and_jacobian_match_mujoco(self):
        mujoco.mj_forward(world.model, world.data)
        plant = world.read_state(Twist.zero())
        jacp = np.zeros((3, world.model.nv))
        jacr = np.zeros((3, world.model.nv))
        for side in world.SIDES:
            state = frames.arm_controller_state(
                plant, side, world.MOUNT_CALIBRATION)
            measured = world.measured_ee_pose(side)
            np.testing.assert_allclose(
                state.ee_pose_world.position_m,
                measured.position_m,
                atol=1e-9,
            )
            np.testing.assert_allclose(
                state.ee_pose_world.rotation,
                measured.rotation,
                atol=1e-9,
            )
            mujoco.mj_jacSite(
                world.model,
                world.data,
                jacp,
                jacr,
                world.ee_site_id[side],
            )
            expected = np.vstack(
                [jacp[:, world.dof_adrs[side]],
                 jacr[:, world.dof_adrs[side]]]
            )
            np.testing.assert_allclose(
                state.jacobian_world, expected, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
