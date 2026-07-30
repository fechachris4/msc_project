"""Pin the production composition root that main.py actually runs.

Every other test file exercises single-arm pieces; these tests call
``arm_flow.build_flows`` itself, so a regression confined to the two-arm
assembly (swapped arm slots, snapshot taken before postures, the
planning-vs-trajectory clash guard, config threading into the planner)
cannot pass the suite unnoticed.
"""

import unittest
from dataclasses import replace

import numpy as np

import arm_flow
from controller import desired_pos, frames
from controller.state import Twist
from controller.transforms import rotation_from_rpy
from planning import planner
from runtime_config import CONFIG
from sim import world


def _fresh_default_plant():
    """The un-posed reset state, for proving posture ordering matters."""
    arm_flow.apply_initial_postures(world.backend, {})
    return world.backend.read_state(Twist.zero())


class BuildFlowsCompositionTest(unittest.TestCase):
    def setUp(self):
        world.backend.configure_torso_driver(None, None)

    def tearDown(self):
        world.backend.configure_torso_driver(None, None)

    def test_shipped_config_composes_both_arms_independently(self):
        default_states = frames.controller_states(
            _fresh_default_plant(), world.MOUNT_CALIBRATION
        )
        static_targets = desired_pos.configured_targets()
        flow = arm_flow.build_flows(
            world.backend, world.MOUNT_CALIBRATION, static_targets
        )
        self.assertEqual(flow.right.kind, "routed_reach")
        self.assertEqual(flow.left.kind, "trajectory")

        # Correct slotting: the dual-arm source must deliver each arm's
        # OWN composed source, not the other arm's.
        flow.left.look_at_object.apply(0.0)
        sampled = flow.source.sample(0.0)
        np.testing.assert_allclose(
            sampled.right.pose.position_m,
            flow.right.source.sample(0.0).pose.position_m,
            rtol=0.0, atol=0.0,
        )
        np.testing.assert_allclose(
            sampled.left.pose.position_m,
            flow.left.source.sample(0.0).pose.position_m,
            rtol=0.0, atol=0.0,
        )

        # Posture ordering: the left trajectory starts at the POSED
        # measured pose, which build_flows must establish before its one
        # shared snapshot.  The backend still holds that posed state.
        posed_states = frames.controller_states(
            world.backend.read_state(Twist.zero()),
            world.MOUNT_CALIBRATION,
        )
        np.testing.assert_allclose(
            sampled.left.pose.position_m,
            posed_states.left.ee_pose_world.position_m,
            rtol=0.0, atol=1e-9,
        )
        # Non-vacuous: the configured posture moves the left arm away
        # from the reset state, so a snapshot-before-postures bug cannot
        # produce the same start pose.
        self.assertGreater(
            float(np.linalg.norm(
                posed_states.left.ee_pose_world.position_m
                - default_states.left.ee_pose_world.position_m
            )),
            0.01,
        )

    def test_routed_reach_interpolates_to_the_goal_orientation(self):
        static_targets = desired_pos.configured_targets()
        flow = arm_flow.build_flows(
            world.backend, world.MOUNT_CALIBRATION, static_targets
        )
        self.assertEqual(flow.right.kind, "routed_reach")
        goal_rotation = rotation_from_rpy(CONFIG.target("right").rpy_rad)
        start_rotation = flow.right.source.sample(0.0).pose.rotation
        end_rotation = flow.right.source.sample(
            flow.right.duration_s
        ).pose.rotation
        np.testing.assert_allclose(
            end_rotation, goal_rotation, rtol=0.0, atol=1e-9
        )
        # Non-vacuous: the shipped goal RPY differs from the measured
        # start orientation, so a reversed or dropped blend cannot pass.
        self.assertGreater(
            float(np.max(np.abs(goal_rotation - start_rotation))), 0.1
        )

    def test_planning_and_trajectory_on_one_arm_is_refused(self):
        config = replace(
            CONFIG,
            planning=replace(CONFIG.planning, enabled=True, arm="left"),
        )
        with self.assertRaisesRegex(ValueError, "both drive"):
            arm_flow.build_flows(
                world.backend,
                world.MOUNT_CALIBRATION,
                desired_pos.configured_targets(),
                config=config,
            )

    def test_planning_one_arm_leaves_the_other_arm_trajectory(self):
        config = replace(
            CONFIG, planning=replace(CONFIG.planning, enabled=True)
        )
        flow = arm_flow.build_flows(
            world.backend,
            world.MOUNT_CALIBRATION,
            desired_pos.configured_targets(),
            config=config,
        )
        self.assertEqual(flow.right.kind, "planned")
        self.assertEqual(flow.left.kind, "trajectory")
        self.assertTrue(flow.right.plan.result.success)

    def test_plan_arm_uses_the_injected_target_config(self):
        plant = world.backend.read_state(Twist.zero())
        custom_target = replace(
            CONFIG.target("right"), position_m=(0.50, -0.30, 0.90)
        )
        plan = planner.plan_arm(
            plant,
            world.MOUNT_CALIBRATION,
            "right",
            CONFIG.planning,
            CONFIG.reactive_pose,
            target_config=custom_target,
        )
        expected = planner.goal_pose_world(
            plant, "right", world.MOUNT_CALIBRATION, custom_target
        )
        np.testing.assert_allclose(
            plan.goal_pose_world.position_m,
            expected.position_m,
            rtol=0.0, atol=0.0,
        )
        np.testing.assert_allclose(
            plan.goal_pose_world.position_m, (0.50, -0.30, 0.90),
            rtol=0.0, atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
