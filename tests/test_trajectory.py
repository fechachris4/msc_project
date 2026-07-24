from dataclasses import replace
import unittest

import mujoco
import numpy as np

from controller import frames
from controller.runner import ReactivePositionRunner
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
from controller.trajectory import (
    CartesianWaypoint,
    CartesianWaypointTrajectory,
    CircleTrajectory,
    HoldTrajectory,
    IndependentArmTargetSource,
    PeriodicTargetSource,
    StaticDualArmTargetSource,
    StaticTargetSource,
    TargetProgram,
    TargetProgramSegment,
    TrajectoryLimits,
    timed_circle_trajectory,
    timed_waypoint_trajectory,
)
from controller.transforms import rotation_about_axis, rotation_from_rpy
from sim import targets, world


def _pose(position, rotation=None):
    return Pose(
        np.asarray(position, dtype=float),
        np.eye(3) if rotation is None else rotation,
    )


def _static(frame, pose):
    return StaticTargetSource(
        FramedTarget(frame, pose, Twist.zero())
    )


def _arm():
    return ArmJointState(np.zeros(7), np.zeros(7))


def _moving_plant():
    return PlantState(
        sample_time_s=5.0,
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


def _vee(skew):
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])


class WaypointTrajectoryTest(unittest.TestCase):
    def setUp(self):
        self.start = _pose(
            [0.1, -0.2, 0.3],
            rotation_from_rpy([0.2, -0.1, 0.3]),
        )
        self.finish = _pose(
            [0.5, 0.4, -0.1],
            self.start.rotation
            @ rotation_about_axis(np.array([0.0, 1.0, 0.0]), np.pi / 2.0),
        )
        self.trajectory = CartesianWaypointTrajectory.from_durations(
            TargetFrame.TORSO,
            (self.start, self.finish),
            (2.0,),
        )

    def test_endpoint_values_and_smooth_endpoint_velocity(self):
        before = self.trajectory.sample(0.0)
        after = self.trajectory.sample(2.0)
        held = self.trajectory.sample(20.0)

        self.assertEqual(before.reference_frame, TargetFrame.TORSO)
        np.testing.assert_array_equal(
            before.pose.position_m, self.start.position_m
        )
        np.testing.assert_array_equal(
            before.pose.rotation, self.start.rotation
        )
        np.testing.assert_array_equal(
            after.pose.position_m, self.finish.position_m
        )
        np.testing.assert_allclose(
            after.pose.rotation, self.finish.rotation, atol=1e-14
        )
        np.testing.assert_array_equal(after.twist.linear_m_s, np.zeros(3))
        np.testing.assert_array_equal(after.twist.angular_rad_s, np.zeros(3))
        np.testing.assert_array_equal(
            held.pose.position_m, self.finish.position_m
        )
        np.testing.assert_array_equal(held.twist.linear_m_s, np.zeros(3))

    def test_translation_and_angular_velocity_match_finite_difference(self):
        time_s = 0.73
        delta_s = 1e-6
        centre = self.trajectory.sample(time_s)
        minus = self.trajectory.sample(time_s - delta_s)
        plus = self.trajectory.sample(time_s + delta_s)

        numerical_linear = (
            plus.pose.position_m - minus.pose.position_m
        ) / (2.0 * delta_s)
        rotation_rate = (
            plus.pose.rotation - minus.pose.rotation
        ) / (2.0 * delta_s)
        numerical_angular = _vee(
            rotation_rate @ centre.pose.rotation.T
        )

        np.testing.assert_allclose(
            centre.twist.linear_m_s,
            numerical_linear,
            rtol=1e-9,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            centre.twist.angular_rad_s,
            numerical_angular,
            rtol=1e-8,
            atol=1e-8,
        )

    def test_orientation_midpoint_follows_principal_so3_path(self):
        midpoint = self.trajectory.sample(1.0)
        expected = (
            self.start.rotation
            @ rotation_about_axis(
                np.array([0.0, 1.0, 0.0]), np.pi / 4.0
            )
        )
        np.testing.assert_allclose(
            midpoint.pose.rotation, expected, atol=1e-12
        )

    def test_near_pi_rotation_keeps_the_directed_principal_path(self):
        axis = np.array([-1.0, 2.0, -3.0])
        axis /= np.linalg.norm(axis)
        near_pi = rotation_about_axis(axis, np.pi - 1e-8)
        trajectory = CartesianWaypointTrajectory.from_durations(
            TargetFrame.WORLD,
            (_pose([0.0, 0.0, 0.0]), _pose([0.0, 0.0, 0.0], near_pi)),
            (1.0,),
        )

        just_before_end = trajectory.sample(1.0 - 1e-4)
        expected = rotation_about_axis(
            axis,
            (np.pi - 1e-8)
            * (
                10.0 * (1.0 - 1e-4) ** 3
                - 15.0 * (1.0 - 1e-4) ** 4
                + 6.0 * (1.0 - 1e-4) ** 5
            ),
        )
        np.testing.assert_allclose(
            just_before_end.pose.rotation, expected, atol=1e-10
        )

    def test_internal_waypoint_has_c2_translation_continuity(self):
        middle = _pose(
            [0.3, 0.1, 0.2],
            self.start.rotation
            @ rotation_about_axis(np.array([1.0, 0.0, 0.0]), 0.4),
        )
        trajectory = CartesianWaypointTrajectory.from_durations(
            TargetFrame.WORLD,
            (self.start, middle, self.finish),
            (0.8, 1.2),
        )
        at_waypoint = trajectory.sample(0.8)
        just_before = trajectory.sample_kinematics(0.8 - 1e-6)
        exact = trajectory.sample_kinematics(0.8)
        just_after = trajectory.sample_kinematics(0.8 + 1e-6)

        np.testing.assert_array_equal(
            at_waypoint.pose.position_m, middle.position_m
        )
        self.assertGreater(
            np.linalg.norm(at_waypoint.twist.linear_m_s), 0.1
        )
        np.testing.assert_array_equal(
            at_waypoint.twist.angular_rad_s, np.zeros(3)
        )
        np.testing.assert_allclose(
            just_before.target.twist.linear_m_s,
            exact.target.twist.linear_m_s,
            atol=3e-6,
        )
        np.testing.assert_allclose(
            just_after.target.twist.linear_m_s,
            exact.target.twist.linear_m_s,
            atol=3e-6,
        )
        np.testing.assert_allclose(
            just_before.linear_acceleration_m_s2,
            exact.linear_acceleration_m_s2,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            just_after.linear_acceleration_m_s2,
            exact.linear_acceleration_m_s2,
            atol=2e-5,
        )

    def test_analytic_linear_acceleration_matches_velocity_difference(self):
        trajectory = CartesianWaypointTrajectory.from_durations(
            TargetFrame.WORLD,
            (
                _pose([0.0, 0.0, 0.0]),
                _pose([0.2, 0.1, -0.1]),
                _pose([0.4, -0.1, 0.2]),
            ),
            (0.8, 1.1),
        )
        time_s = 0.55
        step = 1e-6
        sample = trajectory.sample_kinematics(time_s)
        numerical = (
            trajectory.sample(time_s + step).twist.linear_m_s
            - trajectory.sample(time_s - step).twist.linear_m_s
        ) / (2.0 * step)
        np.testing.assert_allclose(
            sample.linear_acceleration_m_s2,
            numerical,
            rtol=2e-8,
            atol=2e-8,
        )

    def test_invalid_waypoint_timing_and_durations_fail_closed(self):
        with self.assertRaises(ValueError):
            CartesianWaypointTrajectory(
                TargetFrame.WORLD,
                (
                    CartesianWaypoint(0.1, self.start),
                    CartesianWaypoint(1.0, self.finish),
                ),
            )
        with self.assertRaises(ValueError):
            CartesianWaypointTrajectory(
                TargetFrame.WORLD,
                (
                    CartesianWaypoint(0.0, self.start),
                    CartesianWaypoint(0.0, self.finish),
                ),
            )
        with self.assertRaises(ValueError):
            CartesianWaypointTrajectory.from_durations(
                TargetFrame.WORLD,
                (self.start, self.finish),
                (0.0,),
            )
        with self.assertRaises(ValueError):
            self.trajectory.sample(np.nan)


class TargetProgramTest(unittest.TestCase):
    def test_static_holds_and_waypoint_motion_compose_without_a_jump(self):
        first = _pose([0.0, 0.0, 0.0])
        second = _pose([1.0, 0.0, 0.0])
        trajectory = CartesianWaypointTrajectory.from_durations(
            TargetFrame.WORLD, (first, second), (1.0,)
        )
        program = TargetProgram(
            TargetFrame.WORLD,
            (
                TargetProgramSegment(
                    0.5, _static(TargetFrame.WORLD, first)
                ),
                TargetProgramSegment(1.0, trajectory),
                TargetProgramSegment(
                    0.25, _static(TargetFrame.WORLD, second)
                ),
            ),
        )

        np.testing.assert_array_equal(
            program.sample(0.2).pose.position_m, first.position_m
        )
        np.testing.assert_allclose(
            program.sample(1.0).pose.position_m, [0.5, 0.0, 0.0]
        )
        np.testing.assert_array_equal(
            program.sample(program.duration_s + 1.0).pose.position_m,
            second.position_m,
        )

    def test_program_rejects_mixed_frames_and_discontinuous_boundaries(self):
        first = _pose([0.0, 0.0, 0.0])
        second = _pose([1.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            TargetProgram(
                TargetFrame.WORLD,
                (
                    TargetProgramSegment(
                        1.0, _static(TargetFrame.WORLD, first)
                    ),
                    TargetProgramSegment(
                        1.0, _static(TargetFrame.TORSO, first)
                    ),
                ),
            )
        with self.assertRaises(ValueError):
            TargetProgram(
                TargetFrame.WORLD,
                (
                    TargetProgramSegment(
                        1.0, _static(TargetFrame.WORLD, first)
                    ),
                    TargetProgramSegment(
                        1.0, _static(TargetFrame.WORLD, second)
                    ),
                ),
            )

    def test_static_dual_arm_adapter_preserves_existing_values(self):
        right = _static(TargetFrame.WORLD, _pose([0.1, 0.2, 0.3])).target
        left = _static(TargetFrame.BASE, _pose([-0.1, 0.0, 0.4])).target
        retained = DualArmFramedTargets(right, left)
        source = StaticDualArmTargetSource(retained)

        self.assertIs(source.sample(0.0), retained)
        self.assertIs(source.sample(12.0), retained)


class PeriodicTargetSourceTest(unittest.TestCase):
    def test_closed_out_and_back_repeats_without_boundary_jump(self):
        start = _pose([0.1, 0.2, 0.3])
        end = _pose([0.25, 0.2, 0.3])
        closed = CartesianWaypointTrajectory.from_durations(
            TargetFrame.WORLD,
            (start, end, start),
            (2.0, 2.0),
        )
        periodic = PeriodicTargetSource(closed, 4.0)

        np.testing.assert_array_equal(
            periodic.sample(0.0).pose.position_m,
            periodic.sample(4.0).pose.position_m,
        )
        np.testing.assert_array_equal(
            periodic.sample(1.0).pose.position_m,
            periodic.sample(5.0).pose.position_m,
        )
        np.testing.assert_allclose(
            periodic.sample(2.0).twist.linear_m_s,
            np.zeros(3),
            atol=1e-14,
        )
        np.testing.assert_array_equal(
            periodic.sample(4.0).twist.linear_m_s,
            np.zeros(3),
        )

    def test_open_source_cannot_be_repeated_silently(self):
        open_trajectory = CartesianWaypointTrajectory.from_durations(
            TargetFrame.WORLD,
            (_pose([0.0, 0.0, 0.0]), _pose([0.1, 0.0, 0.0])),
            (1.0,),
        )
        with self.assertRaisesRegex(ValueError, "close continuously"):
            PeriodicTargetSource(open_trajectory, 1.0)


class PrimitiveAndConstraintTest(unittest.TestCase):
    def test_explicit_timing_is_rejected_when_it_exceeds_limits(self):
        limits = TrajectoryLimits(
            max_linear_speed_m_s=0.05,
            max_linear_acceleration_m_s2=0.2,
        )
        with self.assertRaisesRegex(ValueError, "violates"):
            timed_waypoint_trajectory(
                TargetFrame.WORLD,
                (_pose([0.0, 0.0, 0.0]), _pose([0.2, 0.0, 0.0])),
                (1.0,),
                limits,
            )

    def test_automatic_waypoint_timing_satisfies_all_limits(self):
        limits = TrajectoryLimits(
            max_linear_speed_m_s=0.12,
            max_linear_acceleration_m_s2=0.25,
            max_angular_speed_rad_s=0.4,
            max_angular_acceleration_rad_s2=0.8,
        )
        poses = (
            _pose([0.0, 0.0, 0.0]),
            _pose(
                [0.08, 0.04, 0.02],
                rotation_from_rpy([0.0, 0.0, 0.1]),
            ),
            _pose(
                [0.14, -0.02, 0.05],
                rotation_from_rpy([0.0, 0.0, 0.2]),
            ),
        )
        trajectory = timed_waypoint_trajectory(
            TargetFrame.WORLD, poses, None, limits
        )
        bounds = trajectory.maximum_rates()
        self.assertLessEqual(bounds.max_linear_speed_m_s, 0.12 + 1e-10)
        self.assertLessEqual(
            bounds.max_linear_acceleration_m_s2, 0.25 + 1e-10
        )
        self.assertLessEqual(bounds.max_angular_speed_rad_s, 0.4 + 1e-10)
        self.assertLessEqual(
            bounds.max_angular_acceleration_rad_s2, 0.8 + 1e-10
        )

    def test_circle_geometry_twist_and_periodic_c2_boundary(self):
        start = _pose([0.3, -0.1, 0.5])
        limits = TrajectoryLimits(
            max_linear_speed_m_s=0.2,
            max_linear_acceleration_m_s2=0.5,
        )
        circle = timed_circle_trajectory(
            TargetFrame.TORSO,
            start,
            0.08,
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            None,
            limits,
        )
        periodic = PeriodicTargetSource(circle, circle.duration_s)
        for time_s in np.linspace(0.0, circle.duration_s, 31):
            sample = circle.sample(time_s)
            radius = sample.pose.position_m - circle.centre_m
            self.assertAlmostEqual(np.linalg.norm(radius), 0.08, places=12)
            self.assertAlmostEqual(radius[2], 0.0, places=12)
            self.assertAlmostEqual(
                np.dot(radius, sample.twist.linear_m_s),
                0.0,
                places=12,
            )
        time_s = 0.37 * circle.duration_s
        step = 1e-6
        numerical = (
            circle.sample(time_s + step).pose.position_m
            - circle.sample(time_s - step).pose.position_m
        ) / (2.0 * step)
        np.testing.assert_allclose(
            circle.sample(time_s).twist.linear_m_s,
            numerical,
            rtol=2e-8,
            atol=2e-8,
        )
        np.testing.assert_allclose(
            periodic.sample_kinematics(0.0).linear_acceleration_m_s2,
            periodic.sample_kinematics(
                circle.duration_s
            ).linear_acceleration_m_s2,
            atol=0.0,
        )

    def test_program_rejects_acceleration_discontinuity(self):
        start = _pose([0.0, 0.0, 0.0])
        line = CartesianWaypointTrajectory.from_durations(
            TargetFrame.WORLD,
            (start, _pose([0.1, 0.0, 0.0])),
            (1.0,),
        )

        class BadHold(HoldTrajectory):
            def sample_kinematics(self, elapsed_time_s):
                sample = super().sample_kinematics(elapsed_time_s)
                return replace(
                    sample,
                    linear_acceleration_m_s2=np.array([1.0, 0.0, 0.0]),
                )

        with self.assertRaisesRegex(ValueError, "C2"):
            TargetProgram(
                TargetFrame.WORLD,
                (
                    TargetProgramSegment(1.0, line),
                    TargetProgramSegment(
                        1.0,
                        BadHold(
                            FramedTarget(
                                TargetFrame.WORLD,
                                line.sample(1.0).pose,
                                Twist.zero(),
                            ),
                            1.0,
                        ),
                    ),
                ),
            )


class TrajectoryFrameSemanticsTest(unittest.TestCase):
    def test_world_pose_round_trips_through_each_target_frame(self):
        plant = _moving_plant()
        mount = MountCalibration(
            right_torso_to_base=_pose([0.2, -0.1, 0.0]),
            left_torso_to_base=_pose([-0.2, 0.1, 0.0]),
        )
        pose_world = _pose(
            [1.2, 2.3, 3.4],
            rotation_from_rpy([0.2, -0.3, 0.4]),
        )

        for frame in TargetFrame:
            with self.subTest(frame=frame):
                expressed = frames.express_pose_in_target_frame(
                    plant, "left", mount, frame, pose_world
                )
                resolved = frames.resolve_target_world(
                    plant,
                    "left",
                    mount,
                    FramedTarget(frame, expressed, Twist.zero()),
                )
                np.testing.assert_allclose(
                    resolved.pose_world.position_m,
                    pose_world.position_m,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    resolved.pose_world.rotation,
                    pose_world.rotation,
                    atol=1e-12,
                )

    def test_world_torso_and_base_trajectories_resolve_from_moving_frame(self):
        plant = _moving_plant()
        mount = MountCalibration(
            right_torso_to_base=_pose([0.2, -0.1, 0.0]),
            left_torso_to_base=_pose([-0.2, 0.1, 0.0]),
        )
        start = _pose([0.2, 0.1, 0.0])
        finish = _pose([0.4, -0.1, 0.2])

        for frame in TargetFrame:
            with self.subTest(frame=frame):
                trajectory = CartesianWaypointTrajectory.from_durations(
                    frame, (start, finish), (1.0,)
                )
                sampled = trajectory.sample(0.5)
                resolved = frames.resolve_target_world(
                    plant, "right", mount, sampled
                )

                if frame == TargetFrame.WORLD:
                    frame_pose = _pose([0.0, 0.0, 0.0])
                    frame_twist = Twist.zero()
                elif frame == TargetFrame.TORSO:
                    frame_pose = plant.torso_pose_world
                    frame_twist = plant.torso_twist_world
                else:
                    frame_pose = frames.compose_pose(
                        plant.torso_pose_world,
                        mount.right_torso_to_base,
                    )
                    mount_offset = (
                        frame_pose.position_m
                        - plant.torso_pose_world.position_m
                    )
                    frame_twist = Twist(
                        plant.torso_twist_world.linear_m_s
                        + np.cross(
                            plant.torso_twist_world.angular_rad_s,
                            mount_offset,
                        ),
                        plant.torso_twist_world.angular_rad_s,
                    )

                offset_world = (
                    frame_pose.rotation @ sampled.pose.position_m
                )
                expected_position = frame_pose.position_m + offset_world
                expected_linear = (
                    frame_twist.linear_m_s
                    + np.cross(
                        frame_twist.angular_rad_s, offset_world
                    )
                    + frame_pose.rotation @ sampled.twist.linear_m_s
                )
                expected_angular = (
                    frame_twist.angular_rad_s
                    + frame_pose.rotation @ sampled.twist.angular_rad_s
                )

                np.testing.assert_allclose(
                    resolved.pose_world.position_m,
                    expected_position,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    resolved.twist_world.linear_m_s,
                    expected_linear,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    resolved.twist_world.angular_rad_s,
                    expected_angular,
                    atol=1e-12,
                )


class _AdvancingBackend:
    def __init__(self, initial_state):
        self.state = initial_state
        self.active = False

    def takeover(self):
        self.active = True
        return self.state

    def exchange(self, _command):
        if not self.active:
            raise RuntimeError("not active")
        self.state = replace(
            self.state,
            sample_time_s=(
                self.state.sample_time_s + self.state.nominal_dt_s
            ),
        )
        return self.state

    def release(self):
        self.active = False


class _CountingDualSource:
    def __init__(self, static_left):
        self.static_left = static_left
        self.sample_times = []

    def sample(self, elapsed_time_s):
        self.sample_times.append(elapsed_time_s)
        right = FramedTarget(
            TargetFrame.WORLD,
            _pose([0.4 + elapsed_time_s, -0.2, 1.1]),
            Twist(np.array([1.0, 0.0, 0.0]), np.zeros(3)),
        )
        return DualArmFramedTargets(right, self.static_left)


class RunnerTargetSourceTest(unittest.TestCase):
    def setUp(self):
        world.backend.release()
        world.backend.configure_torso_driver(None, None)
        world.backend.reset()

    def tearDown(self):
        world.backend.release()
        world.backend.configure_torso_driver(None, None)
        world.backend.reset()

    def test_runner_samples_source_once_per_cycle_from_takeover_time(self):
        initial = replace(
            world.read_state(Twist.zero()), sample_time_s=7.5
        )
        left = _static(
            TargetFrame.WORLD, _pose([0.4, 0.2, 1.1])
        ).target
        source = _CountingDualSource(left)
        runner = ReactivePositionRunner(
            _AdvancingBackend(initial),
            world.MOUNT_CALIBRATION,
            world.PIPELINE_SETUP,
            source,
        )
        runner.start()
        first = runner.cycle()
        second = runner.cycle()
        runner.close()

        np.testing.assert_allclose(source.sample_times, [0.0, 0.002])
        self.assertEqual(first.target_elapsed_time_s, 0.0)
        self.assertAlmostEqual(second.target_elapsed_time_s, 0.002)
        np.testing.assert_allclose(
            second.sampled_targets.right.pose.position_m,
            [0.402, -0.2, 1.1],
            atol=1e-15,
        )

    def test_simulation_markers_are_display_only_not_controller_truth(self):
        initial = world.read_state(Twist.zero())
        right = _static(
            TargetFrame.WORLD, _pose([0.31, -0.22, 1.05])
        ).target
        left = _static(
            TargetFrame.WORLD, _pose([0.29, 0.24, 1.08])
        ).target
        source = IndependentArmTargetSource(
            StaticTargetSource(right),
            StaticTargetSource(left),
        )
        targets.set_target("right", np.array([9.0, 8.0, 7.0]))
        targets.set_target("left", np.array([-9.0, -8.0, -7.0]))

        runner = ReactivePositionRunner(
            _AdvancingBackend(initial),
            world.MOUNT_CALIBRATION,
            world.PIPELINE_SETUP,
            source,
        )
        runner.start()
        cycle = runner.cycle()
        runner.close()

        np.testing.assert_array_equal(
            cycle.resolved_targets.right.pose_world.position_m,
            right.pose.position_m,
        )
        np.testing.assert_array_equal(
            cycle.resolved_targets.left.pose_world.position_m,
            left.pose.position_m,
        )
        self.assertFalse(
            np.array_equal(
                cycle.resolved_targets.right.pose_world.position_m,
                targets.target_position("right"),
            )
        )


if __name__ == "__main__":
    unittest.main()
