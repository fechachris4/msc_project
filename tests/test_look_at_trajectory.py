import tempfile
from pathlib import Path
import unittest

import numpy as np

from controller.look_at import (
    FixedWorldPointSource,
    LookAtTargetSource,
    WorldPointKinematics,
)
from controller.state import FramedTarget, Pose, TargetFrame, Twist
from controller.trajectory import (
    CartesianWaypointTrajectory,
    StaticTargetSource,
)
from controller.trajectory_config import materialize_trajectory
from runtime_config import (
    CONFIG,
    TargetTrajectoryConfig,
    TrajectoryConstraintsConfig,
    TrajectoryOrientationConfig,
    TrajectorySegmentConfig,
    load_config,
)


def _pose(position):
    return Pose(np.asarray(position, dtype=float), np.eye(3))


def _replace_left_trajectory(source, trajectory):
    prefix, remainder = source.split(
        "[targets.left.trajectory]", 1
    )
    _, simulation = remainder.split(
        "[simulation.initial_joint_position_rad]", 1
    )
    return (
        prefix
        + trajectory
        + "\n[simulation.initial_joint_position_rad]"
        + simulation
    )


def _orientation_table(
    *,
    forward="[0.0, 0.0, 1.0]",
    tool_up="[0.0, 1.0, 0.0]",
    world_up="[0.0, 0.0, 1.0]",
):
    return f"""
[targets.left.trajectory.orientation]
policy = "look_at_fixed_world_point"
object_position_world_m = [0.20, -0.10, 1.00]
tool_forward_axis = {forward}
tool_up_axis = {tool_up}
world_up_direction = {world_up}
"""


def _trajectory_toml(
    orientation,
    *,
    reference_frame="world",
    segment_extra="",
):
    return f"""
[targets.left.trajectory]
reference_frame = "{reference_frame}"
start = "measured"
loop = false
open_live_path_plot = false
{orientation}
[[targets.left.trajectory.segments]]
type = "line"
displacement_m = [0.10, 0.0, 0.0]
duration_s = 2.0
{segment_extra}
"""


class LookAtConfigTest(unittest.TestCase):
    def _load(self, trajectory):
        source = Path(CONFIG.source_path).read_text()
        candidate = _replace_left_trajectory(source, trajectory)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "look-at.toml"
            path.write_text(candidate)
            return load_config(path).left_target.trajectory

    def test_separate_trajectory_orientation_table_parses(self):
        trajectory = self._load(
            _trajectory_toml(_orientation_table())
        )
        orientation = trajectory.orientation
        self.assertEqual(
            orientation.policy, "look_at_fixed_world_point"
        )
        self.assertEqual(
            orientation.object_position_world_m,
            (0.20, -0.10, 1.00),
        )
        self.assertEqual(
            orientation.tool_forward_axis, (0.0, 0.0, 1.0)
        )
        self.assertEqual(
            tuple(segment.type for segment in trajectory.segments),
            ("line",),
        )

    def test_non_world_and_degenerate_axes_fail_closed(self):
        candidates = (
            _trajectory_toml(
                _orientation_table(), reference_frame="torso"
            ),
            _trajectory_toml(
                _orientation_table(forward="[0.0, 0.0, 0.0]")
            ),
            _trajectory_toml(
                _orientation_table(
                    forward="[0.0, 0.0, 1.0]",
                    tool_up="[0.0, 0.0, -2.0]",
                )
            ),
            _trajectory_toml(
                _orientation_table(),
                segment_extra="end_rpy_rad = [0.0, 0.0, 0.0]",
            ),
        )
        for index, candidate in enumerate(candidates):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    self._load(candidate)

    def test_absent_orientation_preserves_static_rpy_path_behavior(self):
        specification = TargetTrajectoryConfig(
            reference_frame="world",
            start="measured",
            loop=False,
            open_live_path_plot=False,
            constraints=TrajectoryConstraintsConfig(
                None, None, None, None
            ),
            segments=(
                TrajectorySegmentConfig(
                    type="line",
                    duration_s=2.0,
                    displacement_m=(0.1, 0.0, 0.0),
                ),
            ),
        )
        start = _pose([0.4, 0.2, 1.1])
        result = materialize_trajectory(specification, start)
        self.assertIsNone(specification.orientation)
        np.testing.assert_array_equal(
            result.source.sample(1.0).pose.rotation,
            start.rotation,
        )


class LookAtGeometryTest(unittest.TestCase):
    def _moving_source(self):
        return CartesianWaypointTrajectory.from_durations(
            TargetFrame.WORLD,
            (
                _pose([0.8, -0.4, 0.6]),
                _pose([0.5, 0.6, 0.8]),
            ),
            (2.0,),
        )

    def _look_at(self, source=None, object_position=(0.0, 0.0, 0.0)):
        return LookAtTargetSource(
            self._moving_source() if source is None else source,
            FixedWorldPointSource(object_position),
            tool_forward_axis=[0.0, 0.0, 1.0],
            tool_up_axis=[0.0, 1.0, 0.0],
            world_up_direction=[0.0, 0.0, 1.0],
        )

    def test_forward_axis_and_roll_projection_match_geometry(self):
        source = self._look_at()
        sample = source.sample(0.73)
        expected_forward = -sample.pose.position_m
        expected_forward /= np.linalg.norm(expected_forward)
        actual_forward = sample.pose.rotation @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(
            actual_forward, expected_forward, atol=1e-12
        )

        expected_up = np.array([0.0, 0.0, 1.0])
        expected_up -= (
            expected_forward * float(expected_forward @ expected_up)
        )
        expected_up /= np.linalg.norm(expected_up)
        actual_up = sample.pose.rotation @ np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(actual_up, expected_up, atol=1e-12)
        np.testing.assert_allclose(
            sample.pose.rotation.T @ sample.pose.rotation,
            np.eye(3),
            atol=1e-12,
        )
        self.assertAlmostEqual(
            np.linalg.det(sample.pose.rotation), 1.0, places=12
        )

    def test_analytic_angular_velocity_and_acceleration_are_consistent(self):
        source = self._look_at()
        time_s = 0.73
        step = 1e-5
        centre = source.sample_kinematics(time_s)
        minus = source.sample_kinematics(time_s - step)
        plus = source.sample_kinematics(time_s + step)
        rotation_rate = (
            plus.target.pose.rotation - minus.target.pose.rotation
        ) / (2.0 * step)
        angular_matrix = (
            rotation_rate @ centre.target.pose.rotation.T
        )
        numerical_angular = np.array(
            [
                angular_matrix[2, 1],
                angular_matrix[0, 2],
                angular_matrix[1, 0],
            ]
        )
        numerical_acceleration = (
            plus.target.twist.angular_rad_s
            - minus.target.twist.angular_rad_s
        ) / (2.0 * step)
        np.testing.assert_allclose(
            centre.target.twist.angular_rad_s,
            numerical_angular,
            rtol=2e-9,
            atol=2e-9,
        )
        np.testing.assert_allclose(
            centre.angular_acceleration_rad_s2,
            numerical_acceleration,
            rtol=2e-8,
            atol=2e-8,
        )

    def test_closed_path_is_continuous_with_look_at_policy(self):
        specification = TargetTrajectoryConfig(
            reference_frame="world",
            start="measured",
            loop=True,
            open_live_path_plot=False,
            constraints=TrajectoryConstraintsConfig(
                None, None, None, None
            ),
            segments=(
                TrajectorySegmentConfig(
                    type="line",
                    duration_s=1.0,
                    displacement_m=(0.2, 0.1, 0.0),
                ),
                TrajectorySegmentConfig(
                    type="line",
                    duration_s=1.0,
                    displacement_m=(-0.2, -0.1, 0.0),
                ),
            ),
            orientation=TrajectoryOrientationConfig(
                policy="look_at_fixed_world_point",
                object_position_world_m=(0.0, 0.0, 0.2),
                tool_forward_axis=(0.0, 0.0, 1.0),
                tool_up_axis=(0.0, 1.0, 0.0),
                world_up_direction=(0.0, 0.0, 1.0),
            ),
        )
        result = materialize_trajectory(
            specification, _pose([0.5, -0.3, 0.7])
        )
        oriented_program = result.source
        beginning = oriented_program.sample_kinematics(0.0)
        closing = oriented_program.sample_kinematics(
            result.duration_s
        )
        np.testing.assert_allclose(
            closing.target.pose.rotation,
            beginning.target.pose.rotation,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            closing.target.twist.angular_rad_s,
            beginning.target.twist.angular_rad_s,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            closing.angular_acceleration_rad_s2,
            beginning.angular_acceleration_rad_s2,
            atol=1e-12,
        )
        self.assertTrue(
            np.isinf(result.rate_bounds.max_angular_speed_rad_s)
        )

    def test_zero_look_direction_and_collinear_world_up_are_rejected(self):
        static = StaticTargetSource(
            FramedTarget(
                TargetFrame.WORLD,
                _pose([0.0, 0.0, 0.0]),
                Twist.zero(),
            )
        )
        coincident = self._look_at(
            source=static, object_position=(0.0, 0.0, 0.0)
        )
        collinear = self._look_at(
            source=static, object_position=(0.0, 0.0, 1.0)
        )
        with self.assertRaisesRegex(ValueError, "coincides"):
            coincident.sample(0.0)
        with self.assertRaisesRegex(ValueError, "collinear"):
            collinear.sample(0.0)

    def test_non_world_source_is_rejected(self):
        torso_source = CartesianWaypointTrajectory.from_durations(
            TargetFrame.TORSO,
            (_pose([0.5, 0.0, 0.5]), _pose([0.6, 0.0, 0.5])),
            (1.0,),
        )
        with self.assertRaisesRegex(ValueError, "world-frame"):
            self._look_at(source=torso_source)

    def test_structured_point_source_seam_accepts_sampled_motion(self):
        class MovingWorldPoint:
            def sample_kinematics(self, elapsed_time_s):
                return WorldPointKinematics(
                    [0.1 * elapsed_time_s, -0.2, 0.1],
                    [0.1, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                )

        source = LookAtTargetSource(
            self._moving_source(),
            MovingWorldPoint(),
            tool_forward_axis=[0.0, 0.0, 1.0],
            tool_up_axis=[0.0, 1.0, 0.0],
            world_up_direction=[0.0, 0.0, 1.0],
        )
        time_s = 0.6
        sample = source.sample(time_s)
        point = MovingWorldPoint().sample_kinematics(
            time_s
        ).position_world_m
        expected = point - sample.pose.position_m
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(
            sample.pose.rotation @ np.array([0.0, 0.0, 1.0]),
            expected,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
