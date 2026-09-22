from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import numpy as np

from controller import desired_pos
from controller.state import Pose, Twist
from controller.trajectory_config import materialize_trajectory
from runtime_config import (
    TargetTrajectoryConfig,
    TrajectoryConstraintsConfig,
    TrajectorySegmentConfig,
    load_config,
)
from sim import world
from sim.target_trajectory import (
    apply_initial_postures,
    prepare_target_trajectory,
)

FIXTURE_CONFIG = Path(__file__).resolve().parent / "fixtures" / "control.toml"
CONFIG = load_config(FIXTURE_CONFIG)



def _without_left_trajectory(source):
    prefix, remainder = source.split(
        "[targets.left.trajectory]", 1
    )
    _, simulation = remainder.split(
        "[simulation.initial_joint_position_rad]", 1
    )
    return (
        prefix
        + "[simulation.initial_joint_position_rad]"
        + simulation
    )


def _with_left_trajectory(source, trajectory):
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


class TargetTrajectoryConfigTest(unittest.TestCase):
    def test_optional_trajectory_is_part_of_the_existing_target(self):
        self.assertIsNone(CONFIG.right_target.trajectory)
        trajectory = CONFIG.left_target.trajectory
        self.assertEqual(trajectory.reference_frame, "world")
        self.assertEqual(trajectory.start, "measured")
        self.assertTrue(trajectory.loop)
        self.assertEqual(
            tuple(item.type for item in trajectory.segments),
            ("line", "line"),
        )
        self.assertEqual(
            trajectory.segments[0].displacement_m, (0.4, 0.0, 0.0)
        )
        self.assertEqual(
            trajectory.segments[1].displacement_m, (-0.4, 0.0, 0.0)
        )
        self.assertEqual(trajectory.segments[0].duration_s, 10.0)
        self.assertTrue(trajectory.open_live_path_plot)
        self.assertEqual(
            len(CONFIG.simulation.left_initial_joint_position_rad), 7
        )

    def test_absent_timed_data_preserves_static_target_compatibility(self):
        source = _without_left_trajectory(
            Path(CONFIG.source_path).read_text()
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "static.toml"
            path.write_text(source)
            loaded = load_config(path)
        self.assertIsNone(loaded.right_target.trajectory)
        self.assertIsNone(loaded.left_target.trajectory)
        self.assertEqual(
            loaded.left_target.position_m,
            CONFIG.left_target.position_m,
        )
        self.assertEqual(
            loaded.left_target.rpy_rad,
            CONFIG.left_target.rpy_rad,
        )

    def test_invalid_timed_target_data_fails_closed(self):
        source = Path(CONFIG.source_path).read_text()
        invalid_sources = (
            source.replace(
                'start = "measured"',
                'start = "unknown"',
                1,
            ),
            source.replace(
                "displacement_m = [0.40, 0.0, 0.0]",
                "displacement_m = [0.0, 0.0, 0.0]",
                1,
            ),
            source.replace(
                "duration_s = 10.0",
                "duration_s = 0.0",
                1,
            ),
            source.replace(
                'type = "line"',
                'type = "unknown"',
                1,
            ),
            source.replace(
                "loop = true",
                'loop = "yes"',
                1,
            ),
            source.replace(
                "open_live_path_plot = true",
                'open_live_path_plot = "yes"',
                1,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, invalid in enumerate(invalid_sources):
                path = Path(directory) / f"invalid-{index}.toml"
                path.write_text(invalid)
                with self.subTest(index=index):
                    with self.assertRaises(ValueError):
                        load_config(path)

    def test_general_circle_and_waypoint_sequence_parses(self):
        source = Path(CONFIG.source_path).read_text()
        prefix, remainder = source.split(
            "[targets.left.trajectory]", 1
        )
        _, simulation = remainder.split(
            "[simulation.initial_joint_position_rad]", 1
        )
        general = """
[targets.left.trajectory]
reference_frame = "torso"
start = "measured"
loop = false
open_live_path_plot = true

[targets.left.trajectory.constraints]
max_linear_speed_m_s = 0.10
max_linear_acceleration_m_s2 = 0.20
max_angular_speed_rad_s = 0.50
max_angular_acceleration_rad_s2 = 1.00

[[targets.left.trajectory.segments]]
type = "circle"
radius_m = 0.05
normal = [1.0, 0.0, 0.0]
start_direction = [0.0, 0.0, 1.0]
revolutions = 1
clockwise = false

[[targets.left.trajectory.segments]]
type = "waypoints"
offsets_m = [
  [0.0, 0.0, 0.0],
  [0.0, 0.04, 0.02],
  [0.02, 0.08, 0.00],
]
"""
        candidate = (
            prefix
            + general
            + "\n[simulation.initial_joint_position_rad]"
            + simulation
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "general.toml"
            path.write_text(candidate)
            loaded = load_config(path)
        trajectory = loaded.left_target.trajectory
        self.assertEqual(trajectory.reference_frame, "torso")
        self.assertEqual(
            tuple(item.type for item in trajectory.segments),
            ("circle", "waypoints"),
        )
        self.assertIsNone(trajectory.segments[0].duration_s)
        self.assertIsNone(trajectory.segments[1].durations_s)

    def test_programmatic_request_seam_materializes_circle(self):
        specification = TargetTrajectoryConfig(
            reference_frame="torso",
            start="measured",
            loop=True,
            open_live_path_plot=True,
            constraints=TrajectoryConstraintsConfig(
                max_linear_speed_m_s=0.10,
                max_linear_acceleration_m_s2=0.20,
                max_angular_speed_rad_s=0.50,
                max_angular_acceleration_rad_s2=1.00,
            ),
            segments=(
                TrajectorySegmentConfig(
                    type="circle",
                    radius_m=0.05,
                    normal=(1.0, 0.0, 0.0),
                    start_direction=(0.0, 0.0, 1.0),
                    revolutions=1,
                    clockwise=False,
                ),
            ),
        )
        start = Pose(np.array([0.3, 0.2, 1.1]), np.eye(3))
        result = materialize_trajectory(specification, start)
        self.assertGreater(result.duration_s, 0.0)
        self.assertEqual(result.source.reference_frame.value, "torso")
        np.testing.assert_allclose(
            result.source.sample(0.0).pose.position_m,
            result.source.sample(result.duration_s).pose.position_m,
            atol=1e-12,
        )
        self.assertLessEqual(
            result.rate_bounds.max_linear_speed_m_s, 0.10 + 1e-10
        )

    def test_named_circle_planes_parse_to_exact_circle_directions(self):
        source = Path(CONFIG.source_path).read_text()
        expected = {
            "xy": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
            "horizontal": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
            "xz": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
            "vertical_xz": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
            "yz": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            "vertical_yz": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        }
        with tempfile.TemporaryDirectory() as directory:
            for plane, directions in expected.items():
                trajectory = f"""
[targets.left.trajectory]
reference_frame = "world"
start = "measured"
loop = true
open_live_path_plot = false

[[targets.left.trajectory.segments]]
type = "circle"
plane = "{plane}"
radius_m = 0.05
duration_s = 4.0
revolutions = 1
clockwise = false
"""
                candidate = _with_left_trajectory(source, trajectory)
                path = Path(directory) / f"{plane}.toml"
                path.write_text(candidate)
                with self.subTest(plane=plane):
                    circle = load_config(
                        path
                    ).left_target.trajectory.segments[0]
                    self.assertEqual(circle.normal, directions[0])
                    self.assertEqual(
                        circle.start_direction, directions[1]
                    )

    def test_named_vertical_circle_materializes_from_resolved_start(self):
        source = Path(CONFIG.source_path).read_text()
        trajectory = """
[targets.left.trajectory]
reference_frame = "world"
start = "measured"
loop = true
open_live_path_plot = false

[[targets.left.trajectory.segments]]
type = "circle"
plane = "vertical_yz"
radius_m = 0.05
duration_s = 4.0
revolutions = 1
clockwise = false
"""
        candidate = _with_left_trajectory(source, trajectory)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vertical-circle.toml"
            path.write_text(candidate)
            specification = load_config(path).left_target.trajectory
        start = Pose(np.array([0.3, 0.2, 1.1]), np.eye(3))
        result = materialize_trajectory(specification, start)
        beginning = result.source.sample(0.0).pose.position_m
        halfway = result.source.sample(
            0.5 * result.duration_s
        ).pose.position_m
        closing = result.source.sample(
            result.duration_s
        ).pose.position_m
        np.testing.assert_allclose(beginning, start.position_m, atol=1e-12)
        np.testing.assert_allclose(closing, start.position_m, atol=1e-12)
        np.testing.assert_allclose(
            halfway,
            start.position_m + np.array([0.0, -0.10, 0.0]),
            atol=1e-12,
        )

    def test_named_circle_rejects_ambiguous_or_unknown_geometry(self):
        source = Path(CONFIG.source_path).read_text()
        invalid_geometries = (
            'plane = "vertical_yz"\n'
            "normal = [1.0, 0.0, 0.0]\n"
            "start_direction = [0.0, 0.0, 1.0]",
            'plane = "diagonal"',
            'plane = "vertical_yz"\ncenter_m = [0.3, 0.2, 1.1]',
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, geometry in enumerate(invalid_geometries):
                trajectory = f"""
[targets.left.trajectory]
reference_frame = "world"
start = "measured"
loop = true
open_live_path_plot = false

[[targets.left.trajectory.segments]]
type = "circle"
{geometry}
radius_m = 0.05
duration_s = 4.0
revolutions = 1
clockwise = false
"""
                candidate = _with_left_trajectory(source, trajectory)
                path = Path(directory) / f"invalid-circle-{index}.toml"
                path.write_text(candidate)
                with self.subTest(index=index):
                    with self.assertRaises(ValueError):
                        load_config(path)


class TargetTrajectoryIntegrationTest(unittest.TestCase):
    def tearDown(self):
        world.backend.release()
        world.backend.configure_torso_driver(None, None)

    def _prepare(self, trajectory=None):
        selected = (
            CONFIG.left_target.trajectory
            if trajectory is None
            else trajectory
        )
        apply_initial_postures(
            world.backend,
            {"left": CONFIG.simulation.left_initial_joint_position_rad},
        )
        plant = world.backend.read_state(Twist.zero())
        return selected, prepare_target_trajectory(
            world.backend,
            world.MOUNT_CALIBRATION,
            "left",
            selected,
            plant,
            desired_pos.configured_targets(),
        )

    def _safe_out_and_back(self):
        trajectory = CONFIG.left_target.trajectory
        outward = replace(
            trajectory.segments[0],
            displacement_m=(0.15, 0.0, 0.0),
        )
        returning = replace(
            trajectory.segments[1],
            displacement_m=(-0.15, 0.0, 0.0),
        )
        return replace(
            trajectory,
            segments=(outward, returning),
            orientation=None,
        )

    def test_configured_source_repeats_exactly_from_measured_start(self):
        trajectory = self._safe_out_and_back()
        _, setup = self._prepare(trajectory)
        start = setup.source.sample(0.0)
        forward = setup.source.sample(
            setup.boundary_times_s[0]
        )
        returned = setup.source.sample(
            setup.duration_s
        )
        repeated = setup.source.sample(
            setup.duration_s + 1.0
        )
        first_leg = setup.source.sample(1.0)

        np.testing.assert_array_equal(
            start.pose.position_m,
            setup.start_pose_reference.position_m,
        )
        np.testing.assert_allclose(
            forward.pose.position_m,
            setup.start_pose_reference.position_m
            + np.array([0.15, 0.0, 0.0]),
            atol=1e-14,
        )
        np.testing.assert_array_equal(
            forward.twist.linear_m_s, np.zeros(3)
        )
        np.testing.assert_array_equal(
            returned.pose.position_m,
            setup.start_pose_reference.position_m,
        )
        np.testing.assert_allclose(
            repeated.pose.position_m,
            first_leg.pose.position_m,
            atol=1e-14,
        )

    def test_non_repeating_target_holds_its_endpoint(self):
        base = self._safe_out_and_back()
        trajectory = replace(
            base,
            loop=False,
            segments=(base.segments[0],),
            open_live_path_plot=False,
        )
        _, setup = self._prepare(trajectory)
        at_end = setup.source.sample(
            setup.duration_s
        )
        held = setup.source.sample(
            10.0 * setup.duration_s
        )
        np.testing.assert_array_equal(
            at_end.pose.position_m, held.pose.position_m
        )
        np.testing.assert_array_equal(
            held.twist.linear_m_s, np.zeros(3)
        )

if __name__ == "__main__":
    unittest.main()
