from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import mujoco
import numpy as np

from controller import desired_pos
from controller.runner import ReactivePositionRunner
from controller.state import Pose, Twist
from controller.trajectory import IndependentArmTargetSource, StaticTargetSource
from runtime_config import (
    CONFIG,
    LookAtObjectMotionConfig,
    TargetTrajectoryConfig,
    TrajectoryConstraintsConfig,
    TrajectoryOrientationConfig,
    TrajectorySegmentConfig,
    load_config,
)
from sim import world
from sim.look_at_object import (
    SimulatedMocapPointSource,
    sinusoidal_point_kinematics,
)
from sim.target_trajectory import (
    apply_initial_postures,
    prepare_target_trajectory,
)


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


def _moving_orientation_toml():
    return """
[targets.left.trajectory]
reference_frame = "world"
start = "measured"
loop = false
open_live_path_plot = false

[targets.left.trajectory.orientation]
policy = "look_at_sim_object"
object_body = "look_at_object"
tool_forward_axis = [0.0, 0.0, 1.0]
tool_up_axis = [0.0, 1.0, 0.0]
world_up_direction = [0.0, 0.0, 1.0]

[[targets.left.trajectory.segments]]
type = "line"
displacement_m = [0.02, 0.0, 0.0]
duration_s = 2.0
"""


def _with_object_motion(source):
    header = "[simulation.look_at_object_motion]"
    if header in source:
        source = source.split(header, 1)[0].rstrip()
    motion = """
[simulation.look_at_object_motion]
body_name = "look_at_object"
home_position_world_m = [0.10, -0.20, 1.00]
linear_amplitude_m = [0.04, 0.02, 0.00]
linear_frequency_hz = 0.25

"""
    return source + "\n\n" + motion


def _motion_config():
    return LookAtObjectMotionConfig(
        body_name="look_at_object",
        home_position_world_m=(0.10, -0.20, 1.00),
        linear_amplitude_m=(0.04, 0.02, 0.00),
        linear_frequency_hz=0.25,
    )


def _trajectory_config():
    return TargetTrajectoryConfig(
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
                displacement_m=(0.02, 0.0, 0.0),
            ),
        ),
        orientation=TrajectoryOrientationConfig(
            policy="look_at_sim_object",
            object_position_world_m=None,
            tool_forward_axis=(0.0, 0.0, 1.0),
            tool_up_axis=(0.0, 1.0, 0.0),
            world_up_direction=(0.0, 0.0, 1.0),
            object_body="look_at_object",
        ),
    )


class SimLookAtConfigTest(unittest.TestCase):
    def test_moving_object_policy_and_motion_parse_independently(self):
        source = Path(CONFIG.source_path).read_text()
        candidate = _with_object_motion(
            _replace_left_trajectory(
                source, _moving_orientation_toml()
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "moving-look-at.toml"
            path.write_text(candidate)
            loaded = load_config(path)
        orientation = loaded.left_target.trajectory.orientation
        motion = loaded.simulation.look_at_object_motion
        self.assertEqual(orientation.policy, "look_at_sim_object")
        self.assertEqual(orientation.object_body, "look_at_object")
        self.assertIsNone(orientation.object_position_world_m)
        self.assertEqual(motion.body_name, "look_at_object")
        self.assertEqual(
            motion.linear_amplitude_m, (0.04, 0.02, 0.0)
        )
        self.assertEqual(motion.linear_frequency_hz, 0.25)

    def test_moving_policy_rejects_fixed_point_key(self):
        invalid = _moving_orientation_toml().replace(
            'object_body = "look_at_object"',
            'object_body = "look_at_object"\n'
            "object_position_world_m = [0.1, 0.2, 1.0]",
        )
        source = _with_object_motion(
            _replace_left_trajectory(
                Path(CONFIG.source_path).read_text(), invalid
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.toml"
            path.write_text(source)
            with self.assertRaises(ValueError):
                load_config(path)


class SimLookAtObjectSourceTest(unittest.TestCase):
    def tearDown(self):
        world.backend.release()
        world.backend.configure_torso_driver(None, None)

    def test_sinusoidal_derivatives_match_finite_differences(self):
        config = _motion_config()
        time_s = 0.73
        step = 1e-5
        centre = sinusoidal_point_kinematics(config, time_s)
        minus = sinusoidal_point_kinematics(
            config, time_s - step
        )
        plus = sinusoidal_point_kinematics(
            config, time_s + step
        )
        numerical_velocity = (
            plus.position_world_m - minus.position_world_m
        ) / (2.0 * step)
        numerical_acceleration = (
            plus.velocity_world_m_s - minus.velocity_world_m_s
        ) / (2.0 * step)
        np.testing.assert_allclose(
            centre.velocity_world_m_s,
            numerical_velocity,
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            centre.acceleration_world_m_s2,
            numerical_acceleration,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_source_reads_the_current_mocap_world_position(self):
        source = SimulatedMocapPointSource(
            world.backend, _motion_config()
        )
        source.apply(0.4)
        sample = source.sample_kinematics(0.4)
        body_id = mujoco.mj_name2id(
            world.model,
            mujoco.mjtObj.mjOBJ_BODY,
            source.body_name,
        )
        mocap_index = world.model.body_mocapid[body_id]
        np.testing.assert_array_equal(
            sample.position_world_m,
            world.data.mocap_pos[mocap_index],
        )
        with self.assertRaisesRegex(RuntimeError, "visible cycle"):
            source.sample_kinematics(0.5)

    def test_each_runner_cycle_aims_at_the_updated_sim_object(self):
        trajectory = _trajectory_config()
        apply_initial_postures(
            world.backend,
            {"left": CONFIG.simulation.left_initial_joint_position_rad},
        )
        plant = world.backend.read_state(Twist.zero())
        static_targets = desired_pos.configured_targets()
        setup = prepare_target_trajectory(
            world.backend,
            world.MOUNT_CALIBRATION,
            "left",
            trajectory,
            plant,
            static_targets,
            _motion_config(),
        )
        runner = ReactivePositionRunner(
            world.backend,
            world.MOUNT_CALIBRATION,
            world.PIPELINE_SETUP,
            IndependentArmTargetSource(
                right=StaticTargetSource(
                    static_targets.for_arm("right")
                ),
                left=setup.source,
            ),
            arms=("left",),
        )
        runner.start()
        self.addCleanup(runner.close)

        previous_object = None
        for _ in range(4):
            elapsed = runner.target_elapsed_time_s
            setup.look_at_object.apply(elapsed)
            expected_object = (
                setup.look_at_object.sample_kinematics(
                    elapsed
                ).position_world_m
            )
            cycle = runner.cycle()
            sampled = cycle.sampled_targets.left
            look_direction = (
                expected_object - sampled.pose.position_m
            )
            look_direction /= np.linalg.norm(look_direction)
            np.testing.assert_allclose(
                sampled.pose.rotation @ np.array([0.0, 0.0, 1.0]),
                look_direction,
                atol=1e-12,
            )
            if previous_object is not None:
                self.assertFalse(
                    np.array_equal(expected_object, previous_object)
                )
            previous_object = expected_object

    def test_loop_repeats_position_without_rewinding_object_time(self):
        base = _trajectory_config()
        returning = replace(
            base.segments[0],
            displacement_m=(-0.02, 0.0, 0.0),
        )
        trajectory = replace(
            base,
            loop=True,
            segments=(base.segments[0], returning),
        )
        motion = replace(
            _motion_config(), linear_frequency_hz=0.20
        )
        apply_initial_postures(
            world.backend,
            {"left": CONFIG.simulation.left_initial_joint_position_rad},
        )
        plant = world.backend.read_state(Twist.zero())
        setup = prepare_target_trajectory(
            world.backend,
            world.MOUNT_CALIBRATION,
            "left",
            trajectory,
            plant,
            desired_pos.configured_targets(),
            motion,
        )
        offset_s = 0.4
        setup.look_at_object.apply(offset_s)
        first = setup.source.sample(offset_s)
        setup.look_at_object.apply(setup.duration_s + offset_s)
        repeated = setup.source.sample(
            setup.duration_s + offset_s
        )
        np.testing.assert_allclose(
            repeated.pose.position_m,
            first.pose.position_m,
            atol=1e-14,
        )
        self.assertFalse(
            np.allclose(
                repeated.pose.rotation,
                first.pose.rotation,
                rtol=0.0,
                atol=1e-6,
            )
        )
        object_position = (
            setup.look_at_object.sample_kinematics(
                setup.duration_s + offset_s
            ).position_world_m
        )
        expected = object_position - repeated.pose.position_m
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(
            repeated.pose.rotation @ np.array([0.0, 0.0, 1.0]),
            expected,
            atol=1e-12,
        )

    def test_body_name_mismatch_fails_before_runtime(self):
        trajectory = _trajectory_config()
        bad_motion = LookAtObjectMotionConfig(
            body_name="right_target",
            home_position_world_m=(0.1, -0.2, 1.0),
            linear_amplitude_m=(0.04, 0.0, 0.0),
            linear_frequency_hz=0.25,
        )
        apply_initial_postures(
            world.backend,
            {"left": CONFIG.simulation.left_initial_joint_position_rad},
        )
        plant = world.backend.read_state(Twist.zero())
        with self.assertRaisesRegex(ValueError, "must match"):
            prepare_target_trajectory(
                world.backend,
                world.MOUNT_CALIBRATION,
                "left",
                trajectory,
                plant,
                desired_pos.configured_targets(),
                bad_motion,
            )


if __name__ == "__main__":
    unittest.main()
