from dataclasses import replace
import unittest

import mujoco
import numpy as np

from controller.backend import PlantBackend
from controller.runner import ReactivePositionRunner
from controller.state import JointPositionCommand, Twist
from sim import targets, world


class _ReplyBackend:
    """BaseCyclic-shaped fake: command in, next feedback reply out."""

    def __init__(self, initial_state, next_state):
        self.initial_state = initial_state
        self.next_state = next_state
        self.commands = []
        self.taken_over = False
        self.released = False

    def takeover(self):
        self.taken_over = True
        return self.initial_state

    def exchange(self, command):
        if not self.taken_over:
            raise RuntimeError("not active")
        self.commands.append(command)
        return self.next_state

    def release(self):
        self.released = True
        self.taken_over = False


class BackendContractTest(unittest.TestCase):
    def setUp(self):
        world.backend.release()
        world.backend.configure_torso_driver(None, None)
        world.backend.reset()

    def tearDown(self):
        world.backend.release()
        world.backend.configure_torso_driver(None, None)
        world.backend.reset()

    def test_three_method_reply_backend_runs_same_controller(self):
        initial = world.read_state(Twist.zero())
        following = replace(
            initial,
            sample_time_s=initial.sample_time_s + initial.nominal_dt_s,
        )
        backend = _ReplyBackend(initial, following)
        self.assertIsInstance(backend, PlantBackend)

        runner = ReactivePositionRunner(
            backend,
            world.MOUNT_CALIBRATION,
            world.PIPELINE_SETUP,
            targets.framed_world_targets(),
        )
        runner.start()
        result = runner.cycle()
        runner.close()

        self.assertIs(result.input_state, initial)
        self.assertIs(result.next_state, following)
        self.assertEqual(len(backend.commands), 1)
        self.assertIsInstance(backend.commands[0], JointPositionCommand)
        self.assertTrue(backend.released)

    def test_mujoco_exchange_advances_exactly_one_backend_cycle(self):
        runner = ReactivePositionRunner(
            world.backend,
            world.MOUNT_CALIBRATION,
            world.PIPELINE_SETUP,
            targets.framed_world_targets(),
        )
        initial = runner.start()
        result = runner.cycle()
        runner.close()

        self.assertAlmostEqual(
            result.next_state.sample_time_s - initial.sample_time_s,
            initial.nominal_dt_s,
            places=15,
        )
        for side in world.SIDES:
            np.testing.assert_array_equal(
                world.data.ctrl[world.ctrl_adrs[side]],
                result.command.for_arm(side),
            )
        with self.assertRaises(RuntimeError):
            world.backend.exchange(result.command)

    def test_scripted_torso_is_sampled_at_reply_time(self):
        home = world.data.mocap_pos[
            world.model.body_mocapid[world.torso_body_id]
        ].copy()

        def pose_at(time_s):
            return home + np.array([0.1 * time_s, 0.0, 0.0]), np.zeros(3)

        def twist_at(_time_s):
            return np.array([0.1, 0.0, 0.0]), np.zeros(3)

        world.backend.configure_torso_driver(pose_at, twist_at)
        runner = ReactivePositionRunner(
            world.backend,
            world.MOUNT_CALIBRATION,
            world.PIPELINE_SETUP,
            targets.framed_world_targets(),
        )
        runner.start()
        result = runner.cycle()
        runner.close()

        expected_time = result.input_state.nominal_dt_s
        np.testing.assert_allclose(
            result.next_state.torso_pose_world.position_m,
            home + [0.1 * expected_time, 0.0, 0.0],
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            result.next_state.torso_twist_world.linear_m_s,
            [0.1, 0.0, 0.0],
        )

    def test_lifecycle_errors_do_not_add_controller_reset_paths(self):
        state = world.read_state(Twist.zero())
        backend = _ReplyBackend(state, state)
        runner = ReactivePositionRunner(
            backend,
            world.MOUNT_CALIBRATION,
            world.PIPELINE_SETUP,
            targets.framed_world_targets(),
        )
        with self.assertRaises(RuntimeError):
            runner.cycle()
        runner.start()
        with self.assertRaises(RuntimeError):
            runner.start()
        self.assertFalse(hasattr(runner, "reset"))
        runner.close()


if __name__ == "__main__":
    unittest.main()
