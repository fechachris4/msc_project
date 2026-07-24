import inspect
import unittest

import mujoco
import numpy as np

from controller import frames, position_actuation, reactive_controller, servo
from controller.state import Twist
from sim import targets, world
from tests.control_test_support import apply_cycle, reconstruct_pipeline


class PureControlBoundaryTest(unittest.TestCase):
    def test_control_modules_do_not_import_simulator_or_backend(self):
        for module in (reactive_controller, position_actuation, servo):
            source = inspect.getsource(module)
            self.assertNotIn("import mujoco", source, module.__name__)
            self.assertNotIn("from sim", source, module.__name__)
            self.assertNotIn("import sim", source, module.__name__)

    def test_pipeline_output_matches_controller_and_integrator_stages(self):
        mujoco.mj_resetData(world.model, world.data)
        mujoco.mj_forward(world.model, world.data)
        self.addCleanup(mujoco.mj_resetData, world.model, world.data)

        pipeline = reconstruct_pipeline()
        plant = world.read_state(Twist.zero())
        states = frames.controller_states(plant, world.MOUNT_CALIBRATION)
        world_targets = targets.world_targets()

        controller = reactive_controller.ReactiveController(
            servo.CONTROL, world.PIPELINE_SETUP.right.centering)
        expected = controller.compute(states.right, world_targets.right)
        command, traces = pipeline.step(
            states, world_targets, plant.nominal_dt_s, arms=("right",))

        np.testing.assert_allclose(
            traces["right"].qdot_raw, expected.solve.qdot_raw, atol=1e-12)
        np.testing.assert_allclose(
            command.right_position_rad,
            traces["right"].ctrl_after,
            atol=1e-12,
        )


class ReconstructionResetTest(unittest.TestCase):
    def setUp(self):
        mujoco.mj_resetData(world.model, world.data)
        mujoco.mj_forward(world.model, world.data)

    def tearDown(self):
        mujoco.mj_resetData(world.model, world.data)
        mujoco.mj_forward(world.model, world.data)

    def test_reconstruction_reseeds_from_fresh_plant_state(self):
        first = reconstruct_pipeline()
        initial = first.command()

        targets.set_target(
            "right", targets.target_position("right") + [0.3, 0.0, 0.0])
        apply_cycle(
            first, 1.0, (np.zeros(3), np.zeros(3)), arms=("right",))
        advanced = first.command()
        self.assertFalse(np.array_equal(
            advanced.right_position_rad, initial.right_position_rad))

        fresh_right = (
            world.data.qpos[world.qpos_adrs["right"]].copy() + 0.01
        )
        world.data.qpos[world.qpos_adrs["right"]] = fresh_right
        mujoco.mj_forward(world.model, world.data)
        second = reconstruct_pipeline()

        np.testing.assert_array_equal(
            second.command().right_position_rad, fresh_right)
        np.testing.assert_array_equal(
            first.command().right_position_rad,
            advanced.right_position_rad,
        )
        self.assertFalse(hasattr(second, "reset"))

    def test_unselected_arm_retains_its_persistent_command(self):
        pipeline = reconstruct_pipeline()
        before = pipeline.command().left_position_rad.copy()
        apply_cycle(
            pipeline,
            world.model.opt.timestep,
            (np.zeros(3), np.zeros(3)),
            arms=("right",),
        )
        np.testing.assert_array_equal(
            pipeline.command().left_position_rad, before)


if __name__ == "__main__":
    unittest.main()
