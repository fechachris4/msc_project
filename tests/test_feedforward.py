"""Mount-velocity feedforward: the term behind the 7.5 -> 2.6 mm result.

With feedforward on, the commanded task twist gains
    ff = target_twist - (measured EE twist - J qdot),
i.e. it cancels the EE velocity the mount causes. These tests pin that
identity, and check that the term is exactly zero when it is switched off.
"""

import dataclasses
import unittest

import mujoco
import numpy as np

from controller import frames, reactive_controller, servo
from controller.state import Twist
from sim import targets, world


def _state_with_mount_twist(mount_twist, qdot):
    """Real controller state, with a chosen joint velocity and mount-induced EE twist."""
    mujoco.mj_resetData(world.model, world.data)
    mujoco.mj_forward(world.model, world.data)
    plant = world.read_state(Twist.zero())
    state = frames.controller_states(plant, world.MOUNT_CALIBRATION).right
    joints = dataclasses.replace(state.joints, velocity_rad_s=np.asarray(qdot, float))
    arm_twist = state.jacobian_world @ joints.velocity_rad_s
    total = arm_twist + np.asarray(mount_twist, float)
    ee_twist = Twist(linear_m_s=total[:3], angular_rad_s=total[3:])
    return dataclasses.replace(state, joints=joints, ee_twist_world=ee_twist)


class FeedforwardTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(mujoco.mj_resetData, world.model, world.data)
        self.target = targets.world_targets().right
        self.mount = np.array([0.03, -0.02, 0.05, 0.10, -0.05, 0.02])
        self.qdot = np.array([0.2, -0.1, 0.05, 0.3, -0.2, 0.1, 0.05])

    def _compute(self, feedforward_enabled, state):
        config = dataclasses.replace(
            servo.CONTROL, velocity_feedforward_enabled=feedforward_enabled)
        controller = reactive_controller.ReactiveController(
            config, world.PIPELINE_SETUP.right.centering)
        return controller.compute(state, self.target).solve

    def test_feedforward_cancels_mount_induced_twist(self):
        state = _state_with_mount_twist(self.mount, self.qdot)
        solve = self._compute(True, state)
        target_twist = np.concatenate([
            self.target.twist_world.linear_m_s,
            self.target.twist_world.angular_rad_s,
        ])
        np.testing.assert_allclose(
            solve.ff_twist, target_twist - self.mount, atol=1e-12)

    def test_arm_own_motion_is_not_fed_forward(self):
        still = self._compute(True, _state_with_mount_twist(self.mount, np.zeros(7)))
        moving = self._compute(True, _state_with_mount_twist(self.mount, self.qdot))
        np.testing.assert_allclose(moving.ff_twist, still.ff_twist, atol=1e-12)

    def test_feedforward_off_contributes_nothing(self):
        solve = self._compute(False, _state_with_mount_twist(self.mount, self.qdot))
        np.testing.assert_array_equal(solve.ff_twist, np.zeros(6))


if __name__ == "__main__":
    unittest.main()
