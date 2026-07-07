"""Scripted base motion in sim.motion.

Oracle check: set_torso_pose(t) must land the torso exactly where the
pure-math torso_pose_at(t) says. Behavior check: the closed loop must
keep the world-frame EE error well below the base amplitude while the
torso sways — the disturbance-rejection regression tripwire.
"""

import unittest

import mujoco
import numpy as np

from controller.transforms import rotation_from_rpy

HOME = [0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109,
        1.57079633]


def _restore_world():
    from sim import motion, world

    idx = motion._TORSO_MOCAP_IDX
    world.data.mocap_pos[idx] = motion.HOME_POS
    world.data.mocap_quat[idx] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_resetData(world.model, world.data)
    mujoco.mj_forward(world.model, world.data)


class TorsoPoseOracleTest(unittest.TestCase):
    def test_set_torso_pose_matches_oracle(self):
        from controller import frames
        from sim import motion, world

        self.addCleanup(_restore_world)

        for t in (0.0, 0.31, 0.5, 1.0, 1.77, 4.2):
            motion.set_torso_pose(t)
            mujoco.mj_kinematics(world.model, world.data)

            pos, rot = frames.torso_pose()
            exp_pos, exp_rpy = motion.torso_pose_at(t)
            np.testing.assert_allclose(pos, exp_pos, atol=1e-12)
            np.testing.assert_allclose(
                rot, rotation_from_rpy(exp_rpy), atol=1e-9
            )


class BaseMotionRejectionTest(unittest.TestCase):
    """Uncompensated, a 50 mm base sway shows up as ~50 mm world-frame EE
    error (fixed targets, moving base). First-order disturbance
    sensitivity is w/sqrt(w^2 + KP^2): at 0.5 Hz and KP=2 that is 0.84 —
    barely attenuated (the reactive limitation itself), useless as a
    pass/fail signal. So the test pins its own scenario, independent of
    the motion module's research levers: 50 mm sway at 0.1 Hz, inside
    the bandwidth. Predicted peak 0.30*50 = 15 mm vs 50 mm
    uncompensated; 25 mm proves real rejection with margin; the 5 mm
    floor proves the disturbance actually engaged."""

    SETTLE_SECONDS = 2.0
    TEST_AMPLITUDE = np.array([0.05, 0.0, 0.0])  # m
    TEST_FREQUENCY = 0.1        # Hz
    MOTION_SECONDS = 10.0       # one full period
    PEAK_TOL = 0.025            # m
    PEAK_FLOOR = 0.005          # m

    def test_closed_loop_bounded_under_base_motion(self):
        from controller import frames, servo
        from sim import motion, targets, world

        self.addCleanup(_restore_world)

        mujoco.mj_resetData(world.model, world.data)
        for side in world.SIDES:
            world.data.qpos[frames.qpos_adrs[side]] = HOME
        mujoco.mj_forward(world.model, world.data)

        # Feasible targets by construction: the arms' own FK poses at home.
        for side in world.SIDES:
            pos, rot = frames.ee_pose(side)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, rot.flatten())
            targets.set_target(side, pos)
            targets.set_target_quat(side, quat)
        servo.init_ctrl()

        dt = world.model.opt.timestep
        for _ in range(int(self.SETTLE_SECONDS / dt)):
            servo.apply_ctrl(dt)
            mujoco.mj_step(world.model, world.data)

        t_start = world.data.time  # phase 0 at motion start: no teleport
        peak = {side: 0.0 for side in world.SIDES}
        for _ in range(int(self.MOTION_SECONDS / dt)):
            motion.set_torso_pose(world.data.time - t_start,
                                  linear_amplitude=self.TEST_AMPLITUDE,
                                  linear_frequency=self.TEST_FREQUENCY)
            servo.apply_ctrl(dt)
            mujoco.mj_step(world.model, world.data)
            for side in world.SIDES:
                e_pos, _ = servo.pose_error(side)
                peak[side] = max(peak[side], np.linalg.norm(e_pos))

        for side in world.SIDES:
            self.assertLess(peak[side], self.PEAK_TOL, f"{side}: {peak}")
            self.assertGreater(peak[side], self.PEAK_FLOOR, f"{side}: {peak}")


if __name__ == "__main__":
    unittest.main()
