"""Scripted base motion in sim.motion.

Oracle check: set_torso_pose(t) must land the torso exactly where the
pure-math torso_pose_at(t) says. Behavior check: the closed loop must
keep the world-frame EE error well below the base amplitude while the
torso sways — the disturbance-rejection regression tripwire.
"""

from dataclasses import replace
import unittest

import mujoco
import numpy as np

from controller.transforms import rotation_from_rpy
from controller.state import Twist
from tests.control_test_support import (
    apply_cycle,
    pose_error,
    reconstruct_pipeline,
)

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
    # Pinned scenario, independent of the motion module's research
    # levers; rotation deliberately nonzero to exercise the rpy path.
    SCENARIO = dict(
        linear_amplitude=np.array([0.05, 0.02, 0.01]),
        linear_frequency=0.5,
        rotational_amplitude=np.radians([5.0, 3.0, 8.0]),
        rotational_frequency=0.3,
    )

    def test_public_levers_drive_default_pose_and_twist(self):
        from sim import motion

        lever_names = (
            "LINEAR_AMPLITUDE",
            "LINEAR_FREQUENCY",
            "ROTATIONAL_AMPLITUDE",
            "ROTATIONAL_FREQUENCY",
        )
        for name in lever_names:
            self.assertTrue(hasattr(motion, name), name)

        for amplitude in (
            motion.LINEAR_AMPLITUDE, motion.ROTATIONAL_AMPLITUDE
        ):
            self.assertEqual(np.asarray(amplitude).shape, (3,))
            self.assertTrue(np.all(np.isfinite(amplitude)))
        for frequency in (
            motion.LINEAR_FREQUENCY, motion.ROTATIONAL_FREQUENCY
        ):
            self.assertTrue(np.isscalar(frequency))
            self.assertTrue(np.isfinite(frequency))

        scenario = dict(
            linear_amplitude=motion.LINEAR_AMPLITUDE,
            linear_frequency=motion.LINEAR_FREQUENCY,
            rotational_amplitude=motion.ROTATIONAL_AMPLITUDE,
            rotational_frequency=motion.ROTATIONAL_FREQUENCY,
        )
        t = 1.23
        for actual, expected in zip(
            motion.torso_pose_at(t), motion.torso_pose_at(t, **scenario)
        ):
            np.testing.assert_allclose(actual, expected)
        for actual, expected in zip(
            motion.torso_twist_at(t), motion.torso_twist_at(t, **scenario)
        ):
            np.testing.assert_allclose(actual, expected)

    def test_set_torso_pose_matches_oracle(self):
        from controller import frames
        from sim import motion, world

        self.addCleanup(_restore_world)

        for t in (0.0, 0.31, 0.5, 1.0, 1.77, 4.2):
            motion.set_torso_pose(t, **self.SCENARIO)
            mujoco.mj_kinematics(world.model, world.data)

            plant = world.read_state(Twist.zero())
            pos = plant.torso_pose_world.position_m
            rot = plant.torso_pose_world.rotation
            exp_pos, exp_rpy = motion.torso_pose_at(t, **self.SCENARIO)
            np.testing.assert_allclose(pos, exp_pos, atol=1e-12)
            np.testing.assert_allclose(
                rot, rotation_from_rpy(exp_rpy), atol=1e-9
            )

    def test_zero_amplitude_writes_nothing(self):
        from sim import motion, world

        self.addCleanup(_restore_world)

        # Park the torso off-home (as a viewer hand-drag would).
        idx = motion._TORSO_MOCAP_IDX
        parked_pos = motion.HOME_POS + np.array([0.03, -0.02, 0.05])
        parked_quat = np.array([0.9689124, 0.247404, 0.0, 0.0])  # ~28 deg
        world.data.mocap_pos[idx] = parked_pos
        world.data.mocap_quat[idx] = parked_quat

        motion.set_torso_pose(1.23, linear_amplitude=np.zeros(3),
                              rotational_amplitude=np.zeros(3))

        np.testing.assert_array_equal(world.data.mocap_pos[idx], parked_pos)
        np.testing.assert_array_equal(world.data.mocap_quat[idx], parked_quat)


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
            world.data.qpos[world.qpos_adrs[side]] = HOME
        mujoco.mj_forward(world.model, world.data)

        # Feasible targets by construction: the arms' own FK poses at home.
        for side in world.SIDES:
            state = frames.arm_controller_state(
                world.read_state(Twist.zero()),
                side,
                world.MOUNT_CALIBRATION,
            )
            pos = state.ee_pose_world.position_m
            rot = state.ee_pose_world.rotation
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, rot.flatten())
            targets.set_target(side, pos)
            targets.set_target_quat(side, quat)
        pipeline = reconstruct_pipeline()

        dt = world.model.opt.timestep
        zero_twist = (np.zeros(3), np.zeros(3))
        for _ in range(int(self.SETTLE_SECONDS / dt)):
            apply_cycle(pipeline, dt, zero_twist)
            mujoco.mj_step(world.model, world.data)

        t_start = world.data.time  # phase 0 at motion start: no teleport
        peak = {side: 0.0 for side in world.SIDES}
        for _ in range(int(self.MOTION_SECONDS / dt)):
            t_rel = world.data.time - t_start
            motion.set_torso_pose(t_rel,
                                  linear_amplitude=self.TEST_AMPLITUDE,
                                  linear_frequency=self.TEST_FREQUENCY,
                                  rotational_amplitude=np.zeros(3))
            base_twist = motion.torso_twist_at(
                t_rel, linear_amplitude=self.TEST_AMPLITUDE,
                linear_frequency=self.TEST_FREQUENCY,
                rotational_amplitude=np.zeros(3))
            apply_cycle(pipeline, dt, base_twist)
            mujoco.mj_step(world.model, world.data)
            for side in world.SIDES:
                e_pos, _ = pose_error(side)
                peak[side] = max(peak[side], np.linalg.norm(e_pos))

        for side in world.SIDES:
            self.assertLess(peak[side], self.PEAK_TOL, f"{side}: {peak}")
            self.assertGreater(peak[side], self.PEAK_FLOOR, f"{side}: {peak}")


class PDvsPDisturbanceTest(unittest.TestCase):
    """The D term's acceptance proof. At 0.5 Hz the P law transmits
    0.84 of the base sway to the EE (measured); first-order theory for
    Kd = 0.3 predicts ~0.69. Same pinned scenario run twice — real
    gains vs KD monkeypatched to zero (the P-only law) — the PD peak
    must be strictly lower, with margin for servo-lag effects."""

    SETTLE_SECONDS = 2.0
    AMPLITUDE = np.array([0.05, 0.0, 0.0])  # m
    FREQUENCY = 0.5                          # Hz
    MOTION_SECONDS = 4.0                     # two periods
    PEAK_FLOOR = 0.005                       # m: disturbance engaged

    def _run_peak(self, control=None):
        from controller import frames, servo
        from sim import motion, targets, world

        control = servo.CONTROL if control is None else control

        mujoco.mj_resetData(world.model, world.data)
        for side in world.SIDES:
            world.data.qpos[world.qpos_adrs[side]] = HOME
        mujoco.mj_forward(world.model, world.data)
        for side in world.SIDES:
            state = frames.arm_controller_state(
                world.read_state(Twist.zero()),
                side,
                world.MOUNT_CALIBRATION,
            )
            pos = state.ee_pose_world.position_m
            rot = state.ee_pose_world.rotation
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, rot.flatten())
            targets.set_target(side, pos)
            targets.set_target_quat(side, quat)
        pipeline = reconstruct_pipeline(control)

        dt = world.model.opt.timestep
        zero_twist = (np.zeros(3), np.zeros(3))
        for _ in range(int(self.SETTLE_SECONDS / dt)):
            apply_cycle(pipeline, dt, zero_twist)
            mujoco.mj_step(world.model, world.data)

        t_start = world.data.time
        peak = {side: 0.0 for side in world.SIDES}
        for _ in range(int(self.MOTION_SECONDS / dt)):
            t_rel = world.data.time - t_start
            motion.set_torso_pose(t_rel, linear_amplitude=self.AMPLITUDE,
                                  linear_frequency=self.FREQUENCY,
                                  rotational_amplitude=np.zeros(3))
            base_twist = motion.torso_twist_at(
                t_rel, linear_amplitude=self.AMPLITUDE,
                linear_frequency=self.FREQUENCY,
                rotational_amplitude=np.zeros(3))
            apply_cycle(pipeline, dt, base_twist)
            mujoco.mj_step(world.model, world.data)
            for side in world.SIDES:
                e_pos, _ = pose_error(side)
                peak[side] = max(peak[side], np.linalg.norm(e_pos))
        return peak

    def test_pd_rejects_more_than_p(self):
        from controller import servo
        from sim import world

        self.addCleanup(_restore_world)

        peak_pd = self._run_peak()
        p_only = replace(
            servo.CONTROL, kd_position=0.0, kd_rotation=0.0)
        peak_p = self._run_peak(p_only)

        for side in world.SIDES:
            self.assertGreater(peak_pd[side], self.PEAK_FLOOR, side)
            self.assertLess(
                peak_pd[side], 0.9 * peak_p[side],
                f"{side}: PD {peak_pd[side] * 1000:.1f} mm vs "
                f"P {peak_p[side] * 1000:.1f} mm",
            )


if __name__ == "__main__":
    unittest.main()
