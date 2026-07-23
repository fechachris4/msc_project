"""Scripted EE-target motion in sim.target_motion.

Oracle check: set_target_pose(t, side) must land the target mocap exactly
where the pure-math target_pose_at(t, side) says — pinned with a nonzero
home rotation so the test actually exercises the matrix-composition
rotation path (today's desired_pos.POSES all use rpy=[0,0,0], so going
through apply() would give an identity home rotation and hide a bug
where the composition regresses to motion.py's elementwise-rpy-sum
shortcut). Behavior check: with the base held static and one EE target
driven sinusoidally, the closed loop must track it with an error well
below the commanded amplitude — the reference-tracking counterpart of
test_motion.py's BaseMotionRejectionTest.
"""

import unittest

import mujoco
import numpy as np

from controller.state import Twist

from controller.transforms import rotation_from_quat, rotation_from_rpy

HOME = [0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109,
        1.57079633]


def _restore_world():
    from sim import world

    for side in world.SIDES:
        idx = world.model.body_mocapid[world.target_body_id[side]]
        world.data.mocap_pos[idx] = np.zeros(3)
        world.data.mocap_quat[idx] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_resetData(world.model, world.data)
    mujoco.mj_forward(world.model, world.data)


class TargetPoseOracleTest(unittest.TestCase):
    """Pinned scenario, independent of the target_motion module's research
    levers; rotation deliberately nonzero (both the home and the
    perturbation) to exercise the R_home @ R_local(t) composition path."""

    SCENARIO = dict(
        linear_amplitude=np.array([0.03, -0.02, 0.01]),
        linear_frequency=0.4,
        rotational_amplitude=np.radians([4.0, -6.0, 9.0]),
        rotational_frequency=0.25,
    )
    HOME_POS = np.array([0.4, -0.2, 0.3])
    HOME_RPY = np.radians([10.0, -15.0, 20.0])  # nonzero: not identity

    def test_set_target_pose_matches_oracle(self):
        from sim import target_motion, targets, world

        self.addCleanup(_restore_world)

        side = "right"
        home_rot = rotation_from_rpy(self.HOME_RPY)
        home_quat = np.zeros(4)
        mujoco.mju_mat2Quat(home_quat, home_rot.flatten())
        # Bypass desired_pos.apply() for this one test — it always
        # resolves an identity home rotation today, which would hide the
        # exact Euler-composition bug this test exists to catch.
        targets.set_target(side, self.HOME_POS)
        targets.set_target_quat(side, home_quat)
        target_motion.init_home()

        for t in (0.0, 0.31, 0.5, 1.0, 1.77, 4.2):
            target_motion.set_target_pose(t, side, **self.SCENARIO)

            pos = targets.target_position(side)
            rot = rotation_from_quat(targets.target_quat(side))
            exp_pos, exp_rot = target_motion.target_pose_at(
                t, side, **self.SCENARIO)
            np.testing.assert_allclose(pos, exp_pos, atol=1e-12)
            np.testing.assert_allclose(rot, exp_rot, atol=1e-9)

    def test_zero_amplitude_writes_nothing(self):
        from sim import target_motion, targets, world

        self.addCleanup(_restore_world)

        side = "right"
        targets.set_target(side, self.HOME_POS)
        targets.set_target_quat(side, np.array([1.0, 0.0, 0.0, 0.0]))
        target_motion.init_home()

        # Park the target off-home (as a viewer hand-drag would).
        parked_pos = self.HOME_POS + np.array([0.02, -0.01, 0.03])
        parked_quat = np.array([0.9689124, 0.247404, 0.0, 0.0])  # ~28 deg
        targets.set_target(side, parked_pos)
        targets.set_target_quat(side, parked_quat)

        target_motion.set_target_pose(1.23, side,
                                      linear_amplitude=np.zeros(3),
                                      rotational_amplitude=np.zeros(3))

        np.testing.assert_array_equal(
            targets.target_position(side), parked_pos)
        np.testing.assert_array_equal(targets.target_quat(side), parked_quat)


class TargetTrackingBandwidthTest(unittest.TestCase):
    """Reference-tracking counterpart of BaseMotionRejectionTest: base
    held static, one EE target driven sinusoidally instead of the torso.

    Unlike the base-disturbance case, targets.target_velocity() is
    pinned at zero (no feedforward for a moving target — matches the
    current reactive-baseline phase scope, see target_motion.py's
    module docstring). So the D-term's e_v = 0 - v_ee damps against the
    EE's own tracking velocity instead of assisting it: linearizing the
    task-space law v_cmd = KP_POS*e_pos - KD_POS*v_ee and treating
    v_ee approx v_cmd (well-conditioned J) gives
    v_ee = KP_POS/(1+KD_POS) * e_pos — a first-order tracking law with
    effective bandwidth KP_eff = KP_POS/(1+KD_POS), not KP_POS alone.
    The same high-pass argument as BaseMotionRejectionTest then applies
    with KP_eff in place of KP: |e|/A = w/sqrt(w^2 + KP_eff^2), w = 2*pi*f.

    The test frequency is picked at a fixed fraction of KP_eff (rather
    than a pinned Hz value) so the prediction stays valid if the gains
    are retuned elsewhere; 10% of bandwidth predicts light attenuation
    (~20% of amplitude, since w/sqrt(w^2+KP_eff^2) is not tiny at that
    fraction), i.e. a "well tracked" regime, not a saturation-limit one.
    TOL_MARGIN/FLOOR_MARGIN give slack around the first-order prediction
    for DLS/null-space/discrete-time second-order effects; empirically
    the measured peak lands within ~5% of predicted."""

    SETTLE_SECONDS = 2.0
    TEST_AMPLITUDE = np.array([0.05, 0.0, 0.0])  # m
    BANDWIDTH_FRACTION = 0.1   # test frequency, as a fraction of KP_eff
    N_PERIODS = 3.0
    TOL_MARGIN = 1.3
    FLOOR_MARGIN = 0.5

    def test_closed_loop_tracks_moving_target(self):
        from controller import frames, servo
        from sim import target_motion, targets, world

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
        target_motion.init_home()
        servo.init_ctrl()

        kp_eff = (
            servo.CONTROL.kp_position_s_inv
            / (1.0 + servo.CONTROL.kd_position)
        )
        frequency = self.BANDWIDTH_FRACTION * kp_eff / (2.0 * np.pi)
        w = 2.0 * np.pi * frequency
        predicted_ratio = w / np.sqrt(w**2 + kp_eff**2)
        predicted_peak = predicted_ratio * np.linalg.norm(self.TEST_AMPLITUDE)
        peak_tol = self.TOL_MARGIN * predicted_peak
        peak_floor = self.FLOOR_MARGIN * predicted_peak
        motion_seconds = self.N_PERIODS / frequency

        dt = world.model.opt.timestep
        zero_twist = (np.zeros(3), np.zeros(3))
        for _ in range(int(self.SETTLE_SECONDS / dt)):
            servo.apply_ctrl(dt, zero_twist)
            mujoco.mj_step(world.model, world.data)

        t_start = world.data.time  # phase 0 at motion start: no teleport
        side = "right"
        peak = 0.0
        for _ in range(int(motion_seconds / dt)):
            t_rel = world.data.time - t_start
            target_motion.set_target_pose(
                t_rel, side, linear_amplitude=self.TEST_AMPLITUDE,
                linear_frequency=frequency,
                rotational_amplitude=np.zeros(3))
            servo.apply_ctrl(dt, zero_twist)
            mujoco.mj_step(world.model, world.data)
            e_pos, _ = servo.pose_error(side)
            peak = max(peak, np.linalg.norm(e_pos))

        self.assertLess(peak, peak_tol,
                        f"peak {peak * 1000:.1f} mm, "
                        f"predicted {predicted_peak * 1000:.1f} mm")
        self.assertGreater(peak, peak_floor,
                           f"peak {peak * 1000:.1f} mm, "
                           f"predicted {predicted_peak * 1000:.1f} mm")


if __name__ == "__main__":
    unittest.main()
