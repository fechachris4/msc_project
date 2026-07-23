"""Reactive pose-controller and integrated-position pipeline checks.

Drives the real pipeline (target mocap -> explicit state -> pure controller)
at randomized arm configurations: zero at coincidence, known injected
offsets recovered exactly (e = reference - actual convention).
"""

import unittest
from dataclasses import replace

import mujoco
import numpy as np

from controller import reactive_controller
from controller.transforms import rotation_about_axis
from tests.control_test_support import (
    apply_cycle,
    pose_error,
    reconstruct_pipeline,
)

N_SAMPLES = 20
WRAP_TOL = 1e-9


def _solve_qdot(
    J,
    e_pos,
    e_rot,
    e_v,
    e_w,
    q,
    q_mid,
    null_gain,
    control,
    damping=None,
):
    return reactive_controller.solve_reactive_velocity(
        J,
        e_pos,
        e_rot,
        e_v,
        e_w,
        q,
        q_mid,
        null_gain,
        control,
        damping=damping,
    ).qdot_raw


def _arm_state(side, base_twist=None):
    from controller import frames
    from controller.state import Twist
    from sim import world

    twist = Twist.zero() if base_twist is None else Twist(*base_twist)
    return frames.arm_controller_state(
        world.read_state(twist), side, world.MOUNT_CALIBRATION)


def random_axis_angle(rng):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(-0.95 * np.pi, 0.95 * np.pi)
    return axis, angle


class PoseErrorWrapperTest(unittest.TestCase):
    def _randomize_arm(self, rng, side):
        from sim import world

        for i in range(1, 8):
            jnt_id = mujoco.mj_name2id(
                world.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_joint_{i}"
            )
            if world.model.jnt_limited[jnt_id]:
                low, high = world.model.jnt_range[jnt_id]
            else:
                low, high = -np.pi, np.pi
            world.data.qpos[world.model.jnt_qposadr[jnt_id]] = rng.uniform(
                low, high
            )

    def _check_arm(self, rng, side):
        from controller import reactive_controller
        from sim import targets

        state = _arm_state(side)
        ee_pos = state.ee_pose_world.position_m
        ee_rot = state.ee_pose_world.rotation

        # Target at the FK pose -> zero error.
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, ee_rot.flatten())
        targets.set_target(side, ee_pos)
        targets.set_target_quat(side, quat)
        e_pos, e_rot = reactive_controller.pose_error(
            state, targets.world_target(side))
        np.testing.assert_allclose(e_pos, np.zeros(3), atol=WRAP_TOL)
        np.testing.assert_allclose(e_rot, np.zeros(3), atol=WRAP_TOL)

        # Known offset -> recovered.
        delta_pos = rng.uniform(-0.2, 0.2, 3)
        axis, angle = random_axis_angle(rng)
        mujoco.mju_mat2Quat(
            quat, (rotation_about_axis(axis, angle) @ ee_rot).flatten()
        )
        targets.set_target(side, ee_pos + delta_pos)
        targets.set_target_quat(side, quat)
        e_pos, e_rot = reactive_controller.pose_error(
            state, targets.world_target(side))
        np.testing.assert_allclose(e_pos, delta_pos, atol=WRAP_TOL)
        np.testing.assert_allclose(e_rot, axis * angle, atol=WRAP_TOL)

    def test_wrappers_recover_offsets(self):
        from sim import world

        rng = np.random.default_rng(9)
        for _ in range(5):
            for side in world.SIDES:
                self._randomize_arm(rng, side)
            mujoco.mj_kinematics(world.model, world.data)

            for side in world.SIDES:
                self._check_arm(rng, side)


class QdotFromErrorTest(unittest.TestCase):
    """DLS law, pure math: tracks the commanded twist, tends to pinv."""

    def _random_case(self, rng):
        J = rng.normal(size=(6, 7))  # full rank with probability 1
        e_pos = rng.uniform(-0.2, 0.2, 3)
        e_rot = rng.uniform(-0.5, 0.5, 3)
        return J, e_pos, e_rot

    # q = q_mid makes the null-space term exactly zero, and zero
    # velocity errors make the D term exactly zero, so the DLS
    # properties are tested on the P task term alone.
    _Q0 = np.zeros(7)
    _V0 = np.zeros(3)

    def test_tracks_task_velocity_at_small_damping(self):
        from controller import servo

        rng = np.random.default_rng(10)
        for _ in range(N_SAMPLES):
            J, e_pos, e_rot = self._random_case(rng)
            v = np.concatenate([
                servo.CONTROL.kp_position_s_inv * e_pos,
                servo.CONTROL.kp_rotation_s_inv * e_rot,
            ])
            qdot = _solve_qdot(
                J, e_pos, e_rot, self._V0, self._V0,
                self._Q0, self._Q0,
                servo.CONTROL.null_gain_s_inv,
                servo.CONTROL,
                damping=1e-6,
            )
            np.testing.assert_allclose(J @ qdot, v, atol=1e-8)

    def test_matches_pinv_as_damping_vanishes(self):
        from controller import servo

        rng = np.random.default_rng(11)
        for _ in range(N_SAMPLES):
            J, e_pos, e_rot = self._random_case(rng)
            v = np.concatenate([
                servo.CONTROL.kp_position_s_inv * e_pos,
                servo.CONTROL.kp_rotation_s_inv * e_rot,
            ])
            qdot = _solve_qdot(
                J, e_pos, e_rot, self._V0, self._V0,
                self._Q0, self._Q0,
                servo.CONTROL.null_gain_s_inv,
                servo.CONTROL,
                damping=1e-9,
            )
            np.testing.assert_allclose(
                qdot, np.linalg.pinv(J) @ v, atol=1e-6
            )

    def test_null_space_centering(self):
        """Centering must not disturb the task and must drive the
        centered joints toward q_mid within the null space."""
        from controller import servo

        rng = np.random.default_rng(13)
        k_vec = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        for _ in range(N_SAMPLES):
            J, e_pos, e_rot = self._random_case(rng)
            q = rng.uniform(-2.0, 2.0, 7)
            q_mid = rng.uniform(-1.0, 1.0, 7)

            zero_v = np.zeros(3)
            qdot_plain = _solve_qdot(
                J, e_pos, e_rot, zero_v, zero_v, q_mid, q_mid,
                k_vec, servo.CONTROL, damping=1e-6)
            qdot_cent = _solve_qdot(
                J, e_pos, e_rot, zero_v, zero_v, q, q_mid,
                k_vec, servo.CONTROL, damping=1e-6)
            null_part = qdot_cent - qdot_plain

            # (a) null motion produces no task velocity
            np.testing.assert_allclose(J @ null_part, np.zeros(6),
                                       atol=1e-8)
            # (b) it points toward q_mid on the centered joints
            drive = k_vec * (q - q_mid)
            self.assertLess(float(drive @ null_part), 0.0)


class TwistErrorTest(unittest.TestCase):
    def test_twist_error_is_negated_ee_velocity(self):
        """Static targets => v_des = w_des = 0, so twist_error must equal
        the negated (already 3-leg-validated) frames.ee_velocity at any
        joint state, joint velocity, and base twist."""
        from sim import targets, world

        def reset():
            mujoco.mj_resetData(world.model, world.data)
            mujoco.mj_forward(world.model, world.data)

        self.addCleanup(reset)

        rng = np.random.default_rng(15)
        for _ in range(5):
            for side in world.SIDES:
                world.data.qpos[world.qpos_adrs[side]] = \
                    rng.uniform(-1.0, 1.0, 7)
                world.data.qvel[world.dof_adrs[side]] = \
                    rng.uniform(-1.0, 1.0, 7)
            mujoco.mj_kinematics(world.model, world.data)
            base_twist = (rng.uniform(-0.5, 0.5, 3),
                          rng.uniform(-0.5, 0.5, 3))

            for side in world.SIDES:
                state = _arm_state(side, base_twist)
                e_v, e_w = reactive_controller.twist_error(
                    state, targets.world_target(side))
                v_ee = state.ee_twist_world.linear_m_s
                w_ee = state.ee_twist_world.angular_rad_s
                np.testing.assert_allclose(e_v, -v_ee, atol=1e-12)
                np.testing.assert_allclose(e_w, -w_ee, atol=1e-12)

    def test_zero_at_rest(self):
        """Zero joint velocity + zero base twist -> both errors zero."""
        from sim import targets, world

        def reset():
            mujoco.mj_resetData(world.model, world.data)
            mujoco.mj_forward(world.model, world.data)

        self.addCleanup(reset)

        mujoco.mj_resetData(world.model, world.data)
        mujoco.mj_forward(world.model, world.data)
        zero_twist = (np.zeros(3), np.zeros(3))
        for side in world.SIDES:
            state = _arm_state(side, zero_twist)
            e_v, e_w = reactive_controller.twist_error(
                state, targets.world_target(side))
            np.testing.assert_array_equal(e_v, np.zeros(3))
            np.testing.assert_array_equal(e_w, np.zeros(3))


HOME = [0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109,
        1.57079633]


class ArmSelectionTest(unittest.TestCase):
    """apply_ctrl(arms=...) must update only the selected arm's setpoints."""

    def test_unselected_arm_setpoints_untouched(self):
        from controller import desired_pos, frames, servo
        from sim import world

        def reset():
            mujoco.mj_resetData(world.model, world.data)
            mujoco.mj_forward(world.model, world.data)

        self.addCleanup(reset)

        mujoco.mj_resetData(world.model, world.data)
        for side in world.SIDES:
            world.data.qpos[world.qpos_adrs[side]] = HOME
        mujoco.mj_forward(world.model, world.data)
        desired_pos.apply()
        pipeline = reconstruct_pipeline()

        before = {side: world.data.ctrl[world.ctrl_adrs[side]].copy()
                  for side in world.SIDES}
        apply_cycle(
            pipeline,
            world.model.opt.timestep,
            (np.zeros(3), np.zeros(3)),
            arms=("right",),
        )

        np.testing.assert_array_equal(
            world.data.ctrl[world.ctrl_adrs["left"]], before["left"]
        )
        self.assertTrue(np.any(
            world.data.ctrl[world.ctrl_adrs["right"]] != before["right"]
        ))


class ClosedLoopConvergenceTest(unittest.TestCase):
    """The real proof: under gravity, the P controller must drive both
    arms from home to a feasible world-frame target pose.

    Targets are FK poses of perturbed reachable configurations, feasible
    by construction — this tests the controller, not the reachability of
    any particular task point. (The desired_pos task points are exercised
    in main.py; the left one sits at joint_6's ctrl limit and keeps a
    ~6 mm residual there by design of the clip, not a controller bug.)"""

    SIM_SECONDS = 3.0
    POS_TOL = 0.005  # m
    ROT_TOL = 0.05   # rad

    def test_converges_static_base(self):
        from controller import servo
        from sim import targets, world

        def reset():
            mujoco.mj_resetData(world.model, world.data)
            mujoco.mj_forward(world.model, world.data)

        self.addCleanup(reset)

        mujoco.mj_resetData(world.model, world.data)
        rng = np.random.default_rng(12)
        home = np.array(HOME)

        for side in world.SIDES:
            world.data.qpos[world.qpos_adrs[side]] = \
                home + rng.uniform(-0.3, 0.3, 7)
            mujoco.mj_kinematics(world.model, world.data)
            state = _arm_state(side)
            pos = state.ee_pose_world.position_m
            rot = state.ee_pose_world.rotation
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, rot.flatten())
            targets.set_target(side, pos)
            targets.set_target_quat(side, quat)

        for side in world.SIDES:
            world.data.qpos[world.qpos_adrs[side]] = home
        mujoco.mj_forward(world.model, world.data)
        pipeline = reconstruct_pipeline()

        e0 = {side: np.linalg.norm(pose_error(side)[0])
              for side in world.SIDES}

        dt = world.model.opt.timestep
        zero_twist = (np.zeros(3), np.zeros(3))
        for _ in range(int(self.SIM_SECONDS / dt)):
            apply_cycle(pipeline, dt, zero_twist)
            mujoco.mj_step(world.model, world.data)

        for side in world.SIDES:
            e_pos, e_rot = pose_error(side)
            e_norm = np.linalg.norm(e_pos)
            self.assertLess(e_norm, self.POS_TOL, side)
            self.assertLess(e_norm, e0[side] / 10.0, side)
            self.assertLess(np.linalg.norm(e_rot), self.ROT_TOL, side)


class AntiWindupTest(unittest.TestCase):
    """A physically blocked arm (qpos frozen, task error persisting)
    must not wind the setpoint away from the joint state: |ctrl - qpos|
    stays within CTRL_LEAD. Without the clamp the integrator runs to
    the ctrl range on limited joints (the diagnosed q6 pinning) and
    without bound on the continuous ones, then snaps on release."""

    BLOCKED_SECONDS = 5.0

    def test_ctrl_lead_bounded_while_blocked(self):
        from controller import servo
        from sim import targets, world

        def reset():
            mujoco.mj_resetData(world.model, world.data)
            mujoco.mj_forward(world.model, world.data)

        self.addCleanup(reset)

        mujoco.mj_resetData(world.model, world.data)
        for side in world.SIDES:
            world.data.qpos[world.qpos_adrs[side]] = HOME
        mujoco.mj_forward(world.model, world.data)
        pipeline = reconstruct_pipeline()

        # Unreachable-while-blocked target: 0.5 m above the current EE.
        for side in world.SIDES:
            state = _arm_state(side)
            pos = state.ee_pose_world.position_m
            rot = state.ee_pose_world.rotation
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, rot.flatten())
            targets.set_target(side, pos + np.array([0.0, 0.0, 0.5]))
            targets.set_target_quat(side, quat)

        # Blocked arm: apply_ctrl runs, the sim never steps, qpos and
        # qvel stay frozen — persistent error, zero progress.
        dt = world.model.opt.timestep
        zero_twist = (np.zeros(3), np.zeros(3))
        for _ in range(int(self.BLOCKED_SECONDS / dt)):
            apply_cycle(pipeline, dt, zero_twist)

        for side in world.SIDES:
            lead = np.abs(world.data.ctrl[world.ctrl_adrs[side]]
                          - world.data.qpos[world.qpos_adrs[side]])
            self.assertLessEqual(
                lead.max(), servo.LIMITS.position_lead_rad + 1e-12, side)


class GainInvariantsTest(unittest.TestCase):
    """Structural constraints on the gains, independent of their tuned
    values — the other closed-loop tests use the gains themselves as
    the oracle, so a zeroed channel passes them trivially (commit
    a666077 zeroed KP_ROT for a diagnosis and it went unnoticed)."""

    def test_pose_hold_needs_both_channels(self):
        """The task is a world-frame POSE hold: both position and
        rotation need position-level feedback."""
        from controller import servo

        self.assertGreater(servo.CONTROL.kp_position_s_inv, 0.0)
        self.assertGreater(servo.CONTROL.kp_rotation_s_inv, 0.0)

    def test_defaults_are_the_last_verified_baseline(self):
        from controller import servo
        from runtime_config import CONFIG

        self.assertEqual(servo.CONTROL, CONFIG.reactive_pose)

    def test_later_tuning_is_preserved_only_as_exploratory(self):
        from controller.gain_sets import EXPLORATORY_CANDIDATES

        self.assertEqual(
            EXPLORATORY_CANDIDATES["committed_20_0.9"].as_overrides(),
            {"KP_POS": 20.0, "KP_ROT": 20.0,
             "KD_POS": 0.9, "KD_ROT": 0.9,
             "K_NULL": 0.5, "DAMPING": 0.05},
        )
        self.assertEqual(
            EXPLORATORY_CANDIDATES["working_tree_32_2"].as_overrides(),
            {"KP_POS": 32.0, "KP_ROT": 32.0,
             "KD_POS": 2.0, "KD_ROT": 2.0,
             "K_NULL": 0.5, "DAMPING": 0.05},
        )

    def test_kd_below_discrete_stability_boundary(self):
        """e_v feeds back measured qdot one step delayed — a discrete
        loop with gain ~KD that chatters at the step frequency as
        KD -> 1 (measured qddot 7.3 rad/s^2 at KD_POS = 1.0 vs 2.3 at
        0.3, and the composed-vs-FD velocity skew inflates ~8x)."""
        from controller import servo

        for kd in (
            servo.CONTROL.kd_position,
            servo.CONTROL.kd_rotation,
        ):
            self.assertGreaterEqual(kd, 0.0)
            self.assertLess(kd, 1.0)


class ImmutableControllerConfigTest(unittest.TestCase):
    """Different experiment gains use reconstructed immutable configs."""

    def test_damping_change_moves_qdot(self):
        from controller import servo

        rng = np.random.default_rng(20)
        J = rng.normal(size=(6, 7))
        e_pos = rng.uniform(-0.2, 0.2, 3)
        e_rot = rng.uniform(-0.5, 0.5, 3)
        zero_v = np.zeros(3)
        q = rng.uniform(-1.0, 1.0, 7)
        q_mid = np.zeros(7)

        control_a = replace(servo.CONTROL, dls_damping=0.05)
        qdot_a = _solve_qdot(
            J, e_pos, e_rot, zero_v, zero_v, q, q_mid,
            control_a.null_gain_s_inv, control_a)
        control_b = replace(servo.CONTROL, dls_damping=0.2)
        qdot_b = _solve_qdot(
            J, e_pos, e_rot, zero_v, zero_v, q, q_mid,
            control_b.null_gain_s_inv, control_b)
        self.assertFalse(np.allclose(qdot_a, qdot_b))

    def test_null_gain_vector_preserves_limited_joint_pattern(self):
        from controller import servo
        from sim import world

        control = replace(servo.CONTROL, null_gain_s_inv=3.0)
        for side in world.SIDES:
            mask = world.PIPELINE_SETUP.for_arm(side).centering.enabled
            vec = mask * control.null_gain_s_inv
            np.testing.assert_array_equal(
                vec != 0.0, mask)
            np.testing.assert_allclose(vec[mask], 3.0)


if __name__ == "__main__":
    unittest.main()
