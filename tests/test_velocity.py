"""EE world velocity (frames.ee_velocity) — three independent legs.

Leg 1 (pure math): motion.torso_twist_at vs central finite differences
of motion.torso_pose_at — catches derivative and rpy-rate-map errors
with no simulator in the loop.

Leg 2 (independent library path): the arm term J_world @ qdot, reached
through ee_velocity with a zero base twist, vs Pinocchio's
getFrameVelocity — catches Jacobian and dof-indexing errors.
(jacobian_world itself is separately validated against MuJoCo's
mj_jacSite in test_kinematics.py.)

Leg 3 (end-to-end ground truth): the composed velocity vs central
finite differences of the measured MuJoCo EE pose over a closed-loop
rollout with linear + rotational base sway — the only leg that
exercises the w x r transport term, and the only one whose reference
comes from MuJoCo rather than Pinocchio.
"""

import unittest

import mujoco
import numpy as np
import pinocchio as pin

HOME = [0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109,
        1.57079633]

# Pinned scenario, independent of the motion module's research levers.
# Rotation nonzero (the whole point) but modest: the right arm hits the
# torso near +15.6 deg roll (analysis/diagnosis_report.md), and this
# comparison wants a contact-free rollout.
SCENARIO = dict(
    linear_amplitude=np.array([0.05, 0.02, 0.01]),
    linear_frequency=0.5,
    rotational_amplitude=np.radians([5.0, 3.0, 8.0]),
    rotational_frequency=0.3,
)


def _restore_world():
    from sim import motion, world

    idx = motion._TORSO_MOCAP_IDX
    world.data.mocap_pos[idx] = motion.HOME_POS
    world.data.mocap_quat[idx] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_resetData(world.model, world.data)
    mujoco.mj_forward(world.model, world.data)


class TorsoTwistOracleTest(unittest.TestCase):
    """torso_twist_at == d/dt torso_pose_at, by central differences."""

    H = 1e-6      # s; truncation O(H^2), FD roundoff ~1e-16/H = 1e-10
    ATOL = 1e-6   # m/s and rad/s

    # Second scenario with large angles: the rpy-rate -> w map is only
    # exercised away from identity when pitch/yaw are big.
    BIG = dict(
        linear_amplitude=np.array([0.02, 0.05, 0.03]),
        linear_frequency=0.7,
        rotational_amplitude=np.radians([30.0, 20.0, 40.0]),
        rotational_frequency=0.4,
    )

    def test_twist_matches_finite_difference_of_pose(self):
        from controller.transforms import rotation_from_rpy
        from sim import motion

        for scenario in (SCENARIO, self.BIG):
            for t in (0.0, 0.31, 0.5, 1.0, 1.77, 4.2):
                v, w = motion.torso_twist_at(t, **scenario)

                pos_p, rpy_p = motion.torso_pose_at(t + self.H, **scenario)
                pos_m, rpy_m = motion.torso_pose_at(t - self.H, **scenario)
                v_fd = (pos_p - pos_m) / (2.0 * self.H)
                w_fd = pin.log3(
                    rotation_from_rpy(rpy_p) @ rotation_from_rpy(rpy_m).T
                ) / (2.0 * self.H)

                np.testing.assert_allclose(v, v_fd, atol=self.ATOL)
                np.testing.assert_allclose(w, w_fd, atol=self.ATOL)


class ArmTwistPinocchioTest(unittest.TestCase):
    """ee_velocity with a zero base twist == R_W_K * getFrameVelocity,
    at random configurations, joint rates, and torso poses."""

    N_SAMPLES = 20
    ATOL = 1e-9

    def test_arm_term_matches_pinocchio(self):
        from controller import frames
        from sim import world

        self.addCleanup(_restore_world)

        pin_data = pin.Data(frames.pin_model)  # own scratch, not frames'
        mocap_idx = world.model.body_mocapid[world.torso_body_id]
        rng = np.random.default_rng(29)
        zero_twist = (np.zeros(3), np.zeros(3))

        for _ in range(self.N_SAMPLES):
            world.data.mocap_pos[mocap_idx] = (
                np.array([0.0, 0.0, 1.1]) + rng.uniform(-0.5, 0.5, 3)
            )
            quat = rng.standard_normal(4)
            world.data.mocap_quat[mocap_idx] = quat / np.linalg.norm(quat)

            for side in world.SIDES:
                q = np.empty(7)
                for i in range(1, 8):
                    jnt_id = mujoco.mj_name2id(
                        world.model, mujoco.mjtObj.mjOBJ_JOINT,
                        f"{side}_joint_{i}"
                    )
                    if world.model.jnt_limited[jnt_id]:
                        low, high = world.model.jnt_range[jnt_id]
                    else:
                        low, high = -np.pi, np.pi
                    q[i - 1] = rng.uniform(low, high)
                qdot = rng.uniform(-2.0, 2.0, 7)
                world.data.qpos[frames.qpos_adrs[side]] = q
                world.data.qvel[frames.dof_adrs[side]] = qdot
            mujoco.mj_kinematics(world.model, world.data)

            for side in world.SIDES:
                q = np.asarray(world.data.qpos[frames.qpos_adrs[side]])
                qdot = np.asarray(world.data.qvel[frames.dof_adrs[side]])
                v, w = frames.ee_velocity(side, zero_twist)

                pin.forwardKinematics(frames.pin_model, pin_data, q, qdot)
                pin.updateFramePlacements(frames.pin_model, pin_data)
                v_K = pin.getFrameVelocity(
                    frames.pin_model, pin_data, frames.ee_frame_id,
                    pin.LOCAL_WORLD_ALIGNED,
                )
                _, torso_rot = frames.torso_pose()
                R_W_K = torso_rot @ frames._T_T_K[side][:3, :3]

                np.testing.assert_allclose(v, R_W_K @ v_K.linear,
                                           atol=self.ATOL)
                np.testing.assert_allclose(w, R_W_K @ v_K.angular,
                                           atol=self.ATOL)


class ComposedVsMeasuredFDTest(unittest.TestCase):
    """Composed ee_velocity vs central finite differences of the
    measured MuJoCo EE pose, closed loop under base sway.

    Per iteration the mocap is rewritten for the current time and
    mj_kinematics re-run before sampling, so measured pose, qvel, and
    the analytic base twist all refer to the same instant. The FD
    reference is exact for the base part (smooth trajectory sampled at
    step times); the arm part carries an O(dt * qddot) skew from the
    discrete integrator, which sets the tolerance scale (mm/s at
    dt = 2 ms), not numerical precision.
    """

    SETTLE_SECONDS = 2.0
    MOTION_SECONDS = 5.0
    # Measured worst case 5.7e-4 against 0.33 m/s (rad/s) peaks — 0.17%
    # of peak, the O(dt * qddot) integrator skew. Pinned at ~4x margin.
    LIN_TOL = 2e-3    # m/s, max |composed - FD| per axis
    ANG_TOL = 2e-3    # rad/s
    LIN_FLOOR = 0.05  # m/s: peak measured speed must show real motion

    def test_composed_matches_measured_fd(self):
        from controller import frames, servo
        from sim import motion, targets, world

        self.addCleanup(_restore_world)

        mujoco.mj_resetData(world.model, world.data)
        for side in world.SIDES:
            world.data.qpos[frames.qpos_adrs[side]] = HOME
        mujoco.mj_forward(world.model, world.data)

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

        n = int(self.MOTION_SECONDS / dt)
        t_start = world.data.time  # phase 0 at motion start: no teleport
        p = {s: np.empty((n, 3)) for s in world.SIDES}
        R = {s: np.empty((n, 3, 3)) for s in world.SIDES}
        v = {s: np.empty((n, 3)) for s in world.SIDES}
        w = {s: np.empty((n, 3)) for s in world.SIDES}

        for k in range(n):
            tau = world.data.time - t_start
            motion.set_torso_pose(tau, **SCENARIO)
            mujoco.mj_kinematics(world.model, world.data)
            base_twist = motion.torso_twist_at(tau, **SCENARIO)
            for s in world.SIDES:
                p[s][k], R[s][k] = frames.measured_ee_pose(s)
                v[s][k], w[s][k] = frames.ee_velocity(s, base_twist)
            servo.apply_ctrl(dt)
            mujoco.mj_step(world.model, world.data)

        for s in world.SIDES:
            v_fd = (p[s][2:] - p[s][:-2]) / (2.0 * dt)
            w_fd = np.empty_like(v_fd)
            for k in range(len(w_fd)):
                w_fd[k] = pin.log3(R[s][k + 2] @ R[s][k].T) / (2.0 * dt)

            lin_err = np.abs(v[s][1:-1] - v_fd).max()
            ang_err = np.abs(w[s][1:-1] - w_fd).max()
            self.assertLess(lin_err, self.LIN_TOL, f"{s}: {lin_err}")
            self.assertLess(ang_err, self.ANG_TOL, f"{s}: {ang_err}")
            # the disturbance actually engaged (guards a dead rollout)
            self.assertGreater(np.linalg.norm(v_fd, axis=1).max(),
                               self.LIN_FLOOR, s)


if __name__ == "__main__":
    unittest.main()
