"""Pure per-tick helpers in analysis.dashboard, and a guard on the
servo.py DAMPING fix the dashboard's gain panel depends on.

Hand-computation checks (per project rule: every computed quantity
checkable against ground truth) for headroom_frac, limit_margin_deg,
ctrl_lead_deg, sigma_min; matplotlib.use("Agg") before importing
analysis.dashboard so the module never needs a display.
"""

import unittest

import matplotlib

matplotlib.use("Agg")

import mujoco
import numpy as np

from analysis import dashboard


class HeadroomFracTest(unittest.TestCase):
    """max_i |qdot_raw_i| / QDOT_LIMIT_i, at a known fraction and joint."""

    def test_known_fraction_and_argmax_joint(self):
        from controller import servo

        qdot = np.zeros(7)
        qdot[3] = 0.5 * servo.QDOT_LIMIT[3]
        self.assertAlmostEqual(dashboard.headroom_frac(qdot), 0.5, places=12)

    def test_unclipped_can_exceed_one(self):
        from controller import servo

        qdot = np.zeros(7)
        qdot[0] = 1.5 * servo.QDOT_LIMIT[0]
        self.assertAlmostEqual(dashboard.headroom_frac(qdot), 1.5, places=12)


class LimitMarginDegTest(unittest.TestCase):
    """Min degrees-to-bound, over the limited joints only."""

    def test_hand_picked_bounds(self):
        q = np.array([0.0, 1.0, -0.5, 0.2, 0.0, 0.0, 0.0])
        q_low = np.array([-1.0, 0.0, -1.0, -1.0, -np.inf, -np.inf, -np.inf])
        q_high = np.array([1.0, 2.0, 1.0, 1.0, np.inf, np.inf, np.inf])
        limited = np.array([True, True, True, True, False, False, False])

        # per-joint min(q-low, high-q): 1.0, 1.0, 0.5, 0.8 -> min = 0.5 rad
        margin = dashboard.limit_margin_deg(q, q_low, q_high, limited)
        self.assertAlmostEqual(margin, np.degrees(0.5), places=10)

    def test_unlimited_joints_excluded(self):
        # Only joint 0 is limited, sitting exactly at its lower bound;
        # the other joints are far outside [-1, 1] but must not count.
        q = np.array([-1.0, 5.0, -5.0, 5.0, -5.0, 5.0, -5.0])
        q_low = np.full(7, -1.0)
        q_high = np.full(7, 1.0)
        limited = np.zeros(7, dtype=bool)
        limited[0] = True
        margin = dashboard.limit_margin_deg(q, q_low, q_high, limited)
        self.assertAlmostEqual(margin, 0.0, places=10)


class CtrlLeadDegTest(unittest.TestCase):
    def test_exact_max_abs_difference(self):
        q = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        ctrl = q + np.array([0.01, -0.02, 0.03, -0.19, 0.0, 0.05, -0.02])
        self.assertAlmostEqual(dashboard.ctrl_lead_deg(ctrl, q),
                               np.degrees(0.19), places=10)


class SigmaMinTest(unittest.TestCase):
    def test_matches_numpy_svd(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            J = rng.normal(size=(6, 7))
            expected = np.linalg.svd(J, compute_uv=False).min()
            self.assertAlmostEqual(dashboard.sigma_min(J), expected,
                                   places=10)


class RuntimeDampingTest(unittest.TestCase):
    """Guards the servo.py one-line fix: apply_ctrl must read servo.DAMPING
    at call time, not a def-time-bound default (before the fix, a live
    DAMPING mutation -- e.g. from the dashboard's slider -- would
    silently do nothing)."""

    HOME = [0.0, 0.26179939, 3.14159265, -2.26892803, 0.0,
            0.95993109, 1.57079633]

    def _ctrl_step(self):
        """Reset to HOME, offset the targets, apply_ctrl once at the
        current servo.DAMPING; return {side: ctrl_after - ctrl_before}."""
        from controller import frames, servo
        from sim import targets, world

        mujoco.mj_resetData(world.model, world.data)
        for side in world.SIDES:
            world.data.qpos[frames.qpos_adrs[side]] = self.HOME
        mujoco.mj_forward(world.model, world.data)
        for side in world.SIDES:
            pos, rot = frames.ee_pose(side)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, rot.flatten())
            targets.set_target(side, pos + np.array([0.05, 0.0, 0.05]))
            targets.set_target_quat(side, quat)
        servo.init_ctrl()

        before = {side: world.data.ctrl[world.ctrl_adrs[side]].copy()
                  for side in world.SIDES}
        servo.apply_ctrl(world.model.opt.timestep,
                         (np.zeros(3), np.zeros(3)))
        return {side: world.data.ctrl[world.ctrl_adrs[side]] - before[side]
                for side in world.SIDES}

    def test_apply_ctrl_honors_runtime_damping(self):
        from controller import servo
        from sim import world

        def reset():
            mujoco.mj_resetData(world.model, world.data)
            mujoco.mj_forward(world.model, world.data)

        self.addCleanup(reset)

        step_default = self._ctrl_step()

        original_damping = servo.DAMPING
        servo.DAMPING = 10.0
        try:
            step_huge = self._ctrl_step()
        finally:
            servo.DAMPING = original_damping

        for side in world.SIDES:
            self.assertGreater(np.linalg.norm(step_default[side]), 1e-6,
                              side)
            # Huge damping suppresses the DLS solve towards zero
            # (qdot_task = J^T(JJ^T + damping^2 I)^-1 v -> 0 as
            # damping -> inf), so the setpoint step must shrink markedly
            # if apply_ctrl actually reads DAMPING at call time.
            self.assertLess(
                np.linalg.norm(step_huge[side]),
                0.5 * np.linalg.norm(step_default[side]), side,
            )


if __name__ == "__main__":
    unittest.main()
