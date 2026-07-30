"""Cover the reactive-lag prefilter: its arithmetic and its source wrapper.

Two claims matter here, and both are behavioural rather than purely
numerical:

- ``LeadCompensation`` claims to be the exact first-order inverse of the
  controller's known tracking lag (see the derivation in
  ``controller/lead_compensation.py``).  Expectations below are written out
  longhand from the documented formulas; no test calls the helper it is
  checking (``position_lead_m``, ``rotation_lead_rad``) to build its own
  expected value, because that would only prove the code equals itself.

- Lead compensation claims to REDUCE TRACKING LAG.  Matching the formula
  ``v/Kp`` proves only that the code implements the formula.  The claim
  that matters is closed-loop, so the controller's velocity law is
  simulated here at the real 2 ms period and the deviation from the
  PLANNED path is measured with the prefilter on and off, wrapping a plain
  ``CartesianWaypointTrajectory`` in a ``LeadCompensatedSource``.
"""

import unittest
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from controller.lead_compensation import LeadCompensatedSource, LeadCompensation
from controller.state import FramedTarget, Pose, TargetFrame, Twist
from controller.trajectory import (
    CartesianWaypoint,
    CartesianWaypointTrajectory,
    KinematicTargetSample,
)
from runtime_config import CONFIG


KP_POSITION_S_INV = 2.0
KD_POSITION = 0.3
KP_ROTATION_S_INV = 2.0
KD_ROTATION = 0.3
CONTROL_DT_S = 0.002


def _lead(enabled):
    return LeadCompensation(
        KP_POSITION_S_INV,
        KD_POSITION,
        KP_ROTATION_S_INV,
        KD_ROTATION,
        enabled,
    )


def _rotation_matrix(rotation_vector):
    """Independent SO(3) exponential, from scipy rather than the project."""
    return Rotation.from_rotvec(np.asarray(rotation_vector)).as_matrix()


def _straight_trajectory(start_m, end_m, duration_s, rotation=None):
    """A two-waypoint minimum-jerk path; the simplest moving reference."""
    orientation = np.eye(3) if rotation is None else rotation
    return CartesianWaypointTrajectory(
        TargetFrame.WORLD,
        (
            CartesianWaypoint(
                0.0, Pose(np.asarray(start_m, dtype=float), orientation)
            ),
            CartesianWaypoint(
                float(duration_s),
                Pose(np.asarray(end_m, dtype=float), orientation),
            ),
        ),
    )


class LeadCompensationFormulaTest(unittest.TestCase):
    """The prefilter must be exactly ``v/Kp`` on pose only."""

    def setUp(self):
        self.position_m = np.array([0.40, -0.15, 1.10])
        self.rotation = _rotation_matrix([0.10, -0.20, 0.30])
        self.linear_m_s = np.array([0.12, -0.05, 0.30])
        self.angular_rad_s = np.array([0.20, 0.10, -0.40])
        self.linear_acceleration_m_s2 = np.array([0.60, 0.25, -0.10])
        self.angular_acceleration_rad_s2 = np.array([-0.30, 0.70, 0.15])
        self.sample = KinematicTargetSample(
            FramedTarget(
                TargetFrame.WORLD,
                Pose(self.position_m, self.rotation),
                Twist(self.linear_m_s, self.angular_rad_s),
            ),
            self.linear_acceleration_m_s2,
            self.angular_acceleration_rad_s2,
        )

    def test_disabled_lead_emits_the_planned_pose_exactly(self):
        emitted = _lead(False).apply(self.sample)
        self.assertIs(emitted, self.sample.target)
        np.testing.assert_array_equal(
            emitted.pose.position_m, self.position_m
        )
        np.testing.assert_array_equal(emitted.pose.rotation, self.rotation)

    def test_enabled_lead_adds_the_documented_position_offset(self):
        # Exact inverse is v/Kp ONLY. Because apply() passes the planned
        # twist through unchanged, the closed loop cancels at v/Kp and an
        # acceleration term would be injected error, not a refinement.
        expected = self.position_m + self.linear_m_s / KP_POSITION_S_INV
        emitted = _lead(True).apply(self.sample)
        np.testing.assert_allclose(
            emitted.pose.position_m, expected, rtol=0.0, atol=1e-15
        )

    def test_enabled_lead_premultiplies_the_planned_rotation(self):
        lead_vector = self.angular_rad_s / KP_ROTATION_S_INV
        expected = _rotation_matrix(lead_vector) @ self.rotation
        emitted = _lead(True).apply(self.sample)
        np.testing.assert_allclose(
            emitted.pose.rotation, expected, rtol=0.0, atol=1e-12
        )

    def test_lead_ignores_reference_acceleration(self):
        # Same twist, wildly different acceleration -> identical emitted pose.
        other = KinematicTargetSample(
            self.sample.target,
            self.linear_acceleration_m_s2 * 17.0 + 3.0,
            self.angular_acceleration_rad_s2 * 11.0 - 2.0,
        )
        np.testing.assert_array_equal(
            _lead(True).apply(self.sample).pose.position_m,
            _lead(True).apply(other).pose.position_m,
        )
        np.testing.assert_array_equal(
            _lead(True).apply(self.sample).pose.rotation,
            _lead(True).apply(other).pose.rotation,
        )

    def test_twist_passes_through_unchanged_in_both_modes(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                emitted = _lead(enabled).apply(self.sample)
                np.testing.assert_array_equal(
                    emitted.twist.linear_m_s, self.linear_m_s
                )
                np.testing.assert_array_equal(
                    emitted.twist.angular_rad_s, self.angular_rad_s
                )

    def test_reference_frame_is_preserved(self):
        self.assertEqual(
            _lead(True).apply(self.sample).reference_frame,
            TargetFrame.WORLD,
        )

    def test_lead_vanishes_where_the_plan_is_at_rest(self):
        rest = KinematicTargetSample(
            FramedTarget(
                TargetFrame.WORLD,
                Pose(self.position_m, self.rotation),
                Twist.zero(),
            ),
            np.zeros(3),
            np.zeros(3),
        )
        emitted = _lead(True).apply(rest)
        np.testing.assert_allclose(
            emitted.pose.position_m, self.position_m, rtol=0.0, atol=1e-15
        )
        np.testing.assert_allclose(
            emitted.pose.rotation, self.rotation, rtol=0.0, atol=1e-12
        )

    def test_from_config_drops_kd_when_velocity_feedback_is_off(self):
        without_velocity = replace(
            CONFIG.reactive_pose, velocity_enabled=False
        )
        lead = LeadCompensation.from_config(without_velocity)
        self.assertEqual(lead.kd_position, 0.0)
        self.assertEqual(lead.kd_rotation, 0.0)
        with_velocity = LeadCompensation.from_config(CONFIG.reactive_pose)
        self.assertEqual(
            with_velocity.kd_position, CONFIG.reactive_pose.kd_position
        )
        self.assertEqual(
            with_velocity.kp_position_s_inv,
            CONFIG.reactive_pose.kp_position_s_inv,
        )

    def test_negative_gains_and_non_samples_are_rejected(self):
        with self.assertRaises(ValueError):
            LeadCompensation(-1.0, 0.3, 2.0, 0.3, True)
        with self.assertRaises(ValueError):
            LeadCompensation(0.0, 0.3, 2.0, 0.3, True)
        with self.assertRaises(TypeError):
            _lead(True).apply("not a sample")


class LeadCompensatedSourceTest(unittest.TestCase):
    """The wrapper delivers the leaded pose but exposes the planned one too."""

    def setUp(self):
        self.trajectory = _straight_trajectory(
            [0.40, 0.10, 1.00], [0.60, 0.20, 1.05], 5.0
        )
        self.lead = _lead(True)
        self.source = LeadCompensatedSource(self.trajectory, self.lead)

    def test_sample_equals_lead_applied_to_the_inner_kinematic_sample(self):
        for elapsed_s in (0.0, 1.3, 2.5, 5.0, 7.0):
            with self.subTest(elapsed_s=elapsed_s):
                expected = self.lead.apply(
                    self.trajectory.sample_kinematics(elapsed_s)
                )
                emitted = self.source.sample(elapsed_s)
                self.assertIsInstance(emitted, FramedTarget)
                np.testing.assert_array_equal(
                    emitted.pose.position_m, expected.pose.position_m
                )
                np.testing.assert_array_equal(
                    emitted.pose.rotation, expected.pose.rotation
                )
                np.testing.assert_array_equal(
                    emitted.twist.linear_m_s, expected.twist.linear_m_s
                )
                np.testing.assert_array_equal(
                    emitted.twist.angular_rad_s, expected.twist.angular_rad_s
                )

    def test_planned_sample_returns_the_uncompensated_kinematic_sample(self):
        for elapsed_s in (0.0, 2.5, 5.0):
            with self.subTest(elapsed_s=elapsed_s):
                expected = self.trajectory.sample_kinematics(elapsed_s)
                planned = self.source.planned_sample(elapsed_s)
                self.assertIsInstance(planned, KinematicTargetSample)
                np.testing.assert_array_equal(
                    planned.target.pose.position_m,
                    expected.target.pose.position_m,
                )
                np.testing.assert_array_equal(
                    planned.target.twist.linear_m_s,
                    expected.target.twist.linear_m_s,
                )
                np.testing.assert_array_equal(
                    planned.linear_acceleration_m_s2,
                    expected.linear_acceleration_m_s2,
                )

    def test_reference_frame_duration_and_rates_pass_through(self):
        self.assertEqual(
            self.source.reference_frame, self.trajectory.reference_frame
        )
        self.assertEqual(self.source.duration_s, self.trajectory.duration_s)
        expected_rates = self.trajectory.maximum_rates()
        actual_rates = self.source.maximum_rates()
        self.assertEqual(
            actual_rates.max_linear_speed_m_s,
            expected_rates.max_linear_speed_m_s,
        )
        self.assertEqual(
            actual_rates.max_angular_speed_rad_s,
            expected_rates.max_angular_speed_rad_s,
        )

    def test_rejects_a_non_target_source(self):
        with self.assertRaises(TypeError):
            LeadCompensatedSource("not a source", self.lead)

    def test_rejects_a_non_lead_compensation(self):
        with self.assertRaises(TypeError):
            LeadCompensatedSource(self.trajectory, "not a lead")


class LeadCompensationReducesTrackingLagTest(unittest.TestCase):
    """The behavioural claim: the arm ends up on the PLANNED path.

    ``controller/reactive_controller.py`` resolves the task twist
    ``Kp*(r - x) + Kd*(rdot - xdot)`` into joint velocity, so in Cartesian
    space the closed loop is

        (1 + Kd) * xdot = Kp * (r - x) + Kd * rdot

    with ``rdot`` the reference twist the boundary delivers, which the
    prefilter deliberately leaves equal to the PLANNED velocity.  This is
    integrated at the real 2 ms period.
    """

    START_M = np.array([0.40, 0.10, 1.00])
    END_M = np.array([0.70, 0.20, 0.95])
    DURATION_S = 10.0

    def _simulate(self, enabled):
        """Return the peak distance between the arm and the PLANNED path."""
        source = LeadCompensatedSource(
            _straight_trajectory(self.START_M, self.END_M, self.DURATION_S),
            _lead(enabled),
        )
        position = self.START_M.copy()
        peak_deviation_m = 0.0
        steps = int(round((self.DURATION_S + 3.0) / CONTROL_DT_S))
        for step in range(steps):
            elapsed_s = step * CONTROL_DT_S
            emitted = source.sample(elapsed_s)
            reference_m = emitted.pose.position_m
            reference_velocity_m_s = emitted.twist.linear_m_s
            velocity_m_s = (
                KP_POSITION_S_INV * (reference_m - position)
                + KD_POSITION * reference_velocity_m_s
            ) / (1.0 + KD_POSITION)
            position = position + velocity_m_s * CONTROL_DT_S
            planned_m = source.planned_sample(elapsed_s).target.pose.position_m
            peak_deviation_m = max(
                peak_deviation_m,
                float(np.linalg.norm(position - planned_m)),
            )
        return peak_deviation_m

    def test_uncompensated_peak_matches_the_analytic_lag(self):
        # Minimum-jerk peak speed is 1.875 * distance / duration, and the
        # documented steady-state lag is that speed divided by Kp.
        distance_m = float(np.linalg.norm(self.END_M - self.START_M))
        analytic_lag_m = (
            1.875 * distance_m / self.DURATION_S
        ) / KP_POSITION_S_INV
        measured_m = self._simulate(False)
        self.assertAlmostEqual(
            measured_m, analytic_lag_m, delta=0.10 * analytic_lag_m
        )

    def test_compensation_shrinks_the_peak_deviation_at_least_fivefold(self):
        uncompensated_m = self._simulate(False)
        compensated_m = self._simulate(True)
        self.assertGreater(uncompensated_m, 0.0)
        self.assertGreater(compensated_m, 0.0)
        self.assertGreaterEqual(uncompensated_m / compensated_m, 5.0)
        self.assertLess(compensated_m, 0.005)


if __name__ == "__main__":
    unittest.main()
