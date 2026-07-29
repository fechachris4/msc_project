"""Cover the planning layer's controller seam, not just its arithmetic.

Three claims in ``planning/plan_source.py`` and ``planning/planner.py``
are behavioural rather than numerical, so they are tested by behaviour:

- The planner must be acceptable to the UNMODIFIED Runner.  That is a
  structural claim about ``controller.trajectory.DualArmTargetSource``,
  so it is checked with ``isinstance`` against the runtime-checkable
  protocol itself rather than by duck-typing a call.

- Lead compensation claims to REDUCE TRACKING LAG.  Matching the formula
  ``v/Kp + Kd*a/Kp^2`` proves only that the code implements the formula.
  The claim that matters is closed-loop, so the controller's velocity
  law is simulated here at the real 2 ms period and the deviation from
  the PLANNED path is measured with the prefilter on and off.

- Phase ownership exists so a replan is continuous.  A source without it
  is not merely inelegant: sampled at the Runner's large elapsed time it
  clamps to the new plan's FINAL waypoint, i.e. it commands a jump.  The
  test constructs that failure explicitly and then shows ``replace``
  avoids it.

Expectations are written out longhand from the documented formulas.  No
test calls the helper it is checking (``position_lead_m``,
``rotation_lead_rad``, ``_rotation_from_vector``) to build its own
expected value, because that would only prove the code equals itself.
"""

import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from controller.cylinder_router import CylinderKeepout
from controller.state import (
    DualArmFramedTargets,
    FramedTarget,
    Pose,
    TargetFrame,
    Twist,
)
from controller.trajectory import (
    CartesianWaypoint,
    CartesianWaypointTrajectory,
    DualArmTargetSource,
    KinematicTargetSample,
    TargetSource,
    TrajectoryLimits,
)
from planning import planner
from planning.path_optimizer import OptimizerParams
from planning.plan_source import (
    LeadCompensation,
    PlannedArmPath,
    PlannedDualArmSource,
    hold_source,
)
from runtime_config import CONFIG, PlanningConfig, load_config
from sim import world


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


class PlannedSourceSatisfiesRunnerSeamTest(unittest.TestCase):
    """The Runner must accept the planner with no controller change."""

    def setUp(self):
        self.path = PlannedArmPath(
            _straight_trajectory(
                [0.40, 0.10, 1.00], [0.60, 0.20, 1.05], 5.0
            ),
            _lead(True),
        )
        self.hold = PlannedArmPath(
            hold_source(Pose(np.array([0.4, -0.2, 1.1]), np.eye(3))),
            _lead(True),
        )
        self.source = PlannedDualArmSource(self.path, self.hold)

    def test_source_satisfies_the_dual_arm_protocol(self):
        self.assertIsInstance(self.source, DualArmTargetSource)

    def test_each_arm_path_satisfies_the_single_arm_protocol(self):
        self.assertIsInstance(self.path, TargetSource)
        self.assertIsInstance(self.hold, TargetSource)
        self.assertEqual(self.path.reference_frame, TargetFrame.WORLD)

    def test_sample_returns_dual_arm_framed_targets(self):
        targets = self.source.sample(1.0)
        self.assertIsInstance(targets, DualArmFramedTargets)
        self.assertIsInstance(targets.right, FramedTarget)
        self.assertIsInstance(targets.left, FramedTarget)
        self.assertIs(targets.for_arm("right"), targets.right)
        self.assertIs(targets.for_arm("left"), targets.left)

    def test_hold_arm_never_moves(self):
        first = self.source.sample(0.0).left.pose.position_m
        later = self.source.sample(7.5).left.pose.position_m
        np.testing.assert_array_equal(first, later)
        np.testing.assert_array_equal(
            self.source.sample(7.5).left.twist.linear_m_s, np.zeros(3)
        )

    def test_elapsed_time_must_be_finite_and_non_negative(self):
        with self.assertRaises(ValueError):
            self.source.sample(-0.001)
        with self.assertRaises(ValueError):
            self.source.sample(float("nan"))

    def test_unknown_arm_is_rejected(self):
        with self.assertRaises(ValueError):
            self.source.for_arm("third")


class LeadCompensationFormulaTest(unittest.TestCase):
    """The prefilter must be exactly ``v/Kp + Kd*a/Kp^2`` on pose only."""

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
        path = PlannedArmPath(
            _straight_trajectory(self.START_M, self.END_M, self.DURATION_S),
            _lead(enabled),
        )
        position = self.START_M.copy()
        peak_deviation_m = 0.0
        steps = int(round((self.DURATION_S + 3.0) / CONTROL_DT_S))
        for step in range(steps):
            elapsed_s = step * CONTROL_DT_S
            emitted = path.sample(elapsed_s)
            reference_m = emitted.pose.position_m
            reference_velocity_m_s = emitted.twist.linear_m_s
            velocity_m_s = (
                KP_POSITION_S_INV * (reference_m - position)
                + KD_POSITION * reference_velocity_m_s
            ) / (1.0 + KD_POSITION)
            position = position + velocity_m_s * CONTROL_DT_S
            planned_m = path.planned_sample(elapsed_s).target.pose.position_m
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


class PhaseOwnershipTest(unittest.TestCase):
    """A replan must start the new plan, not clamp to its last waypoint."""

    FIRST_START_M = np.array([0.40, 0.10, 1.00])
    FIRST_END_M = np.array([0.55, 0.10, 1.00])
    SECOND_END_M = np.array([0.55, 0.40, 1.20])
    PLAN_DURATION_S = 4.0
    LATE_ELAPSED_S = 100.0

    def setUp(self):
        self.first = _straight_trajectory(
            self.FIRST_START_M, self.FIRST_END_M, self.PLAN_DURATION_S
        )
        self.second = _straight_trajectory(
            self.FIRST_END_M, self.SECOND_END_M, self.PLAN_DURATION_S
        )
        self.path = PlannedArmPath(self.first, _lead(False))

    def test_a_source_without_phase_ownership_would_jump_to_the_goal(self):
        # This is the failure the phase origin exists to prevent: the
        # Runner's elapsed time never resets, so a plan pinned to that
        # clock is sampled far past its own duration.
        naive = PlannedArmPath(self.second, _lead(False))
        self.assertEqual(naive.phase_origin_s, 0.0)
        self.assertGreater(self.LATE_ELAPSED_S, self.second.duration_s)
        np.testing.assert_allclose(
            naive.sample(self.LATE_ELAPSED_S).pose.position_m,
            self.SECOND_END_M,
            rtol=0.0,
            atol=1e-12,
        )

    def test_replace_restarts_the_plan_clock_at_the_replan_instant(self):
        np.testing.assert_allclose(
            self.path.sample(self.LATE_ELAPSED_S).pose.position_m,
            self.FIRST_END_M,
            rtol=0.0,
            atol=1e-12,
        )

        self.path.replace(self.second, self.LATE_ELAPSED_S)

        self.assertEqual(self.path.phase_origin_s, self.LATE_ELAPSED_S)
        self.assertEqual(self.path.plan_time_s(self.LATE_ELAPSED_S), 0.0)
        np.testing.assert_allclose(
            self.path.sample(self.LATE_ELAPSED_S).pose.position_m,
            self.FIRST_END_M,
            rtol=0.0,
            atol=1e-12,
        )

    def test_the_replanned_path_then_advances_towards_its_own_goal(self):
        self.path.replace(self.second, self.LATE_ELAPSED_S)
        midway_m = self.path.sample(
            self.LATE_ELAPSED_S + 0.5 * self.PLAN_DURATION_S
        ).pose.position_m
        finished_m = self.path.sample(
            self.LATE_ELAPSED_S + self.PLAN_DURATION_S
        ).pose.position_m

        travelled_m = float(
            np.linalg.norm(midway_m - self.FIRST_END_M)
        )
        remaining_m = float(np.linalg.norm(midway_m - self.SECOND_END_M))
        self.assertGreater(travelled_m, 0.0)
        self.assertGreater(remaining_m, 0.0)
        np.testing.assert_allclose(
            finished_m, self.SECOND_END_M, rtol=0.0, atol=1e-12
        )

    def test_plan_time_never_goes_negative(self):
        self.path.replace(self.second, self.LATE_ELAPSED_S)
        self.assertEqual(
            self.path.plan_time_s(self.LATE_ELAPSED_S - 10.0), 0.0
        )

    def test_replace_rejects_a_non_source(self):
        with self.assertRaises(TypeError):
            self.path.replace(object(), 1.0)


class PlanFromStateTest(unittest.TestCase):
    """The composition root, exercised against a real plant state."""

    @classmethod
    def setUpClass(cls):
        world.backend.configure_torso_driver(None, None)
        cls.plant = world.backend.takeover()
        try:
            cls.calibration = world.MOUNT_CALIBRATION
        finally:
            world.backend.release()

    def test_enabled_cylinder_keepout_is_refused(self):
        with self.assertRaisesRegex(ValueError, "cylinder"):
            planner.plan_from_state(
                self.plant,
                self.calibration,
                cylinder_keepout=CylinderKeepout(enabled=True),
            )

    def test_disabled_cylinder_keepout_plans_successfully(self):
        outcome = planner.plan_from_state(
            self.plant,
            self.calibration,
            cylinder_keepout=CylinderKeepout(enabled=False),
        )
        self.assertIsInstance(outcome.source, PlannedDualArmSource)
        self.assertIsInstance(outcome.source, DualArmTargetSource)
        self.assertEqual(
            tuple(plan.side for plan in outcome.plans),
            planner.planned_sides(CONFIG.planning),
        )
        for plan in outcome.plans:
            self.assertTrue(plan.result.success, plan.result.message)
            self.assertGreaterEqual(plan.result.final_min_clearance_m, 0.0)
            self.assertGreater(plan.result.duration_s, 0.0)
        np.testing.assert_array_equal(
            outcome.torso_pose_world.position_m,
            self.plant.torso_pose_world.position_m,
        )

    def test_omitted_cylinder_keepout_refuses_while_the_router_is_enabled(self):
        # The guard must fire on the DEFAULT call path, because the shipped
        # config enables the router and it would silently rewrite the plan.
        self.assertTrue(CONFIG.cylinder_keepout.cylinder_keepout_enabled)
        with self.assertRaises(ValueError):
            planner.plan_from_state(self.plant, self.calibration)

    def test_plan_starts_at_the_measured_pose_and_ends_at_the_target(self):
        outcome = planner.plan_from_state(
            self.plant,
            self.calibration,
            cylinder_keepout=CylinderKeepout(enabled=False),
        )
        for plan in outcome.plans:
            path = outcome.source.for_arm(plan.side)
            np.testing.assert_allclose(
                path.sample(0.0).pose.position_m,
                plan.start_pose_world.position_m,
                rtol=0.0,
                atol=1e-9,
            )
            np.testing.assert_allclose(
                path.sample(path.duration_s + 1.0).pose.position_m,
                plan.goal_pose_world.position_m,
                rtol=0.0,
                atol=1e-9,
            )

    def test_unplanned_arm_holds_its_measured_pose(self):
        outcome = planner.plan_from_state(
            self.plant,
            self.calibration,
            cylinder_keepout=CylinderKeepout(enabled=False),
        )
        planned = planner.planned_sides(CONFIG.planning)
        for side in ("right", "left"):
            if side in planned:
                continue
            path = outcome.source.for_arm(side)
            np.testing.assert_array_equal(
                path.sample(0.0).pose.position_m,
                path.sample(50.0).pose.position_m,
            )

    def test_phase_origin_follows_the_replan_instant(self):
        outcome = planner.plan_from_state(
            self.plant,
            self.calibration,
            cylinder_keepout=CylinderKeepout(enabled=False),
            elapsed_time_s=37.5,
        )
        for plan in outcome.plans:
            path = outcome.source.for_arm(plan.side)
            self.assertEqual(path.phase_origin_s, 37.5)
            np.testing.assert_allclose(
                path.sample(37.5).pose.position_m,
                plan.start_pose_world.position_m,
                rtol=0.0,
                atol=1e-9,
            )

    def test_remaining_clearance_is_positive_for_a_fresh_plan(self):
        outcome = planner.plan_from_state(
            self.plant,
            self.calibration,
            cylinder_keepout=CylinderKeepout(enabled=False),
        )
        clearance_m = planner.remaining_clearance(
            outcome, self.plant, CONFIG.planning
        )
        self.assertGreater(clearance_m, 0.0)

    def test_wrong_argument_types_are_rejected(self):
        with self.assertRaises(TypeError):
            planner.plan_from_state("not a plant", self.calibration)
        with self.assertRaises(TypeError):
            planner.plan_from_state(self.plant, "not a calibration")
        with self.assertRaises(TypeError):
            planner.plan_from_state(
                self.plant, self.calibration, cylinder_keepout="on"
            )


class PlanningConfigMappingTest(unittest.TestCase):
    """Configuration must reach the optimiser without silent renaming."""

    def test_planned_sides_expands_both_and_passes_single_arms_through(self):
        self.assertEqual(
            planner.planned_sides(replace(CONFIG.planning, arm="both")),
            ("right", "left"),
        )
        self.assertEqual(
            planner.planned_sides(replace(CONFIG.planning, arm="right")),
            ("right",),
        )
        self.assertEqual(
            planner.planned_sides(replace(CONFIG.planning, arm="left")),
            ("left",),
        )

    def test_planned_sides_rejects_a_non_planning_config(self):
        with self.assertRaises(TypeError):
            planner.planned_sides(CONFIG.reactive_pose)

    def test_optimizer_params_carry_every_weight(self):
        configured = replace(
            CONFIG.planning,
            waypoint_count=5,
            dense_samples=41,
            clearance_margin_m=0.07,
            smoothness_weight=2.5,
            obstacle_weight=13.0,
            max_iterations=77,
            tool_radius_m=0.03,
        )
        params = planner.optimizer_params(configured)
        self.assertIsInstance(params, OptimizerParams)
        self.assertEqual(params.waypoint_count, 5)
        self.assertEqual(params.dense_samples, 41)
        self.assertEqual(params.clearance_margin_m, 0.07)
        self.assertEqual(params.smoothness_weight, 2.5)
        self.assertEqual(params.obstacle_weight, 13.0)
        self.assertEqual(params.max_iterations, 77)
        self.assertEqual(params.tool_radius_m, 0.03)

    def test_trajectory_limits_carry_every_rate(self):
        configured = replace(
            CONFIG.planning,
            max_linear_speed_m_s=0.11,
            max_linear_acceleration_m_s2=0.22,
            max_angular_speed_rad_s=0.33,
            max_angular_acceleration_rad_s2=0.44,
        )
        limits = planner.trajectory_limits(configured)
        self.assertIsInstance(limits, TrajectoryLimits)
        self.assertEqual(limits.max_linear_speed_m_s, 0.11)
        self.assertEqual(limits.max_linear_acceleration_m_s2, 0.22)
        self.assertEqual(limits.max_angular_speed_rad_s, 0.33)
        self.assertEqual(limits.max_angular_acceleration_rad_s2, 0.44)

    def test_committed_config_maps_onto_the_optimiser_defaults_in_use(self):
        params = planner.optimizer_params(CONFIG.planning)
        self.assertEqual(
            params.waypoint_count, CONFIG.planning.waypoint_count
        )
        self.assertEqual(
            params.obstacle_weight, CONFIG.planning.obstacle_weight
        )
        limits = planner.trajectory_limits(CONFIG.planning)
        self.assertEqual(
            limits.max_linear_speed_m_s,
            CONFIG.planning.max_linear_speed_m_s,
        )


class PlanningConfigParsingTest(unittest.TestCase):
    """``[planning]`` must fail loudly, like every other config section."""

    def _planning_variant(self, key, literal):
        """Rewrite one key INSIDE the [planning] block.

        ``enabled`` and ``arm`` are demo-tunable, and bare keys like
        ``arm = ...`` also occur in other sections, so anchoring on the
        section keeps these tests independent of the shipped values.
        """
        source = Path(CONFIG.source_path).read_text()
        start = source.index("[planning]")
        end = source.find("\n[", start + 1)
        end = len(source) if end == -1 else end
        block = source[start:end]
        rewritten, substitutions = re.subn(
            rf"^{re.escape(key)} = .*$",
            f"{key} = {literal}",
            block,
            count=1,
            flags=re.MULTILINE,
        )
        # Check the KEY was found, not that the text changed: rewriting a
        # key to its existing value is a legitimate no-op.
        self.assertEqual(substitutions, 1, f"{key} not found in [planning]")
        source = source[:start] + rewritten + source[end:]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "variant.toml"
            path.write_text(source)
            return load_config(path)

    def _load_variant(self, *replacements):
        source = Path(CONFIG.source_path).read_text()
        for old, new in replacements:
            self.assertIn(old, source)
            source = source.replace(old, new, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "variant.toml"
            path.write_text(source)
            return load_config(path)

    def test_committed_planning_section_parses(self):
        self.assertIsInstance(CONFIG.planning, PlanningConfig)
        # enabled/arm are demo-tunable; assert their CONTRACT, not a value.
        self.assertIsInstance(CONFIG.planning.enabled, bool)
        self.assertIn(CONFIG.planning.arm, ("right", "left", "both"))
        self.assertEqual(CONFIG.planning.waypoint_count, 3)
        self.assertEqual(CONFIG.planning.dense_samples, 60)
        self.assertEqual(CONFIG.planning.clearance_margin_m, 0.05)
        self.assertEqual(CONFIG.planning.smoothness_weight, 1.0)
        self.assertEqual(CONFIG.planning.obstacle_weight, 40.0)
        self.assertEqual(CONFIG.planning.max_iterations, 200)
        self.assertEqual(CONFIG.planning.tool_radius_m, 0.0)
        self.assertTrue(CONFIG.planning.lead_compensation_enabled)
        self.assertEqual(CONFIG.planning.replan_clearance_trigger_m, 0.01)
        self.assertTrue(CONFIG.planning.include_floor)
        self.assertEqual(CONFIG.planning.floor_height_world_m, 0.0)
        self.assertTrue(CONFIG.planning.include_torso_box)
        self.assertEqual(
            CONFIG.planning.torso_box_half_extent_m, (0.12, 0.18, 0.28)
        )
        self.assertEqual(CONFIG.planning.max_linear_speed_m_s, 0.2)
        self.assertEqual(
            CONFIG.planning.max_linear_acceleration_m_s2, 0.5
        )
        self.assertEqual(CONFIG.planning.max_angular_speed_rad_s, 0.5)
        self.assertEqual(
            CONFIG.planning.max_angular_acceleration_rad_s2, 1.0
        )

    def test_planning_section_is_immutable(self):
        with self.assertRaises(AttributeError):
            CONFIG.planning.obstacle_weight = 1.0

    def test_unknown_planning_key_fails(self):
        with self.assertRaisesRegex(ValueError, "planning keys differ"):
            self._load_variant(
                ("waypoint_count = 3", "waypoint_count = 3\nunexpected = 1")
            )

    def test_missing_planning_key_fails(self):
        with self.assertRaisesRegex(ValueError, "planning keys differ"):
            self._load_variant(("lead_compensation_enabled = true\n", ""))

    def test_negative_weights_fail(self):
        with self.assertRaisesRegex(
            ValueError, "planning.smoothness_weight"
        ):
            self._load_variant(
                ("smoothness_weight = 1.0", "smoothness_weight = -1.0")
            )
        with self.assertRaisesRegex(ValueError, "planning.obstacle_weight"):
            self._load_variant(
                ("obstacle_weight = 40.0", "obstacle_weight = -40.0")
            )
        with self.assertRaisesRegex(
            ValueError, "planning.clearance_margin_m"
        ):
            self._load_variant(
                ("clearance_margin_m = 0.05", "clearance_margin_m = -0.05")
            )

    def test_zero_waypoint_count_fails(self):
        with self.assertRaisesRegex(
            ValueError, "planning.waypoint_count must be a positive integer"
        ):
            self._load_variant(
                ("waypoint_count = 3", "waypoint_count = 0")
            )

    def test_planning_arm_accepts_only_supported_choices(self):
        for arm in ("right", "left", "both"):
            with self.subTest(arm=arm):
                loaded = self._planning_variant("arm", f'"{arm}"')
                self.assertEqual(loaded.planning.arm, arm)
        with self.assertRaisesRegex(ValueError, "planning.arm"):
            self._planning_variant("arm", '"upper"')

    def test_non_boolean_switch_fails(self):
        with self.assertRaisesRegex(
            ValueError, "planning.lead_compensation_enabled"
        ):
            self._load_variant(
                (
                    "lead_compensation_enabled = true",
                    "lead_compensation_enabled = 1",
                )
            )

    def test_torso_box_half_extent_must_hold_three_numbers(self):
        with self.assertRaisesRegex(
            ValueError, "planning.torso_box_half_extent_m"
        ):
            self._load_variant(
                (
                    "torso_box_half_extent_m = [0.12, 0.18, 0.28]",
                    "torso_box_half_extent_m = [0.12, 0.18]",
                )
            )


if __name__ == "__main__":
    unittest.main()
