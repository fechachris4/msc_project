"""Cover the planning layer's configuration schema and its controller seam.

Two claims matter here:

- ``[planning]`` must fail loudly on the same terms as every other config
  section: unknown/missing keys, out-of-range values, and boolean
  strictness are all checked directly against the committed TOML.

- ``plan_arm`` is the composition root exercised against a real plant
  state: it must return an ``ArmPlan`` whose delivered ``source`` starts at
  the measured end-effector pose and ends at the resolved goal, with the
  optimiser's own evidence (``success``, clearance, duration) intact.
"""

import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from controller.trajectory import TrajectoryLimits
from planning import planner
from planning.path_optimizer import OptimizerParams
from runtime_config import PlanningConfig, load_config
from sim import world

FIXTURE_CONFIG = Path(__file__).resolve().parent / "fixtures" / "control.toml"
CONFIG = load_config(FIXTURE_CONFIG)



class PlanArmTest(unittest.TestCase):
    """The composition root, exercised against a real plant state."""

    @classmethod
    def setUpClass(cls):
        world.backend.configure_torso_driver(None, None)
        cls.plant = world.backend.takeover()
        try:
            cls.calibration = world.MOUNT_CALIBRATION
        finally:
            world.backend.release()

    def test_plan_arm_returns_an_arm_plan_for_each_side(self):
        for side in ("right", "left"):
            with self.subTest(side=side):
                plan = planner.plan_arm(self.plant, self.calibration, side)
                self.assertIsInstance(plan, planner.ArmPlan)
                self.assertEqual(plan.side, side)

    def test_plan_starts_at_the_measured_pose_and_ends_at_the_goal(self):
        # Lead compensation adds pdot/Kp only while MOVING: at t=0 and at
        # rest at the end the reference twist is zero, so the lead vanishes
        # there and both endpoints are exact regardless of whether lead
        # compensation is enabled.
        for side in ("right", "left"):
            with self.subTest(side=side):
                plan = planner.plan_arm(self.plant, self.calibration, side)
                np.testing.assert_allclose(
                    plan.source.sample(0.0).pose.position_m,
                    plan.start_pose_world.position_m,
                    rtol=0.0,
                    atol=1e-9,
                )
                np.testing.assert_allclose(
                    plan.source.sample(
                        plan.source.duration_s + 1.0
                    ).pose.position_m,
                    plan.goal_pose_world.position_m,
                    rtol=0.0,
                    atol=1e-9,
                )

    def test_result_is_a_sane_optimizer_outcome(self):
        for side in ("right", "left"):
            with self.subTest(side=side):
                plan = planner.plan_arm(self.plant, self.calibration, side)
                self.assertTrue(plan.result.success, plan.result.message)
                self.assertGreaterEqual(
                    plan.result.final_min_clearance_m, 0.0
                )
                self.assertGreater(plan.result.duration_s, 0.0)

    def test_torso_pose_is_echoed_from_the_plant(self):
        plan = planner.plan_arm(self.plant, self.calibration, "right")
        np.testing.assert_array_equal(
            plan.torso_pose_world.position_m,
            self.plant.torso_pose_world.position_m,
        )

    def test_remaining_clearance_is_positive_for_a_fresh_plan(self):
        plans = [
            planner.plan_arm(self.plant, self.calibration, side)
            for side in ("right", "left")
        ]
        clearance_m = planner.remaining_clearance(
            plans, self.plant, CONFIG.planning
        )
        self.assertGreater(clearance_m, 0.0)

    def test_wrong_argument_types_are_rejected(self):
        with self.assertRaises(TypeError):
            planner.plan_arm("not a plant", self.calibration, "right")
        with self.assertRaises(TypeError):
            planner.plan_arm(self.plant, "not a calibration", "right")
        with self.assertRaises(TypeError):
            planner.plan_arm(
                self.plant,
                self.calibration,
                "right",
                planning_config="not a config",
            )

    def test_unknown_side_is_rejected(self):
        with self.assertRaises(ValueError):
            planner.plan_arm(self.plant, self.calibration, "third")


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
        # Anchor on the section HEADER line, not any substring occurrence:
        # prose comments elsewhere in the file reference "[planning]" too.
        header = re.search(r"^\[planning\]$", source, flags=re.MULTILINE)
        self.assertIsNotNone(header, "[planning] header not found")
        start = header.start()
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
