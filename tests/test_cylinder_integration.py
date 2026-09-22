"""Cylinder keep-out wired into the simulation: config, composition, viewer.

Covers the parts the pure-geometry tests cannot: the strict TOML schema, the
disabled passthrough, and a single world-vertical visualization using the
same ``CylinderKeepout`` the router consumes.

Routing is a COMPOSITION-time path transformation now (``arm_flow.py``): the
Runner no longer routes at runtime, so the routing coverage here builds
``ArmFlow``s with ``arm_flow.build_arm_flow`` directly and checks the
composed source, not a per-cycle follower status.

All lengths are metres.
"""

from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import mujoco
import numpy as np

import arm_flow
from analysis import cylinder_demo
from controller.cylinder_router import CylinderKeepout
from controller.runner import ReactivePositionRunner
from controller.state import (
    DualArmFramedTargets,
    FramedTarget,
    Pose,
    TargetFrame,
    Twist,
)
from controller.trajectory import IndependentArmTargetSource, StaticTargetSource
from runtime_config import load_config
from sim import cylinder_view, world

FIXTURE_CONFIG = Path(__file__).resolve().parent / "fixtures" / "control.toml"
CONFIG = load_config(FIXTURE_CONFIG)



CYLINDER_KEYS = (
    "cylinder_keepout_enabled",
    "cylinder_keepout_center_x_m",
    "cylinder_keepout_center_y_m",
    "cylinder_keepout_radius_m",
    "cylinder_keepout_z_min_m",
    "cylinder_keepout_z_max_m",
    "cylinder_keepout_clearance_m",
    "cylinder_waypoint_tolerance_m",
)


def _with_side_target(targets, side, target):
    """Replace one arm's framed target, leaving the other arm untouched."""
    values = {"right": targets.right, "left": targets.left}
    values[side] = target
    return DualArmFramedTargets(right=values["right"], left=values["left"])


class CylinderConfigSchemaTest(unittest.TestCase):
    def test_committed_config_exposes_the_cpp_field_names(self):
        config = CONFIG.cylinder_keepout
        for key in CYLINDER_KEYS:
            self.assertTrue(
                hasattr(config, key), f"missing config field {key}")

    def test_committed_world_cylinder_matches_the_scene_person(self):
        config = CONFIG.cylinder_keepout
        self.assertTrue(config.cylinder_keepout_enabled)
        self.assertEqual(config.cylinder_keepout_center_x_m, 0.0)
        self.assertEqual(config.cylinder_keepout_center_y_m, 0.0)
        self.assertEqual(config.cylinder_keepout_radius_m, 0.25)
        self.assertEqual(config.cylinder_keepout_z_min_m, 0.0)
        self.assertEqual(config.cylinder_keepout_z_max_m, 1.8)
        self.assertEqual(config.cylinder_keepout_clearance_m, 0.10)
        self.assertEqual(config.cylinder_waypoint_tolerance_m, 0.01)
        np.testing.assert_allclose(
            world.model.body_pos[world.torso_body_id][:2],
            [
                config.cylinder_keepout_center_x_m,
                config.cylinder_keepout_center_y_m,
            ],
            atol=0.0,
            rtol=0.0,
        )

    def test_committed_config_enables_the_central_keepout(self):
        self.assertTrue(
            CONFIG.cylinder_keepout.cylinder_keepout_enabled,
            "the central human keep-out must be enabled",
        )

    def test_keepout_from_config_copies_every_field(self):
        keepout = arm_flow.keepout_from_config(CONFIG.cylinder_keepout)
        config = CONFIG.cylinder_keepout
        self.assertEqual(keepout.enabled, config.cylinder_keepout_enabled)
        self.assertEqual(
            keepout.center_xy_m,
            (config.cylinder_keepout_center_x_m,
             config.cylinder_keepout_center_y_m),
        )
        self.assertEqual(keepout.radius_m, config.cylinder_keepout_radius_m)
        self.assertEqual(keepout.z_min_m, config.cylinder_keepout_z_min_m)
        self.assertEqual(keepout.z_max_m, config.cylinder_keepout_z_max_m)
        self.assertEqual(
            keepout.clearance_m, config.cylinder_keepout_clearance_m)
        self.assertEqual(
            keepout.waypoint_tolerance_m,
            config.cylinder_waypoint_tolerance_m,
        )

    def test_invalid_bounds_are_rejected_like_the_cpp(self):
        source = Path(CONFIG.source_path).read_text()
        bad_values = (
            ("cylinder_keepout_radius_m = 0.25",
             "cylinder_keepout_radius_m = 0.0"),
            ("cylinder_keepout_z_max_m = 1.8",
             "cylinder_keepout_z_max_m = 0.0"),
            ("cylinder_keepout_clearance_m = 0.10",
             "cylinder_keepout_clearance_m = -0.1"),
            ("cylinder_waypoint_tolerance_m = 0.01",
             "cylinder_waypoint_tolerance_m = 0.0"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control.toml"
            for original, replacement in bad_values:
                path.write_text(source.replace(original, replacement))
                with self.assertRaises(ValueError, msg=replacement):
                    load_config(path)

    def test_unknown_cylinder_key_is_rejected(self):
        source = Path(CONFIG.source_path).read_text()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control.toml"
            path.write_text(source.replace(
                "cylinder_keepout_enabled = true",
                "cylinder_keepout_enabled = true\ncylinder_unexpected = 1",
            ))
            with self.assertRaises(ValueError):
                load_config(path)


class ArmFlowRoutingCompositionTest(unittest.TestCase):
    """Keep-out routing is composed once, before the Runner ever starts."""

    def setUp(self):
        self.keepout, self.targets, self.start_world, self.target_world = (
            cylinder_demo.build_scenario())
        self.plant = world.read_state(Twist.zero())
        self.side = cylinder_demo.SIDE

    def tearDown(self):
        # The scenario seeds a posture into the shared backend; restore
        # defaults so module ordering cannot leak state into other tests.
        world.backend.release()
        world.backend.reset()

    def _build(self, targets=None, keepout=None):
        return arm_flow.build_arm_flow(
            world.backend,
            world.MOUNT_CALIBRATION,
            self.side,
            self.plant,
            self.targets if targets is None else targets,
            self.keepout if keepout is None else keepout,
        )

    def test_blocked_target_is_routed_around_the_keepout(self):
        flow = self._build()
        self.assertEqual(flow.kind, "routed_reach")
        self.assertIsNotNone(flow.route)
        self.assertNotEqual(flow.route.kind, "direct")
        self.assertGreater(len(flow.route.waypoints_world_m), 1)

    def test_routed_source_is_world_frame(self):
        flow = self._build()
        self.assertEqual(flow.source.reference_frame, TargetFrame.WORLD)

    def test_routed_path_stays_outside_the_inflated_cylinder(self):
        flow = self._build()
        keepout = self.keepout
        minimum_radial_m = float("inf")
        for elapsed in np.linspace(0.0, flow.duration_s, 400):
            position = np.asarray(
                flow.source.sample(elapsed).pose.position_m)
            if not (
                keepout.obstacle_z_min_m
                <= position[2]
                <= keepout.obstacle_z_max_m
            ):
                continue
            radial = float(np.linalg.norm(position[:2] - keepout.center))
            minimum_radial_m = min(minimum_radial_m, radial)
        self.assertGreaterEqual(
            minimum_radial_m, keepout.obstacle_radius_m - 1e-6)

    def test_routed_path_starts_at_measured_ee_and_ends_at_effective_target(
        self,
    ):
        flow = self._build()
        start_sample = flow.source.sample(0.0)
        np.testing.assert_allclose(
            np.asarray(start_sample.pose.position_m),
            self.start_world,
            atol=1e-9,
        )
        end_sample = flow.source.sample(flow.duration_s)
        np.testing.assert_allclose(
            np.asarray(end_sample.pose.position_m),
            flow.route.effective_target_world_m,
            atol=1e-9,
        )

    def test_sampled_twist_matches_the_finite_difference_of_position(self):
        flow = self._build()
        dt = 1e-4
        t = flow.duration_s / 2.0
        center = flow.source.sample(t)
        left = np.asarray(flow.source.sample(t - dt).pose.position_m)
        right = np.asarray(flow.source.sample(t + dt).pose.position_m)
        finite_difference_m_s = (right - left) / (2.0 * dt)
        np.testing.assert_allclose(
            np.asarray(center.twist.linear_m_s),
            finite_difference_m_s,
            atol=1e-4,
        )

    def test_clear_target_returns_a_static_flow(self):
        original = self.targets.for_arm(self.side)
        clear_target = FramedTarget(
            TargetFrame.WORLD,
            Pose(
                self.start_world + np.array([0.0, 0.0, 0.05]),
                original.pose.rotation,
            ),
            Twist.zero(),
        )
        flow = self._build(
            targets=_with_side_target(self.targets, self.side, clear_target))
        self.assertEqual(flow.kind, "static")
        self.assertIsInstance(flow.source, StaticTargetSource)
        self.assertIsNone(flow.route)

    def test_disabled_keepout_returns_a_static_flow_even_when_blocked(self):
        flow = self._build(keepout=replace(self.keepout, enabled=False))
        self.assertEqual(flow.kind, "static")
        self.assertIsInstance(flow.source, StaticTargetSource)
        self.assertIsNone(flow.route)


class RoutedReachClosedLoopTest(unittest.TestCase):
    """Closed-loop demonstration: the composed detour actually gets walked.

    Mirrors the old demo test's step budget, but the composed source and
    duration are now fixed at build time -- the Runner just samples them.
    """

    def tearDown(self):
        world.backend.release()
        world.backend.reset()

    def test_reactive_control_converges_on_the_effective_target(self):
        keepout, targets, start_world, target_world = (
            cylinder_demo.build_scenario())
        side = cylinder_demo.SIDE
        other_side = cylinder_demo.OTHER_SIDE
        plant = world.read_state(Twist.zero())

        flow = arm_flow.build_arm_flow(
            world.backend, world.MOUNT_CALIBRATION, side, plant,
            targets, keepout,
        )
        self.assertEqual(flow.kind, "routed_reach")

        held = StaticTargetSource(targets.for_arm(other_side))
        sources = {side: flow.source, other_side: held}
        source = IndependentArmTargetSource(
            right=sources["right"], left=sources["left"])

        world.backend.configure_torso_driver(None, None)
        runner = ReactivePositionRunner(
            world.backend,
            world.MOUNT_CALIBRATION,
            world.PIPELINE_SETUP,
            source,
            arms=(side,),
            human_safety_config=replace(
                CONFIG.human_safety, enabled=False
            ),
        )
        runner.start()
        try:
            cycle = None
            for _ in range(12000):
                cycle = runner.cycle()
        finally:
            runner.close()

        self.assertGreater(cycle.target_elapsed_time_s, flow.duration_s)
        final_position_m = np.asarray(
            cycle.controller_states.for_arm(side).ee_pose_world.position_m
        )
        error_m = float(np.linalg.norm(
            final_position_m - flow.route.effective_target_world_m))
        self.assertLess(
            error_m, 0.005,
            "reactive control did not converge on the routed target "
            f"within 12000 cycles (final error {error_m * 1000.0:.2f} mm)",
        )


class CylinderViewTest(unittest.TestCase):
    def setUp(self):
        self.scene = mujoco.MjvScene(world.model, maxgeom=200)

    def test_disabled_draws_nothing(self):
        self.scene.ngeom = 0
        added = cylinder_view.draw(
            self.scene, CylinderKeepout(enabled=False))
        self.assertEqual(added, 0)
        self.assertEqual(self.scene.ngeom, 0)

    def test_enabled_draws_one_physical_and_one_clearance_cylinder(self):
        keepout = CylinderKeepout(
            enabled=True, radius_m=0.25, clearance_m=0.1,
            z_min_m=0.0, z_max_m=1.0)
        self.scene.ngeom = 0
        added = cylinder_view.draw(self.scene, keepout)
        self.assertEqual(added, 2)

        # MjvGeom stores size/pos as float32, so compare at float32 precision.
        radii = sorted(
            float(self.scene.geoms[i].size[0]) for i in range(self.scene.ngeom)
        )
        self.assertAlmostEqual(radii[0], 0.25, places=6)
        self.assertAlmostEqual(radii[-1], 0.35, places=6)

    def test_drawn_geometry_is_world_centred_and_world_vertical(self):
        keepout = CylinderKeepout(
            enabled=True, radius_m=0.25, clearance_m=0.1,
            z_min_m=0.0, z_max_m=1.0, center_xy_m=(0.1, -0.2))
        self.scene.ngeom = 0
        cylinder_view.draw(self.scene, keepout)
        expected = np.array([0.1, -0.2, 0.5])

        # float32 scene storage: 1e-5 m is ~100x the representable resolution
        # here and still far below any geometric error that would matter.
        positions = [
            np.array(self.scene.geoms[i].pos) for i in range(self.scene.ngeom)
        ]
        np.testing.assert_allclose(positions[0], expected, atol=1e-5, rtol=0.0)
        matrices = [
            np.array(self.scene.geoms[i].mat).reshape(3, 3)
            for i in range(self.scene.ngeom)
        ]
        for matrix in matrices:
            np.testing.assert_allclose(
                matrix, np.eye(3), atol=1e-6, rtol=0.0)

    def test_drawn_geometry_is_visualization_only(self):
        """user_scn geometry must not introduce collidable model geoms."""
        before = world.model.ngeom
        keepout = CylinderKeepout(enabled=True, radius_m=0.25)
        self.scene.ngeom = 0
        cylinder_view.draw(self.scene, keepout)
        self.assertEqual(world.model.ngeom, before)

    def test_route_drawing_adds_segments_and_waypoints_with_no_highlight(
        self,
    ):
        """Composed routes have no live cursor: no active-waypoint sphere.

        Each RouteReport-like route contributes len(waypoints)-1 line
        segments plus len(waypoints) spheres -- one fewer geom per route
        than the old follower status, which also drew a highlight sphere.
        """
        keepout = CylinderKeepout(
            enabled=True, radius_m=0.25, clearance_m=0.1,
            z_min_m=0.0, z_max_m=1.0)
        waypoints = (
            np.array([0.3, 0.0, 0.5]),
            np.array([0.3, 0.2, 0.5]),
            np.array([0.1, 0.3, 0.5]),
        )
        route = arm_flow.RouteReport(
            kind="counter-clockwise",
            waypoints_world_m=waypoints,
            requested_target_world_m=waypoints[-1],
            effective_target_world_m=waypoints[-1],
            target_adjusted=False,
        )
        self.assertFalse(hasattr(route, "active_waypoint_world_m"))

        self.scene.ngeom = 0
        added = cylinder_view.draw(self.scene, keepout, {"right": route})
        expected = 2 + (len(waypoints) - 1) + len(waypoints)
        self.assertEqual(added, expected)

    def test_link_diagnostic_reports_without_changing_routing(self):
        shoulder = world.data.xpos[world.backend.arm_base_id["right"]]

        swallowing = CylinderKeepout(
            enabled=True,
            center_xy_m=(shoulder[0], shoulder[1]),
            radius_m=1.5, clearance_m=0.1,
            z_min_m=shoulder[2] - 2.0, z_max_m=shoulder[2] + 2.0)
        findings = cylinder_view.link_intersections(
            world.model, world.data, swallowing)
        self.assertGreater(len(findings), 0)
        self.assertFalse(
            any(name.endswith("_target") for _, name, _ in findings))
        message = cylinder_view.format_link_intersections(findings)
        self.assertIn("not whole-arm avoidance", message)

    def test_link_diagnostic_is_silent_when_disabled(self):
        self.assertEqual(
            cylinder_view.link_intersections(
                world.model, world.data,
                CylinderKeepout(enabled=False)),
            [],
        )
        self.assertIsNone(cylinder_view.format_link_intersections([]))

    def test_banner_states_enabled_or_disabled(self):
        disabled = cylinder_view.describe(
            CylinderKeepout(enabled=False), world.SIDES)
        enabled = cylinder_view.describe(
            CylinderKeepout(enabled=True), world.SIDES)
        self.assertIn("DISABLED", disabled)
        self.assertIn("ENABLED", enabled)
        self.assertIn("WORLD", enabled)
        self.assertIn("one central cylinder", enabled)
        self.assertIn("NOT whole-arm", enabled)


if __name__ == "__main__":
    unittest.main()
