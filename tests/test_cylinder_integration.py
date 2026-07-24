"""Cylinder keep-out wired into the simulation: config, Runner, viewer.

Covers the parts the pure-geometry tests cannot: that the TOML schema matches
the C++ field names, that a disabled keep-out leaves the simulation's original
direct-target behaviour untouched, that an enabled keep-out routes the
end effector while preserving the requested orientation, and that the viewer
geometry is visualization-only and uses the same transform as the router.

All lengths are metres.
"""

import tempfile
from pathlib import Path
import unittest

import mujoco
import numpy as np

from controller.cylinder_router import CylinderKeepout, CylinderRouteKind
from controller.runner import ReactivePositionRunner, keepout_from_config
from runtime_config import CONFIG, load_config
from sim import cylinder_view, targets, world


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


def _run_one_cycle(keepout):
    """One Runner cycle against the shared MuJoCo backend."""
    world.backend.configure_torso_driver(None, None)
    runner = ReactivePositionRunner(
        world.backend,
        world.MOUNT_CALIBRATION,
        world.PIPELINE_SETUP,
        targets.framed_world_targets(),
        cylinder_keepout=keepout,
    )
    runner.start()
    try:
        return runner.cycle()
    finally:
        runner.close()


def _blocking_keepout(cycle, side="right"):
    """Build a keep-out that sits between the EE and its target, base frame."""
    base_pose = cylinder_view.base_pose_world(
        cycle.input_state, world.MOUNT_CALIBRATION, side)
    rotation = np.asarray(base_pose.rotation, dtype=float)
    origin = np.asarray(base_pose.position_m, dtype=float)

    ee_base = rotation.T @ (
        cycle.controller_states.for_arm(side).ee_pose_world.position_m - origin
    )
    target_base = rotation.T @ (
        cycle.resolved_targets.for_arm(side).pose_world.position_m - origin
    )
    midpoint = 0.5 * (ee_base + target_base)
    separation = float(np.linalg.norm(target_base[:2] - ee_base[:2]))
    return CylinderKeepout(
        enabled=True,
        center_xy_m=(midpoint[0], midpoint[1]),
        radius_m=max(0.05, 0.20 * separation),
        z_min_m=midpoint[2] - 1.0,
        z_max_m=midpoint[2] + 1.0,
        clearance_m=0.02,
        waypoint_tolerance_m=0.01,
    )


class CylinderConfigSchemaTest(unittest.TestCase):
    def test_committed_config_exposes_the_cpp_field_names(self):
        config = CONFIG.cylinder_keepout
        for key in CYLINDER_KEYS:
            self.assertTrue(
                hasattr(config, key), f"missing config field {key}")

    def test_defaults_match_the_hardware_controller(self):
        config = CONFIG.cylinder_keepout
        self.assertFalse(config.cylinder_keepout_enabled)
        self.assertEqual(config.cylinder_keepout_center_x_m, 0.0)
        self.assertEqual(config.cylinder_keepout_center_y_m, 0.0)
        self.assertEqual(config.cylinder_keepout_radius_m, 0.25)
        self.assertEqual(config.cylinder_keepout_z_min_m, 0.0)
        self.assertEqual(config.cylinder_keepout_z_max_m, 1.8)
        self.assertEqual(config.cylinder_keepout_clearance_m, 0.10)
        self.assertEqual(config.cylinder_waypoint_tolerance_m, 0.01)

    def test_keepout_from_config_copies_every_field(self):
        keepout = keepout_from_config(CONFIG.cylinder_keepout)
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
                "cylinder_keepout_enabled = false",
                "cylinder_keepout_enabled = false\ncylinder_unexpected = 1",
            ))
            with self.assertRaises(ValueError):
                load_config(path)


class DisabledKeepoutTest(unittest.TestCase):
    def test_disabled_leaves_the_resolved_target_untouched(self):
        cycle = _run_one_cycle(
            CylinderKeepout(enabled=False, radius_m=0.25))
        self.assertEqual(cycle.cylinder_routes, {})
        for side in world.SIDES:
            np.testing.assert_array_equal(
                cycle.routed_targets.for_arm(side).pose_world.position_m,
                cycle.resolved_targets.for_arm(side).pose_world.position_m,
            )
        self.assertIs(cycle.routed_targets, cycle.resolved_targets)

    def test_committed_default_config_is_disabled(self):
        self.assertFalse(
            CONFIG.cylinder_keepout.cylinder_keepout_enabled,
            "the committed default must preserve original behaviour",
        )


class EnabledKeepoutRunnerTest(unittest.TestCase):
    def setUp(self):
        self.baseline = _run_one_cycle(CylinderKeepout(enabled=False))
        self.keepout = _blocking_keepout(self.baseline, "right")
        self.cycle = _run_one_cycle(self.keepout)

    def test_blocked_target_produces_a_detour_route(self):
        status = self.cycle.cylinder_routes["right"]
        self.assertNotEqual(status.kind, "direct")
        self.assertGreater(status.waypoint_count, 1)
        self.assertTrue(status.route_changed)

    def test_routed_position_differs_from_the_direct_target(self):
        routed = self.cycle.routed_targets.right.pose_world.position_m
        direct = self.cycle.resolved_targets.right.pose_world.position_m
        self.assertGreater(float(np.linalg.norm(routed - direct)), 1e-6)

    def test_requested_orientation_is_preserved_on_intermediate_waypoints(self):
        status = self.cycle.cylinder_routes["right"]
        self.assertFalse(status.at_final_waypoint)
        np.testing.assert_allclose(
            self.cycle.routed_targets.right.pose_world.rotation,
            self.cycle.resolved_targets.right.pose_world.rotation,
            atol=0.0, rtol=0.0,
        )

    def test_route_waypoints_stay_outside_the_inflated_cylinder(self):
        from controller.cylinder_router import CylinderRouter

        router = CylinderRouter(self.keepout)
        base_pose = cylinder_view.base_pose_world(
            self.cycle.input_state, world.MOUNT_CALIBRATION, "right")
        rotation = np.asarray(base_pose.rotation, dtype=float)
        origin = np.asarray(base_pose.position_m, dtype=float)

        status = self.cycle.cylinder_routes["right"]
        points = [rotation.T @ (np.asarray(p, dtype=float) - origin)
                  for p in status.waypoints_world_m]
        start = rotation.T @ (
            self.cycle.controller_states.right.ee_pose_world.position_m
            - origin
        )
        previous = start
        for point in points:
            self.assertFalse(
                router.segment_intersects(previous, point),
                "a routed segment enters the inflated cylinder",
            )
            previous = point

    def test_target_is_never_refused(self):
        status = self.cycle.cylinder_routes["right"]
        self.assertGreater(status.waypoint_count, 0)
        self.assertTrue(
            np.all(np.isfinite(status.active_waypoint_world_m)))

    def test_route_survives_repeated_cycles_without_replanning(self):
        world.backend.configure_torso_driver(None, None)
        runner = ReactivePositionRunner(
            world.backend,
            world.MOUNT_CALIBRATION,
            world.PIPELINE_SETUP,
            targets.framed_world_targets(),
            cylinder_keepout=self.keepout,
        )
        runner.start()
        try:
            first = runner.cycle()
            changed = [first.cylinder_routes["right"].route_changed]
            for _ in range(5):
                changed.append(
                    runner.cycle().cylinder_routes["right"].route_changed)
        finally:
            runner.close()
        self.assertTrue(changed[0], "first cycle must accept the target")
        self.assertFalse(
            any(changed[1:]),
            "a static target must not be replanned every cycle",
        )


class CylinderViewTest(unittest.TestCase):
    def setUp(self):
        self.cycle = _run_one_cycle(CylinderKeepout(enabled=False))
        self.base_poses = cylinder_view.base_poses_world(
            self.cycle.input_state, world.MOUNT_CALIBRATION, world.SIDES)
        self.scene = mujoco.MjvScene(world.model, maxgeom=200)

    def test_disabled_draws_nothing(self):
        self.scene.ngeom = 0
        added = cylinder_view.draw(
            self.scene, CylinderKeepout(enabled=False), self.base_poses)
        self.assertEqual(added, 0)
        self.assertEqual(self.scene.ngeom, 0)

    def test_enabled_draws_physical_and_clearance_cylinders_per_arm(self):
        keepout = CylinderKeepout(
            enabled=True, radius_m=0.25, clearance_m=0.1,
            z_min_m=0.0, z_max_m=1.0)
        self.scene.ngeom = 0
        added = cylinder_view.draw(self.scene, keepout, self.base_poses)
        self.assertEqual(added, 2 * len(self.base_poses))

        # MjvGeom stores size/pos as float32, so compare at float32 precision.
        radii = sorted(
            float(self.scene.geoms[i].size[0]) for i in range(self.scene.ngeom)
        )
        self.assertAlmostEqual(radii[0], 0.25, places=6)
        self.assertAlmostEqual(radii[-1], 0.35, places=6)

    def test_drawn_geometry_uses_the_same_base_transform_as_the_router(self):
        keepout = CylinderKeepout(
            enabled=True, radius_m=0.25, clearance_m=0.1,
            z_min_m=0.0, z_max_m=1.0, center_xy_m=(0.1, -0.2))
        self.scene.ngeom = 0
        cylinder_view.draw(self.scene, keepout, self.base_poses)

        pose = self.base_poses["right"]
        rotation = np.asarray(pose.rotation, dtype=float)
        origin = np.asarray(pose.position_m, dtype=float)
        expected = origin + rotation @ np.array([0.1, -0.2, 0.5])

        # float32 scene storage: 1e-5 m is ~100x the representable resolution
        # here and still far below any geometric error that would matter.
        positions = [
            np.array(self.scene.geoms[i].pos) for i in range(self.scene.ngeom)
        ]
        self.assertTrue(
            any(float(np.linalg.norm(p - expected)) < 1e-5 for p in positions),
            "no drawn cylinder sits at the router's centre in world",
        )
        matrices = [
            np.array(self.scene.geoms[i].mat).reshape(3, 3)
            for i in range(self.scene.ngeom)
        ]
        self.assertTrue(
            any(float(np.max(np.abs(m - rotation))) < 1e-6 for m in matrices),
            "no drawn cylinder uses the arm base rotation",
        )

    def test_drawn_geometry_is_visualization_only(self):
        """user_scn geometry must not introduce collidable model geoms."""
        before = world.model.ngeom
        keepout = CylinderKeepout(enabled=True, radius_m=0.25)
        self.scene.ngeom = 0
        cylinder_view.draw(self.scene, keepout, self.base_poses)
        self.assertEqual(world.model.ngeom, before)

    def test_link_diagnostic_reports_without_changing_routing(self):
        pose = self.base_poses["right"]
        rotation = np.asarray(pose.rotation, dtype=float)
        origin = np.asarray(pose.position_m, dtype=float)
        shoulder = world.data.xpos[world.backend.arm_base_id["right"]]
        centre_base = rotation.T @ (np.asarray(shoulder, dtype=float) - origin)

        swallowing = CylinderKeepout(
            enabled=True,
            center_xy_m=(centre_base[0], centre_base[1]),
            radius_m=1.5, clearance_m=0.1,
            z_min_m=centre_base[2] - 2.0, z_max_m=centre_base[2] + 2.0)
        findings = cylinder_view.link_intersections(
            world.model, world.data, swallowing, self.base_poses)
        self.assertGreater(len(findings), 0)
        message = cylinder_view.format_link_intersections(findings)
        self.assertIn("not whole-arm avoidance", message)

    def test_link_diagnostic_is_silent_when_disabled(self):
        self.assertEqual(
            cylinder_view.link_intersections(
                world.model, world.data,
                CylinderKeepout(enabled=False), self.base_poses),
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
        self.assertIn("NOT whole-arm", enabled)


class OppositeSidesDemonstrationTest(unittest.TestCase):
    """Closed-loop demonstration: start and target on opposite sides.

    Runs ``analysis.cylinder_demo`` headless and checks that the arm actually
    walks the detour to its final waypoint, rather than only that a route was
    planned.
    """

    def tearDown(self):
        # The demo seeds a posture into the shared backend; restore defaults
        # so module ordering cannot leak state into other tests.
        world.backend.release()
        world.backend.reset()

    def test_demo_walks_the_detour_to_its_final_waypoint(self):
        from analysis import cylinder_demo
        from controller.cylinder_router import CylinderRouter

        keepout, demo_targets, start_base, target_base = (
            cylinder_demo.build_scenario())

        # The straight line really is blocked, and both endpoints are outside.
        router = CylinderRouter(keepout)
        self.assertTrue(router.segment_intersects(start_base, target_base))
        for point in (start_base, target_base):
            radial = float(
                np.linalg.norm(point[:2] - keepout.center))
            self.assertGreater(radial, keepout.obstacle_radius_m)

        world.backend.configure_torso_driver(None, None)
        runner = ReactivePositionRunner(
            world.backend,
            world.MOUNT_CALIBRATION,
            world.PIPELINE_SETUP,
            demo_targets,
            arms=(cylinder_demo.SIDE,),
            cylinder_keepout=keepout,
        )
        runner.start()
        try:
            reached_final = False
            kinds = set()
            for _ in range(12000):
                cycle = runner.cycle()
                status = cycle.cylinder_routes[cylinder_demo.SIDE]
                kinds.add(status.kind)
                if status.at_final_waypoint:
                    reached_final = True
                    break
        finally:
            runner.close()

        self.assertTrue(
            reached_final,
            "demo never reached its final waypoint within 12000 steps")
        self.assertNotIn("direct", kinds)
        self.assertEqual(len(kinds), 1, f"route changed mid-run: {kinds}")
        self.assertIn(
            kinds.pop(), {"clockwise", "counter-clockwise", "over"})

    def test_demo_route_segments_stay_outside_the_inflated_cylinder(self):
        from analysis import cylinder_demo
        from controller.cylinder_router import CylinderRouter

        keepout, _, start_base, target_base = cylinder_demo.build_scenario()
        router = CylinderRouter(keepout)
        route = router.plan(start_base, target_base)

        self.assertNotEqual(route.kind, CylinderRouteKind.DIRECT)
        previous = start_base
        for point in route.waypoints:
            self.assertFalse(
                router.segment_intersects(previous, point),
                "a demonstration segment enters the inflated cylinder",
            )
            previous = point
        np.testing.assert_allclose(route.waypoints[-1], target_base)


if __name__ == "__main__":
    unittest.main()
