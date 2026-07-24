"""Cylinder-router port tests.

Two layers:

1. The cases ported from the hardware controller's
   ``tests/test_control_logic.cpp`` ``TestCylinderRouter()``, plus the extra
   cases requested for this port (direct, clockwise, counter-clockwise, over,
   inside target, exact boundary, disabled, waypoint advancement, and an
   opposite-sides demonstration).

2. A cross-check that compiles the UNMODIFIED C++ ``CylinderRouter.cpp`` and
   compares route kind, effective target, waypoint count and waypoint
   coordinates against this Python port. That layer skips (it does not fail)
   when the C++ tree, g++ or Eigen are unavailable, so the suite still runs on
   a machine that only has the simulation.

All lengths are metres.
"""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from controller.cylinder_router import (
    ARC_STEP_RAD,
    MAX_WAYPOINTS,
    ROUTE_PADDING_M,
    CylinderKeepout,
    CylinderRouteFollower,
    CylinderRouteKind,
    CylinderRouter,
    route_kind_name,
)


CPP_ROOT = Path(
    "/home/christian/Desktop/HumanSL_MAIN/Christian_control/basic_control"
)
CPP_SOURCE = CPP_ROOT / "src" / "control" / "CylinderRouter.cpp"
DUMPER_SOURCE = Path(__file__).resolve().parent / "cpp_reference" / "route_dump.cpp"
EIGEN_INCLUDE = Path("/usr/include/eigen3")

_BUILD_CACHE = {}


def _reference_binary():
    """Compile the C++ dumper once per process; None when unavailable."""
    if "path" in _BUILD_CACHE:
        return _BUILD_CACHE["path"]
    _BUILD_CACHE["path"] = None
    if not (CPP_SOURCE.is_file() and DUMPER_SOURCE.is_file()):
        return None
    if shutil.which("g++") is None or not EIGEN_INCLUDE.is_dir():
        return None
    directory = Path(tempfile.mkdtemp(prefix="cylinder_router_ref_"))
    binary = directory / "route_dump"
    result = subprocess.run(
        [
            "g++", "-std=c++17", "-O2",
            f"-I{EIGEN_INCLUDE}",
            f"-I{CPP_ROOT / 'src'}",
            str(DUMPER_SOURCE),
            str(CPP_SOURCE),
            "-o", str(binary),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not binary.is_file():
        return None
    _BUILD_CACHE["path"] = binary
    return binary


def _reference_routes(cases):
    """Run the C++ implementation over ``cases``; one dict per case."""
    binary = _reference_binary()
    if binary is None:
        return None
    payload = "\n".join(
        " ".join(
            str(int(value)) if index == 0 else repr(float(value))
            for index, value in enumerate(case)
        )
        for case in cases
    )
    result = subprocess.run(
        [str(binary)], input=payload, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise AssertionError(f"C++ reference failed: {result.stderr}")
    lines = [line for line in result.stdout.strip().split("\n") if line]
    if len(lines) != len(cases):
        raise AssertionError(
            f"C++ reference returned {len(lines)} rows for {len(cases)} cases"
        )
    return [json.loads(line) for line in lines]


def _case(keepout, start, target):
    """Flatten one scenario into the dumper's input row."""
    return (
        1 if keepout.enabled else 0,
        keepout.center_xy_m[0],
        keepout.center_xy_m[1],
        keepout.radius_m,
        keepout.z_min_m,
        keepout.z_max_m,
        keepout.clearance_m,
        keepout.waypoint_tolerance_m,
        start[0], start[1], start[2],
        target[0], target[1], target[2],
    )


def _keepout(**overrides):
    """The C++ test's keep-out (test_control_logic.cpp lines 159-166)."""
    values = dict(
        enabled=True,
        center_xy_m=(0.0, 0.0),
        radius_m=0.5,
        clearance_m=0.1,
        z_min_m=0.0,
        z_max_m=2.0,
        waypoint_tolerance_m=0.01,
    )
    values.update(overrides)
    return CylinderKeepout(**values)


LEFT = np.array([-1.0, 0.0, 1.0])
RIGHT = np.array([1.0, 0.0, 1.0])


class PortedCppCasesTest(unittest.TestCase):
    """Line-for-line port of TestCylinderRouter() in test_control_logic.cpp."""

    def setUp(self):
        self.keepout = _keepout()
        self.router = CylinderRouter(self.keepout)

    def test_segment_through_cylinder_is_detected(self):
        self.assertTrue(self.router.segment_intersects(LEFT, RIGHT))

    def test_segment_above_cylinder_is_clear(self):
        self.assertFalse(
            self.router.segment_intersects(
                np.array([-1.0, 0.0, 2.11]), np.array([1.0, 0.0, 2.11])
            )
        )

    def test_crossing_target_gets_an_automatic_detour(self):
        route = self.router.plan(LEFT, RIGHT)
        self.assertNotEqual(route.kind, CylinderRouteKind.DIRECT)
        self.assertGreater(route.size, 1)
        self.assertLess(
            float(np.linalg.norm(route.effective_target - RIGHT)), 1e-12
        )

    def test_every_detour_segment_stays_outside_the_cylinder(self):
        route = self.router.plan(LEFT, RIGHT)
        previous = LEFT
        for point in route.waypoints:
            self.assertFalse(
                self.router.segment_intersects(previous, point),
                f"segment {previous} -> {point} enters the inflated cylinder",
            )
            previous = point

    def test_target_inside_cylinder_is_adjusted_not_refused(self):
        inside = np.array([0.1, 0.0, 1.0])
        route = self.router.plan(LEFT, inside)
        self.assertTrue(route.target_adjusted)
        self.assertGreater(
            float(
                np.linalg.norm(
                    route.effective_target[:2] - self.keepout.center
                )
            ),
            self.keepout.radius_m + self.keepout.clearance_m,
        )
        self.assertGreater(route.size, 0)

    def test_target_exactly_on_inflated_boundary_is_moved_outside(self):
        # radius + clearance = 0.6 exactly.
        route = self.router.plan(LEFT, np.array([0.6, 0.0, 1.0]))
        self.assertTrue(route.target_adjusted)
        self.assertGreater(
            float(
                np.linalg.norm(
                    route.effective_target[:2] - self.keepout.center
                )
            ),
            self.keepout.radius_m + self.keepout.clearance_m,
        )

    def test_short_over_the_top_route_is_selected_when_appropriate(self):
        low = _keepout(z_max_m=0.2)
        router = CylinderRouter(low)
        start = np.array([-1.0, 0.0, 0.1])
        target = np.array([1.0, 0.0, 0.1])
        route = router.plan(start, target)
        self.assertEqual(route.kind, CylinderRouteKind.OVER)
        previous = start
        for point in route.waypoints:
            self.assertFalse(router.segment_intersects(previous, point))
            previous = point

    def test_disabled_keepout_preserves_direct_motion(self):
        route = CylinderRouter(_keepout(enabled=False)).plan(LEFT, RIGHT)
        self.assertEqual(route.kind, CylinderRouteKind.DIRECT)
        self.assertEqual(route.size, 1)
        np.testing.assert_allclose(route.waypoints[0], RIGHT)

    def test_follower_commands_a_detour_then_advances(self):
        follower = CylinderRouteFollower(self.keepout)
        follower.reset(LEFT)
        follower.set_target(LEFT, RIGHT)
        waypoint = follower.update(LEFT)
        self.assertGreater(float(np.linalg.norm(waypoint - RIGHT)), 1e-6)
        waypoint = follower.update(waypoint)
        self.assertTrue(np.all(np.isfinite(waypoint)))


class RouteKindSelectionTest(unittest.TestCase):
    """Direct, clockwise, counter-clockwise and over, each explicitly."""

    def test_clear_segment_is_direct_with_one_waypoint(self):
        router = CylinderRouter(_keepout())
        start = np.array([-1.0, -1.0, 1.0])
        target = np.array([-1.0, 1.0, 1.0])
        route = router.plan(start, target)
        self.assertEqual(route.kind, CylinderRouteKind.DIRECT)
        self.assertEqual(route.size, 1)
        self.assertFalse(route.target_adjusted)

    def test_counter_clockwise_and_clockwise_are_both_reachable(self):
        """Mirrored targets must pick mirrored arc senses."""
        router = CylinderRouter(_keepout())
        kinds = set()
        for target_y in (0.35, -0.35):
            start = np.array([-1.0, -target_y, 1.0])
            target = np.array([1.0, target_y, 1.0])
            kinds.add(route_kind_name(router.plan(start, target).kind))
        self.assertEqual(kinds, {"clockwise", "counter-clockwise"})

    def test_over_route_uses_the_same_top_height_rule(self):
        low = _keepout(z_max_m=0.2, clearance_m=0.1)
        router = CylinderRouter(low)
        start = np.array([-1.0, 0.0, 0.1])
        target = np.array([1.0, 0.0, 0.1])
        route = router.plan(start, target)
        self.assertEqual(route.kind, CylinderRouteKind.OVER)
        # C++ OverCandidate: max(z_max + clearance + 0.05, start.z, target.z).
        expected_top = max(0.2 + 0.1 + 0.05, 0.1, 0.1)
        self.assertAlmostEqual(route.waypoints[0][2], expected_top, places=12)
        self.assertAlmostEqual(route.waypoints[1][2], expected_top, places=12)
        np.testing.assert_allclose(route.waypoints[-1], target)

    def test_shortest_candidate_wins(self):
        """The chosen route is no longer than any candidate it beat."""
        router = CylinderRouter(_keepout())
        start = np.array([-1.0, 0.05, 1.0])
        target = np.array([1.0, -0.05, 1.0])
        route = router.plan(start, target)
        self.assertNotEqual(route.kind, CylinderRouteKind.DIRECT)
        straight = float(np.linalg.norm(target - start))
        self.assertGreater(route.length_m, straight)
        self.assertLess(route.length_m, 4.0 * straight)


class RouteGeometryInvariantTest(unittest.TestCase):
    """Every nominal waypoint segment must stay outside the inflated cylinder."""

    def test_all_segments_clear_across_a_sweep_of_crossing_requests(self):
        keepout = _keepout()
        router = CylinderRouter(keepout)
        checked = 0
        detours = 0
        for angle in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False):
            start = np.array(
                [1.4 * np.cos(angle), 1.4 * np.sin(angle), 1.0])
            for offset in (np.pi, 0.75 * np.pi, 1.25 * np.pi):
                target = np.array([
                    1.4 * np.cos(angle + offset),
                    1.4 * np.sin(angle + offset),
                    1.0,
                ])
                route = router.plan(start, target)
                if route.kind != CylinderRouteKind.DIRECT:
                    detours += 1
                previous = start
                for point in route.waypoints:
                    self.assertFalse(
                        router.segment_intersects(previous, point),
                        f"{route.kind} segment {previous} -> {point} "
                        "enters the inflated cylinder",
                    )
                    previous = point
                    checked += 1
        self.assertGreater(detours, 0, "sweep produced no detours")
        self.assertGreater(checked, 100)

    def test_chord_compensation_keeps_arc_waypoints_outside(self):
        """route_radius must exceed the inflated radius by the C++ formula."""
        keepout = _keepout()
        expected = (
            keepout.obstacle_radius_m / np.cos(ARC_STEP_RAD / 2.0)
            + ROUTE_PADDING_M
        )
        self.assertAlmostEqual(keepout.route_radius_m, expected, places=15)
        self.assertGreater(keepout.route_radius_m, keepout.obstacle_radius_m)

    def test_route_never_exceeds_the_fixed_capacity(self):
        keepout = _keepout()
        router = CylinderRouter(keepout)
        route = router.plan(
            np.array([-1.0, 1e-9, 1.0]), np.array([-1.0, -1e-9, 1.0])
        )
        self.assertLessEqual(route.size, MAX_WAYPOINTS)


class WaypointAdvancementTest(unittest.TestCase):
    def setUp(self):
        self.keepout = _keepout()
        self.follower = CylinderRouteFollower(self.keepout)

    def test_advances_only_within_tolerance_and_reports_final_last(self):
        self.follower.reset(LEFT)
        self.follower.set_target(LEFT, RIGHT)
        route = self.follower.route
        self.assertGreater(route.size, 2)

        # Standing still at the start does not advance past waypoint 0.
        first = self.follower.update(LEFT)
        self.assertEqual(self.follower.index, 0)
        self.assertFalse(self.follower.at_final_waypoint())

        # Just outside tolerance: still no advance.
        just_outside = route.waypoints[0] + np.array(
            [self.keepout.waypoint_tolerance_m * 1.5, 0.0, 0.0])
        self.follower.update(just_outside)
        self.assertEqual(self.follower.index, 0)

        # Walking the route reaches, and stops at, the final waypoint.
        for point in route.waypoints:
            self.follower.update(point)
        self.assertTrue(self.follower.at_final_waypoint())
        self.assertEqual(self.follower.index, route.size - 1)
        np.testing.assert_allclose(
            self.follower.update(route.waypoints[-1]), RIGHT)
        self.assertIs(first is None, False)

    def test_arrival_is_not_reported_at_intermediate_waypoints(self):
        self.follower.reset(LEFT)
        self.follower.set_target(LEFT, RIGHT)
        route = self.follower.route
        for point in route.waypoints[:-1]:
            self.follower.update(point)
            if self.follower.index < route.size - 1:
                self.assertFalse(
                    self.follower.at_final_waypoint(),
                    "arrival reported before the final waypoint",
                )

    def test_route_is_only_rebuilt_on_a_new_target(self):
        self.follower.reset(LEFT)
        self.follower.set_target(LEFT, RIGHT)
        identity = id(self.follower.route)
        for _ in range(5):
            self.follower.update(LEFT)
        self.assertEqual(id(self.follower.route), identity)
        self.follower.set_target(LEFT, np.array([1.0, 0.5, 1.0]))
        self.assertNotEqual(id(self.follower.route), identity)
        self.assertEqual(self.follower.index, 0)


class OppositeSidesDemonstrationTest(unittest.TestCase):
    """The requested demonstration: start and target on opposite sides."""

    def test_opposite_sides_route_goes_around_and_stays_clear(self):
        keepout = _keepout()
        router = CylinderRouter(keepout)
        start = np.array([-1.0, 0.0, 1.0])
        target = np.array([1.0, 0.0, 1.0])

        self.assertTrue(router.segment_intersects(start, target))
        route = router.plan(start, target)

        self.assertIn(
            route.kind,
            (CylinderRouteKind.CLOCKWISE,
             CylinderRouteKind.COUNTER_CLOCKWISE),
        )
        self.assertGreater(route.size, 2)
        np.testing.assert_allclose(route.waypoints[-1], target)
        self.assertFalse(route.target_adjusted)

        # Detour is longer than the blocked straight line, and every arc
        # waypoint sits on the padded route radius.
        self.assertGreater(route.length_m, float(np.linalg.norm(target - start)))
        for point in route.waypoints[:-1]:
            radial = float(np.linalg.norm(point[:2] - keepout.center))
            self.assertGreaterEqual(radial, keepout.obstacle_radius_m)

        previous = start
        for point in route.waypoints:
            self.assertFalse(router.segment_intersects(previous, point))
            previous = point


class CppCrossCheckTest(unittest.TestCase):
    """Compare the port against the compiled C++ implementation."""

    def setUp(self):
        if _reference_binary() is None:
            self.skipTest(
                "C++ reference unavailable (needs the basic_control tree, "
                "g++ and Eigen)"
            )

    def _compare(self, cases):
        reference = _reference_routes(cases)
        self.assertIsNotNone(reference)
        worst = 0.0
        for case, expected in zip(cases, reference):
            keepout = CylinderKeepout(
                enabled=bool(case[0]),
                center_xy_m=(case[1], case[2]),
                radius_m=case[3],
                z_min_m=case[4],
                z_max_m=case[5],
                clearance_m=case[6],
                waypoint_tolerance_m=case[7],
            )
            router = CylinderRouter(keepout)
            start = np.array(case[8:11], dtype=float)
            target = np.array(case[11:14], dtype=float)
            route = router.plan(start, target)

            self.assertEqual(
                route_kind_name(route.kind), expected["kind"],
                f"route kind differs for {case}")
            self.assertEqual(
                route.size, expected["size"],
                f"waypoint count differs for {case}")
            self.assertEqual(
                bool(route.target_adjusted), expected["target_adjusted"],
                f"target_adjusted differs for {case}")
            self.assertEqual(
                router.segment_intersects(start, target),
                expected["segment_intersects"],
                f"segment_intersects differs for {case}")

            worst = max(worst, float(np.max(np.abs(
                route.effective_target
                - np.array(expected["effective_target"], dtype=float)))))
            np.testing.assert_allclose(
                route.effective_target,
                np.array(expected["effective_target"], dtype=float),
                atol=1e-12, rtol=0.0,
                err_msg=f"effective target differs for {case}")
            for index, (got, want) in enumerate(
                zip(route.waypoints, expected["waypoints"])
            ):
                worst = max(worst, float(np.max(np.abs(
                    got - np.array(want, dtype=float)))))
                np.testing.assert_allclose(
                    got, np.array(want, dtype=float),
                    atol=1e-12, rtol=0.0,
                    err_msg=f"waypoint {index} differs for {case}")
            self.assertAlmostEqual(
                route.length_m, expected["length_m"], places=12)
        return worst

    def test_matches_cpp_on_the_ported_scenarios(self):
        keepout = _keepout()
        low = _keepout(z_max_m=0.2)
        disabled = _keepout(enabled=False)
        cases = [
            _case(keepout, LEFT, RIGHT),                      # detour
            _case(keepout, LEFT, [0.1, 0.0, 1.0]),            # inside
            _case(keepout, LEFT, [0.6, 0.0, 1.0]),            # exact boundary
            _case(keepout, [-1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]),  # direct
            _case(low, [-1.0, 0.0, 0.1], [1.0, 0.0, 0.1]),    # over
            _case(disabled, LEFT, RIGHT),                     # disabled
            _case(keepout, [-1.0, 0.35, 1.0], [1.0, -0.35, 1.0]),
            _case(keepout, [-1.0, -0.35, 1.0], [1.0, 0.35, 1.0]),
        ]
        self._compare(cases)

    def test_matches_cpp_across_an_offset_centre_sweep(self):
        """Non-zero centre, mixed heights, both senses, over and direct."""
        cases = []
        for center in ((0.0, 0.0), (0.15, -0.22)):
            for z_max in (2.0, 0.2):
                keepout = _keepout(center_xy_m=center, z_max_m=z_max)
                for start in ([-1.0, 0.0, 1.0], [-1.0, 0.3, 1.0],
                              [0.8, 0.8, 0.5], [-1.0, 0.0, 0.1]):
                    for target in ([1.0, 0.0, 1.0], [0.1, 0.0, 1.0],
                                   [1.0, 0.4, 1.6], [0.0, 0.0, 1.0],
                                   [-0.9, 0.2, 0.9], [1.0, 0.0, 0.1]):
                        cases.append(_case(keepout, start, target))
        worst = self._compare(cases)
        self.assertLess(worst, 1e-12)

    def test_cpp_reference_exercises_every_route_kind(self):
        keepout = _keepout()
        low = _keepout(z_max_m=0.2)
        cases = [
            _case(keepout, [-1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]),
            _case(keepout, [-1.0, 0.35, 1.0], [1.0, -0.35, 1.0]),
            _case(keepout, [-1.0, -0.35, 1.0], [1.0, 0.35, 1.0]),
            _case(low, [-1.0, 0.0, 0.1], [1.0, 0.0, 0.1]),
        ]
        kinds = {row["kind"] for row in _reference_routes(cases)}
        self.assertEqual(
            kinds,
            {"direct", "clockwise", "counter-clockwise", "over"},
        )


if __name__ == "__main__":
    unittest.main()
